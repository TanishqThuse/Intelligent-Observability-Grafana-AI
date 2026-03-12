"""
FastAPI Backend — Main application for the Intelligent Anomaly Detection System.

Endpoints:
  GET  /api/metrics              — Fetch live metrics from Prometheus
  GET  /api/anomaly-detection    — Run anomaly detection on latest data
  POST /api/train-model          — Train a model on historical data
  GET  /api/anomaly-history      — Get past anomaly records
  GET  /api/system-health        — Health of all monitored services
  GET  /api/engine-status        — ML engine status (models, thresholds)
  GET  /                         — Serve the web dashboard
"""

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging

from prometheus_client import (
    get_all_metrics_snapshot,
    get_metric_timeseries,
    get_service_health,
    METRIC_QUERIES,
)
from models.anomaly_engine import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligent Time-Series Anomaly Detection",
    description="AI-powered anomaly detection for server and application monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the dashboard static files
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")
if os.path.isdir(DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")


# ── Dashboard ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard."""
    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found</h1>")


# ── Metrics ────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
async def get_metrics():
    """Get current snapshot of all monitored metrics."""
    snapshot = await get_all_metrics_snapshot()
    return {"status": "ok", "metrics": snapshot}


@app.get("/api/metrics/{metric_name}")
async def get_metric_series(
    metric_name: str,
    duration: int = Query(default=30, description="Duration in minutes"),
    step: str = Query(default="15s", description="Query step interval"),
):
    """Get time-series data for a specific metric."""
    result = await get_metric_timeseries(metric_name, duration_minutes=duration, step=step)
    return {"status": "ok", **result}


@app.get("/api/metrics-list")
async def list_available_metrics():
    """List all available metric names."""
    return {"status": "ok", "metrics": list(METRIC_QUERIES.keys())}


# ── Anomaly Detection ─────────────────────────────────────────────────────

@app.get("/api/anomaly-detection")
async def detect_anomalies(
    metric: str = Query(default="request_rate", description="Metric name to analyze"),
    model: str = Query(default=None, description="Model to use (lstm_autoencoder, tcn, transformer)"),
    duration: int = Query(default=30, description="Duration in minutes of data to analyze"),
):
    """Run anomaly detection on a metric's latest data."""
    # Fetch time-series from Prometheus
    ts_data = await get_metric_timeseries(metric, duration_minutes=duration)

    if not ts_data["values"]:
        return {"status": "error", "error": "No data available from Prometheus for this metric"}

    # Run detection
    result = engine.detect_anomalies(metric, ts_data["values"], model_name=model)

    return {
        "status": "ok",
        "data_points": len(ts_data["values"]),
        "timestamps": ts_data["timestamps"],
        "raw_values": ts_data["values"],
        **result,
    }


@app.get("/api/anomaly-detection/all")
async def detect_all_anomalies(
    duration: int = Query(default=30, description="Duration in minutes"),
):
    """Run anomaly detection on all key metrics at once."""
    results = {}
    for metric_name in ["request_rate", "cpu_usage", "memory_usage", "latency_p95"]:
        ts_data = await get_metric_timeseries(metric_name, duration_minutes=duration)
        if ts_data["values"]:
            detection = engine.detect_anomalies(metric_name, ts_data["values"])
            results[metric_name] = {
                "anomaly_score": detection["latest_anomaly_score"],
                "severity": detection["severity"],
                "data_points": len(ts_data["values"]),
            }
        else:
            results[metric_name] = {
                "anomaly_score": 0,
                "severity": "no_data",
                "data_points": 0,
            }

    return {"status": "ok", "results": results}


# ── Training ───────────────────────────────────────────────────────────────

@app.post("/api/train-model")
async def train_model(
    metric: str = Query(default="request_rate", description="Metric to train on"),
    model: str = Query(default="lstm_autoencoder", description="Model to train"),
    duration: int = Query(default=60, description="Duration in minutes of training data"),
    epochs: int = Query(default=50, description="Training epochs"),
):
    """Train a model on historical metric data."""
    # Fetch historical data
    ts_data = await get_metric_timeseries(metric, duration_minutes=duration, step="15s")

    if not ts_data["values"] or len(ts_data["values"]) < 60:
        return {
            "status": "error",
            "error": f"Not enough data: got {len(ts_data.get('values', []))} points, need at least 60",
        }

    # Train
    result = engine.train_model(model, metric, ts_data["values"], epochs=epochs)

    return {"status": "ok", **result}


# ── History ────────────────────────────────────────────────────────────────

@app.get("/api/anomaly-history")
async def get_anomaly_history(
    limit: int = Query(default=100, description="Max records to return"),
):
    """Get recent anomaly detection history."""
    history = engine.get_history(limit=limit)
    return {"status": "ok", "count": len(history), "history": history}


# ── System Health ──────────────────────────────────────────────────────────

@app.get("/api/system-health")
async def system_health():
    """Check health of all monitored services."""
    services = await get_service_health()
    return {
        "status": "ok",
        "services": services,
        "total": len(services),
        "up": sum(1 for s in services if s["status"] == "up"),
        "down": sum(1 for s in services if s["status"] == "down"),
    }


@app.get("/api/engine-status")
async def engine_status():
    """Get ML engine status (models, thresholds, device)."""
    return {"status": "ok", **engine.get_status()}


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
