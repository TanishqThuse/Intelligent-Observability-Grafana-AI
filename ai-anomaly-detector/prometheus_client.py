"""
Prometheus Client — Queries live metrics from Prometheus using PromQL.
Connects to the Prometheus instance running inside the Docker network.
"""

import httpx
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090/prometheus")


# ── Predefined PromQL Queries ──────────────────────────────────────────────
METRIC_QUERIES = {
    "request_rate": 'sum(rate(http_server_duration_seconds_count[1m]))',
    "cpu_usage": 'sum(rate(process_cpu_seconds_total[1m]))',
    "memory_usage": 'sum(process_resident_memory_bytes)',
    "error_rate": 'sum(rate(http_server_duration_seconds_count{http_status_code=~"5.."}[1m]))',
    "latency_p95": 'histogram_quantile(0.95, sum(rate(http_server_duration_seconds_bucket[5m])) by (le))',
    "request_rate_per_service": 'sum by (service_name) (rate(http_server_duration_seconds_count[1m]))',
    "cpu_per_service": 'sum by (service_name) (rate(process_cpu_seconds_total[1m]))',
    "memory_per_service": 'sum by (service_name) (process_resident_memory_bytes)',
}


async def query_prometheus(query: str) -> dict:
    """Execute an instant PromQL query."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
        return {"status": "error", "error": str(e), "data": {"result": []}}


async def query_prometheus_range(query: str, start: datetime, end: datetime, step: str = "15s") -> dict:
    """Execute a range PromQL query for time-series data."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start.timestamp(),
                    "end": end.timestamp(),
                    "step": step,
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Prometheus range query failed: {e}")
        return {"status": "error", "error": str(e), "data": {"result": []}}


async def get_metric_timeseries(metric_name: str, duration_minutes: int = 30, step: str = "15s") -> dict:
    """
    Fetch a time-series for a named metric.
    Returns {"timestamps": [...], "values": [...], "metric_name": "..."}
    """
    query = METRIC_QUERIES.get(metric_name)
    if not query:
        return {"error": f"Unknown metric: {metric_name}", "timestamps": [], "values": []}

    end = datetime.utcnow()
    start = end - timedelta(minutes=duration_minutes)

    result = await query_prometheus_range(query, start, end, step)

    timestamps = []
    values = []

    if result.get("status") == "success":
        results = result["data"].get("result", [])
        if results:
            # Take the first result series
            series = results[0]
            for point in series.get("values", []):
                ts, val = point
                timestamps.append(float(ts))
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    values.append(0.0)

    return {
        "metric_name": metric_name,
        "timestamps": timestamps,
        "values": values,
        "count": len(values),
    }


async def get_all_metrics_snapshot() -> dict:
    """Get current instant values for all key metrics."""
    snapshot = {}
    for name, query in METRIC_QUERIES.items():
        # skip per-service queries for the snapshot
        if "per_service" in name:
            continue
        result = await query_prometheus(query)
        value = 0.0
        if result.get("status") == "success":
            results = result["data"].get("result", [])
            if results:
                try:
                    value = float(results[0]["value"][1])
                except (ValueError, TypeError, IndexError, KeyError):
                    value = 0.0
        snapshot[name] = round(value, 6)
    return snapshot


async def get_service_health() -> list:
    """Check which services are up/down using the 'up' metric."""
    result = await query_prometheus("up")
    services = []
    if result.get("status") == "success":
        for item in result["data"].get("result", []):
            labels = item.get("metric", {})
            value = item.get("value", [0, "0"])
            services.append({
                "job": labels.get("job", "unknown"),
                "instance": labels.get("instance", "unknown"),
                "status": "up" if value[1] == "1" else "down",
            })
    return services
