"""
Anomaly Engine — Unified interface for all AI models with dynamic thresholding.

Provides:
  - Model management (LSTM, TCN, Transformer)
  - Training pipeline
  - Inference with anomaly scoring
  - Dynamic threshold calculation
  - Severity classification (normal / warning / critical)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from models.lstm_autoencoder import create_lstm_autoencoder, LSTMAutoencoder
from models.tcn_model import create_tcn_model, TCNModel
from models.transformer_model import create_transformer_model, TransformerAnomalyDetector
from preprocessing import preprocess_pipeline, MinMaxScaler, create_sequences

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/saved_models")
WINDOW_SIZE = 30

# ── Anomaly History Store ──────────────────────────────────────────────────
anomaly_history: List[dict] = []


class AnomalyEngine:
    """
    Unified anomaly detection engine that manages all three models.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create models
        self.models: Dict[str, nn.Module] = {
            "lstm_autoencoder": create_lstm_autoencoder(input_dim=1, hidden_dim=64, latent_dim=32, num_layers=2),
            "tcn": create_tcn_model(input_dim=1, hidden_dim=64, num_levels=4, kernel_size=3),
            "transformer": create_transformer_model(input_dim=1, d_model=64, nhead=4, num_layers=3),
        }

        # Move to device
        for name, model in self.models.items():
            self.models[name] = model.to(self.device)

        # Dynamic thresholds per metric (learned during training)
        self.thresholds: Dict[str, Dict[str, float]] = {}

        # Scalers per metric
        self.scalers: Dict[str, MinMaxScaler] = {}

        # Training state
        self.is_trained: Dict[str, bool] = {name: False for name in self.models}
        self.active_model: str = "lstm_autoencoder"

        # Load saved models if they exist
        self._load_models()

    def _load_models(self):
        """Load pre-trained model weights if available."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(MODEL_DIR, f"{name}.pt")
            if os.path.exists(path):
                try:
                    model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                    self.is_trained[name] = True
                    logger.info(f"Loaded saved model: {name}")
                except Exception as e:
                    logger.warning(f"Could not load model {name}: {e}")

        # Load thresholds
        threshold_path = os.path.join(MODEL_DIR, "thresholds.json")
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                self.thresholds = json.load(f)

    def _save_model(self, name: str):
        """Save model weights to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        torch.save(self.models[name].state_dict(), path)
        logger.info(f"Saved model: {name} → {path}")

    def _save_thresholds(self):
        """Save dynamic thresholds to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "thresholds.json")
        with open(path, "w") as f:
            json.dump(self.thresholds, f, indent=2)

    # ── Training ───────────────────────────────────────────────────────────

    def train_model(
        self,
        model_name: str,
        metric_name: str,
        values: List[float],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict:
        """
        Train a model on historical data for a specific metric.

        Returns training stats.
        """
        if model_name not in self.models:
            return {"error": f"Unknown model: {model_name}"}

        model = self.models[model_name]
        model.train()

        # Preprocess
        sequences, scaler = preprocess_pipeline(values, window_size=WINDOW_SIZE)
        self.scalers[metric_name] = scaler

        if len(sequences) < 2:
            return {"error": "Not enough data for training (need at least 60 data points)"}

        # Convert to tensors
        X = torch.FloatTensor(sequences).unsqueeze(-1).to(self.device)  # (N, window, 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(X.size(0))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, X.size(0), batch_size):
                batch = X[perm[i:i + batch_size]]
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

        # Compute threshold from training data
        model.eval()
        with torch.no_grad():
            output = model(X)
            errors = torch.mean((X - output) ** 2, dim=(1, 2)).cpu().numpy()

        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))

        self.thresholds[metric_name] = {
            "mean": mean_error,
            "std": std_error,
            "warning": mean_error + 2 * std_error,
            "critical": mean_error + 3 * std_error,
            "p95": float(np.percentile(errors, 95)),
            "p99": float(np.percentile(errors, 99)),
        }

        self.is_trained[model_name] = True
        self._save_model(model_name)
        self._save_thresholds()

        logger.info(f"Training complete: {model_name} on {metric_name}, final_loss={losses[-1]:.6f}")

        return {
            "model": model_name,
            "metric": metric_name,
            "epochs": epochs,
            "final_loss": round(losses[-1], 6),
            "threshold_warning": round(self.thresholds[metric_name]["warning"], 6),
            "threshold_critical": round(self.thresholds[metric_name]["critical"], 6),
            "training_samples": len(sequences),
        }

    # ── Inference ──────────────────────────────────────────────────────────

    def detect_anomalies(
        self,
        metric_name: str,
        values: List[float],
        model_name: Optional[str] = None,
    ) -> Dict:
        """
        Run anomaly detection on a metric's time-series data.

        Returns anomaly scores and classifications.
        """
        use_model = model_name or self.active_model
        model = self.models.get(use_model)
        if model is None:
            return {"error": f"Unknown model: {use_model}"}

        model.eval()

        # If we have a saved scaler for this metric, use it; otherwise fit a new one
        if metric_name in self.scalers:
            scaler = self.scalers[metric_name]
            data = np.array(values, dtype=np.float64)
            data_normalized = scaler.transform(data)
            sequences = create_sequences(data_normalized, window_size=WINDOW_SIZE)
        else:
            sequences, scaler = preprocess_pipeline(values, window_size=WINDOW_SIZE)

        if len(sequences) == 0:
            return {"error": "Not enough data for detection"}

        # Run inference
        X = torch.FloatTensor(sequences).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            output = model(X)
            errors = torch.mean((X - output) ** 2, dim=(1, 2)).cpu().numpy()

        # Get threshold
        thresholds = self.thresholds.get(metric_name, {
            "warning": 0.01,
            "critical": 0.05,
            "mean": 0.005,
            "std": 0.003,
        })

        # Classify each sequence
        classifications = []
        for score in errors:
            if score >= thresholds["critical"]:
                classifications.append("critical")
            elif score >= thresholds["warning"]:
                classifications.append("warning")
            else:
                classifications.append("normal")

        # Latest anomaly score
        latest_score = float(errors[-1])
        latest_severity = classifications[-1]

        # Get actual vs reconstructed for the latest window
        actual_window = values[-WINDOW_SIZE:] if len(values) >= WINDOW_SIZE else values
        with torch.no_grad():
            last_seq = X[-1:] 
            last_reconstructed = model(last_seq).cpu().numpy().flatten()
            last_original = last_seq.cpu().numpy().flatten()

        # Record in history
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric": metric_name,
            "model": use_model,
            "anomaly_score": round(latest_score, 6),
            "severity": latest_severity,
            "threshold_warning": round(thresholds["warning"], 6),
            "threshold_critical": round(thresholds["critical"], 6),
        }
        anomaly_history.append(record)
        # Keep only last 1000 records
        if len(anomaly_history) > 1000:
            anomaly_history.pop(0)

        return {
            "metric_name": metric_name,
            "model_used": use_model,
            "is_trained": self.is_trained[use_model],
            "latest_anomaly_score": round(latest_score, 6),
            "severity": latest_severity,
            "scores": [round(float(s), 6) for s in errors],
            "classifications": classifications,
            "thresholds": thresholds,
            "actual_window": [round(float(v), 4) for v in last_original],
            "reconstructed_window": [round(float(v), 4) for v in last_reconstructed],
        }

    def get_history(self, limit: int = 100) -> List[dict]:
        """Get recent anomaly detection history."""
        return anomaly_history[-limit:]

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            "active_model": self.active_model,
            "models": {
                name: {
                    "is_trained": self.is_trained[name],
                    "parameters": sum(p.numel() for p in model.parameters()),
                }
                for name, model in self.models.items()
            },
            "thresholds": self.thresholds,
            "history_count": len(anomaly_history),
            "device": str(self.device),
        }


# Global singleton
engine = AnomalyEngine()
