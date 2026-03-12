"""
Transformer-Based Time-Series Model — Uses self-attention to capture
complex temporal dependencies in metric data.

Architecture:
  - Positional Encoding: preserves temporal order
  - Multi-Head Self-Attention: captures global dependencies
  - Feed-Forward layers: per-position transformations
  - Anomaly detection via prediction/reconstruction error
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Inject positional information into the sequence embeddings."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerAnomalyDetector(nn.Module):
    """
    Transformer model for time-series anomaly detection.

    Architecture:
      Input → Linear Embedding → Positional Encoding → Transformer Encoder → Reconstruction Head

    The model learns to reconstruct normal time-series patterns.
    Anomalies produce high reconstruction error.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reconstruction head
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            reconstructed: (batch, seq_len, input_dim)
        """
        # Project to d_model dimensions
        embedded = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        embedded = self.pos_encoder(embedded)

        # Transformer encoding
        encoded = self.transformer_encoder(embedded)  # (batch, seq_len, d_model)

        # Reconstruct
        reconstructed = self.output_projection(encoded)  # (batch, seq_len, input_dim)
        return reconstructed

    def compute_anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """Compute reconstruction error as anomaly score."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
            return mse.cpu().numpy()

    def get_reconstruction(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Get actual vs reconstructed for visualization."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            return x.cpu().numpy(), reconstructed.cpu().numpy()


def create_transformer_model(
    input_dim: int = 1,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
    dim_feedforward: int = 128,
) -> TransformerAnomalyDetector:
    """Factory function to create a Transformer anomaly detector."""
    return TransformerAnomalyDetector(input_dim, d_model, nhead, num_layers, dim_feedforward)
