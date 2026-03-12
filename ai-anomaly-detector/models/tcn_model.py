"""
Temporal Convolutional Network (TCN) — Detects long-range temporal patterns
using causal dilated convolutions.

Architecture:
  - Causal convolutions: ensures no future data leaks
  - Dilated convolutions: exponentially increasing receptive field
  - Residual connections: stable gradient flow
  - Prediction-based anomaly scoring
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class CausalConv1d(nn.Module):
    """Causal convolution — pads only on the left to prevent future leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove the future-padding from the right
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN residual block with causal dilated convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for time-series anomaly detection.

    Uses stacked causal dilated convolutions with exponentially increasing
    dilation factors to capture long-range dependencies.
    Detects anomalies via prediction error.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Stack of TCN blocks with increasing dilation
        layers = []
        for i in range(num_levels):
            in_ch = input_dim if i == 0 else hidden_dim
            dilation = 2 ** i  # 1, 2, 4, 8...
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            predicted: (batch, seq_len, input_dim)
        """
        # TCN expects (batch, channels, seq_len)
        out = x.permute(0, 2, 1)
        out = self.network(out)
        # Back to (batch, seq_len, channels)
        out = out.permute(0, 2, 1)
        predicted = self.fc(out)
        return predicted

    def compute_anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """Compute prediction error as anomaly score."""
        self.eval()
        with torch.no_grad():
            predicted = self.forward(x)
            # Shift: predict next step from current
            # Compare predicted[:-1] with actual[1:]
            mse = torch.mean((x - predicted) ** 2, dim=(1, 2))
            return mse.cpu().numpy()

    def get_prediction(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Get actual vs predicted for visualization."""
        self.eval()
        with torch.no_grad():
            predicted = self.forward(x)
            return x.cpu().numpy(), predicted.cpu().numpy()


def create_tcn_model(
    input_dim: int = 1,
    hidden_dim: int = 64,
    num_levels: int = 4,
    kernel_size: int = 3,
) -> TCNModel:
    """Factory function to create a TCN model."""
    return TCNModel(input_dim, hidden_dim, num_levels, kernel_size)
