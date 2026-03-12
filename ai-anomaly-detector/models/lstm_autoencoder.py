"""
LSTM Autoencoder Model — Learns normal time-series patterns and detects
anomalies based on reconstruction error.

Architecture:
  Encoder: LSTM layers compress the input sequence into a latent representation
  Decoder: LSTM layers reconstruct the original sequence from the latent space
  Anomaly Score: MSE between input and reconstruction (high error = anomaly)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class LSTMEncoder(nn.Module):
    """LSTM Encoder — compresses time-series into latent vector."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        # x: (batch, seq_len, input_dim)
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, (hidden, cell)


class LSTMDecoder(nn.Module):
    """LSTM Decoder — reconstructs time-series from latent vector."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Tuple) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        outputs, _ = self.lstm(x, hidden)
        reconstructed = self.fc(outputs)
        return reconstructed


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder for time-series anomaly detection.

    The model learns to reconstruct 'normal' sequences.
    High reconstruction error indicates an anomaly.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers)
        self.encoder_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_decoder = nn.Linear(latent_dim, hidden_dim)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, num_layers, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) — input time-series sequences
        Returns:
            reconstructed: (batch, seq_len, input_dim) — reconstructed sequences
        """
        batch_size, seq_len, _ = x.shape

        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Compress to latent space
        latent = self.encoder_to_latent(hidden[-1])  # (batch, latent_dim)

        # Expand from latent space
        decoder_hidden = self.latent_to_decoder(latent).unsqueeze(0)  # (1, batch, hidden_dim)
        decoder_hidden = decoder_hidden.repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        decoder_cell = torch.zeros_like(decoder_hidden)

        # Decode — reconstruct the original sequence
        reconstructed = self.decoder(x, (decoder_hidden, decoder_cell))
        return reconstructed

    def compute_anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute per-sequence anomaly score (MSE reconstruction error).

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            scores: (batch,) — anomaly score for each sequence
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # MSE per sequence
            mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
            return mse.cpu().numpy()

    def get_reconstruction(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both the original and reconstructed sequences for visualization.

        Returns:
            (original, reconstructed) as numpy arrays
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            return x.cpu().numpy(), reconstructed.cpu().numpy()


def create_lstm_autoencoder(
    input_dim: int = 1,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    num_layers: int = 2,
) -> LSTMAutoencoder:
    """Factory function to create an LSTM Autoencoder."""
    model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, num_layers)
    return model
