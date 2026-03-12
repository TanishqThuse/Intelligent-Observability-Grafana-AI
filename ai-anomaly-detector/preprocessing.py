"""
Data Preprocessing Module — Prepares time-series data for deep learning models.

Implements:
  - Sliding window segmentation
  - Min-Max normalization
  - Missing value interpolation
  - Sequence generation for PyTorch DataLoaders
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ── Normalization ──────────────────────────────────────────────────────────

class MinMaxScaler:
    """Min-Max normalization to [0, 1] range."""

    def __init__(self):
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        self.min_val = float(np.nanmin(data))
        self.max_val = float(np.nanmax(data))
        if self.max_val == self.min_val:
            self.max_val = self.min_val + 1.0  # prevent division by zero
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_val is None:
            self.fit(data)
        return (data - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.max_val - self.min_val) + self.min_val

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)


# ── Missing Value Handling ─────────────────────────────────────────────────

def interpolate_missing(values: np.ndarray) -> np.ndarray:
    """Linear interpolation for NaN/missing values."""
    nans = np.isnan(values)
    if not np.any(nans):
        return values

    indices = np.arange(len(values))
    valid = ~nans
    if np.sum(valid) < 2:
        return np.nan_to_num(values, nan=0.0)

    values[nans] = np.interp(indices[nans], indices[valid], values[valid])
    return values


# ── Sliding Window Segmentation ───────────────────────────────────────────

def create_sequences(data: np.ndarray, window_size: int = 30, step: int = 1) -> np.ndarray:
    """
    Create sliding window sequences from 1D time-series data.

    Args:
        data: 1D array of values
        window_size: length of each sequence
        step: stride between windows

    Returns:
        2D array of shape (num_sequences, window_size)
    """
    if len(data) < window_size:
        # Pad with the mean if data is too short
        pad_size = window_size - len(data)
        mean_val = np.nanmean(data) if len(data) > 0 else 0.0
        data = np.concatenate([np.full(pad_size, mean_val), data])

    sequences = []
    for i in range(0, len(data) - window_size + 1, step):
        sequences.append(data[i:i + window_size])

    return np.array(sequences)


# ── PyTorch Dataset ────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time-series sequences."""

    def __init__(self, sequences: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


# ── Full Pipeline ──────────────────────────────────────────────────────────

def preprocess_pipeline(
    values: list,
    window_size: int = 30,
    step: int = 1,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Full preprocessing pipeline:
    1. Convert to numpy array
    2. Interpolate missing values
    3. Normalize to [0, 1]
    4. Create sliding window sequences

    Returns:
        (sequences, scaler) — sequences ready for model input, and the scaler for inverse transform
    """
    data = np.array(values, dtype=np.float64)

    # Handle missing values
    data = interpolate_missing(data)

    # Normalize
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    # Create sequences
    sequences = create_sequences(data_normalized, window_size=window_size, step=step)

    logger.info(f"Preprocessed {len(values)} points → {len(sequences)} sequences (window={window_size})")
    return sequences, scaler


def create_dataloader(sequences: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Create a PyTorch DataLoader from sequences."""
    dataset = TimeSeriesDataset(sequences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
