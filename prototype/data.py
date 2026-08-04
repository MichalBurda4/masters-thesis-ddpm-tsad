"""Synthetic time series: normal patterns + injected anomalies."""

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_normal_series(length: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, length)
    trend = 0.02 * np.arange(length)
    seasonal = 0.8 * np.sin(t) + 0.3 * np.sin(2.5 * t + rng.uniform(0, np.pi))
    noise = rng.normal(0, 0.05, length)
    return (trend + seasonal + noise).astype(np.float32)


def inject_anomaly(series: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (corrupted series, binary mask: 1 = anomaly)."""
    out = series.copy()
    mask = np.zeros(len(series), dtype=np.float32)
    n_spikes = rng.integers(1, 4)
    for _ in range(n_spikes):
        start = rng.integers(0, max(1, len(series) - 20))
        width = rng.integers(3, 15)
        end = min(len(series), start + width)
        kind = rng.integers(0, 3)
        if kind == 0:
            out[start:end] += rng.uniform(1.5, 3.0)
        elif kind == 1:
            out[start:end] *= rng.uniform(0.2, 0.5)
        else:
            out[start:end] = rng.uniform(-2, 2, end - start)
        mask[start:end] = 1.0
    return out, mask


class NormalSeriesDataset(Dataset):
    """Training data: only normal series (DDPM learns 'healthy' patterns)."""

    def __init__(self, n_samples: int = 500, length: int = 128, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.series = np.stack(
            [generate_normal_series(length, rng) for _ in range(n_samples)]
        )
        # Per-dataset normalization for stable training
        self.mean = self.series.mean()
        self.std = self.series.std() + 1e-6
        self.series = (self.series - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.series[idx]
        return torch.from_numpy(x).unsqueeze(0)  # (1, L)


def make_test_set(
    n_normal: int = 30,
    n_anomaly: int = 30,
    length: int = 128,
    seed: int = 123,
) -> dict:
    rng = np.random.default_rng(seed)
    train_stats = NormalSeriesDataset(n_samples=100, length=length, seed=42)
    mean, std = train_stats.mean, train_stats.std

    samples, labels, masks = [], [], []
    for _ in range(n_normal):
        s = generate_normal_series(length, rng)
        s = (s - mean) / std
        samples.append(s)
        labels.append(0)
        masks.append(np.zeros(length, dtype=np.float32))

    for _ in range(n_anomaly):
        s = generate_normal_series(length, rng)
        s = (s - mean) / std
        s, m = inject_anomaly(s, rng)
        samples.append(s)
        labels.append(1)
        masks.append(m)

    return {
        "series": np.stack(samples),
        "label": np.array(labels),
        "mask": np.stack(masks),
        "mean": mean,
        "std": std,
    }
