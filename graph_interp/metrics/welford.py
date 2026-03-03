"""Welford online mean/variance accumulator for arbitrary-shape numpy arrays."""
from __future__ import annotations

import numpy as np


class WelfordAccumulator:
    """Numerically stable online computation of mean and variance.

    Each call to :meth:`update` ingests a single sample (a numpy array of
    arbitrary but consistent shape).  After all samples have been added,
    :meth:`finalize` returns ``(mean, std, count)``.
    """

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: np.ndarray | None = None
        self.M2: np.ndarray | None = None

    def update(self, x: np.ndarray) -> None:
        """Add a new sample.  *x* can be any shape; all samples must share the same shape."""
        x = np.asarray(x, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.M2 = np.zeros_like(x)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def finalize(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Return ``(mean, std, count)``.  *std* is NaN if ``count < 2``."""
        if self.count < 2:
            std = np.full_like(self.mean, np.nan) if self.mean is not None else np.array(np.nan)
            mean = self.mean if self.mean is not None else np.array(0.0)
            return mean, std, self.count
        var = self.M2 / (self.count - 1)
        return self.mean.copy(), np.sqrt(np.clip(var, 0.0, None)), self.count
