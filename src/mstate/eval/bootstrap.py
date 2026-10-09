"""Moving-block bootstrap over time (EXPERIMENTS.md §4): 1,000 resamples, block 20 (daily)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def block_indices(n: int, block_len: int, n_resamples: int, seed: int = 0) -> np.ndarray:
    """``[n_resamples, n]`` time indices built from blocks of consecutive dates."""
    rng = np.random.default_rng(seed)
    b = min(block_len, n)
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_resamples, -1)
    return idx[:, :n]


@dataclass
class Interval:
    value: float
    lo: float
    hi: float

    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


def mean_ci(x: np.ndarray, block_len: int = 20, n_resamples: int = 1000, alpha: float = 0.05,
            seed: int = 0) -> Interval:
    """CI for the mean of a daily series."""
    idx = block_indices(len(x), block_len, n_resamples, seed)
    stats = x[idx].mean(1)
    return Interval(float(x.mean()), *np.quantile(stats, [alpha / 2, 1 - alpha / 2]).tolist())


def pct_reduction_ci(loss: np.ndarray, base: np.ndarray, block_len: int = 20,
                     n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """Percent loss reduction of ``loss`` vs ``base`` (paired daily series, same blocks).

    ``100 * (mean(base) - mean(loss)) / mean(base)``; positive means ``loss`` is better.
    """
    idx = block_indices(len(loss), block_len, n_resamples, seed)
    b, m = base[idx].mean(1), loss[idx].mean(1)
    stats = 100.0 * (b - m) / b
    point = 100.0 * (base.mean() - loss.mean()) / base.mean()
    return Interval(float(point), *np.quantile(stats, [alpha / 2, 1 - alpha / 2]).tolist())
