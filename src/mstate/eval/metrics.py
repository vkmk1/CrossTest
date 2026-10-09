"""Pre-registered metrics (EXPERIMENTS.md §4). Row-level losses; aggregate with evaluate.py."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PRIMARY = {"lrv": "mse"}


def mse(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Squared error on log RV."""
    return (y - yhat) ** 2


def qlike(y_log: np.ndarray, yhat_log: np.ndarray) -> np.ndarray:
    """QLIKE on variances, from log RV: RV/RV_hat - log(RV/RV_hat) - 1 (>= 0, 0 iff equal)."""
    d = y_log - yhat_log
    return np.exp(d) - d - 1.0


def pinball(y: np.ndarray, q_pred: np.ndarray, taus: Sequence[float]) -> np.ndarray:
    """Mean pinball over quantiles: y ``[n]``, q_pred ``[n, Q]`` -> ``[n]``."""
    t = np.asarray(taus)[None, :]
    u = y[:, None] - q_pred
    return np.mean(np.maximum(t * u, (t - 1) * u), axis=1)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return ((y >= lo) & (y <= hi)).astype(np.float64)


ROW_METRICS = {"mse": mse, "qlike": qlike}
