"""Panel windows for Stage 1 (LSM) training and context export.

Shapes: ``x [B, T, N, F]``, ``present [B, T, N]``, ``pool [B, N]``.

Inputs are the cross-sectionally normalized features ``xs_<f>`` (DATA.md §4.3) plus the
``mkt_*`` level features, the latter z-scored over time with train-period moments (they are
raw levels, e.g. median log dollar volume ~ 15). Universe for breadth N: assets with
``cap_rank <= N`` in the year of the window's last date; ``full`` means every asset present.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import torch

from mstate.data.datasets import DensePanel
from mstate.data.features import BASE_FEATURES
from mstate.data.splits import Splits, assert_no_test


def lsm_columns(features: Sequence[str]) -> list[str]:
    """Config names -> frame columns: base feature f -> xs_f; mkt_* unchanged."""
    return [f"xs_{f}" if f in BASE_FEATURES else f for f in features]


class LSMPanel:
    def __init__(self, df: pd.DataFrame, features: Sequence[str], lookback: int, splits: Splits,
                 universe: str | int = "full", allow_test: bool = False,
                 dtype: np.dtype = np.float32):
        if not allow_test:
            assert_no_test(df["date"], splits)
        self.cols = lsm_columns(features)
        self.lookback, self.splits = lookback, splits
        self.universe = universe
        panel = DensePanel.from_long(df, [*self.cols, "cap_rank"])
        self.calendar, self.permnos = panel.calendar, panel.permnos
        X = panel.stack(self.cols)  # [Td, Na, F]
        self.present = np.isfinite(X[..., self.cols.index("xs_ret")]) if "xs_ret" in self.cols \
            else np.isfinite(X).any(-1)
        tr = (self.calendar >= splits.train[0]) & (self.calendar <= splits.train[1])
        for j, c in enumerate(self.cols):
            if c.startswith("mkt_"):
                v = X[tr, :, j][self.present[tr]]
                X[..., j] = (X[..., j] - v.mean()) / (v.std() + 1e-8)
        self.finite = np.isfinite(X)
        self.X = np.nan_to_num(X).astype(dtype)
        self.cap_rank = panel.values["cap_rank"]

    # ------------------------------------------------------------------------------------------
    def members(self, t: int) -> np.ndarray:
        """Asset indices in the LSM universe at date index t."""
        pres = self.present[t]
        if self.universe == "full":
            return np.flatnonzero(pres)
        return np.flatnonzero(pres & (self.cap_rank[t] <= int(self.universe)))

    def end_dates(self, split: str) -> np.ndarray:
        """Window end indices t for a split. Train: [t-L+1, t] inside train (targets at <= t).
        Val: t inside val."""
        idx = np.arange(len(self.calendar))
        lo, hi = self.splits.range(split)
        if split == "train":
            ok = (idx - self.lookback + 1 >= 0) & (self.calendar >= lo) & (self.calendar <= hi)
            ok &= self.calendar[np.clip(idx - self.lookback + 1, 0, None)] >= lo
        else:
            ok = (idx - self.lookback + 1 >= 0) & (self.calendar >= lo) & (self.calendar <= hi)
        return idx[ok]

    def window(self, t: int, assets: np.ndarray) -> tuple[np.ndarray, ...]:
        """Left-padded window ending at t: x [T, n, F], present [T, n], finite [T, n, F]."""
        T = self.lookback
        lo = max(0, t - T + 1)
        pad = T - (t - lo + 1)
        x = self.X[lo:t + 1][:, assets]
        p = self.present[lo:t + 1][:, assets]
        f = self.finite[lo:t + 1][:, assets]
        if pad:
            x = np.concatenate([np.zeros((pad, *x.shape[1:]), x.dtype), x])
            p = np.concatenate([np.zeros((pad, p.shape[1]), bool), p])
            f = np.concatenate([np.zeros((pad, *f.shape[1:]), bool), f])
        return x, p, f

    def batches(self, split: str, batch_size: int, max_assets: int, shuffle: bool,
                rng: np.random.Generator, device: torch.device
                ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield (x [B,T,n,F], present [B,T,n], finite [B,T,n,F]); n padded within the batch."""
        ends = self.end_dates(split)
        if shuffle:
            ends = rng.permutation(ends)
        for s in range(0, len(ends), batch_size):
            parts = []
            for t in ends[s:s + batch_size]:
                a = self.members(int(t))
                if len(a) > max_assets:
                    a = np.sort(rng.choice(a, max_assets, replace=False))
                parts.append(self.window(int(t), a))
            n = max(p[0].shape[1] for p in parts)
            B, T, F = len(parts), self.lookback, self.X.shape[-1]
            x = np.zeros((B, T, n, F), np.float32)
            pr = np.zeros((B, T, n), bool)
            fi = np.zeros((B, T, n, F), bool)
            for b, (xx, pp, ff) in enumerate(parts):
                k = xx.shape[1]
                x[b, :, :k], pr[b, :, :k], fi[b, :, :k] = xx, pp, ff
            yield (torch.from_numpy(x).to(device), torch.from_numpy(pr).to(device),
                   torch.from_numpy(fi).to(device))
