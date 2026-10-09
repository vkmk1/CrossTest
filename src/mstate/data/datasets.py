"""Dense panels and the Stage 2 (forecaster) dataset.

Shapes: panels are ``[n_dates, n_assets, F]``; forecaster batches are ``x [B, L, F]``,
``c [B, L, d]``, ``y [B]``.

Forecaster inputs are the asset's own raw base features (DATA.md §4.1), standardized with
train-period mean/std, so that the ``none`` baseline still sees its own volatility level. The
context slot is the only place market-wide information enters.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from mstate.data.features import target_column
from mstate.data.splits import Splits, assert_no_test, sample_mask

SPLITS = ("train", "val", "test")


@dataclass
class DensePanel:
    calendar: pd.DatetimeIndex
    permnos: np.ndarray
    values: dict[str, np.ndarray]  # column -> [n_dates, n_assets], NaN where absent

    @classmethod
    def from_long(cls, df: pd.DataFrame, columns: Sequence[str],
                  permnos: np.ndarray | None = None,
                  calendar: pd.DatetimeIndex | None = None) -> DensePanel:
        cal = calendar if calendar is not None else pd.DatetimeIndex(np.sort(df["date"].unique()))
        pn = np.sort(df["permno"].unique()) if permnos is None else np.asarray(permnos)
        sub = df[df["permno"].isin(pn)]
        ti = cal.get_indexer(sub["date"])
        ai = np.searchsorted(pn, sub["permno"].to_numpy())
        keep = ti >= 0
        vals = {}
        for c in columns:
            arr = np.full((len(cal), len(pn)), np.nan, dtype=np.float32)
            arr[ti[keep], ai[keep]] = sub[c].to_numpy(dtype=np.float32)[keep]
            vals[c] = arr
        return cls(cal, pn, vals)

    def stack(self, columns: Sequence[str]) -> np.ndarray:
        return np.stack([self.values[c] for c in columns], axis=-1)


def window_index(t: np.ndarray, lookback: int) -> np.ndarray:
    """Date indices of the lookback window ending at t, inclusive: ``[len(t), L]``."""
    return t[:, None] + np.arange(-lookback + 1, 1)[None, :]


class ForecastData:
    """Samples (t, asset) for one target/horizon over the evaluation universe.

    ``contexts`` is ``[n_dates, n_assets, d]`` aligned to this object's calendar/permnos (build it
    with :func:`mstate.data.contexts.build_contexts`), or None for zeros.
    """

    def __init__(self, df: pd.DataFrame, features: Sequence[str], target: str, horizon: int,
                 lookback: int, eval_universe: int, splits: Splits, allow_test: bool = False,
                 context_dim: int = 16):
        if not allow_test:
            assert_no_test(df["date"], splits)
        splits.check_lookback(lookback, horizon)
        self.features = list(features)
        self.target = target_column(target, horizon)
        self.horizon, self.lookback, self.splits = horizon, lookback, splits
        self.context_dim = context_dim

        members = df[df["cap_rank"] <= eval_universe]
        permnos = np.sort(members["permno"].unique())
        calendar = pd.DatetimeIndex(np.sort(df["date"].unique()))
        cols = [*self.features, self.target, "cap_rank", "lrv"]
        self.panel = DensePanel.from_long(df, list(dict.fromkeys(cols)), permnos, calendar)
        self.calendar, self.permnos = calendar, permnos

        X = self.panel.stack(self.features)  # [T, N, F]
        Y = self.panel.values[self.target]
        member = self.panel.values["cap_rank"] <= eval_universe
        finite_x = np.isfinite(X).all(-1)
        # window fully observed: rolling count of finite rows over the last L dates
        cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(finite_x, axis=0)])
        full_win = np.zeros_like(finite_x)
        full_win[lookback - 1:] = (cs[lookback:] - cs[:-lookback]) == lookback
        candidate = member & np.isfinite(Y) & full_win

        tt, aa = np.nonzero(candidate)
        self.index: dict[str, np.ndarray] = {}
        allowed = ("train", "val", "test") if allow_test else ("train", "val")
        for split in allowed:
            m = sample_mask(calendar, tt, lookback, horizon, split, splits)
            self.index[split] = np.stack([tt[m], aa[m]], axis=1)

        tr = self.index["train"]
        if len(tr) == 0:
            raise ValueError("no training samples")
        rows = X[tr[:, 0], tr[:, 1]]
        self.x_mean = rows.mean(0).astype(np.float32)
        self.x_std = (rows.std(0) + 1e-6).astype(np.float32)
        ytr = Y[tr[:, 0], tr[:, 1]]
        self.y_mean, self.y_std = float(ytr.mean()), float(ytr.std() + 1e-6)

        self.X = np.nan_to_num((X - self.x_mean) / self.x_std).astype(np.float32)
        self.Y = np.nan_to_num(Y).astype(np.float32)
        self.C = np.zeros((*X.shape[:2], context_dim), dtype=np.float32)
        self._dev: dict[str, torch.Tensor] = {}
        self.check_causality()

    # ----------------------------------------------------------------------------------------
    def set_contexts(self, C: np.ndarray) -> None:
        if C.shape != (len(self.calendar), len(self.permnos), self.context_dim):
            raise ValueError(f"context shape {C.shape} does not match panel "
                             f"{(len(self.calendar), len(self.permnos), self.context_dim)}")
        self.C = np.nan_to_num(C).astype(np.float32)
        self._dev = {}

    def to(self, device: torch.device) -> ForecastData:
        self._dev = {k: torch.from_numpy(v).to(device) for k, v in
                     (("X", self.X), ("C", self.C), ("Y", self.Y))}
        self._device = device
        return self

    def n(self, split: str) -> int:
        return len(self.index[split])

    def batches(self, split: str, batch_size: int, shuffle: bool = False,
                rng: np.random.Generator | None = None
                ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
        """Yield ``(x [B,L,F], c [B,L,d], y [B], idx [B,2])`` on the device set by ``to``."""
        if not self._dev:
            self.to(torch.device("cpu"))
        idx = self.index[split]
        order = (rng or np.random.default_rng()).permutation(len(idx)) if shuffle else \
            np.arange(len(idx))
        dev = self._device
        for s in range(0, len(idx), batch_size):
            b = idx[order[s:s + batch_size]]
            w = torch.from_numpy(window_index(b[:, 0], self.lookback)).to(dev)
            a = torch.from_numpy(b[:, 1]).to(dev)[:, None]
            x = self._dev["X"][w, a]
            c = self._dev["C"][w, a]
            y = self._dev["Y"][torch.from_numpy(b[:, 0]).to(dev), a[:, 0]]
            yield x, c, y, b

    # ----------------------------------------------------------------------------------------
    def check_causality(self, n: int = 256, seed: int = 0) -> None:
        """Assert on a sample of rows: inputs at t use dates <= t; the target uses (t, t+h] only;
        training windows and targets stay inside the training range."""
        rng = np.random.default_rng(seed)
        lrv = self.panel.values["lrv"]
        tr_lo, tr_hi = self.splits.train
        for split, idx in self.index.items():
            if len(idx) == 0:
                continue
            pick = idx[rng.choice(len(idx), size=min(n, len(idx)), replace=False)]
            w = window_index(pick[:, 0], self.lookback)
            assert (w.max(1) == pick[:, 0]).all(), "input window reaches past t"
            fut = pick[:, 0][:, None] + np.arange(1, self.horizon + 1)[None, :]
            if self.target.startswith("y_lrv"):
                expect = np.log(np.exp(lrv[fut, pick[:, 1][:, None]].astype(np.float64)).sum(1))
                got = self.Y[pick[:, 0], pick[:, 1]]
                assert np.allclose(got, expect, atol=1e-3), "target is not built from (t, t+h]"
            if split == "train":
                assert self.calendar[w.min()] >= tr_lo and self.calendar[fut.max()] <= tr_hi, \
                    "training sample leaves the training range"
            else:
                lo, hi = self.splits.range(split)
                assert self.calendar[pick[:, 0].min()] >= lo and self.calendar[fut.max()] <= hi
