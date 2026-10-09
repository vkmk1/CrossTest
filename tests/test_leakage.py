"""Leakage tests, DATA.md §6. Must pass before any evaluation run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mstate.data.bars import clean_daily
from mstate.data.datasets import ForecastData, window_index
from mstate.data.features import BASE_FEATURES, MAD_SCALE, build_daily_features
from mstate.data.splits import HeldOutAccessError, Splits, restrict

T0 = pd.Timestamp("2019-06-14")
H = 5


def _perturb_future(raw: pd.DataFrame, t0: pd.Timestamp, seed: int = 7) -> pd.DataFrame:
    """Replace every value after t0 with noise (prices kept >= $1.5 so filters see the same rows)."""
    rng = np.random.default_rng(seed)
    out = raw.copy()
    fut = out["date"] > t0
    n = int(fut.sum())
    px = np.exp(rng.normal(3.0, 1.0, n)) + 1.5
    out.loc[fut, "prc"] = px
    out.loc[fut, "openprc"] = px * np.exp(rng.normal(0, 0.05, n))
    out.loc[fut, "askhi"] = px * np.exp(np.abs(rng.normal(0, 0.1, n)) + 0.05)
    out.loc[fut, "bidlo"] = px * np.exp(-np.abs(rng.normal(0, 0.1, n)) - 0.05)
    out.loc[fut, "ret"] = np.where(out.loc[fut, "ret"].notna(), rng.normal(0, 0.2, n), np.nan)
    out.loc[fut, "vol"] = rng.integers(1, 10**8, n).astype(float)
    out.loc[fut, "shrout"] = rng.integers(1_000, 10**6, n).astype(float)
    return out


def test_shift_future_noise_does_not_change_features(raw: pd.DataFrame,
                                                     features: pd.DataFrame) -> None:
    """1. Set all data after t0 to noise; no feature at t <= t0 may change, nor any target whose
    window (t, t+h] ends by t0."""
    noisy = build_daily_features(clean_daily(_perturb_future(raw, T0)), horizons=(1, H))
    key = ["permno", "date"]
    a = features[features["date"] <= T0].set_index(key).sort_index()
    b = noisy[noisy["date"] <= T0].set_index(key).sort_index()
    assert a.index.equals(b.index)
    feat_cols = [c for c in a.columns if not c.startswith("y_")]
    pd.testing.assert_frame_equal(a[feat_cols], b[feat_cols])

    cal = pd.DatetimeIndex(np.sort(features["date"].unique()))
    safe = cal[: cal.get_loc(T0) + 1 - H]  # t with t+H <= T0
    tgt_cols = [c for c in a.columns if c.startswith("y_")]
    sa = a[a.index.get_level_values("date").isin(safe)][tgt_cols]
    sb = b[b.index.get_level_values("date").isin(safe)][tgt_cols]
    pd.testing.assert_frame_equal(sa, sb)
    # and the perturbation really did change something just after t0
    later = features["date"] > T0
    assert not np.allclose(features.loc[later, "xs_lrv"].to_numpy()[:100],
                           build_daily_features(clean_daily(_perturb_future(raw, T0)), (1,))
                           .loc[lambda d: d["date"] > T0, "xs_lrv"].to_numpy()[:100])


def test_normalization_uses_only_same_timestamp(features: pd.DataFrame) -> None:
    """2. mkt_median_f at t equals the median of that timestamp's rows alone (same for MAD)."""
    rng = np.random.default_rng(0)
    for d in rng.choice(features["date"].unique(), size=25, replace=False):
        rows = features[features["date"] == d]
        for f in BASE_FEATURES:
            x = rows[f].to_numpy(dtype=np.float64)
            x = x[np.isfinite(x)]
            med = np.median(x)
            mad = np.median(np.abs(x - med)) * MAD_SCALE
            assert np.allclose(rows[f"mkt_median_{f}"], med, rtol=1e-5, atol=1e-5)
            assert np.allclose(rows[f"mkt_mad_{f}"], mad, rtol=1e-4, atol=1e-5)
            z = np.clip((rows[f] - med) / (mad + 1e-8), -5, 5)
            assert np.allclose(rows[f"xs_{f}"].dropna(), z.dropna(), atol=1e-4)


@pytest.mark.parametrize("horizon", [1, 5])
def test_split_no_training_window_in_val_or_test(features: pd.DataFrame, splits: Splits,
                                                 horizon: int) -> None:
    """3. No training window (inputs or target) has any timestamp in validation or test.
    Uses synthetic data that extends into the test range, with allow_test=True."""
    ds = ForecastData(features, BASE_FEATURES, "lrv", horizon, 60, 50, splits, allow_test=True)
    tr = ds.index["train"]
    assert len(tr) > 0
    w = window_index(tr[:, 0], 60)
    assert ds.calendar[w.min()] >= splits.train[0]
    assert ds.calendar[tr[:, 0].max() + horizon] <= splits.train[1]
    assert ds.calendar[tr[:, 0].max() + horizon] < splits.val[0]
    for split in ("val", "test"):
        idx = ds.index[split]
        lo, hi = splits.range(split)
        assert ds.calendar[idx[:, 0].min()] >= lo
        assert ds.calendar[idx[:, 0].max() + horizon] <= hi
    # train / val / test targets are disjoint in time
    t_hi = ds.index["train"][:, 0].max() + horizon
    assert t_hi < ds.index["val"][:, 0].min() + 1


def test_heldout_guard(features: pd.DataFrame, splits: Splits) -> None:
    """Development code may not receive test-period rows."""
    with pytest.raises(HeldOutAccessError):
        ForecastData(features, BASE_FEATURES, "lrv", 1, 60, 50, splits)
    ds = ForecastData(restrict(features, splits), BASE_FEATURES, "lrv", 1, 60, 50, splits)
    assert "test" not in ds.index


def test_hmm_context_is_filtered(features: pd.DataFrame, splits: Splits) -> None:
    """4. HMMContext at t equals the filtered posterior from the forward pass truncated at t."""
    baselines = pytest.importorskip("mstate.models.baselines")
    mkt = baselines.market_frame(restrict(features, splits))
    hmm = baselines.HMMContext(n_states=3, dim=3, series="mkt_median_lrv")
    hmm.fit(mkt[mkt.index <= splits.train[1]])
    full = hmm.transform(mkt).to_numpy()
    rng = np.random.default_rng(0)
    for t in rng.choice(np.arange(10, len(mkt)), size=20, replace=False):
        trunc = hmm.transform(mkt.iloc[: t + 1]).to_numpy()
        np.testing.assert_allclose(full[: t + 1], trunc, atol=1e-8)
        # independent check: hmmlearn smoothing at the last point equals filtering at t
        x = mkt["mkt_median_lrv"].to_numpy()[: t + 1, None]
        ref = hmm.model.predict_proba(x)[-1][hmm.order]
        np.testing.assert_allclose(full[t], ref, atol=1e-6)
