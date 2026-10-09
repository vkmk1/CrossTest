"""Baseline sanity tests: causality of the HMM and PCA contexts, shapes, padding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mstate.data.contexts import build_contexts
from mstate.data.datasets import ForecastData
from mstate.data.features import BASE_FEATURES
from mstate.data.splits import Splits, restrict
from mstate.models.baselines import HMMContext, PCAContext, market_frame


def _perturb_after(mkt: pd.DataFrame, t: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = mkt.copy()
    out.iloc[t + 1:] = rng.normal(0, 5, size=out.iloc[t + 1:].shape) + out.mean().to_numpy()
    return out


def test_hmm_context_ignores_future_rows(features: pd.DataFrame, splits: Splits) -> None:
    mkt = market_frame(restrict(features, splits))
    hmm = HMMContext(3, 16).fit(mkt[mkt.index <= splits.train[1]])
    base = hmm.transform(mkt).to_numpy()
    assert base.shape == (len(mkt), 16)
    assert np.allclose(base[:, :3].sum(1), 1.0) and np.allclose(base[:, 3:], 0.0)
    for t in (50, 400, len(mkt) - 30):
        pert = hmm.transform(_perturb_after(mkt, t)).to_numpy()
        np.testing.assert_allclose(base[: t + 1], pert[: t + 1], atol=1e-10)
        assert not np.allclose(base[t + 1:], pert[t + 1:])


def test_hmm_states_ordered_by_mean(features: pd.DataFrame, splits: Splits) -> None:
    mkt = market_frame(restrict(features, splits))
    hmm = HMMContext(3, 16).fit(mkt[mkt.index <= splits.train[1]])
    means = hmm.model.means_[hmm.order, 0]
    assert np.all(np.diff(means) > 0)


def test_pca_context_ignores_future_rows(features: pd.DataFrame, splits: Splits) -> None:
    mkt = market_frame(restrict(features, splits))
    pca = PCAContext(16, lags=1).fit(mkt[mkt.index <= splits.train[1]])
    base = pca.transform(mkt).to_numpy()
    assert base.shape == (len(mkt), 16)
    t = 300
    pert = pca.transform(_perturb_after(mkt, t)).to_numpy()
    np.testing.assert_allclose(base[: t + 1], pert[: t + 1], atol=1e-10)


def test_build_contexts_shapes(features: pd.DataFrame, splits: Splits) -> None:
    df = restrict(features, splits)
    ds = ForecastData(df, BASE_FEATURES, "lrv", 1, 60, 50, splits)
    for src in ("none", "hmm", "pca"):
        C, _ = build_contexts({"source": src, "dim": 16}, df, ds, splits)
        assert C.shape == (len(ds.calendar), len(ds.permnos), 16)
        assert np.isfinite(C).all()
        # market-level: identical across assets at a date
        assert np.allclose(C[:, 0], C[:, -1])
