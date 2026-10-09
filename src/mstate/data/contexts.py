"""Build the forecaster's context array ``C [n_dates, n_assets, d]`` for one ablation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mstate.data.datasets import DensePanel, ForecastData
from mstate.data.splits import Splits
from mstate.models.baselines import make_market_context, market_frame
from mstate.utils.paths import data_path

MARKET_SOURCES = ("none", "hmm", "pca", "jump")
LSM_SOURCES = ("lsm", "lsm_noxs", "lsm_n50")


def build_contexts(ctx: dict[str, Any], df: pd.DataFrame, ds: ForecastData,
                   splits: Splits) -> tuple[np.ndarray, Any]:
    """Return (C aligned to ds.calendar x ds.permnos, fitted context object or None).

    Market-level sources are fitted on train dates only, then broadcast across assets.
    LSM sources read the frozen export ``contexts_path`` (permno, date, c_0..c_{d-1}).
    """
    dim = int(ctx["dim"])
    src = ctx["source"]
    T, N = len(ds.calendar), len(ds.permnos)
    if src in MARKET_SOURCES:
        mkt = market_frame(df).reindex(ds.calendar)
        model = make_market_context(dict(ctx)).fit(mkt[mkt.index <= splits.train[1]])
        c = model.transform(mkt).to_numpy(dtype=np.float32)
        return np.broadcast_to(c[:, None, :], (T, N, dim)).copy(), model
    if src.startswith("lsm"):
        if ctx.get("mode") == "e2e" or src == "lsm_e2e":
            raise NotImplementedError("lsm_e2e is a Phase 1 control")
        path = data_path(ctx["contexts_path"])
        z = pd.read_parquet(path, filters=[("date", "<=", ds.calendar[-1])])
        cols = [f"c_{i}" for i in range(dim)]
        panel = DensePanel.from_long(z, cols, ds.permnos, ds.calendar)
        return panel.stack(cols), None
    raise ValueError(f"unknown context source {src}")
