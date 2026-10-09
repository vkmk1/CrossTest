"""Where the feature frame comes from: the real Parquet panel or the in-memory synthetic one.

``cfg.data.source`` is ``real`` (default) or ``synthetic``. Synthetic data is for smoke tests and
pipeline checks only and is never reported.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from omegaconf import DictConfig

from mstate.data.splits import Splits, read_features, restrict


@lru_cache(maxsize=2)
def _synthetic(n_assets: int, seed: int, end: str) -> pd.DataFrame:
    from mstate.data.bars import clean_daily
    from mstate.data.features import build_daily_features
    from mstate.data.synthetic import make_crsp_like

    raw, _ = make_crsp_like(n_assets=n_assets, start="2010-01-01", end=end, seed=seed)
    return build_daily_features(clean_daily(raw), horizons=(1, 5))


def load_feature_frame(cfg: DictConfig, splits: Splits, allow_test: bool = False) -> pd.DataFrame:
    data = cfg.get("data") or {}
    if data.get("source", "real") == "synthetic":
        end = "2025-12-31" if allow_test else str(splits.val[1].date())
        df = _synthetic(int(data.get("n_assets", 400)), int(data.get("seed", 0)), end)
        return restrict(df, splits, allow_test)
    return read_features(str(cfg.freq), splits, allow_test)
