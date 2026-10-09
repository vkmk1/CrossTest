from __future__ import annotations

import pandas as pd
import pytest

from mstate.data.bars import clean_daily
from mstate.data.features import build_daily_features
from mstate.data.splits import Splits
from mstate.data.synthetic import make_crsp_like


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    df, _ = make_crsp_like(n_assets=120, start="2017-01-01", end="2023-06-30", seed=1)
    return df


@pytest.fixture(scope="session")
def features(raw: pd.DataFrame) -> pd.DataFrame:
    return build_daily_features(clean_daily(raw), horizons=(1, 5))


@pytest.fixture(scope="session")
def splits() -> Splits:
    return Splits.load("daily")
