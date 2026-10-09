"""Chronological splits and the held-out guard (DATA.md §5, CLAUDE.md rule 3).

Sample membership for a sample at market-date index t with lookback L and horizon h:

- train: the whole span [t-L+1, t+h] lies inside the train range. No training window or target
  touches validation or test, which is the purge.
- val / test: t and t+h lie inside the split. The input window may extend back into earlier
  splits (that is past data, available at prediction time); the target never leaves the split.

Test-period rows are only loaded when ``allow_test=True``, which only ``mstate.eval.evaluate``
passes, from its ``--split test`` path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mstate.utils import config


class HeldOutAccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Splits:
    train: tuple[pd.Timestamp, pd.Timestamp]
    val: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]
    purge_bars: int

    @classmethod
    def load(cls, freq: str = "daily") -> Splits:
        s = config.splits(freq)
        rng = {k: (pd.Timestamp(s[k][0]), pd.Timestamp(s[k][1])) for k in ("train", "val", "test")}
        return cls(rng["train"], rng["val"], rng["test"], int(s.purge_bars))

    def range(self, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        return {"train": self.train, "val": self.val, "test": self.test}[split]

    def last_allowed(self, allow_test: bool) -> pd.Timestamp:
        return self.test[1] if allow_test else self.val[1]

    def check_lookback(self, lookback: int, horizon: int) -> None:
        if self.purge_bars < lookback + horizon:
            raise ValueError(f"purge_bars={self.purge_bars} < lookback+horizon={lookback + horizon}")


def restrict(df: pd.DataFrame, splits: Splits, allow_test: bool = False) -> pd.DataFrame:
    """Drop rows after the last date development code is allowed to see."""
    return df[df["date"] <= splits.last_allowed(allow_test)]


def assert_no_test(dates: pd.Series | np.ndarray | pd.DatetimeIndex, splits: Splits) -> None:
    d = pd.to_datetime(np.asarray(dates))
    if len(d) and d.max() >= splits.test[0]:
        raise HeldOutAccessError(
            f"development code received dates up to {d.max().date()}, inside the test split "
            f"(starts {splits.test[0].date()}). Only evaluate.py --split test may read it."
        )


def read_features(freq: str, splits: Splits, allow_test: bool = False,
                  columns: list[str] | None = None) -> pd.DataFrame:
    from mstate.data.features import features_path

    last = splits.last_allowed(allow_test)
    df = pd.read_parquet(features_path(freq), columns=columns, filters=[("date", "<=", last)])
    if not allow_test:
        assert_no_test(df["date"], splits)
    return df


def sample_mask(calendar: pd.DatetimeIndex, t: np.ndarray, lookback: int, horizon: int,
                split: str, splits: Splits) -> np.ndarray:
    """Boolean mask over candidate sample indices ``t`` (positions in ``calendar``)."""
    n = len(calendar)
    t = np.asarray(t)
    lo_i, hi_i = t - lookback + 1, t + horizon
    ok = (lo_i >= 0) & (hi_i < n)
    lo_d = calendar[np.clip(lo_i, 0, n - 1)]
    t_d = calendar[np.clip(t, 0, n - 1)]
    hi_d = calendar[np.clip(hi_i, 0, n - 1)]
    start, end = splits.range(split)
    if split == "train":
        ok &= (lo_d >= start) & (hi_d <= end)
    else:
        ok &= (t_d >= start) & (hi_d <= end)
    return np.asarray(ok)
