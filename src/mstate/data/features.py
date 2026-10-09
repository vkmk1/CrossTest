"""Daily feature panel (DATA.md §4).

Output: one long frame, one row per (permno, date), with

- ``tidx``: index of the date in the market calendar (sorted unique dates in the bars)
- ``cap_rank``: market-cap rank on the first trading day of the row's calendar year (1 = largest)
- raw base features ``ret, lrv, lvol, range, turn`` (DATA.md §4.1)
- ``xs_<f>``: cross-sectionally normalized ``(x - median_t) / (1.4826 * MAD_t)`` clipped to [-5, 5]
- ``mkt_median_<f>, mkt_mad_<f>``: the per-date median and (scaled) MAD, same on every row of t
- targets ``y_lrv_h<h>`` = log(sum of RV over (t, t+h]) and ``y_ret_h<h>`` = sum of log returns
  over (t, t+h]; NaN unless all h following market dates are present for the asset

Everything except the targets at t uses rows with date <= t only; cross-sectional statistics use
only the rows at t. ``tests/test_leakage.py`` checks both.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from mstate.utils.paths import features_dir

BASE_FEATURES = ("ret", "lrv", "lvol", "range", "turn")
RV_FLOOR = 1e-8  # floor on daily variance before the log (0.01% daily vol)
MAD_SCALE = 1.4826  # makes MAD a consistent estimator of sigma under normality
MAD_EPS = 1e-8
CLIP = 5.0
GK_C = 2.0 * np.log(2.0) - 1.0


def mkt_columns(features: Sequence[str] = BASE_FEATURES) -> list[str]:
    return [f"mkt_{s}_{f}" for f in features for s in ("median", "mad")]


def xs_columns(features: Sequence[str] = BASE_FEATURES) -> list[str]:
    return [f"xs_{f}" for f in features]


def target_column(target: str, h: int) -> str:
    return f"y_{target}_h{h}"


def garman_klass(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray,
                 ret: np.ndarray) -> np.ndarray:
    """Daily variance: Garman-Klass if O,H,L,C present; Parkinson if only H,L; else ret^2."""
    with np.errstate(invalid="ignore", divide="ignore"):
        hh = np.fmax(np.fmax(h, o), c)
        ll = np.fmin(np.fmin(lo, o), c)
        hl = np.log(hh / ll)
        co = np.log(c / o)
        gk = 0.5 * hl**2 - GK_C * co**2
        park = np.log(h / lo) ** 2 / (4.0 * np.log(2.0))
    var = np.where(np.isfinite(gk), gk, np.where(np.isfinite(park), park, ret**2))
    return np.maximum(var, RV_FLOOR)


def base_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars[["permno", "date", "year", "mcap"]].copy()
    r = np.log1p(bars["ret"].to_numpy(dtype=np.float64))
    df["ret"] = r
    df["lrv"] = np.log(garman_klass(bars["openprc"].to_numpy(), bars["askhi"].to_numpy(),
                                    bars["bidlo"].to_numpy(), bars["prc"].to_numpy(), r))
    dollar_vol = bars["vol"].to_numpy() * bars["prc"].to_numpy()
    df["lvol"] = np.log1p(np.fmax(dollar_vol, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rng = np.log(np.fmax(bars["askhi"].to_numpy(), bars["prc"].to_numpy())
                     / np.fmin(bars["bidlo"].to_numpy(), bars["prc"].to_numpy()))
    df["range"] = np.where(np.isfinite(rng), rng, np.nan)
    df["turn"] = bars["vol"].to_numpy() / (bars["shrout"].to_numpy() * 1000.0)
    return df


def cross_sectional_normalize(df: pd.DataFrame,
                              features: Sequence[str] = BASE_FEATURES) -> pd.DataFrame:
    """Add xs_<f>, mkt_median_<f>, mkt_mad_<f>. Statistics use only rows sharing the date."""
    out = df.copy()
    g = out.groupby("date", sort=False)
    for f in features:
        med = g[f].transform("median")
        mad = (out[f] - med).abs().groupby(out["date"], sort=False).transform("median") * MAD_SCALE
        out[f"mkt_median_{f}"] = med
        out[f"mkt_mad_{f}"] = mad
        out[f"xs_{f}"] = ((out[f] - med) / (mad + MAD_EPS)).clip(-CLIP, CLIP)
    return out


def cap_ranks(df: pd.DataFrame) -> pd.Series:
    """Rank by market cap on the first market trading day of each calendar year (1 = largest).

    The rank is fixed for the year; assets not trading on that day are unranked (NaN).
    """
    first_day = df.groupby("year")["date"].transform("min")
    on_first = df[df["date"] == first_day]
    rank = on_first.groupby("year")["mcap"].rank(ascending=False, method="first")
    key = pd.Series(rank.to_numpy(), index=pd.MultiIndex.from_frame(on_first[["permno", "year"]]))
    idx = pd.MultiIndex.from_frame(df[["permno", "year"]])
    return pd.Series(key.reindex(idx).to_numpy(), index=df.index, name="cap_rank")


def add_targets(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """y_lrv_h = log(sum_{k=1..h} RV_{t+k}); y_ret_h = sum_{k=1..h} ret_{t+k}.

    Requires the asset to be present on each of the next h market dates.
    """
    out = df.sort_values(["permno", "tidx"]).copy()
    g = out.groupby("permno", sort=False)
    rv = np.exp(out["lrv"])
    for h in horizons:
        rv_sum = np.zeros(len(out))
        ret_sum = np.zeros(len(out))
        ok = np.ones(len(out), dtype=bool)
        for k in range(1, h + 1):
            ti = g["tidx"].shift(-k).to_numpy()
            ok &= ti == out["tidx"].to_numpy() + k
            rv_sum += np.nan_to_num(rv.groupby(out["permno"], sort=False).shift(-k).to_numpy())
            ret_sum += np.nan_to_num(g["ret"].shift(-k).to_numpy())
        with np.errstate(divide="ignore"):
            out[target_column("lrv", h)] = np.where(ok, np.log(rv_sum), np.nan)
        out[target_column("ret", h)] = np.where(ok, ret_sum, np.nan)
    return out.sort_values(["date", "permno"]).reset_index(drop=True)


def build_daily_features(bars: pd.DataFrame, horizons: Sequence[int] = (1, 5)) -> pd.DataFrame:
    df = base_features(bars)
    cal = np.sort(df["date"].unique())
    df["tidx"] = np.searchsorted(cal, df["date"].to_numpy()).astype(np.int32)
    df["cap_rank"] = cap_ranks(df)
    df = cross_sectional_normalize(df)
    df = add_targets(df, horizons)
    float_cols = [c for c in df.columns if df[c].dtype == np.float64 and c != "mcap"]
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def features_path(freq: str = "daily") -> Path:
    return features_dir() / f"{freq}.parquet"


def write_features(df: pd.DataFrame, freq: str = "daily") -> Path:
    p = features_path(freq)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p
