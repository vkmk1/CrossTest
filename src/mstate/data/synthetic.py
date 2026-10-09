"""Synthetic CRSP-daily-like panel with a known latent market regime.

Used by tests and pipeline smoke runs when WRDS data is not present. It is never used for
any reported number. The generator produces the same columns as ``wrds_pull.COLUMNS`` so the
whole bars -> features -> model path runs unchanged.

Model: a 3-state persistent Markov regime scales market and idiosyncratic volatility; each asset
has a beta, a persistent AR(1) log-volatility component, staggered listing/delisting, and
volume that co-moves with volatility. OHLC is approximated around the close-to-close move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_MKT_VOL = np.array([0.006, 0.011, 0.028])
REGIME_IDIO_MULT = np.array([1.0, 1.3, 2.2])
TRANSITION = np.array(
    [
        [0.985, 0.013, 0.002],
        [0.015, 0.975, 0.010],
        [0.010, 0.040, 0.950],
    ]
)


def make_crsp_like(
    n_assets: int = 300,
    start: str = "2010-01-01",
    end: str = "2022-12-31",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (raw rows with wrds_pull.COLUMNS, regime series indexed by date)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T, N = len(dates), n_assets

    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        regime[t] = rng.choice(3, p=TRANSITION[regime[t - 1]])
    f_mkt = rng.standard_normal(T) * REGIME_MKT_VOL[regime]

    beta = rng.uniform(0.5, 1.6, N)
    idio_base = np.exp(rng.normal(np.log(0.014), 0.35, N))
    # persistent asset-specific log-vol deviation
    phi, s = 0.97, 0.08
    lv = np.zeros((T, N))
    eps = rng.standard_normal((T, N)) * s
    for t in range(1, T):
        lv[t] = phi * lv[t - 1] + eps[t]
    idio_vol = idio_base[None, :] * REGIME_IDIO_MULT[regime][:, None] * np.exp(lv)
    ret = beta[None, :] * f_mkt[:, None] + rng.standard_normal((T, N)) * idio_vol
    ret = np.clip(ret, -0.6, 1.0)
    sigma = np.sqrt((beta[None, :] * REGIME_MKT_VOL[regime][:, None]) ** 2 + idio_vol**2)

    p0 = np.exp(rng.normal(3.3, 0.9, N))
    close = p0[None, :] * np.cumprod(1.0 + ret, axis=0)
    prev_close = np.vstack([p0[None, :], close[:-1]])
    gap = rng.standard_normal((T, N)) * 0.25 * sigma
    openp = prev_close * np.exp(gap)
    hi_ext = np.abs(rng.standard_normal((T, N))) * 0.6 * sigma
    lo_ext = np.abs(rng.standard_normal((T, N))) * 0.6 * sigma
    high = np.maximum(openp, close) * np.exp(hi_ext)
    low = np.minimum(openp, close) * np.exp(-lo_ext)

    shrout = np.exp(rng.normal(np.log(80_000), 1.0, N))  # thousands of shares
    base_turn = np.exp(rng.normal(np.log(0.006), 0.4, N))
    turn = base_turn[None, :] * (sigma / sigma.mean(axis=0, keepdims=True)) ** 1.2
    turn *= np.exp(rng.normal(0.0, 0.3, (T, N)))
    vol = np.round(turn * shrout[None, :] * 1000.0)

    # staggered listing: 70% present from the start, others list later; 15% delist early
    first = np.where(rng.random(N) < 0.7, 0, rng.integers(0, max(T - 300, 1), N))
    last = np.where(rng.random(N) < 0.15, first + rng.integers(260, T, N), T - 1)
    last = np.minimum(last, T - 1)
    alive = (np.arange(T)[:, None] >= first[None, :]) & (np.arange(T)[:, None] <= last[None, :])

    tt, ii = np.nonzero(alive)
    df = pd.DataFrame(
        {
            "permno": 10000 + ii,
            "date": dates[tt],
            "prc": close[tt, ii],
            "ret": ret[tt, ii],
            "vol": vol[tt, ii],
            "shrout": shrout[ii],
            "openprc": openp[tt, ii],
            "askhi": high[tt, ii],
            "bidlo": low[tt, ii],
        }
    )
    # CRSP conventions: first day has no return; a few closes are negative bid/ask averages
    df.loc[tt == first[ii], "ret"] = np.nan
    neg = rng.random(len(df)) < 0.002
    df.loc[neg, "prc"] = -df.loc[neg, "prc"]
    df = df.sort_values(["date", "permno"]).reset_index(drop=True)
    return df, pd.Series(regime, index=dates, name="regime")
