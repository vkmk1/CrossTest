"""Daily bars from raw CRSP rows (DATA.md §2 cleaning).

Rules, in order:
1. ``prc = |prc|`` (negative means bid/ask average); same for open/high/low; zero prices -> NaN.
2. Drop rows with missing ``ret``.
3. Price filter: drop an (asset, year) if the asset's first close of that year is below $1.
4. Drop assets with fewer than 252 valid days over the whole sample.
5. ``mcap = prc * shrout`` (shrout is in thousands, so mcap is in $ thousands).

Note on rule 4: it is a full-sample filter, so the universe at t depends on whether an asset
survives 252 days in total. This is the spec'd standard filter; it is a mild survivorship
choice, not a feature leak (feature values at t never use data after t).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mstate.utils.paths import bars_dir, raw_dir

MIN_VALID_DAYS = 252
MIN_PRICE = 1.0


def clean_daily(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("prc", "openprc", "askhi", "bidlo"):
        df[c] = df[c].abs().replace(0.0, np.nan)
    df = df[df["ret"].notna() & df["prc"].notna()]
    df = df.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"])

    df["year"] = df["date"].dt.year
    first_px = df.groupby(["permno", "year"])["prc"].transform("first")
    df = df[first_px >= MIN_PRICE]

    n_valid = df.groupby("permno")["date"].transform("size")
    df = df[n_valid >= MIN_VALID_DAYS]

    df["mcap"] = df["prc"] * df["shrout"]
    cols = ["permno", "date", "year", "prc", "ret", "vol", "shrout", "openprc", "askhi", "bidlo",
            "mcap"]
    return df[cols].sort_values(["date", "permno"]).reset_index(drop=True)


def read_raw_daily(src: Path | None = None) -> pd.DataFrame:
    src = src or raw_dir("crsp_daily")
    files = sorted(p for p in src.glob("*.parquet") if p.stem.isdigit())
    if not files:
        raise FileNotFoundError(f"no yearly raw CRSP files in {src}; run mstate.data.wrds_pull")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def write_bars(bars: pd.DataFrame, out: Path | None = None) -> None:
    out = out or bars_dir("daily")
    out.mkdir(parents=True, exist_ok=True)
    for year, g in bars.groupby("year"):
        g.drop(columns="year").to_parquet(out / f"{year}.parquet", index=False)


def read_bars(src: Path | None = None) -> pd.DataFrame:
    src = src or bars_dir("daily")
    files = sorted(src.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no daily bars in {src}; run scripts/build_bars.py")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df["year"] = df["date"].dt.year
    return df
