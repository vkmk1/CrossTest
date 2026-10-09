"""Build the daily feature panel: data/bars/daily/ -> data/features/daily.parquet."""

from mstate.data.bars import read_bars
from mstate.data.features import build_daily_features, write_features

if __name__ == "__main__":
    feats = build_daily_features(read_bars(), horizons=(1, 5))
    p = write_features(feats)
    print(f"{len(feats):,} rows, {feats.shape[1]} columns -> {p}")
