"""Build daily bars from the raw CRSP pull: data/raw/crsp_daily/*.parquet -> data/bars/daily/."""

from mstate.data.bars import clean_daily, read_raw_daily, write_bars

if __name__ == "__main__":
    raw = read_raw_daily()
    bars = clean_daily(raw)
    write_bars(bars)
    print(f"raw {len(raw):,} rows -> bars {len(bars):,} rows, {bars['permno'].nunique():,} permnos")
