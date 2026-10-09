# DATA.md: sources, pipelines, features, splits

## 1. Access

All data comes from WRDS. Use the `wrds` Python package with the user's credentials in `~/.pgpass`; never hard-code credentials. Query with SQL through `wrds.Connection().raw_sql`. For TAQ, the WRDS Cloud is the right place to run the aggregation job (the tables are too large to pull raw); write the aggregation as a SQL query that returns bars, not trades.

Libraries:
- Daily: `crsp.dsf` (prices, volume, shares), `crsp.dsenames` (identifiers, exchange, share code).
- Intraday: `taqm_YYYY.ctm_YYYYMMDD` (consolidated trades) and `taqm_YYYY.cqm_YYYYMMDD` (consolidated quotes), or the `nbbom_YYYYMMDD` NBBO tables where available, which are already the best bid and offer and are much smaller. Prefer NBBO for quote features.

## 2. Daily universe and pull (Phase 0)

- Universe: all CRSP common shares (`shrcd ∈ {10, 11}`) on NYSE, NASDAQ, AMEX (`exchcd ∈ {1, 2, 3}`), 2010-01-01 to 2025-12-31.
- Fields: `permno, date, prc, ret, vol, shrout, openprc, askhi, bidlo`.
- Clean: take `abs(prc)` (negative means bid/ask average), drop rows with missing `ret`, drop assets with fewer than 252 valid days, drop price below $1 at the start of each year (standard filter). Record the filter rule in the data section of the thesis.
- Store `data/bars/daily/{year}.parquet` with one row per (permno, date).

For the breadth experiment (H3), subsets of N ∈ {50, 200, 1000, full} are drawn by market cap rank at the start of each year, fixed for the year. The forecaster's evaluation set is always the same: the 200 largest names, so breadth changes only what the LSM saw.

## 3. Intraday pull (Phase 1+, only after Phase 0 passes)

- Universe: the 500 largest names by prior-year market cap, one calendar year to start (2024), regular hours 09:30–16:00 ET.
- Bars: 1-minute first. Built by a SQL aggregation on WRDS Cloud: for trades, `count, sum(size), sum(price*size)/sum(size) as vwap, first/last price, max/min price, sum(size where buyer-initiated)` with trade signing by the Lee-Ready tie-breaker against the prevailing NBBO midpoint; for quotes, time-weighted mean spread, mean inside depth, count of NBBO updates.
- Filters: standard TAQ cleaning (Holden and Jacobsen): drop trades with correction codes ≠ 0, drop sale conditions that are not regular, drop quotes with spread ≤ 0 or > 25% of mid, drop locked/crossed.
- Store `data/bars/minute/{year}/{month}.parquet`.
- Second bars are built from the same job with a 1-second bucket, only if minute results pass.

## 4. Features (per asset, per bar)

### 4.1 Base features (all frequencies)

| Feature | Definition |
|---|---|
| `ret` | log return of close over previous close |
| `lrv` | log realized variance over the bar (daily: Garman-Klass from OHLC; intraday: sum of squared sub-bar returns, or squared bar return if no sub-bars) |
| `lvol` | log dollar volume |
| `range` | log(high/low) |
| `turn` | volume / shares outstanding (daily) or volume / 20-bar median volume (intraday) |

### 4.2 Intraday-only features

| Feature | Definition |
|---|---|
| `lspread` | log time-weighted quoted spread / mid |
| `ldepth` | log mean inside depth (bid + ask shares) |
| `imb` | (buy volume − sell volume) / total volume |
| `qrate` | log(1 + NBBO update count) |
| `trate` | log(1 + trade count) |

### 4.3 Cross-sectional normalization (critical)

At each timestamp t, each feature is z-scored across the assets present at t: `x̃_{i,t} = (x_{i,t} − median_i x_{i,t}) / MAD_i x_{i,t}`, clipped to [−5, 5]. The per-timestamp median and MAD are stored as additional "market" features `mkt_median_f, mkt_mad_f` and appended to every asset's feature vector, so the level information is not lost, it is just separated from the relative information. The PCA baseline is fitted on the `mkt_*` vectors. This uses only data at t: no lookahead.

### 4.4 Targets

`y_{i,t,h}` = feature value aggregated over `(t, t+h]`: log RV summed over the next h bars; return summed; spread averaged. Computed in `build_features.py` and stored alongside, with the horizon in the column name.

## 5. Splits, `configs/splits.yaml`

Chronological, with purge gaps equal to the longest horizon plus lookback so no window straddles a boundary.

| Frequency | Train | Validation | Test (untouched) |
|---|---|---|---|
| Daily | 2010-01-01 to 2019-12-31 | 2020-01-01 to 2022-12-31 | 2023-01-01 to 2025-12-31 |
| Minute (2024) | Jan–Aug | Sep–Oct | Nov–Dec |

Validation deliberately contains March 2020 so the HMM and LSM are both selected on a period with a regime change. Test contains 2023–2025, which includes the volatile days of 2025 used for the regime-detection check (H5).

Labeled event dates for H5 (`configs/events.yaml`): FOMC decision days, CPI release days, 2020-02-24 to 2020-03-23, and the ten highest cross-sectional-mean-RV days in test. Quiet days: the 20% of days with lowest cross-sectional-mean RV in each split.

## 6. Leakage tests, `tests/test_leakage.py`

Required before any evaluation:
1. Shift test: shuffle the future (set all t+h data to noise) and confirm no feature at t changes.
2. Normalization test: confirm `mkt_median` at t equals the median computed from that timestamp's rows alone.
3. Split test: confirm no window in the training dataset has any timestamp in validation or test.
4. HMM test: confirm the context at t from `HMMContext` equals the filtered posterior from the forward pass truncated at t.

## 7. Storage estimates

Daily full universe 2010–2025: ~15M rows, under 2 GB Parquet. Minute, 500 names, one year: ~50M rows, ~5 GB. Second, 500 names, one month: ~300M rows; start with 50 names if going to seconds.
