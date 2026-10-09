# EXPERIMENTS.md: phased registry with gates

Each phase ends with a table appended to `RESULTS.md` in the format of §6 and a one-paragraph verdict. Do not start a phase before the previous gate is recorded.

## Phase 0: daily pilot (target: two weeks)

Goal: establish that the pipeline works and that the core claims (S1, S3) are at least plausible on daily data.

Runs:
1. Data: pull CRSP daily, build bars and features, pass leakage tests.
2. Baselines: `none`, `hmm` (K=3), `pca` (d=16). Forecaster with `film` injection, 5 seeds each, target log RV at h=1 and h=5. Evaluation universe: 200 largest names.
3. LSM: train with `recon` objective on the full universe. Freeze. Export contexts. Forecaster with `lsm` context, 5 seeds.
4. LSM no cross-section: train `lsm_noxs` (matched parameters). Forecaster with `lsm_noxs` context, 5 seeds.
5. First breadth point: train LSM on N=50 (top 50 by cap). Forecaster with that context.

Gate (on **validation**): S1 (`lsm` < `none` and `lsm` < `hmm`, interval excludes zero) and S3 (`lsm` < `lsm_noxs`, interval excludes zero). Also report whether `lsm` < `pca` and whether the N=50 gain is smaller than the full-universe gain.

Outcomes:
- Pass: proceed to Phase 1.
- S1 passes, S3 fails: the state helps but is not cross-sectional. Reassess; likely pivot to a per-asset representation thesis. Do not proceed to intraday.
- S1 fails: try `latent` objective and `concat` injection once each. If still failing, stop and report.

## Phase 1: daily, full grid (target: by Nov 30)

1. Full breadth sweep: LSM at N ∈ {50, 200, 1000, full}; forecaster at each; 5 seeds. This is the H3 figure.
2. Objective comparison: `recon` versus `latent` at full N.
3. Injection robustness: `film` versus `concat` for `lsm` and `none`.
4. `jump` baseline added.
5. `lsm_e2e` control.
6. Transfer (H4): train LSM on names ranked 1–1000, evaluate forecaster on names ranked 1001–1200 with `lsm` versus `none`.
7. Secondary target: return quantiles at h=1, 5.
8. Regime detection (H5): `z_t` versus HMM on event days, detection lead at matched false-alarm rate.
9. **Test evaluation, once.** Every configuration above, test split, written to RESULTS.md with git hash. Also record the validation-minus-test gap per configuration (the proxy-versus-gold number).

## Phase 2: minute (target: by Feb 15)

1. Build TAQ minute bars for 2024, 500 names.
2. Repeat Phase 0 runs at minute frequency, h ∈ {5, 30}, targets log RV and log spread.
3. Breadth sweep at N ∈ {50, 200, 500}.
4. Test evaluation, once.

## Phase 3: scale and seconds (conditional, Feb–Mar)

- Scale LSM width and depth at full N on daily and minute; report gain versus parameter count.
- Second bars for 50 names, one month, h ∈ {10, 60} s, only if minute S1 and S3 hold.
- One non-financial panel (electricity load or traffic sensors, public dataset) run through the identical pipeline for the generalization claim. Separate repo subfolder; same code.

## 4. Metrics, `src/mstate/eval/metrics.py`

- Log RV: MSE, QLIKE. Primary is MSE on log RV.
- Returns: mean pinball over the five quantiles; 90% interval coverage; directional accuracy at h as a secondary.
- Spread: MSE on log spread.
- Regime detection: for each labeled event, the first t at which a standardized change statistic on `z_t` (CUSUM on the first principal direction of z) exceeds a threshold set to give a fixed false-alarm rate on quiet days; same for HMM posterior of the high-vol state. Lead = HMM time minus z time, in bars.

Uncertainty: moving-block bootstrap over time with 1,000 resamples; block length 20 (daily) or 390 bars = 1 day (minute). Seeds pooled. Report 95% intervals on the difference versus `none`.

## 5. Breadth figure specification

x-axis: N assets the LSM was trained on (log scale). y-axis: percent reduction in test MSE on log RV versus `none`, with 95% interval. One line per frequency. Horizontal dashed lines for `hmm` and `pca`. This is the thesis's headline figure; produce it from `scripts/plot_breadth.py` so it regenerates from RESULTS.md.

## 6. Results table format (RESULTS.md)

```
| phase | freq | target | h | context | inject | N_lsm | objective | seeds | split | metric | value | ci_lo | ci_hi | lsm_params | fc_params | git |
```

One row per evaluated configuration. Rows are appended, never edited. A verdict paragraph follows each phase's block.

## 7. Reporting discipline

- Validation numbers may be looked at freely during a phase.
- Test numbers are produced by one scripted command at phase end, and the script refuses to run twice for the same phase without a `--force` flag that writes a note to CHANGELOG.md.
- Any metric, split, or configuration added after seeing validation results is marked `post_hoc: true` in its config and in RESULTS.md.
