# SPEC.md: research design

## 1. One-sentence question

Does a self-supervised market state learned jointly across a large cross-section of assets improve single-asset forecasting over classical regime estimators, and does that improvement grow with the breadth of the cross-section it is learned from?

## 2. Why two models

A forecaster trained end to end will learn whatever helps its loss; nothing in it is identifiable as a "regime." Separating the system into a state model and a forecaster turns the question into an ablation: hold the forecaster fixed, swap only what enters its context slot, and compare. Everything that differs between runs is the context source. The design is deliberately not adversarial (not a GAN): the two networks cooperate, and the first is frozen before the second is trained.

```
panel of all assets ──► Stage 1: latent state model ──► z_t (market), z_{i,t} (asset)
                                                              │
   recent bars of one asset ──► Stage 2: forecaster ◄── context ∈ {z, HMM, PCA, none}
                                        │
                                        ▼
                              prediction for asset i at t+h
```

## 3. Hypotheses, pre-registered

- **H1 (usefulness).** With the forecaster fixed, LSM context yields lower out-of-sample loss than `none`, than HMM posteriors, and than PCA factors of equal dimension, at every frequency tested.
- **H2 (cross-section matters).** An LSM trained with cross-asset attention removed (per-asset only, equal parameter count) yields a smaller gain than the full LSM. If H2 fails, the "market state" is not cross-sectional and the thesis reverts to a per-asset representation study.
- **H3 (breadth scaling).** The gain from LSM context increases with the number of assets `N` the LSM was trained on (N ∈ {50, 200, 1000, full}), with the forecaster and evaluation set fixed.
- **H4 (transfer).** The LSM state helps forecasting on assets excluded from LSM training.
- **H5 (regime detection, secondary).** The market state `z_t` flags labeled event periods earlier than the HMM at matched false-alarm rate on quiet periods.

H1–H3 are the thesis. H4 and H5 are supporting results. H3 is the main-conference claim and the one figure the thesis is built around.

## 4. Targets and horizons

Targets are per asset. Default is next-period realized variance (log), because a regime state should primarily help second-moment prediction. Secondary targets are next-period log return (point and quantiles) and next-period average bid-ask spread (intraday only).

| Frequency | Bar | Horizon h | Lookback L | Data source |
|---|---|---|---|---|
| Daily | 1 trading day | 1, 5 days | 60 bars | CRSP daily |
| Minute | 1 minute, regular hours | 5, 30 min | 120 bars | TAQ aggregated |
| Second | 1 second, regular hours | 10, 60 s | 300 bars | TAQ aggregated |

Only daily is in scope until Phase 0 passes. Minute follows. Second is conditional on minute.

## 5. The context sources (the ablation grid)

| Name | Dimension | How produced |
|---|---|---|
| `none` | 0 | zeros; forecaster sees only its own bars |
| `hmm` | K (default 3) | Gaussian HMM on cross-sectional mean realized volatility, posterior state probabilities at t, fitted on train only, filtered (not smoothed) at inference |
| `jump` | K | Nystrup-style jump model on the same series; optional, stronger regime baseline |
| `pca` | d | first d principal components of the cross-sectional feature panel at t, loadings fitted on train only |
| `lsm` | d | frozen Stage 1 state `[z_t ; z_{i,t}]` |
| `lsm_noxs` | d | same as `lsm` but Stage 1 trained without cross-asset attention |
| `lsm_e2e` | d | Stage 1 and Stage 2 trained jointly on the forecasting loss (control: shows what "just a bigger forecaster" does) |

`d` is fixed across `pca`, `lsm`, `lsm_noxs`, `lsm_e2e` so comparisons are at equal context dimension. Default d = 16 (8 market + 8 asset).

## 6. Success criteria

Report on the test split only, once per phase, with 95% block-bootstrap intervals over time (block = 20 trading days daily, 1 day intraday).

- **S1.** `lsm` beats `none` and `hmm` on log-RV loss (QLIKE or MSE on log RV) with the bootstrap interval excluding zero. Effect size reported as percent loss reduction.
- **S2.** `lsm` beats `pca` at equal d with interval excluding zero.
- **S3.** `lsm` beats `lsm_noxs` with interval excluding zero.
- **S4.** Gain versus `none` is monotone non-decreasing in N across {50, 200, 1000, full}, and the gain at full is at least 1.5× the gain at 50.
- **S5 (regime detection).** On labeled event days, median detection lead of `z_t` over HMM is positive at matched false-alarm rate.

Phase 0 gate: S1 and S3 on daily validation (not test). If either fails, stop and reassess before building intraday.

## 7. What counts as the thesis contribution

1. A decoupled, frozen, self-supervised market state with an explicit market/asset bottleneck, evaluated by a fixed-forecaster ablation against HMM, jump-model, and PCA baselines.
2. The breadth-scaling curve (H3): first measurement of how forecasting gain from a learned regime state scales with cross-sectional coverage.
3. A cross-frequency replication (daily, minute, optionally second) with the same code and the same metrics.
4. Pre-registered success criteria and proxy-versus-gold reporting (validation versus untouched test), so the size of the selection effect is itself a reported number.

## 8. Out of scope

Market simulators, execution or trading strategy backtests, non-price data, reinforcement learning. These may be future work and must not enter the codebase.
