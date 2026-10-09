# DECISIONS

Interpretations of SPEC/ARCHITECTURE/DATA where the documents were silent, ambiguous, or
internally inconsistent. Each entry: what was decided, why, and where it lives. Review these
before Phase 0 results are interpreted; changing any of them after results exist goes in
CHANGELOG.md.

## Data

1. **Pandas + PyArrow instead of Polars.** The machine had <200 MB free when the package was
   installed and Polars 2.x did not fit. Daily scale (~15M rows) is fine in pandas. Revisit for
   TAQ minute bars. (`src/mstate/data/*`)
2. **Garman-Klass fallbacks and floor.** GK when O,H,L,C are all present (H/L widened to contain
   O and C); Parkinson if open is missing; squared return if H/L missing; variance floored at
   1e-8 before the log. (`features.garman_klass`)
3. **Scaled MAD.** Cross-sectional normalization divides by 1.4826 * MAD (sigma-consistent), so
   the [-5, 5] clip is in sigma units. `mkt_mad_f` stores the scaled value.
4. **lvol = log(1 + vol * |prc|)**, turn = vol / (shrout * 1000) (CRSP shrout is in thousands).
5. **Multi-day target** `y_lrv_h` = log(sum of daily GK variance over t+1..t+h), i.e. log of
   the h-day variance (not the sum of logs). Requires the asset on all h following market dates.
6. **Universe ranks** use market cap on the first market trading day of each calendar year;
   assets not trading that day are unranked for the year. The $1 filter uses each asset's first
   close in the year.
7. **The 252-valid-day filter is full-sample, as specified.** This is a mild survivorship choice
   (the universe at t knows whether an asset will reach 252 days). Feature values never use
   future data; the leakage tests keep row sets fixed when perturbing the future.
8. **Delisting returns are not merged** (DATA.md lists only `crsp.dsf` fields).
9. **CRSP CIZ.** WRDS froze the legacy SIZ tables (`crsp.dsf`) at the 2024 year-end update,
   so 2025 may only exist in `crsp.dsf_v2`. `wrds_pull --source ciz` maps CIZ to the legacy
   column names; its filters are verified on a one-month pull before use.

## Splits

10. **Purge semantics.** A train sample's whole span [t-L+1, t+h] lies in train. A val/test
    sample needs t and t+h in the split, but its input window may reach back into earlier
    splits. Requiring windows to start inside validation would drop the first 60 trading days
    of 2020, which is the COVID crash that DATA.md puts in validation on purpose.
    `purge_bars` is checked to be >= L+h.

## Forecaster

11. **Forecaster inputs are the asset's own raw base features**, z-scored with train-period
    moments, not the cross-sectionally normalized ones. With xs-normalized inputs the `none`
    baseline could not see its own volatility level, so any context carrying the market level
    would win trivially. Market-wide information enters only through the context slot.
12. **Context is zero-padded to d = 16 for every source** (`none` all zeros, `hmm` K = 3
    posteriors plus 13 zeros). The forecaster, including its FiLM MLPs, then has identical
    shape and parameter count (736,769) under every ablation.
13. **FiLM**: one 2-layer MLP (d -> D -> 2D) per injection point (after embed + after each of 3
    blocks), output layer zero-initialized so FiLM starts as the identity. The FiLM MLPs add
    139k parameters to the ~598k core; the spec's "about 600k" counts the core only.
14. **Targets are standardized internally** (head output * train std + train mean), so the loss
    is still MSE on log RV.
15. **bf16 autocast on CUDA only**; MPS/CPU run fp32.
16. **Window completeness**: a forecaster sample requires all L bars of the asset present and
    finite.

## Baselines

17. **PCA input dimension.** The mkt_* vector is 10-dim (5 features x median/MAD), fewer than
    d = 16. PCA is fitted on [mkt_t ; mkt_{t-1}] (20 dims, causal lag), inputs z-scored and
    components whitened on train. `lags: 1` is set in `configs/ablations/pca.yaml`.
18. **HMM**: fitted on train `mkt_median_lrv`, best of 5 random inits by log-likelihood, states
    ordered by mean (0 = calm). The context is the forward-filtered posterior from a log-space
    forward pass, never `predict_proba`.

## LSM

19. **Cross-asset block = induced ("pack/unpack") attention through K = 64 summary tokens that
    share the block's Q/K/V/O projections.** A cross block then has exactly the parameters of a
    temporal block plus K x D summary queries, so `lsm_noxs` (N_t = 6, N_x = 0) matches `lsm`
    within 1.35% (1,196,175 vs 1,212,559). A Set-Transformer ISAB (two full attention+MLP
    sub-blocks) would have needed N_t = 8 for `lsm_noxs` to match.
20. **Pretraining head reads [z_t ; z_{i,t} ; h_{i,t}]** as specified. Because h bypasses the
    bottleneck, nothing forces z to carry information. See the open question in the session
    summary; `head_input` is a candidate config switch.
21. **Recon targets** are next-bar features inside the window (positions 0..T-2 predict
    1..T-1), all 15 input features, masked Huber (delta 1).
22. **Training windows subsample up to `train.max_assets: 512` assets** (random per window) for
    memory; z_t is then a mean over a random subset. Export uses all assets.
23. **mkt_* inputs to the LSM are z-scored over time with train moments**, since they are raw
    levels; `xs_*` features are already normalized.
24. **Breadth runs (N < full)**: summaries and z_t pool over the LSM's own universe only;
    evaluation-universe assets outside it read the summaries but never write to them. Contexts
    are exported for eval-universe assets only, for dates through the end of validation.
25. **Missing bars inside a window** are zero-filled and masked as keys in temporal attention
    (a query can always attend to itself).
