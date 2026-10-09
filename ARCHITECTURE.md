# ARCHITECTURE.md: the two networks, tensor by tensor

Notation: B batch, T lookback length, N assets in the panel, F features per bar, D model width, d state dimension. All blocks are pre-norm transformer blocks (LayerNorm → attention → residual → LayerNorm → MLP → residual) unless stated.

## 1. Stage 1: latent state model (LSM)

### 1.1 Input

A window of the full panel: `X ∈ ℝ^{B × T × N × F}`. Features are defined in DATA.md §4 and are already cross-sectionally normalized per timestamp. Missing assets (not trading on that bar) carry a mask `M ∈ {0,1}^{B × T × N}`.

Asset identity is not given as an embedding by default (so the model transfers to unseen assets, H4). An optional learned asset embedding is a config flag `lsm.asset_embed: false`.

### 1.2 Blocks, in order

| Block | Operation | Output shape | Notes |
|---|---|---|---|
| Embed | Linear(F → D) + learned time-of-day/day-of-week encoding (intraday) or sinusoidal position over T (daily) | B×T×N×D | |
| Temporal ×N_t | Self-attention over T, independently per asset, causal mask | B×T×N×D | Reshape to (B·N)×T×D |
| Cross-asset ×N_x | Self-attention over N at each timestamp, masked by M | B×T×N×D | Reshape to (B·T)×N×D. For large N use K learned summary tokens (K=64) that attend to all assets, then assets attend to summaries (Perceiver-style), to keep cost O(N·K). |
| Bottleneck | Masked mean-pool over N → Linear(D → d_m) for z_t; Linear(D → d_a) per asset for z_{i,t} | z_t: B×T×d_m, z_a: B×T×N×d_a | d_m = d_a = 8 default |
| Pretraining head | Linear(d_m + d_a + D → F) predicting next-bar features per asset | B×T×N×F | Dropped after training |

Default depth N_t = 4, N_x = 2, D = 128, 4 heads, dropout 0.1. Roughly 1–2M parameters. This is deliberately small for Phase 0; scale in Phase 2.

Interleaving: run all temporal blocks first, then cross-asset blocks (simplest). A config flag allows alternating (T, X, T, X).

### 1.3 Pretraining objective

Two options, config `lsm.objective`:

- `recon` (default for Phase 0): predict next-bar features `X_{t+1}` from the state. Loss is masked Huber on normalized features. Simple, stable, easy to debug.
- `latent` (Phase 1+): JEPA-style. A target encoder (EMA copy of the online encoder) produces `z^{target}_{t+1}`; the online model predicts it from `z_t, z_{i,t}` through a small MLP predictor. Loss is MSE in latent space plus a VICReg variance/covariance regularizer on `z` to prevent collapse. Regularizer weights: variance 1.0, covariance 0.04. The claim behind this objective is that predicting only in latent space keeps what persists (the regime) and discards tick noise; test it against `recon`, do not assume it.

Both objectives are strictly causal: targets at t+1 are never visible to the encoder at t.

### 1.4 Regularization of the state

- `z_t` is passed through a LayerNorm with no affine parameters so its scale is fixed.
- Optional: a small L1 penalty on `z_t` to encourage sparse, interpretable dimensions (`lsm.z_l1: 0.0` default).

### 1.5 Output used downstream

For asset i at time t the context vector is `c_{i,t} = [z_t ; z_{i,t}] ∈ ℝ^{d}`, d = 16. Computed once over the whole dataset with the frozen LSM and stored as a Parquet column set, so Stage 2 never loads Stage 1 weights.

Ablation `lsm_noxs`: identical model with N_x = 0 and the parameter budget moved into N_t so counts match (N_t = 6).

## 2. Stage 2: forecaster

### 2.1 Input

For one asset: its own recent bars `x ∈ ℝ^{B × T × F}` and a context sequence `c ∈ ℝ^{B × T × d}` (the context at each timestamp of the lookback, so the forecaster sees how the state evolved, not only its current value). For `none`, c is zeros of the right shape.

### 2.2 Blocks

| Block | Operation | Output shape |
|---|---|---|
| Embed | Linear(F → D) + positional encoding | B×T×D |
| Context injection | see 2.3 | B×T×D |
| Sequence encoder ×M | causal self-attention over T | B×T×D |
| Pool | take last timestep | B×D |
| Output head | Linear(D → outputs) | B×outputs |

Default M = 3, D = 128, 4 heads, dropout 0.1, about 600k parameters. **This block never changes between ablations.**

### 2.3 Context injection, config `fc.inject`

- `concat`: `Linear(F + d → D)` at embed. The baseline.
- `film` (default): context produces per-timestep scale and shift `γ_t, β_t = MLP(c_t)`; applied as `h_t ← γ_t ⊙ h_t + β_t` after the embed and after every encoder block. Usually strongest for slowly varying conditioning.
- `xattn`: one cross-attention layer per encoder block, queries from h, keys/values from c. Most expressive; most prone to overfit.

Phase 0 runs `film` only. Phase 1 adds `concat` as a robustness check.

### 2.4 Output head and loss

- Realized variance target: predict log RV; loss MSE on log RV (report QLIKE as a second metric).
- Return target: predict quantiles at τ ∈ {0.05, 0.25, 0.5, 0.75, 0.95} with pinball loss; report pinball and coverage. Quantiles are preferred over point forecasts because a regime state is expected to help the tails.
- Spread target (intraday): log spread, MSE.

### 2.5 Training recipe (fixed)

AdamW, lr 3e-4, weight decay 0.01, cosine decay with 500-step warmup, batch 256, gradient clip 1.0, bf16 autocast, early stopping on validation loss with patience 5 epochs, max 50 epochs. Five seeds per configuration; report mean and the bootstrap interval across time blocks, pooling seeds.

## 3. Baselines, `src/mstate/models/baselines.py`

- `HMMContext`: `hmmlearn.GaussianHMM(n_components=K, covariance_type="full")` fitted on train-period cross-sectional mean log RV (and optionally mean log spread). At inference, run the forward algorithm causally to get filtered posteriors; never use `predict_proba` on the full series (that smooths with future data).
- `JumpContext`: Nystrup, Kolm and Lindström jump model; implement the simple version with a fixed jump penalty λ chosen on validation.
- `PCAContext`: `sklearn.decomposition.PCA(n_components=d)` fitted on the train-period feature panel (assets × features stacked per timestamp); transform each timestamp's cross-section. Since the panel is N×F per timestamp, PCA is fitted on the N×F flattened vector of cross-sectional means and dispersions (DATA.md §4.3) so the dimension is fixed as N changes.

## 4. Parameter counts to report

Report LSM and forecaster parameter counts in every results table. The `lsm_noxs` ablation must be within 2% of `lsm`.

## 5. Compute expectations

Daily, full CRSP universe (~4,000 assets), 2010–2025: LSM pretraining is under an hour on one A100 at the default size. Minute, 500 assets, one year: a few hours. Forecaster runs are minutes each; the ablation grid at five seeds is a few hundred short runs, which is why they must be fully scripted.
