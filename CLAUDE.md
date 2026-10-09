# CLAUDE.md

Standing instructions for Claude Code working in this repository. Read SPEC.md, ARCHITECTURE.md, DATA.md and EXPERIMENTS.md before writing code.

## What this project is

A two-stage deep learning system for financial time series:

1. **Stage 1, latent state model (LSM).** A cross-sectional transformer trained self-supervised on a panel of many assets. It outputs a market-wide state vector `z_t` and per-asset states `z_{i,t}` at each timestamp. It is trained once per frequency, then frozen.
2. **Stage 2, forecaster.** A fixed-design sequence model that predicts a target for one asset at horizon `h`, given recent bars for that asset plus a context vector. The context vector is swapped between runs: the LSM state, an HMM regime posterior, PCA factors, or zeros.

The research question is whether a learned cross-sectional market state improves forecasting over the baselines, and whether the gain grows with the breadth of the cross-section the state is learned from. See SPEC.md.

## Non-negotiable rules

1. **The forecaster never changes between ablations.** Same architecture, depth, width, optimizer, schedule, seed set, early-stopping rule. Only the context input and the injection mechanism are config switches. If you think the forecaster needs a change, make it, re-run every ablation, and record it in `CHANGELOG.md`.
2. **Stage 1 is trained separately and frozen before Stage 2 sees it.** No end-to-end training in the main experiments. The end-to-end variant exists only as a named ablation (`lsm.mode: e2e`).
3. **Held-out periods are never touched during development.** Splits are defined in DATA.md and `configs/splits.yaml`. Code that reads a held-out period outside `evaluate.py` is a bug. Validation is for model selection; test is reported once, at the end of each phase, and the number is written to `RESULTS.md` with the git hash.
4. **No lookahead.** Every feature at timestamp `t` uses only data with timestamp `≤ t`. Every target at `t` uses data in `(t, t+h]`. Cross-sectional normalization uses only the cross-section at `t`. Rolling statistics are causal. Add an assertion in the dataset class that checks this on a sample of rows.
5. **Every run is reproducible.** One YAML config fully specifies a run. Seeds are set for Python, NumPy, and PyTorch. Each run writes `config.yaml`, `git_hash.txt`, `metrics.json`, and `env.txt` to its own directory under `runs/`.
6. **Pre-registered metrics only.** The metrics and success criteria in EXPERIMENTS.md are the ones reported. Do not add a metric after seeing results without recording that it was added post hoc.
7. **Raw TAQ is read once.** Aggregate to bars in a single pass and store as Parquet. No experiment code touches raw trade or quote tables.
8. **Prefer boring choices.** Standard pre-norm transformer blocks, AdamW, cosine schedule, gradient clipping at 1.0, bf16 on GPU. Novelty lives in the research design, not the training recipe.

## Repository layout

```
.
├── CLAUDE.md
├── SPEC.md  ARCHITECTURE.md  DATA.md  EXPERIMENTS.md
├── RESULTS.md            # one row per completed evaluation, appended only
├── CHANGELOG.md          # any change that affects comparability
├── configs/
│   ├── splits.yaml       # train / val / test date ranges per frequency
│   ├── lsm_daily.yaml    # Stage 1 config for daily
│   ├── fc_daily.yaml     # Stage 2 config for daily
│   └── ablations/        # one file per context source
├── data/
│   ├── raw/              # never committed; WRDS pulls land here
│   ├── bars/             # Parquet, one file per (frequency, year)
│   └── features/         # Parquet, feature panel per frequency
├── src/mstate/
│   ├── data/             # wrds pulls, bar building, feature engineering, datasets
│   ├── models/           # lsm.py, forecaster.py, baselines.py (hmm, pca, jump)
│   ├── train/            # train_lsm.py, train_fc.py, common loop utilities
│   ├── eval/             # evaluate.py, metrics.py, bootstrap.py
│   └── utils/            # seeding, logging, config loading
├── scripts/              # thin CLIs: build_bars.py, build_features.py, run.py
├── tests/                # leakage tests, shape tests, baseline sanity tests
├── runs/                 # never committed
└── notebooks/            # exploration only, never imported by src
```

## Conventions

- Python 3.11, PyTorch 2.x, Polars for data, Hydra or plain YAML + OmegaConf for configs, `hmmlearn` for the HMM baseline, `scikit-learn` for PCA.
- Type hints everywhere. `ruff` and `mypy` clean.
- Tensors are `[batch, time, assets, features]` for panels and `[batch, time, features]` for single-asset sequences. Document shapes in docstrings.
- Log to stdout and to `runs/<id>/log.txt`. Use Weights & Biases only if `WANDB_API_KEY` is set; never make it required.
- Tests must pass before any evaluation run: `pytest tests/`.
- Commit after each working unit. Commit messages describe the change in one line, then a body if needed.

## How to start

1. `pip install -e .` from the `pyproject.toml` you create.
2. Implement `src/mstate/data/wrds_pull.py` for CRSP daily (DATA.md §2). Pull 2010–2025, full universe, to `data/raw/crsp_daily/`.
3. Implement bar and feature construction for daily (DATA.md §4) and write `tests/test_leakage.py`.
4. Implement the baselines (`hmm`, `pca`, `none`) and the forecaster. Run the forecaster with `context: none` and confirm it trains and evaluates on validation.
5. Implement the LSM, train it on daily, freeze, and run the full ablation grid for Phase 0 (EXPERIMENTS.md).
6. Append Phase 0 results to `RESULTS.md` and stop. The user decides whether to proceed.

## What not to do

- Do not add news, fundamentals, or any non-price data. The information set is prices, volumes, and (intraday) quotes only, at every frequency.
- Do not change the sampling frequency mid-phase.
- Do not tune the forecaster on test.
- Do not build the TAQ pipeline before Phase 0 passes.
- Do not add a market simulator.
