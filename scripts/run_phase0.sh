#!/usr/bin/env bash
# Phase 0 grid (EXPERIMENTS.md): daily, target log RV, h in {1,5}, film injection, 5 seeds.
# Validation only. Never touches the test split.
#
#   scripts/run_phase0.sh                 # real data (needs data/raw/crsp_daily from wrds_pull)
#   scripts/run_phase0.sh --synthetic     # smoke test of the whole grid on synthetic data;
#                                         # writes to runs/_smoke and runs/_smoke_data only
# Optional overrides (smoke tests only; real Phase 0 runs use the configs as committed):
#   FC_ARGS="--seeds 0 --horizons 1 --set train.max_epochs=1"  LSM_ARGS="--set train.max_epochs=1"
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

SYN=()
if [[ "${1:-}" == "--synthetic" ]]; then
  shift
  SYN=(--synthetic)
  export MSTATE_DATA_ROOT=${MSTATE_DATA_ROOT:-runs/_smoke_data}
  export MSTATE_RUNS_ROOT=${MSTATE_RUNS_ROOT:-runs/_smoke}
fi
GROUP=${GROUP:-phase0}

echo "== tests"
$PY -m pytest -q tests/

if [[ ${#SYN[@]} -eq 0 ]]; then
  FEAT="${MSTATE_DATA_ROOT:-data}/features/daily.parquet"
  if [[ ! -f "$FEAT" ]]; then
    echo "== bars + features"
    $PY scripts/build_bars.py
    $PY scripts/build_features.py
  fi
fi

echo "== Stage 1: LSM full, lsm_noxs, N=50 (train on train split, freeze, export contexts)"
$PY -m mstate.train.train_lsm --config configs/lsm_daily.yaml "${SYN[@]+"${SYN[@]}"}" ${LSM_ARGS:-}
$PY -m mstate.train.train_lsm --config configs/lsm_daily.yaml --override configs/lsm_overrides/noxs.yaml \
    "${SYN[@]+"${SYN[@]}"}" ${LSM_ARGS:-}
$PY -m mstate.train.train_lsm --config configs/lsm_daily.yaml --override configs/lsm_overrides/n50.yaml \
    "${SYN[@]+"${SYN[@]}"}" ${LSM_ARGS:-}

echo "== Stage 2: forecaster ablations"
for abl in none hmm pca lsm lsm_noxs lsm_n50; do
  $PY scripts/run.py --config configs/fc_daily.yaml --ablation "configs/ablations/$abl.yaml" \
      --group "$GROUP" --skip-tests "${SYN[@]+"${SYN[@]}"}" ${FC_ARGS:-}
done

echo "== Phase 0 gate (validation)"
$PY scripts/phase0_gate.py --group "$GROUP"
