"""Evaluation with moving-block bootstrap intervals (EXPERIMENTS.md §4, §6, §7).

Seeds are pooled: per date, the loss is averaged over all (asset, seed) predictions, giving one
daily loss series per configuration; the bootstrap resamples blocks of dates.

Validation (default) reads ``predictions_val.parquet`` written by training. Test is the only
code path allowed to load held-out rows; it re-predicts from each run's saved model, refuses to
run twice for a phase without ``--force`` (which writes a CHANGELOG note), and appends rows to
RESULTS.md.

Examples::

    python -m mstate.eval.evaluate --runs 'runs/phase0/lsm_h1_s*' --baseline 'runs/phase0/none_h1_s*'
    python -m mstate.eval.evaluate --split test --phase 0 --runs ... --baseline ...  # once, at end
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mstate.eval.bootstrap import Interval, mean_ci, pct_reduction_ci
from mstate.eval.metrics import ROW_METRICS
from mstate.utils.paths import REPO_ROOT, runs_root
from mstate.utils.run_dir import git_hash


def expand(patterns: list[str] | str) -> list[Path]:
    pats = [patterns] if isinstance(patterns, str) else patterns
    out = sorted({Path(p) for pat in pats for p in glob.glob(pat)
                  if (Path(p) / "config.yaml").exists()})
    if not out:
        raise FileNotFoundError(f"no run dirs match {pats}")
    return out


def load_predictions(run_dirs: list[Path], split: str = "val") -> pd.DataFrame:
    if split == "test":
        return pd.concat([_predict_test(r) for r in run_dirs], ignore_index=True)
    parts = []
    for r in run_dirs:
        p = pd.read_parquet(r / f"predictions_{split}.parquet")
        p["run"] = r.name
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def _predict_test(run_dir: Path) -> pd.DataFrame:
    """The single place where test-period rows are loaded."""
    import torch

    from mstate.train.train_fc import build_model, predict, prepare
    from mstate.utils.seeding import device

    cfg = OmegaConf.load(run_dir / "config.yaml")
    _, ds, _ = prepare(cfg, int(cfg.horizon), allow_test=True)
    dev = device()
    model = build_model(cfg, ds)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    ds.to(dev)
    p = predict(model.to(dev), ds, "test")
    p["run"] = run_dir.name
    return p


def daily_loss(pred: pd.DataFrame, metric: str) -> pd.Series:
    loss = ROW_METRICS[metric](pred["y"].to_numpy(np.float64), pred["yhat"].to_numpy(np.float64))
    return pd.Series(loss, index=pred["date"]).groupby(level=0).mean().sort_index()


def summarize(pred: pd.DataFrame, metric: str, block_len: int, n_resamples: int) -> Interval:
    return mean_ci(daily_loss(pred, metric).to_numpy(), block_len, n_resamples)


def compare(pred: pd.DataFrame, base: pd.DataFrame, metric: str, block_len: int,
            n_resamples: int) -> Interval:
    """Percent reduction in ``metric`` vs the baseline on the dates both cover."""
    a, b = daily_loss(pred, metric), daily_loss(base, metric)
    common = a.index.intersection(b.index)
    return pct_reduction_ci(a[common].to_numpy(), b[common].to_numpy(), block_len, n_resamples)


def evaluate(runs: list[Path], baseline: list[Path] | None, split: str = "val",
             metrics: tuple[str, ...] = ("mse", "qlike")) -> dict[str, Any]:
    cfg = OmegaConf.load(runs[0] / "config.yaml")
    bl, nr = int(cfg.eval.block_len), int(cfg.eval.bootstrap_resamples)
    pred = load_predictions(runs, split)
    base = load_predictions(baseline, split) if baseline else None
    out: dict[str, Any] = {"split": split, "context": str(cfg.context.source),
                           "horizon": int(cfg.horizon), "seeds": len(runs),
                           "n_dates": int(pred["date"].nunique())}
    for m in metrics:
        iv = summarize(pred, m, bl, nr)
        out[m] = {"value": iv.value, "ci_lo": iv.lo, "ci_hi": iv.hi}
        if base is not None:
            d = compare(pred, base, m, bl, nr)
            out[f"{m}_pct_reduction_vs_baseline"] = {"value": d.value, "ci_lo": d.lo, "ci_hi": d.hi,
                                                     "excludes_zero": d.excludes_zero()}
    return out


# --- test-split discipline ----------------------------------------------------------------------
def _lock(phase: str, runs: list[Path]) -> Path:
    """One lock per (phase, configuration) so each configuration's test number is produced once."""
    cfg = OmegaConf.load(runs[0] / "config.yaml")
    key = f"{cfg.context.source}_{cfg.inject}_{cfg.target}_h{cfg.horizon}"
    return runs_root() / "test_evals" / f"phase{phase}_{key}.done"


def _results_row(res: dict[str, Any], runs: list[Path], phase: str, metric: str) -> str:
    cfg = OmegaConf.load(runs[0] / "config.yaml")
    m = json.loads((runs[0] / "metrics.json").read_text())
    ctx = cfg.context
    n_lsm = ctx.get("n_lsm", "full" if str(ctx.source).startswith("lsm") else "-")
    v = res[metric]
    cells = [phase, cfg.freq, cfg.target, res["horizon"], ctx.source, cfg.inject, n_lsm,
             ctx.get("objective", "-"), res["seeds"], res["split"], metric,
             f"{v['value']:.5f}", f"{v['ci_lo']:.5f}", f"{v['ci_hi']:.5f}",
             ctx.get("lsm_params", "-"), m.get("fc_params", "-"), git_hash()]
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--phase", help="required for --split test")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", help="write the result JSON here")
    a = ap.parse_args()

    runs = expand(a.runs)
    if a.split == "test":
        if not a.phase:
            ap.error("--split test requires --phase")
        lock = _lock(a.phase, runs)
        if lock.exists() and not a.force:
            raise SystemExit(f"test evaluation for phase {a.phase} already ran ({lock}); "
                             "use --force, which records a note in CHANGELOG.md")
        if lock.exists():
            with open(REPO_ROOT / "CHANGELOG.md", "a") as f:
                f.write(f"\n- {date.today()}: test evaluation for phase {a.phase} re-run with "
                        f"--force on {a.runs} (git {git_hash()}).\n")

    res = evaluate(runs, expand(a.baseline) if a.baseline else None, a.split)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
    if a.split == "test":
        with open(REPO_ROOT / "RESULTS.md", "a") as f:
            for m in ("mse", "qlike"):
                f.write(_results_row(res, runs, str(a.phase), m) + "\n")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.touch()


if __name__ == "__main__":
    main()


def latest_runs(group_dir: Path, source: str, horizon: int) -> list[Path]:
    """Most recent completed run dir per seed for ``<source>_h<h>_s<seed>_<stamp>``."""
    best: dict[str, Path] = {}
    for d in sorted(group_dir.glob(f"{source}_h{horizon}_s*")):
        if not (d / "predictions_val.parquet").exists():
            continue
        seed = d.name[len(f"{source}_h{horizon}_"):].split("_")[0]
        best[seed] = d  # sorted by stamp, so the last wins
    return [best[k] for k in sorted(best)]
