"""Run one forecaster ablation over all seeds and horizons, then evaluate on validation.

    python scripts/run.py --config configs/fc_daily.yaml --ablation configs/ablations/hmm.yaml
    python scripts/run.py ... --seeds 0 --horizons 1 --set train.max_epochs=2 --synthetic

The base config is never edited; the ablation file and ``--set`` dotlist are merged on top and
the merged config is saved in each run dir. Validation only: there is no flag for test here.
Run dirs: ``runs/<group>/<context>_h<h>_s<seed>_<stamp>``; summary:
``runs/<group>/summary_<context>_h<h>.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from omegaconf import OmegaConf

from mstate.eval.evaluate import evaluate, latest_runs
from mstate.train.train_fc import prepare, train_one
from mstate.utils import config
from mstate.utils.paths import REPO_ROOT, data_path, runs_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fc_daily.yaml")
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotlist overrides (recorded in config)")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--horizons", nargs="*", type=int)
    ap.add_argument("--group", default="phase0")
    ap.add_argument("--synthetic", action="store_true", help="in-memory synthetic data (smoke only)")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()

    if not a.skip_tests:  # CLAUDE.md: tests must pass before any evaluation run
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"], cwd=REPO_ROOT, check=True)

    extra = []
    if a.synthetic:
        extra.append(config.from_dotlist(["data.source=synthetic", "data.n_assets=400"]))
    extra.append(config.from_dotlist(a.set))  # --set wins
    cfg = config.merge(a.config, a.ablation, *extra)
    src = str(cfg.context.source)
    if src == "lsm_e2e" or cfg.context.get("mode") == "e2e":
        raise SystemExit("lsm_e2e is a Phase 1 control and is not implemented")
    if "contexts_path" in cfg.context:  # record which frozen LSM produced the contexts
        meta = data_path(cfg.context.contexts_path).with_suffix(".json")
        if meta.exists():
            m = json.loads(meta.read_text())
            cfg = OmegaConf.merge(cfg, {"context": {"lsm_params": m["lsm_params"],
                                                    "objective": m["objective"],
                                                    "n_lsm": m["universe"],
                                                    "lsm_run": m["run_dir"]}})
    seeds = a.seeds if a.seeds is not None else list(cfg.seeds)
    horizons = a.horizons if a.horizons is not None else list(cfg.horizons)

    for h in horizons:
        _, ds, _ = prepare(cfg, h)
        dirs = [train_one(cfg, ds, s, h, group=a.group).path for s in seeds]
        res = evaluate(dirs, None, "val")
        base = latest_runs(runs_root() / a.group, "none", h)
        if src != "none" and base:
            res = evaluate(dirs, base, "val")
        res["runs"] = [str(d) for d in dirs]
        out = runs_root() / a.group / f"summary_{src}_h{h}.json"
        out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
