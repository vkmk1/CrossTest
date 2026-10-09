"""Phase 0 gate on VALIDATION (EXPERIMENTS.md, Phase 0). Never reads test.

S1: lsm beats none and hmm (MSE on log RV, 95% block-bootstrap interval excludes zero).
S3: lsm beats lsm_noxs. Also reported: lsm vs pca, and whether the N=50 gain < full gain.

    python scripts/phase0_gate.py --group phase0
"""

from __future__ import annotations

import argparse
import json

from mstate.eval.evaluate import evaluate, latest_runs
from mstate.utils.paths import runs_root

PAIRS = [("S1_vs_none", "lsm", "none"), ("S1_vs_hmm", "lsm", "hmm"),
         ("S3_vs_noxs", "lsm", "lsm_noxs"), ("vs_pca", "lsm", "pca"),
         ("n50_vs_none", "lsm_n50", "none"), ("full_vs_none", "lsm", "none"),
         ("hmm_vs_none", "hmm", "none"), ("pca_vs_none", "pca", "none"),
         ("noxs_vs_none", "lsm_noxs", "none")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="phase0")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 5])
    a = ap.parse_args()
    g = runs_root() / a.group
    report: dict[str, dict] = {}
    for h in a.horizons:
        rows = {}
        for name, x, b in PAIRS:
            rx, rb = latest_runs(g, x, h), latest_runs(g, b, h)
            if not rx or not rb:
                rows[name] = {"missing": [s for s, r in ((x, rx), (b, rb)) if not r]}
                continue
            r = evaluate(rx, rb, "val", metrics=("mse",))["mse_pct_reduction_vs_baseline"]
            r["better_and_significant"] = r["value"] > 0 and r["ci_lo"] > 0
            rows[name] = r
        ok = lambda k: rows.get(k, {}).get("better_and_significant", False)  # noqa: E731
        rows["S1_pass"] = ok("S1_vs_none") and ok("S1_vs_hmm")
        rows["S3_pass"] = ok("S3_vs_noxs")
        if "value" in rows.get("n50_vs_none", {}) and "value" in rows.get("full_vs_none", {}):
            rows["n50_gain_smaller_than_full"] = (rows["n50_vs_none"]["value"]
                                                 < rows["full_vs_none"]["value"])
        report[f"h{h}"] = rows
    report["gate_pass"] = all(report[f"h{h}"]["S1_pass"] and report[f"h{h}"]["S3_pass"]
                              for h in a.horizons)
    out = g / "gate_val.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
