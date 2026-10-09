"""Per-run output directory: config.yaml, git_hash.txt, env.txt, metrics.json, log.txt."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mstate.utils.paths import REPO_ROOT, runs_root


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT) != 0
        return h + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _env_text() -> str:
    import numpy
    import pandas
    import sklearn
    import torch

    lines = [
        f"python {sys.version.split()[0]}",
        f"platform {platform.platform()}",
        f"torch {torch.__version__} cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()}",
        f"numpy {numpy.__version__}",
        f"pandas {pandas.__version__}",
        f"sklearn {sklearn.__version__}",
    ]
    try:
        import hmmlearn

        lines.append(f"hmmlearn {hmmlearn.__version__}")
    except ImportError:
        pass
    return "\n".join(lines) + "\n"


class RunDir:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, cfg: DictConfig, name: str, group: str | None = None) -> RunDir:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = runs_root() / group if group else runs_root()
        rd = cls(base / f"{name}_{stamp}")
        OmegaConf.save(cfg, rd.path / "config.yaml", resolve=True)
        (rd.path / "git_hash.txt").write_text(git_hash() + "\n")
        (rd.path / "env.txt").write_text(_env_text())
        return rd

    def logger(self, name: str = "mstate") -> logging.Logger:
        log = logging.getLogger(f"{name}.{self.path.name}")
        log.setLevel(logging.INFO)
        log.propagate = False
        if not log.handlers:
            fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
            for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(self.path / "log.txt")):
                h.setFormatter(fmt)
                log.addHandler(h)
        return log

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        (self.path / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
