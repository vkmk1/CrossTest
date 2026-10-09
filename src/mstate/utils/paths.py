"""Filesystem locations.

Bulk data lives under ``MSTATE_DATA_ROOT`` (default ``<repo>/data``) so it can be pointed at
external or cloud-synced storage without touching code. Runs live under ``MSTATE_RUNS_ROOT``
(default ``<repo>/runs``).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def data_root() -> Path:
    return Path(os.environ.get("MSTATE_DATA_ROOT", REPO_ROOT / "data"))


def runs_root() -> Path:
    return Path(os.environ.get("MSTATE_RUNS_ROOT", REPO_ROOT / "runs"))


def raw_dir(name: str) -> Path:
    return data_root() / "raw" / name


def bars_dir(freq: str) -> Path:
    return data_root() / "bars" / freq


def features_dir() -> Path:
    return data_root() / "features"


def config_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p
