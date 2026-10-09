"""YAML + OmegaConf config loading. One merged YAML fully specifies a run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mstate.utils.paths import config_path


def load(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(config_path(path))
    assert isinstance(cfg, DictConfig)
    return cfg


def merge(base: str | Path, *overrides: str | Path | dict[str, Any] | DictConfig) -> DictConfig:
    """Load ``base`` and merge each override (path, dict, or DictConfig) on top, in order."""
    cfg = load(base)
    for ov in overrides:
        if isinstance(ov, (str, Path)):
            ov = load(ov)
        elif isinstance(ov, dict):
            ov = OmegaConf.create(ov)
        cfg = OmegaConf.merge(cfg, ov)
    assert isinstance(cfg, DictConfig)
    return cfg


def from_dotlist(items: list[str]) -> DictConfig:
    """``["train.max_epochs=2", "context.source=pca"]`` -> DictConfig."""
    return OmegaConf.from_dotlist(items)


def splits(freq: str) -> DictConfig:
    s = load("configs/splits.yaml")[freq]
    assert isinstance(s, DictConfig)
    return s
