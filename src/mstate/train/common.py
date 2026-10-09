"""Shared training utilities: AdamW, warmup + cosine schedule, clipping, autocast, early stop."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator

import torch
from torch import nn


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "ln" in n or "norm" in n else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def warmup_cosine(opt: torch.optim.Optimizer, warmup: int, total: int
                  ) -> torch.optim.lr_scheduler.LambdaLR:
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def autocast(device: torch.device, precision: str) -> contextlib.AbstractContextManager[None]:
    """bf16 autocast on CUDA when requested; full fp32 on CPU/MPS."""
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience, self.best, self.bad, self.best_epoch = patience, math.inf, 0, -1
        self.best_state: dict[str, torch.Tensor] | None = None

    def step(self, value: float, epoch: int, model: nn.Module) -> bool:
        """Record ``value``; return True if training should stop."""
        if value < self.best:
            self.best, self.bad, self.best_epoch = value, 0, epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience


def chunks(n: int, size: int) -> Iterator[slice]:
    for s in range(0, n, size):
        yield slice(s, min(n, s + size))
