"""Stage 2 forecaster (ARCHITECTURE.md §2). This module never changes between ablations.

Input ``x [B, T, F]`` (the asset's own bars) and ``c [B, T, d]`` (context over the lookback).
Output ``[B, n_out]``. Predictions are returned on the target's original scale: the head
predicts a standardized value that is mapped back with train-period target moments.
"""

from __future__ import annotations

import torch
from torch import nn

from mstate.models.blocks import MHA, Block, causal_mask, sinusoidal


class FiLM(nn.Module):
    """``h <- (1 + gamma(c)) * h + beta(c)``; zero-initialized output layer, so identity at init."""

    def __init__(self, context_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim, d_model), nn.GELU(),
                                 nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(c).chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class CrossAttn(nn.Module):
    """Queries from h, keys/values from the context sequence, causal over time."""

    def __init__(self, context_dim: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(context_dim, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads, dropout)

    def forward(self, h: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return h + self.attn(self.ln(h), self.proj(c), mask)


class Forecaster(nn.Module):
    def __init__(self, n_features: int, context_dim: int = 16, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1, inject: str = "film",
                 max_len: int = 512, n_out: int = 1, y_mean: float = 0.0, y_std: float = 1.0):
        super().__init__()
        if inject not in ("film", "concat", "xattn"):
            raise ValueError(inject)
        self.inject = inject
        in_dim = n_features + context_dim if inject == "concat" else n_features
        self.embed = nn.Linear(in_dim, d_model)
        self.register_buffer("pe", sinusoidal(max_len, d_model), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        if inject == "film":
            self.film = nn.ModuleList(FiLM(context_dim, d_model) for _ in range(n_layers + 1))
        elif inject == "xattn":
            self.xattn = nn.ModuleList(CrossAttn(context_dim, d_model, n_heads, dropout)
                                       for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_out)
        self.register_buffer("y_mean", torch.tensor(float(y_mean)))
        self.register_buffer("y_std", torch.tensor(float(y_std)))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x ``[B, T, F]``, c ``[B, T, d]`` -> ``[B, n_out]`` on the target scale."""
        T = x.shape[1]
        if self.inject == "concat":
            x = torch.cat([x, c], dim=-1)
        h = self.drop(self.embed(x) + self.pe[:T])
        mask = causal_mask(T, x.device)
        if self.inject == "film":
            h = self.film[0](h, c)
        for i, blk in enumerate(self.blocks):
            h = blk(h, mask)
            if self.inject == "film":
                h = self.film[i + 1](h, c)
            elif self.inject == "xattn":
                h = self.xattn[i](h, c, mask)
        out = self.head(self.ln_f(h[:, -1]))
        return self.y_mean + self.y_std * out


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
