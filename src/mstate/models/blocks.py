"""Shared building blocks: sinusoidal positions, masked multi-head attention, pre-norm blocks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal(length: int, dim: int) -> torch.Tensor:
    """``[length, dim]`` fixed sinusoidal position encoding."""
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[L, L]``, True where attention is allowed (key index <= query index)."""
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


class MHA(nn.Module):
    """Multi-head attention with separate query and key/value inputs and a boolean mask.

    ``mask`` is broadcastable to ``[B, heads, Lq, Lk]``, True = attend. Rows with no allowed key
    produce zeros instead of NaN.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.h, self.dk).transpose(1, 2)

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).reshape(b, n, self.h * self.dk)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               mask: torch.Tensor | None) -> torch.Tensor:
        """q, k, v already split into heads: ``[B, h, L, dk]``."""
        p = self.dropout if self.training else 0.0
        if mask is None:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        any_key = mask.any(-1, keepdim=True)
        return torch.where(any_key, out, torch.zeros_like(out))

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.attend(self.split(self.q(xq)), self.split(self.k(xkv)), self.split(self.v(xkv)),
                          mask)
        return self.o(self.merge(out))


class MLP(nn.Sequential):
    def __init__(self, d_model: int, dropout: float, mult: int = 4):
        super().__init__(nn.Linear(d_model, mult * d_model), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(mult * d_model, d_model), nn.Dropout(dropout))


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN x) then x + MLP(LN x)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads, dropout)
        self.mlp = MLP(d_model, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, mask))
        return x + self.mlp(self.ln2(x))
