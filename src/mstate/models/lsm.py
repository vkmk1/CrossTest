"""Stage 1 latent state model (ARCHITECTURE.md §1).

Input ``x [B, T, N, F]`` (cross-sectionally normalized features), ``present [B, T, N]`` (asset
trades on that bar) and optional ``pool [B, N]`` (assets allowed to contribute to the market
summary and z_t; default all). Outputs

- ``z_m [B, T, d_m]``: market state (LayerNorm without affine parameters)
- ``z_a [B, T, N, d_a]``: per-asset state
- ``h [B, T, N, D]``: final hidden states
- ``pred [B, T, N, F]``: pretraining head, next-bar features (dropped after training)

Causality: temporal attention is causal; cross-asset attention mixes assets at the same t only.
So every output at t depends only on inputs at dates <= t (tests/test_lsm.py).

Cross-asset attention uses K learned summary tokens (Perceiver/Luna-style "pack and unpack") so
cost is O(N K). The summaries pool keys and values through the block's own projections, so a
cross-asset block has exactly the parameters of a temporal block plus the K x D summary queries.
That is what makes ``lsm_noxs`` (n_cross = 0, n_temporal = 6) parameter-matched within 2%.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mstate.models.blocks import MHA, MLP, Block, causal_mask, sinusoidal


class InducedCrossAssetBlock(nn.Module):
    """Pre-norm block over the asset axis via K summary tokens. ``h [B', N, D]``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, n_summary: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads, dropout)
        self.summary = nn.Parameter(torch.randn(n_summary, d_model) / math.sqrt(d_model))
        self.mlp = MLP(d_model, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        """h ``[B', N, D]``; pool ``[B', N]`` bool, which assets the summaries may read."""
        a = self.attn
        x = self.ln1(h)
        k, v = a.split(a.k(x)), a.split(a.v(x))  # [B', H, N, dk]
        q_s = a.split(self.summary[None].expand(h.shape[0], -1, -1))  # [B', H, K, dk]
        read = pool[:, None, None, :]  # [B', 1, 1, N]
        s_k = a.attend(q_s, k, k, read)  # pack: summaries read keys and values of pooled assets
        s_v = a.attend(q_s, k, v, read)
        out = a.attend(a.split(a.q(x)), s_k, s_v, None)  # unpack: every asset reads summaries
        h = h + self.drop(a.o(a.merge(out)))
        return h + self.mlp(self.ln2(h))


class LSM(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 4, n_temporal: int = 4,
                 n_cross: int = 2, summary_tokens: int = 64, d_market: int = 8, d_asset: int = 8,
                 dropout: float = 0.1, max_len: int = 512, interleave: bool = False,
                 asset_embed: bool = False):
        super().__init__()
        if asset_embed:
            raise NotImplementedError("lsm.asset_embed is off for Phase 0 (H4 transfer)")
        self.embed = nn.Linear(n_features, d_model)
        self.register_buffer("pe", sinusoidal(max_len, d_model), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.temporal = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_temporal))
        self.cross = nn.ModuleList(InducedCrossAssetBlock(d_model, n_heads, dropout, summary_tokens)
                                   for _ in range(n_cross))
        if interleave:
            seq, ti, xi = [], 0, 0
            while ti < n_temporal or xi < n_cross:
                if ti < n_temporal:
                    seq.append(("t", ti))
                    ti += 1
                if xi < n_cross:
                    seq.append(("x", xi))
                    xi += 1
        else:
            seq = [("t", i) for i in range(n_temporal)] + [("x", i) for i in range(n_cross)]
        self.order = seq
        self.ln_f = nn.LayerNorm(d_model)
        self.to_market = nn.Linear(d_model, d_market)
        self.market_norm = nn.LayerNorm(d_market, elementwise_affine=False)
        self.to_asset = nn.Linear(d_model, d_asset)
        self.head = nn.Linear(d_market + d_asset + d_model, n_features)
        self.d_market, self.d_asset = d_market, d_asset

    def forward(self, x: torch.Tensor, present: torch.Tensor,
                pool: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B, T, N, _ = x.shape
        D = self.embed.out_features
        pool_t = present & (pool[:, None, :] if pool is not None else True)  # [B, T, N]
        h = self.drop(self.embed(x) + self.pe[:T, None, :])

        # temporal mask per asset sequence: causal and key present (self always allowed)
        pres_bn = present.permute(0, 2, 1).reshape(B * N, T)  # [B*N, T]
        cm = causal_mask(T, x.device)
        eye = torch.eye(T, dtype=torch.bool, device=x.device)
        tmask = (cm[None] & (pres_bn[:, None, :] | eye[None]))[:, None]  # [B*N, 1, T, T]
        pool_bt = pool_t.reshape(B * T, N)

        for kind, i in self.order:
            if kind == "t":
                hb = h.permute(0, 2, 1, 3).reshape(B * N, T, D)
                hb = self.temporal[i](hb, tmask)
                h = hb.view(B, N, T, D).permute(0, 2, 1, 3)
            else:
                hb = self.cross[i](h.reshape(B * T, N, D), pool_bt)
                h = hb.view(B, T, N, D)
        h = self.ln_f(h)

        w = pool_t.to(h.dtype)[..., None]  # [B, T, N, 1]
        pooled = (h * w).sum(2) / w.sum(2).clamp_min(1.0)  # [B, T, D]
        z_m = self.market_norm(self.to_market(pooled))
        z_a = self.to_asset(h)
        head_in = torch.cat([z_m[:, :, None, :].expand(B, T, N, self.d_market), z_a, h], dim=-1)
        return {"z_m": z_m, "z_a": z_a, "h": h, "pred": self.head(head_in)}

    def context(self, x: torch.Tensor, present: torch.Tensor,
                pool: torch.Tensor | None = None) -> torch.Tensor:
        """``c = [z_t ; z_{i,t}]`` at the last bar of the window: ``[B, N, d_m + d_a]``."""
        out = self.forward(x, present, pool)
        z_m = out["z_m"][:, -1]
        z_a = out["z_a"][:, -1]
        return torch.cat([z_m[:, None, :].expand(-1, z_a.shape[1], -1), z_a], dim=-1)


def recon_loss(out: dict[str, torch.Tensor], x: torch.Tensor, present: torch.Tensor,
               target_ok: torch.Tensor | None = None, delta: float = 1.0) -> torch.Tensor:
    """Masked Huber between pred at t and features at t+1 (both inside the window)."""
    pred, tgt = out["pred"][:, :-1], x[:, 1:]
    m = (present[:, :-1] & present[:, 1:])[..., None].expand_as(tgt)
    if target_ok is not None:
        m = m & target_ok[:, 1:]
    loss = nn.functional.huber_loss(pred, tgt, reduction="none", delta=delta)
    return (loss * m).sum() / m.sum().clamp_min(1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
