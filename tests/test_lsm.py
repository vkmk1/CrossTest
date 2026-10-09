"""LSM shape, causality, pooling and parameter-matching tests."""

from __future__ import annotations

import torch

from mstate.models.lsm import LSM, n_params, recon_loss
from mstate.utils import config


def _model(**kw: int) -> LSM:
    torch.manual_seed(0)
    return LSM(15, d_model=32, n_heads=4, n_temporal=kw.get("nt", 2), n_cross=kw.get("nx", 1),
               summary_tokens=8, dropout=0.0, max_len=20).eval()


def _batch(B: int = 2, T: int = 20, N: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, T, N, 15, generator=g)
    p = torch.rand(B, T, N, generator=g) > 0.1
    return x, p


def test_shapes_and_loss() -> None:
    m = _model()
    x, p = _batch()
    out = m(x, p)
    assert out["z_m"].shape == (2, 20, 8) and out["z_a"].shape == (2, 20, 12, 8)
    assert out["pred"].shape == x.shape
    assert m.context(x, p).shape == (2, 12, 16)
    loss = recon_loss(out, x, p)
    assert torch.isfinite(loss)
    loss.backward()


def test_causal_in_time() -> None:
    """Outputs at t do not change when every input after t is replaced."""
    m = _model()
    x, p = _batch()
    t = 11
    x2, p2 = x.clone(), p.clone()
    x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:]) * 10
    p2[:, t + 1:] = ~p2[:, t + 1:]
    a, b = m(x, p), m(x2, p2)
    for k in ("z_m", "z_a", "h", "pred"):
        torch.testing.assert_close(a[k][:, : t + 1], b[k][:, : t + 1])
    assert not torch.allclose(a["z_m"][:, t + 1:], b["z_m"][:, t + 1:])


def test_assets_outside_pool_do_not_write_market_state() -> None:
    """Breadth runs: non-pool assets read summaries but do not change z_t or pooled assets."""
    m = _model()
    x, p = _batch()
    pool = torch.zeros(2, 12, dtype=torch.bool)
    pool[:, :6] = True
    x2 = x.clone()
    x2[:, :, 6:] = torch.randn_like(x2[:, :, 6:]) * 10
    a, b = m(x, p, pool), m(x2, p, pool)
    torch.testing.assert_close(a["z_m"], b["z_m"])
    torch.testing.assert_close(a["z_a"][:, :, :6], b["z_a"][:, :, :6])


def test_cross_asset_attention_mixes_assets() -> None:
    m = _model()
    x, p = _batch()
    x2 = x.clone()
    x2[:, :, 0] += 5.0
    a, b = m(x, p), m(x2, p)
    assert not torch.allclose(a["z_a"][:, :, 1], b["z_a"][:, :, 1])
    n = _model(nx=0, nt=3)
    assert torch.allclose(n(x, p)["z_a"][:, :, 1], n(x2, p)["z_a"][:, :, 1])  # noxs: no mixing


def test_noxs_parameter_count_within_2pct() -> None:
    from mstate.train.train_lsm import build_lsm

    base = config.load("configs/lsm_daily.yaml")
    full = n_params(build_lsm(base))
    noxs = n_params(build_lsm(config.merge("configs/lsm_daily.yaml",
                                           "configs/lsm_overrides/noxs.yaml")))
    assert abs(full - noxs) / full < 0.02, (full, noxs)
