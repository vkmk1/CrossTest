"""Train the Stage 2 forecaster for one (config, horizon, seed). Recipe: ARCHITECTURE.md §2.5.

Artifacts in the run dir: config.yaml, git_hash.txt, env.txt, log.txt, history.json,
loss_curve.png, model.pt, predictions_val.parquet, metrics.json.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from mstate.data.contexts import build_contexts
from mstate.data.datasets import ForecastData
from mstate.data.source import load_feature_frame
from mstate.data.splits import Splits
from mstate.eval.metrics import mse, qlike
from mstate.models.forecaster import Forecaster, n_params
from mstate.train.common import EarlyStopping, autocast, make_optimizer, warmup_cosine
from mstate.utils.run_dir import RunDir
from mstate.utils.seeding import device as default_device
from mstate.utils.seeding import seed_everything


def prepare(cfg: DictConfig, horizon: int, allow_test: bool = False
            ) -> tuple[pd.DataFrame, ForecastData, Splits]:
    """Seed-independent: the feature frame, the dataset, and the context array."""
    splits = Splits.load(str(cfg.freq))
    df = load_feature_frame(cfg, splits, allow_test)
    ds = ForecastData(df, list(cfg.features), str(cfg.target), horizon, int(cfg.lookback),
                      int(cfg.eval_universe), splits, allow_test=allow_test,
                      context_dim=int(cfg.context.dim))
    C, _ = build_contexts(OmegaConf.to_container(cfg.context, resolve=True), df, ds, splits)
    ds.set_contexts(C)
    return df, ds, splits


def build_model(cfg: DictConfig, ds: ForecastData) -> Forecaster:
    m = cfg.model
    return Forecaster(len(cfg.features), int(cfg.context.dim), int(m.d_model), int(m.n_heads),
                      int(m.n_layers), float(m.dropout), str(cfg.inject),
                      max_len=int(cfg.lookback), y_mean=ds.y_mean, y_std=ds.y_std)


@torch.no_grad()
def predict(model: Forecaster, ds: ForecastData, split: str, batch_size: int = 4096,
            precision: str = "fp32") -> pd.DataFrame:
    model.eval()
    dev = next(model.parameters()).device
    ys, yh, idx = [], [], []
    for x, c, y, b in ds.batches(split, batch_size):
        with autocast(dev, precision):
            out = model(x, c)[:, 0]
        ys.append(y.float().cpu().numpy())
        yh.append(out.float().cpu().numpy())
        idx.append(b)
    if not idx:
        return pd.DataFrame(columns=["date", "permno", "y", "yhat"])
    ii = np.concatenate(idx)
    return pd.DataFrame({"date": ds.calendar[ii[:, 0]], "permno": ds.permnos[ii[:, 1]],
                         "y": np.concatenate(ys), "yhat": np.concatenate(yh)})


def _plot(history: list[dict[str, float]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ep, [h["train_mse"] for h in history], label="train MSE")
    ax.plot(ep, [h["val_mse"] for h in history], label="val MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on log RV")
    ax.set_title(title, fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_one(cfg: DictConfig, ds: ForecastData, seed: int, horizon: int,
              group: str | None = None, name: str | None = None,
              dev: torch.device | None = None) -> RunDir:
    run_cfg = OmegaConf.merge(cfg, {"seed": seed, "horizon": horizon})
    name = name or f"{cfg.context.source}_h{horizon}_s{seed}"
    rd = RunDir.create(run_cfg, name, group)
    log = rd.logger()
    seed_everything(seed)
    dev = dev or default_device()
    ds.to(dev)
    model = build_model(cfg, ds).to(dev)
    tr = cfg.train
    bs = int(tr.batch_size)
    steps_per_epoch = math.ceil(ds.n("train") / bs)
    total = steps_per_epoch * int(tr.max_epochs)
    opt = make_optimizer(model, float(tr.lr), float(tr.weight_decay))
    sched = warmup_cosine(opt, int(tr.warmup_steps), total)
    stopper = EarlyStopping(int(tr.patience))
    rng = np.random.default_rng(seed)
    log.info(f"{name}: device={dev} params={n_params(model):,} train={ds.n('train'):,} "
             f"val={ds.n('val'):,} steps/epoch={steps_per_epoch}")

    history: list[dict[str, Any]] = []
    for epoch in range(1, int(tr.max_epochs) + 1):
        model.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for x, c, y, _ in ds.batches("train", bs, shuffle=True, rng=rng):
            with autocast(dev, str(tr.precision)):
                pred = model(x, c)[:, 0]
            loss = torch.mean((pred.float() - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.grad_clip))
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            cnt += len(y)
        val = predict(model, ds, "val", precision=str(tr.precision))
        val_mse = float(mse(val["y"].to_numpy(), val["yhat"].to_numpy()).mean())
        history.append({"epoch": epoch, "train_mse": tot / cnt, "val_mse": val_mse,
                        "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        log.info(f"epoch {epoch:3d} train {tot / cnt:.4f} val {val_mse:.4f} "
                 f"({time.time() - t0:.0f}s)")
        if stopper.step(val_mse, epoch, model):
            log.info(f"early stop at epoch {epoch}; best epoch {stopper.best_epoch}")
            break

    assert stopper.best_state is not None
    model.load_state_dict(stopper.best_state)
    torch.save(stopper.best_state, rd.path / "model.pt")
    (rd.path / "history.json").write_text(json.dumps(history, indent=1))
    _plot(history, rd.path / "loss_curve.png", f"{name} ({cfg.get('data', {}).get('source', 'real')})")
    val = predict(model, ds, "val", precision=str(tr.precision))
    val.to_parquet(rd.path / "predictions_val.parquet", index=False)
    rd.write_metrics({
        "split": "val", "best_epoch": stopper.best_epoch, "epochs_run": len(history),
        "val_mse": float(mse(val["y"].to_numpy(), val["yhat"].to_numpy()).mean()),
        "val_qlike": float(qlike(val["y"].to_numpy(), val["yhat"].to_numpy()).mean()),
        "n_val": len(val), "n_train": ds.n("train"), "fc_params": n_params(model),
    })
    log.info(f"done: best val MSE {stopper.best:.4f} at epoch {stopper.best_epoch}")
    return rd
