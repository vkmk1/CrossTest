"""Train the Stage 1 LSM (``recon`` objective), freeze it, export contexts.

Contexts ``c_{i,t} = [z_t ; z_{i,t}]`` are written for every date up to the end of validation
(test only via ``allow_test``, which only the phase-end test script passes) for the assets in
the forecaster's evaluation universe, so Stage 2 never loads Stage 1 weights:
``<data_root>/features/contexts_<name>.parquet`` plus ``contexts_<name>.json`` metadata.

For breadth runs (N < full) the summaries and z_t pool only over the LSM's own universe;
evaluation assets outside it read the summaries but never write to them, so breadth changes
only what the LSM saw.

Usage::

    python -m mstate.train.train_lsm --config configs/lsm_daily.yaml [--override configs/lsm_overrides/noxs.yaml]
"""

from __future__ import annotations

import argparse
import os
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from mstate.data.lsm_data import LSMPanel
from mstate.data.source import load_feature_frame
from mstate.data.splits import Splits
from mstate.models.lsm import LSM, n_params, recon_loss
from mstate.train.common import EarlyStopping, autocast, make_optimizer, warmup_cosine
from mstate.utils import config
from mstate.utils.paths import features_dir
from mstate.utils.run_dir import RunDir
from mstate.utils.seeding import device as default_device
from mstate.utils.seeding import seed_everything


def build_lsm(cfg: DictConfig) -> LSM:
    m = cfg.model
    if cfg.objective != "recon":
        raise NotImplementedError("the latent (JEPA) objective is Phase 1")
    return LSM(len(cfg.features), int(m.d_model), int(m.n_heads), int(m.n_temporal),
               int(m.n_cross), int(m.summary_tokens), int(m.d_market), int(m.d_asset),
               float(m.dropout), max_len=int(cfg.lookback),
               interleave=bool(m.get("interleave", False)), asset_embed=bool(m.asset_embed))


@torch.no_grad()
def eval_loss(model: LSM, panel: LSMPanel, split: str, cfg: DictConfig, dev: torch.device) -> float:
    model.eval()
    rng = np.random.default_rng(0)  # fixed asset subsample so the val loss is comparable
    tot, n = 0.0, 0
    for x, p, f in panel.batches(split, int(cfg.train.batch_size), int(cfg.train.max_assets),
                                 False, rng, dev):
        with autocast(dev, str(cfg.train.precision)):
            out = model(x, p)
        tot += recon_loss(out, x, p, f).item() * len(x)
        n += len(x)
    return tot / max(n, 1)


def train(cfg: DictConfig, df: pd.DataFrame, splits: Splits, group: str = "lsm",
          dev: torch.device | None = None) -> tuple[LSM, RunDir, LSMPanel]:
    rd = RunDir.create(cfg, str(cfg.name), group)
    log = rd.logger()
    seed_everything(int(cfg.seed))
    dev = dev or default_device()
    panel = LSMPanel(df, list(cfg.features), int(cfg.lookback), splits, cfg.universe)
    model = build_lsm(cfg).to(dev)
    tr = cfg.train
    steps = math.ceil(len(panel.end_dates("train")) / int(tr.batch_size))
    opt = make_optimizer(model, float(tr.lr), float(tr.weight_decay))
    sched = warmup_cosine(opt, int(tr.warmup_steps), steps * int(tr.max_epochs))
    stopper = EarlyStopping(int(tr.patience))
    rng = np.random.default_rng(int(cfg.seed))
    log.info(f"{cfg.name}: params={n_params(model):,} universe={cfg.universe} "
             f"assets={len(panel.permnos):,} train windows={len(panel.end_dates('train')):,} "
             f"steps/epoch={steps} device={dev}")
    history = []
    for epoch in range(1, int(tr.max_epochs) + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, p, f in panel.batches("train", int(tr.batch_size), int(tr.max_assets), True, rng,
                                     dev):
            with autocast(dev, str(tr.precision)):
                out = model(x, p)
                loss = recon_loss(out, x, p, f)
                if float(cfg.z_l1) > 0:
                    loss = loss + float(cfg.z_l1) * out["z_m"].abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.grad_clip))
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
            n += len(x)
        val = eval_loss(model, panel, "val", cfg, dev)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val,
                        "sec": time.time() - t0})
        log.info(f"epoch {epoch:3d} train {tot / n:.4f} val {val:.4f} ({time.time() - t0:.0f}s)")
        if stopper.step(val, epoch, model):
            log.info(f"early stop; best epoch {stopper.best_epoch}")
            break
    assert stopper.best_state is not None
    model.load_state_dict(stopper.best_state)
    for prm in model.parameters():  # frozen from here on
        prm.requires_grad_(False)
    model.eval()
    torch.save(stopper.best_state, rd.path / "model.pt")
    (rd.path / "history.json").write_text(json.dumps(history, indent=1))
    rd.write_metrics({"best_val_recon": stopper.best, "best_epoch": stopper.best_epoch,
                      "lsm_params": sum(p.numel() for p in model.parameters())})
    return model, rd, panel


@torch.no_grad()
def export_contexts(model: LSM, panel: LSMPanel, cfg: DictConfig, rd: RunDir,
                    dev: torch.device, out_dir: Path | None = None) -> Path:
    """Context for every date in the panel, for eval-universe assets (plus LSM members)."""
    model.eval()
    k_export = int(cfg.get("export_universe", 200))
    batch = int(cfg.get("export_batch", 8))
    d = int(cfg.model.d_market) + int(cfg.model.d_asset)
    rows: list[pd.DataFrame] = []
    T = len(panel.calendar)
    for s in range(0, T, batch):
        ts = list(range(s, min(T, s + batch)))
        sets = []
        for t in ts:
            lsm_m = panel.members(t)
            evals = np.flatnonzero(panel.present[t] & (panel.cap_rank[t] <= k_export))
            assets = np.union1d(lsm_m, evals)
            sets.append((assets, np.isin(assets, lsm_m), evals))
        n = max(len(a) for a, _, _ in sets)
        if n == 0:
            continue
        F = panel.X.shape[-1]
        x = np.zeros((len(ts), panel.lookback, n, F), np.float32)
        pr = np.zeros((len(ts), panel.lookback, n), bool)
        pool = np.zeros((len(ts), n), bool)
        for b, (t, (a, pm, _)) in enumerate(zip(ts, sets, strict=True)):
            xx, pp, _ = panel.window(t, a)
            x[b, :, :len(a)], pr[b, :, :len(a)], pool[b, :len(a)] = xx, pp, pm
        with autocast(dev, str(cfg.train.precision)):
            c = model.context(torch.from_numpy(x).to(dev), torch.from_numpy(pr).to(dev),
                              torch.from_numpy(pool).to(dev)).float().cpu().numpy()
        for b, (t, (a, _, ev)) in enumerate(zip(ts, sets, strict=True)):
            keep = np.isin(a, ev)
            frame = pd.DataFrame(c[b, :len(a)][keep], columns=[f"c_{i}" for i in range(d)])
            frame.insert(0, "date", panel.calendar[t])
            frame.insert(0, "permno", panel.permnos[a[keep]])
            rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out_dir = out_dir or features_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"contexts_{cfg.name}.parquet"
    out.to_parquet(path, index=False)
    meta = {"name": str(cfg.name), "run_dir": str(rd.path), "universe": str(cfg.universe),
            "objective": str(cfg.objective), "lsm_params": sum(p.numel() for p in model.parameters()),
            "last_date": str(panel.calendar[-1].date()), "rows": len(out)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/lsm_daily.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. train.max_epochs=2")
    ap.add_argument("--synthetic", action="store_true", help="smoke test on in-memory synthetic data")
    ap.add_argument("--group", default="lsm")
    a = ap.parse_args()
    extra = [config.from_dotlist(a.set)]
    if a.synthetic:
        extra.append(config.from_dotlist(["data.source=synthetic", "data.n_assets=400"]))
    cfg = config.merge(a.config, *a.override, *extra)
    splits = Splits.load(str(cfg.freq))
    df = load_feature_frame(cfg, splits)
    dev = default_device()
    model, rd, panel = train(cfg, df, splits, a.group, dev)
    if cfg.export_contexts:
        if a.synthetic and "MSTATE_DATA_ROOT" not in os.environ:
            raise SystemExit("--synthetic export needs MSTATE_DATA_ROOT set to a scratch dir")
        print(f"contexts -> {export_contexts(model, panel, cfg, rd, dev)}")


if __name__ == "__main__":
    main()
