"""Weekend queue stage 5 -- G3 full-data deployment checkpoint.

100% of labelled data (`src.folds.get_full_data_split`, additive, touches no
existing split path), NO fold holdout, 30 epochs, batch_size 90 (the
VRAM-probed value for this exact model -- see configs/g3_checkpointed.yaml's
header). `out_dir runs/deploy_g3_full`. No EMA/SWA -- plain weights only,
this is a deployment checkpoint, not a variance-reduction arm.

NO VALIDATION SET EXISTS BY CONSTRUCTION, same as scripts/59_full_data_ema.py
(this script's structure and resumption discipline is that script's, minus
the EMA shadow tracking -- ema_shadow is optional in
src.train.train_one_epoch, so it is simply omitted rather than tracked and
discarded).

Resumable: `checkpoints/last.pt` written atomically every epoch;
skip-validated: does nothing if `weights_fp32.pt` already exists.

    python scripts/61_full_data_g3.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO_ROOT, "logs")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, "runs", "deploy_g3_full")
CONFIG_PATH = "configs/g3_checkpointed.yaml"   # batch_size 90, save_checkpoint true
SEED = 0


def log(msg: str) -> None:
    print(f"[g3full {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def train_full_data() -> Dict[str, Any]:
    import dataclasses

    import pandas as pd
    import torch
    import torch.nn as nn
    from tqdm.auto import tqdm

    from src.config import Config
    from src.data import BarrettDataset, make_loader, make_sampler
    from src.folds import get_full_data_split
    from src.model import build_model
    from src.seeding import seed_everything
    from src.train import (
        CKPT_LAST, CKPT_WEIGHTS, append_jsonl, make_scaler, make_scheduler,
        rng_state, save_checkpoint, seed_epoch, set_rng_state, train_one_epoch,
    )

    cfg = Config.from_yaml(CONFIG_PATH)
    cfg = dataclasses.replace(cfg, seed=SEED, out_dir=str(OUT_DIR))
    run_dir = os.path.join(OUT_DIR, f"full_s{cfg.seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")

    weights_path = os.path.join(ckpt_dir, CKPT_WEIGHTS)
    if os.path.exists(weights_path):
        log("full-data G3 checkpoint already complete; skipping (skip-validated).")
        with open(os.path.join(run_dir, "summary.json")) as fh:
            return json.load(fh)

    wall_t0 = time.perf_counter()
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.to_yaml(os.path.join(run_dir, "config.yaml"))

    manifest_df = pd.read_csv(cfg.manifest)
    train_fps = get_full_data_split(cfg.manifest)
    log(f"full-data split: {len(train_fps)} training images, 0 held out "
       f"(NO VALIDATION SET EXISTS BY CONSTRUCTION)")
    train_ds = BarrettDataset(train_fps, manifest_df, cfg, train=True, build_cache=True)

    train_gen = torch.Generator()
    loader_gen = torch.Generator()
    loader_gen.manual_seed(cfg.seed)
    seed_epoch(train_gen, cfg.seed, 0)
    train_sampler = make_sampler(train_ds, cfg, train_gen)
    train_loader = make_loader(train_ds, cfg, shuffle=False, sampler=train_sampler,
                               generator=loader_gen)

    model = build_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = make_scheduler(optimizer, total_steps, cfg.warmup_frac)
    scaler = make_scaler(cfg.precision, device)

    start_epoch = 0
    history = []
    last_path = os.path.join(ckpt_dir, CKPT_LAST)
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        train_gen.set_state(ck["train_generator"].cpu())
        if "loader_generator" in ck:
            loader_gen.set_state(ck["loader_generator"].cpu())
        set_rng_state(ck["rng"])
        start_epoch = int(ck["epoch"])
        history = list(ck["history"])
        log(f"resumed from {last_path} at epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="full-data G3 epochs",
                      unit="ep", initial=start_epoch, total=cfg.epochs,
                      file=sys.stderr):
        if os.path.exists(os.path.join(REPO_ROOT, "logs", "gpu_hold_kill")):
            log(f"logs/gpu_hold_kill present -- stopping before epoch {epoch + 1}; "
               f"checkpoints/last.pt has epoch {start_epoch}..{epoch}, re-run to resume")
            return {"run_dir": run_dir, "n_train_images": len(train_fps), "n_val_images": 0,
                    "note": "STOPPED by gpu_hold_kill mid-run, not complete",
                    "epochs_completed": epoch, "wall_seconds": time.perf_counter() - wall_t0,
                    "final_train_loss": history[-1]["train_loss"] if history else None,
                    "history": history, "weights_path": None, "stopped": True}
        seed_epoch(train_gen, cfg.seed, epoch)
        train_ds.set_epoch(epoch)
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                             scaler, cfg, device, epoch, progress=True)
        record = {"event": "epoch", "epoch": epoch + 1, "time": time.time(),
                  "train_loss": tr["train_loss"], "train_seconds": tr["train_seconds"]}
        history.append(record)
        append_jsonl(log_path, record)
        payload = {
            "epoch": epoch + 1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "train_generator": train_gen.get_state(),
            "loader_generator": loader_gen.get_state(), "rng": rng_state(),
            "history": history, "config": cfg.to_dict(),
        }
        save_checkpoint(last_path, payload)
        tqdm.write(f"epoch {epoch + 1:3d}/{cfg.epochs}  train {tr['train_loss']:.4f}  "
                  f"{tr['train_seconds']:.1f}s (no validation set)")

    raw_weights = {"epoch": cfg.epochs,
                  "model": {k: v.detach().cpu().float() for k, v in model.state_dict().items()},
                  "config": cfg.to_dict(), "selection": "last", "canonical": True,
                  "note": "trained on 100% of labelled data; no validation set exists"}
    save_checkpoint(weights_path, raw_weights)
    if os.path.exists(last_path):
        os.remove(last_path)

    summary = {
        "run_dir": run_dir, "n_train_images": len(train_fps), "n_val_images": 0,
        "note": "NO VALIDATION SET EXISTS BY CONSTRUCTION -- 100% of labelled "
                "data was used for training.",
        "epochs_completed": cfg.epochs,
        "wall_seconds": time.perf_counter() - wall_t0,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "history": history, "weights_path": weights_path,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def render(summary: Dict[str, Any]) -> None:
    L = []
    A = L.append
    A("# G3 full-data deployment checkpoint -- no validation set")
    A("")
    A(f"`{CONFIG_PATH}` (batch_size 90, VRAM-probed; aug/arch/optim otherwise "
     f"unchanged from G3's screening config), seed {SEED}, `runs/deploy_g3_full`. "
     f"{summary['n_train_images']} training images, "
     f"**{summary['n_val_images']} held out** -- {summary['note']}")
    A("")
    if summary.get('final_train_loss') is not None:
        A(f"Epochs completed: {summary['epochs_completed']}/30. Wall clock: "
         f"{summary['wall_seconds'] / 3600:.2f} h. Final train loss: "
         f"{summary['final_train_loss']:.4f}.")
    A("")
    A(f"Weights: `{summary['weights_path']}`.")
    A("")
    A("| epoch | train_loss | seconds |")
    A("|---|---|---|")
    for r in summary["history"]:
        A(f"| {r['epoch']} | {r['train_loss']:.4f} | {r['train_seconds']:.1f} |")
    A("")
    A("**Status: weights exist. NOT wired into any submission container** -- "
     "G3-lineage weights carry the GastroNet licence exposure "
     "(reports/job_e_a4_pinned_container.md), so this checkpoint is a "
     "research/paper artefact, not a deployment candidate.")
    A("")
    with open(os.path.join(REPORT_DIR, "g3_full_data.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    log("written: reports/g3_full_data.md")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    log("G3 full-data checkpoint stage starting.")
    summary = train_full_data()
    render(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
