"""STEP 8 BRANCH B, stage 4 (2026-08-07/08) -- full-data EMA checkpoint.
tmux session `ema`.

100% of labelled data (`src.folds.get_full_data_split`, additive, touches
no existing split path), NO fold holdout, 30 epochs, passive EMA
(decay=0.999/step, reusing `src.train.train_one_epoch`'s existing EMA
support unmodified). `out_dir runs/deploy_full_ema`.

NO VALIDATION SET EXISTS BY CONSTRUCTION. This script does not build a
val_loader, does not call `src.train.validate`, and writes no prediction
parquet -- there is nothing to predict ON that wasn't trained on. Per-epoch
records carry train_loss and epoch_seconds only. Anyone reading
`runs/deploy_full_ema/full_s0/summary.json` and expecting a val_loss or
ROC-AUC will not find one; that absence is deliberate, not a bug.

Standalone rather than a `src.train.run()` code path: `run()`'s resume/
history/prediction-dump machinery is built entirely around having a
validation split, and threading a "there is no val set" branch through it
end-to-end was judged riskier than reusing its already-tested BUILDING
BLOCKS (BarrettDataset, make_loader, build_model, train_one_epoch -- EMA
support included, unmodified) directly here.

Resumable: `checkpoints/last.pt` (model + EMA shadow + optimiser/scheduler/
scaler/RNG state + completed epochs) written atomically every epoch;
`--resume` implied automatically if `last.pt` exists and `weights_fp32.pt`
does not. Skip-validated: if `weights_fp32.pt` and `weights_ema_fp32.pt`
already exist, does nothing.

    python scripts/59_full_data_ema.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PID_FILE = os.path.join(LOG_DIR, "ema_chain.pid")
SENTINEL_DONE = os.path.join(LOG_DIR, "ema_chain_done")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "deploy_full_ema")
CONFIG_PATH = "configs/a4_pinned085_swa.yaml"
SEED = 0


def log(msg: str) -> None:
    print(f"[ema-chain {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body or time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


def train_full_data() -> Dict[str, Any]:
    import dataclasses

    import numpy as np
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
        CKPT_LAST, CKPT_WEIGHTS, CKPT_WEIGHTS_EMA,
        _avg_state_init, _ema_update, append_jsonl, make_scaler, make_scheduler,
        rng_state, save_checkpoint, seed_epoch, set_rng_state, train_one_epoch,
    )

    cfg = Config.from_yaml(CONFIG_PATH)
    cfg = dataclasses.replace(cfg, seed=SEED, out_dir=str(OUT_DIR),
                              swa_enabled=False)  # EMA only for this artefact
    run_dir = os.path.join(OUT_DIR, f"full_s{cfg.seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")

    weights_path = os.path.join(ckpt_dir, CKPT_WEIGHTS)
    weights_ema_path = os.path.join(ckpt_dir, CKPT_WEIGHTS_EMA)
    if os.path.exists(weights_path) and os.path.exists(weights_ema_path):
        log("full-data EMA checkpoint already complete; skipping (skip-validated).")
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
    ema_shadow = _avg_state_init(model)

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
        for k, v in ck["ema"].items():
            ema_shadow[k] = v.to(ema_shadow[k].device).float()
        start_epoch = int(ck["epoch"])
        history = list(ck["history"])
        log(f"resumed from {last_path} at epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="full-data EMA epochs",
                      unit="ep", initial=start_epoch, total=cfg.epochs,
                      file=sys.stderr):
        seed_epoch(train_gen, cfg.seed, epoch)
        train_ds.set_epoch(epoch)
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                             scaler, cfg, device, epoch, progress=True,
                             ema_shadow=ema_shadow)
        record = {"event": "epoch", "epoch": epoch + 1, "time": time.time(),
                  "train_loss": tr["train_loss"],
                  "train_seconds": tr["train_seconds"]}
        history.append(record)
        append_jsonl(log_path, record)
        payload = {
            "epoch": epoch + 1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "train_generator": train_gen.get_state(),
            "loader_generator": loader_gen.get_state(), "rng": rng_state(),
            "history": history, "config": cfg.to_dict(),
            "ema": {k: v.cpu() for k, v in ema_shadow.items()},
        }
        save_checkpoint(last_path, payload)
        tqdm.write(f"epoch {epoch + 1:3d}/{cfg.epochs}  "
                  f"train {tr['train_loss']:.4f}  {tr['train_seconds']:.1f}s "
                  f"(no validation set)")

    raw_weights = {"epoch": cfg.epochs,
                  "model": {k: v.detach().cpu().float() for k, v in model.state_dict().items()},
                  "config": cfg.to_dict(), "selection": "last", "canonical": True,
                  "note": "trained on 100% of labelled data; no validation set exists"}
    ema_weights = dict(raw_weights)
    ema_weights["model"] = {k: v.detach().cpu().float() for k, v in ema_shadow.items()}
    ema_weights["selection"] = "ema"
    save_checkpoint(weights_path, raw_weights)
    save_checkpoint(weights_ema_path, ema_weights)
    if os.path.exists(last_path):
        os.remove(last_path)

    summary = {
        "run_dir": run_dir, "n_train_images": len(train_fps),
        "n_val_images": 0,
        "note": "NO VALIDATION SET EXISTS BY CONSTRUCTION -- 100% of "
                "labelled data was used for training.",
        "epochs_completed": cfg.epochs,
        "wall_seconds": time.perf_counter() - wall_t0,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "history": history,
        "weights_path": weights_path, "weights_ema_path": weights_ema_path,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def render(summary: Dict[str, Any]) -> None:
    L = []
    A = L.append
    A("# Full-data EMA checkpoint -- deployment artefact, no validation set")
    A("")
    A(f"`{CONFIG_PATH}` (aug/arch/optim unchanged; full-data split, EMA "
     f"decay=0.999/step), seed {SEED}, `runs/deploy_full_ema`. "
     f"{summary['n_train_images']} training images, "
     f"**{summary['n_val_images']} held out** -- "
     f"{summary['note']}")
    A("")
    A(f"Epochs completed: {summary['epochs_completed']}/30. Wall clock: "
     f"{summary['wall_seconds'] / 3600:.2f} h. Final train loss: "
     f"{summary['final_train_loss']:.4f}." if summary.get('final_train_loss') is not None
     else "")
    A("")
    A(f"Weights: `{summary['weights_path']}` (raw), "
     f"`{summary['weights_ema_path']}` (EMA).")
    A("")
    A("| epoch | train_loss | seconds |")
    A("|---|---|---|")
    for r in summary["history"]:
        A(f"| {r['epoch']} | {r['train_loss']:.4f} | {r['train_seconds']:.1f} |")
    A("")
    A("**Status: weights exist. NOT wired into any submission container "
     "tonight** (out of scope: 'DO NOT build or upload any container').")
    A("")

    with open(os.path.join(REPORT_DIR, "full_data_ema.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    log("written: reports/full_data_ema.md")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    touch(PID_FILE, f"{os.getpid()}\n")
    log("full-data EMA chain starting.")
    exit_code = 0
    try:
        try:
            summary = train_full_data()
            render(summary)
            touch(SENTINEL_DONE)
        except Exception as exc:  # noqa: BLE001
            log(f"FAILED: {exc!r} -- falling through to step 5's launcher, "
               "which will check reports/pauc_smoke.json independently. "
               "Re-run this script to resume from checkpoints/last.pt.")
            exit_code = 3

        # Step 5 gate: ONLY if the pAUC smoke was clean. Checked here (not
        # inside run_pauc_p1_chain.sh) so the reason is on record in this log.
        smoke_path = os.path.join(REPORT_DIR, "pauc_smoke.json")
        proceed = False
        if os.path.exists(smoke_path):
            with open(smoke_path) as fh:
                smoke = json.load(fh)
            proceed = not smoke.get("diverged", True) and not smoke.get("gradient_collapsed", True)
            log(f"pauc_smoke.json: diverged={smoke.get('diverged')} "
               f"collapsed={smoke.get('gradient_collapsed')} -> "
               f"step 5 {'PROCEEDS' if proceed else 'SKIPPED'}")
        else:
            log("reports/pauc_smoke.json not found -- step 5 SKIPPED "
               "(cannot confirm the smoke was clean)")

        if proceed:
            subprocess.run(["bash", os.path.join(SCRIPTS_DIR, "run_pauc_p1_chain.sh")],
                           cwd=REPO_ROOT)
        else:
            touch(os.path.join(LOG_DIR, "pauc_p1_chain_report_done"),
                 "skipped: pauc smoke was not clean, or missing\n")
        return exit_code
    finally:
        try:
            with open(PID_FILE) as fh:
                if int(fh.read().strip()) == os.getpid():
                    os.remove(PID_FILE)
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
