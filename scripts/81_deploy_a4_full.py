"""STAGE 1 -- A4-corrected full-data deployment checkpoint.

ConvNeXt-Base, 30 epochs, 100% of labelled data (`src.folds.get_full_data_split`,
the same additive helper `scripts/59_full_data_ema.py` used for the RN50/EMA
full-data artefact) -- NO HOLDOUT, NO VALIDATION SET BY CONSTRUCTION. Config is
`configs/sweep_a4_checkpointed.yaml` verbatim except seed, out_dir, and
batch_size (see below); this is the same aug/arch/optim block that produced
the shipping r0_f0_s0 checkpoint (`reports/e2_checkpoint_identity.md`,
`reports/stage0_artifact_verification.md`), just trained without a held-out
fold.

BATCH SIZE IS MEASURED, NOT REUSED. `runs/vram_probe_a4_full.json`
(`scripts/vram_probe.py`, this exact arch/image_size/precision/grad_checkpointing
cell) found max_bs=38 at 65.6 img/s, 14.98 GiB reserved -- NOT the 90 used
for RN50/G3, a different architecture with a different memory footprint.
This script reads that probe file, refuses to run if the matching row is
missing, and asserts the ACTUAL first training batch's shape equals the
probed value before epoch 1 starts -- a config-level match alone would not
catch a loader silently overriding batch_size (e.g. via drop_last mismatch
on a short final batch).

out_dir: runs/deploy_a4_full_s{seed} (seed baked into the directory name
per the instruction, not a subdirectory of a shared parent).

Resumable, same pattern as 59_full_data_ema.py: checkpoints/last.pt written
atomically every epoch, --resume implied if it exists and weights_fp32.pt
does not.

    python scripts/81_deploy_a4_full.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO_ROOT, "logs")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
VRAM_PROBE_JSON = os.path.join(REPO_ROOT, "runs", "vram_probe_a4_full.json")
CONFIG_PATH = "configs/sweep_a4_checkpointed.yaml"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def log(msg: str, seed: int) -> None:
    print(f"[deploy-a4-full s{seed} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def probed_batch_size(image_size: int, precision: str, grad_checkpointing: bool) -> int:
    if not os.path.exists(VRAM_PROBE_JSON):
        raise SystemExit(
            f"{VRAM_PROBE_JSON} not found -- run scripts/vram_probe.py for this "
            f"config first. Reusing the RN50/G3 batch size (90) for a different "
            f"architecture is exactly the mistake this script exists to prevent."
        )
    with open(VRAM_PROBE_JSON) as fh:
        data = json.load(fh)
    if data.get("arch") != "convnext_base.fb_in1k":
        raise SystemExit(f"{VRAM_PROBE_JSON} was probed for arch={data.get('arch')!r}, "
                         f"not convnext_base.fb_in1k -- re-probe before running.")
    for row in data["rows"]:
        if (row["image_size"] == image_size and row["precision"] == precision
                and row["grad_checkpointing"] == grad_checkpointing and row["batch_size"] > 0):
            return int(row["batch_size"])
    raise SystemExit(f"no matching row in {VRAM_PROBE_JSON} for image_size={image_size} "
                     f"precision={precision} grad_checkpointing={grad_checkpointing}")


def train_full_data(seed: int) -> Dict[str, Any]:
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

    base_cfg = Config.from_yaml(CONFIG_PATH)
    out_dir = os.path.join(REPO_ROOT, "runs", f"deploy_a4_full_s{seed}")

    probed_bs = probed_batch_size(base_cfg.image_size, base_cfg.precision,
                                  base_cfg.grad_checkpointing)
    log(f"probed batch size for image_size={base_cfg.image_size} "
       f"precision={base_cfg.precision} grad_checkpointing={base_cfg.grad_checkpointing}: "
       f"{probed_bs} (from {VRAM_PROBE_JSON})", seed)

    cfg = dataclasses.replace(base_cfg, seed=seed, out_dir=str(out_dir),
                              batch_size=probed_bs)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")

    weights_path = os.path.join(ckpt_dir, CKPT_WEIGHTS)
    if os.path.exists(weights_path):
        log("full-data checkpoint already complete; skipping (skip-validated).", seed)
        with open(os.path.join(out_dir, "summary.json")) as fh:
            return json.load(fh)

    wall_t0 = time.perf_counter()
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.to_yaml(os.path.join(out_dir, "config.yaml"))

    manifest_df = pd.read_csv(cfg.manifest)
    train_fps = get_full_data_split(cfg.manifest)
    log(f"full-data split: {len(train_fps)} training images, 0 held out "
       f"(NO VALIDATION SET EXISTS BY CONSTRUCTION)", seed)
    train_ds = BarrettDataset(train_fps, manifest_df, cfg, train=True, build_cache=True)

    train_gen = torch.Generator()
    loader_gen = torch.Generator()
    loader_gen.manual_seed(cfg.seed)
    seed_epoch(train_gen, cfg.seed, 0)
    train_sampler = make_sampler(train_ds, cfg, train_gen)
    train_loader = make_loader(train_ds, cfg, shuffle=False, sampler=train_sampler,
                               generator=loader_gen)

    # BIT-IDENTITY ASSERTION BEFORE EPOCH 1: pull one real batch through the
    # actual loader and check its shape against the probed value, not just
    # the config field that was set to request it.
    probe_batch = next(iter(train_loader))
    actual_bs = probe_batch[0].shape[0] if isinstance(probe_batch, (list, tuple)) else probe_batch.shape[0]
    assert actual_bs == probed_bs, (
        f"loader's first batch has size {actual_bs}, probed value was {probed_bs} -- "
        f"refusing to train on an unverified batch size"
    )
    log(f"assert PASSED: running batch size {actual_bs} == probed {probed_bs}", seed)

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
        log(f"resumed from {last_path} at epoch {start_epoch}", seed)

    epoch_start_wall = time.perf_counter()
    for epoch in tqdm(range(start_epoch, cfg.epochs), desc=f"deploy_a4_full s{seed}",
                      unit="ep", initial=start_epoch, total=cfg.epochs, file=sys.stderr):
        if os.path.exists(os.path.join(REPO_ROOT, "logs", "gpu_hold_kill")):
            log(f"logs/gpu_hold_kill present -- stopping before epoch {epoch + 1}; "
               f"checkpoints/last.pt has epoch {start_epoch}..{epoch}, re-run to resume", seed)
            return {"run_dir": out_dir, "seed": seed, "probed_batch_size": probed_bs,
                    "n_train_images": len(train_fps), "n_val_images": 0,
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
                  f"{tr['train_seconds']:.1f}s  (no validation set)")

        # in-process count, NOT epoch+1: after a resume, start_epoch > 0, so
        # epoch+1==3 can fire after this process has only run ONE epoch --
        # dividing that single epoch's wall time by a hardcoded 3 would
        # under-estimate the per-epoch cost 3x. Divide by what actually ran
        # in THIS process instead.
        epochs_run_this_process = epoch - start_epoch + 1
        if epochs_run_this_process == 3:
            per_epoch = (time.perf_counter() - epoch_start_wall) / epochs_run_this_process
            remaining = cfg.epochs - (epoch + 1)
            log(f"MEASURED per-epoch wall-clock ({epochs_run_this_process} epoch(s) "
               f"timed in this process, resumed from epoch {start_epoch}): "
               f"{per_epoch:.1f}s/epoch avg, ~{remaining * per_epoch / 60:.1f} min "
               f"remaining ({remaining * per_epoch / 3600:.2f} h) -- not extrapolated "
               f"from G3", seed)

    raw_weights = {"epoch": cfg.epochs,
                  "model": {k: v.detach().cpu().float() for k, v in model.state_dict().items()},
                  "config": cfg.to_dict(), "selection": "last", "canonical": True,
                  "note": "trained on 100% of labelled data; no validation set exists"}
    save_checkpoint(weights_path, raw_weights)
    if os.path.exists(last_path):
        os.remove(last_path)

    summary = {
        "run_dir": out_dir, "seed": seed, "probed_batch_size": probed_bs,
        "n_train_images": len(train_fps), "n_val_images": 0,
        "note": "NO VALIDATION SET EXISTS BY CONSTRUCTION -- 100% of "
                "labelled data was used for training.",
        "epochs_completed": cfg.epochs,
        "wall_seconds": time.perf_counter() - wall_t0,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "history": history, "weights_path": weights_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    pid_file = os.path.join(LOG_DIR, f"deploy_a4_full_s{args.seed}.pid")
    with open(pid_file, "w") as fh:
        fh.write(f"{os.getpid()}\n")
    log("chain starting.", args.seed)
    try:
        summary = train_full_data(args.seed)
        if summary.get("stopped"):
            # gpu_hold_kill (or any other early-return path) is NOT completion.
            # A "_done" sentinel written here would be indistinguishable from a
            # real finish to anything that later gates on it -- found live by
            # the 2026-08-11 gpu_hold_kill test, which stopped this run
            # correctly at epoch 2 but this branch used to mark it done anyway.
            log(f"STOPPED, not complete: {summary['epochs_completed']}/30 epochs "
               f"({summary.get('note')}) -- no _done sentinel written; "
               f"re-run the same command to resume", args.seed)
            return 4
        log(f"done: {summary['epochs_completed']} epochs, "
           f"{summary['wall_seconds'] / 3600:.2f} h, "
           f"final_train_loss={summary.get('final_train_loss')}", args.seed)
        with open(os.path.join(LOG_DIR, f"deploy_a4_full_s{args.seed}_done"), "w") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
        return 0
    except Exception as exc:  # noqa: BLE001
        log(f"FAILED: {exc!r} -- re-run the same command to resume from "
           f"checkpoints/last.pt", args.seed)
        return 3
    finally:
        try:
            with open(pid_file) as fh:
                if int(fh.read().strip()) == os.getpid():
                    os.remove(pid_file)
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
