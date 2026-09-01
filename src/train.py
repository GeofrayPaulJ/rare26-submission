"""One (repeat, fold, seed) of plain supervised training, per invocation.

DELIBERATELY UNAMBITIOUS. AdamW + cosine schedule + BCEWithLogitsLoss, and
nothing else. No EMA, no SAM, no partial-AUC surrogate loss, no TTA, no
ensembling. This step exists to produce a working checkpoint and trustworthy
timings, not a good model.

CHECKPOINT SELECTION -- read this before touching anything downstream.
The canonical output of a run is the LAST epoch, always -- never
best-by-ROC-AUC. "Best" is by ROC-AUC and never by PPV@90R. A validation fold
holds ~31 positives, so 90% recall is the operating point set by roughly three
images; per-fold PPV@90R moves several points between epochs on noise alone.
It is logged and plotted because its shape over training is informative, and
it is never, ever used to choose a model. Even best-by-ROC-AUC is a mild
selection bias on 617 rows -- ``diagnostic_best_roc_auc`` in summary.json
records which epoch it was and its score, but no checkpoint is ever written
for it; there is nothing downstream that is allowed to select on it.

CHECKPOINT POLICY -- disk, not just correctness.
``cfg.save_checkpoint`` (default False) decides what survives a *successful*
run:

    False (screening, the default): nothing does. Only the per-epoch and
    canonical predictions parquets remain.
    True (a run explicitly designated as a final ensemble member):
    checkpoints/weights_fp32.pt survives -- the LAST epoch's model weights
    only, cast to fp32, no optimiser/scheduler/scaler/RNG/history.

Regardless of that flag, checkpoints/last.pt is written atomically every
epoch while the run is incomplete -- this is the rolling *resume* checkpoint,
not a retained artefact, and it is deleted the moment the schedule completes
successfully (replaced by weights_fp32.pt when save_checkpoint=True). This is
what keeps a 95-unit screening sweep from accumulating ~1 GB/unit of
optimiser state for units that were never going anywhere.

TMUX DURABILITY. Every epoch appends one JSON record to train_log.jsonl and
writes checkpoints/last.pt atomically. ``--resume`` restores model, optimiser,
scheduler, scaler, sampler generator and every RNG, so a killed run continues
on the same trajectory rather than a merely similar one. Batch order is seeded
per epoch from (seed, epoch), which is what makes that guarantee exact instead
of approximate; tests/test_resume.py pins it to 4+ decimal places. Because
last.pt is deleted on success, ``--resume`` only ever applies to a run that
did not complete last time -- resuming a finished run raises FileNotFoundError,
which is the correct failure (there is nothing to resume).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import Config, autocast
from .data import BarrettDataset, make_loader, make_sampler, stratum_draw_counts
from .folds import get_holdout_split, get_split
from .io import build_frame, durable_replace, fsync_dir, write_predictions
from .metrics import evaluate
from .model import build_model, param_counts
from .seeding import seed_everything

logger = logging.getLogger("train")

LOG_NAME = "train_log.jsonl"
SUMMARY_NAME = "summary.json"
CKPT_LAST = "last.pt"          # rolling resume checkpoint; deleted on success
CKPT_WEIGHTS = "weights_fp32.pt"  # retained artefact, save_checkpoint=True only
CKPT_WEIGHTS_EMA = "weights_ema_fp32.pt"  # ema_enabled + save_checkpoint only
CKPT_WEIGHTS_SWA = "weights_swa_fp32.pt"  # swa_enabled + save_checkpoint only

# Fraction of total physical VRAM the spill guard treats as "about to page to
# system RAM". See VramSpillError below for why this exists.
VRAM_SPILL_FRACTION = 0.95


class VramSpillError(RuntimeError):
    """Raised when the CUDA allocator has reserved more than
    VRAM_SPILL_FRACTION of physical VRAM.

    This GPU runs under the Windows WDDM driver, which lets the allocator
    oversubscribe the device and quietly back the overflow with system RAM
    over PCIe instead of raising OutOfMemoryError. The run does not crash --
    it just runs roughly 40x slower with no indication why. A silent 15-hour
    run at 1/40th speed is much worse than a crash with a clear cause, so this
    checks the peak reservation after the first epoch (by which point the
    steady-state footprint -- AdamW state included -- is established) and
    aborts loudly rather than let the rest of the schedule page silently.
    """

BEST_WARNING = (
    "DIAGNOSTIC MARKER ONLY -- best validation ROC-AUC epoch on a single "
    "617-row fold. NOT the canonical output of this run and no checkpoint is "
    "kept for it; the canonical model is always the LAST epoch."
)


# ---------------------------------------------------------------------------
# GPU utilisation sampling
# ---------------------------------------------------------------------------
class GpuMonitor:
    """Poll GPU utilisation on a background thread.

    Answers the only question that matters when a training loop looks slow: is
    the GPU actually busy, or is it waiting on the loader? nvml reports the
    fraction of the sampling window in which any kernel was resident, so a mean
    well under ~90% means the device is idling between batches.
    """

    def __init__(self, index: int = 0, interval: float = 0.25) -> None:
        self.index = index
        self.interval = interval
        self.enabled = torch.cuda.is_available()
        self._samples: List[int] = []
        self._all: List[int] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                util = int(torch.cuda.utilization(self.index))
            except Exception:  # nvml hiccup must never kill a training run
                continue
            with self._lock:
                self._samples.append(util)
                self._all.append(util)

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def drain(self) -> Dict[str, float]:
        """Summarise and clear the samples taken since the last drain."""
        with self._lock:
            s = self._samples[:]
            self._samples.clear()
        return summarise_util(s)

    def overall(self) -> Dict[str, float]:
        with self._lock:
            return summarise_util(self._all[:])


def summarise_util(samples: Sequence[int]) -> Dict[str, float]:
    if not samples:
        return {"mean": float("nan"), "p50": float("nan"), "min": float("nan"), "n": 0}
    a = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p50": float(np.median(a)),
        "min": float(a.min()),
        "n": int(a.size),
    }


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def make_scaler(precision: str, device: str) -> torch.amp.GradScaler:
    """Gradient scaler, enabled only for fp16.

    fp16 has five exponent bits, so small gradients underflow to zero without
    loss scaling. bf16 carries fp32's exponent range and needs none; fp32
    obviously needs none. Disabled, GradScaler is a transparent pass-through --
    scale() returns the loss object untouched and step() forwards straight to
    the optimiser -- which is what lets fp32 be a true no-op path rather than a
    second code path that merely behaves the same.
    """
    return torch.amp.GradScaler(
        device, enabled=(precision == "fp16" and device == "cuda")
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over the first ``warmup_frac`` of steps, then cosine to zero.

    Stepped per optimiser step, not per epoch, so the shape does not change when
    the batch size does. ``total_steps`` is derived from the configured epoch
    count and stays fixed across a resume -- a resumed run must ride the same
    curve as the run it replaces, so this can never be recomputed from the
    epochs that remain.
    """
    warmup_steps = max(1, int(round(total_steps * warmup_frac))) if warmup_frac > 0 else 0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# RNG plumbing for exact resume
# ---------------------------------------------------------------------------
def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    # every one of these must be a CPU ByteTensor, whatever device the
    # checkpoint happened to be staged through
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def seed_epoch(generator: torch.Generator, seed: int, epoch: int) -> None:
    """Seed the ordering RNG as a pure function of (seed, epoch).

    This is the mechanism that makes resume exact. If the sampler simply carried
    its state forward, a resumed run would depend on how far the killed one got
    mid-epoch; deriving the seed from the epoch index means epoch k draws the
    same batches no matter how many times the process died before reaching it.
    """
    generator.manual_seed((seed * 1_000_003 + epoch * 9_176_053 + 17) % (2 ** 63 - 1))


# ---------------------------------------------------------------------------
# Checkpoint / log helpers
# ---------------------------------------------------------------------------
def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append one record and fsync it.

    Append-only by design: a resumed run adds to this file rather than rewriting
    it, so after a crash the log may hold two records for the same epoch (the
    interrupted attempt and the replayed one). Readers should take the LAST
    record per epoch; the ``history`` list inside last.pt is authoritative.
    """
    existed = os.path.exists(path)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    if not existed:
        # First record CREATES the file, so its directory entry is new and
        # itself unflushed. fsync on the handle makes the bytes durable but
        # not the name pointing at them. Only on creation -- every later
        # append writes into an entry that is already on disk.
        fsync_dir(os.path.dirname(os.path.abspath(path)))


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    """Write atomically: a kill during torch.save must not leave a torn file
    where a valid checkpoint used to be.

    os.replace() alone only guarantees the RENAME is atomic -- it says nothing
    about whether the tmp file's content had actually reached durable storage
    before the rename. Confirmed the hard way: an unclean host/VM reset during
    the 3-repeat gating run renamed a last.pt whose data was still sitting in
    the page cache, leaving a file with the right name that failed to load
    ("PytorchStreamReader ... failed finding central directory") on the very
    next resume. A plain SIGKILL of this process wouldn't show the bug -- the
    page cache survives a killed process -- which is why it didn't turn up in
    the mid-epoch SIGKILL proof. fsync before replace closes that gap.
    """
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    durable_replace(tmp, path)


# fields that define the trajectory; a resume that changes one of these is not a
# resume, it is a different experiment wearing the old run's directory
TRAJECTORY_FIELDS = (
    "arch", "seed", "repeat", "fold", "holdout_centre", "epochs", "lr",
    "weight_decay", "warmup_frac", "batch_size", "image_size", "cache_size",
    "precision", "sampler", "crop_mode", "grad_checkpointing", "init_weights",
)


def check_resume_compatible(saved: Dict[str, Any], current: Dict[str, Any]) -> None:
    diffs = [
        f"{k}: checkpoint={saved.get(k)!r} -> config={current.get(k)!r}"
        for k in TRAJECTORY_FIELDS
        if saved.get(k) != current.get(k)
    ]
    if diffs:
        raise ValueError(
            "refusing to resume: the config differs from the checkpoint in fields "
            "that change the training trajectory:\n  " + "\n  ".join(diffs)
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def split_name(cfg: Config) -> str:
    """Filename/directory stem identifying this run's split.

    CV runs are ``r{repeat}_f{fold}_s{seed}``; LOCO runs are
    ``loco_c{centre}_s{seed}``. Two namespaces on purpose -- a LOCO run and a CV
    run must never be able to land on the same path and overwrite each other.
    """
    if cfg.holdout_centre is not None:
        return f"loco_c{cfg.holdout_centre}_s{cfg.seed}"
    return f"r{cfg.repeat}_f{cfg.fold}_s{cfg.seed}"


def build_datasets(cfg: Config, manifest_df: pd.DataFrame):
    if cfg.holdout_centre is not None:
        train_fps, val_fps = get_holdout_split(cfg.holdout_centre, cfg.manifest)
        logger.info(
            "LOCO holdout center_%d: %d train / %d held-out images",
            cfg.holdout_centre, len(train_fps), len(val_fps),
        )
    else:
        train_fps, val_fps = get_split(cfg.repeat, cfg.fold, cfg.manifest)
        logger.info(
            "split r%d f%d: %d train / %d val images",
            cfg.repeat, cfg.fold, len(train_fps), len(val_fps),
        )
    train_ds = BarrettDataset(train_fps, manifest_df, cfg, train=True, build_cache=True)
    val_ds = BarrettDataset(val_fps, manifest_df, cfg, train=False, build_cache=True)
    return train_ds, val_ds


def val_metadata(manifest_df: pd.DataFrame, val_fps: Sequence[str]) -> pd.DataFrame:
    """Per-filepath manifest columns the prediction dump carries, indexed by
    filepath so rows can be aligned to whatever order the loader returns."""
    cols = ["centre", "class_label", "visibility", "group_id_v2"]
    return manifest_df.set_index("filepath").loc[list(val_fps), cols]


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Weight averaging (EMA / SWA). PASSIVE by construction: both read the live
# weights and never write them back, consume no RNG, and touch no optimiser
# state -- so a run with these enabled produces bit-identical raw parquets
# and raw checkpoints to the same run with them disabled. That passivity is
# the whole point: the averaged variants are EXTRA artefacts, not a change
# to the run they average.
# ---------------------------------------------------------------------------
def _avg_state_init(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().float().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point}


@torch.no_grad()
def _ema_update(shadow: Dict[str, torch.Tensor], model, decay: float) -> None:
    for k, v in model.state_dict().items():
        if k in shadow:
            shadow[k].lerp_(v.detach().float(), 1.0 - decay)


@torch.no_grad()
def _swa_accumulate(swa: Dict[str, Any], model) -> None:
    swa["count"] += 1
    for k, v in model.state_dict().items():
        if k in swa["sum"]:
            swa["sum"][k] += v.detach().float()


def _swa_mean(swa: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {k: v / max(1, swa["count"]) for k, v in swa["sum"].items()}


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler,
    cfg: Config, device: str, epoch: int, progress: bool,
    ema_shadow: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    model.train()
    total_loss, n_seen = 0.0, 0
    t0 = time.perf_counter()

    bar = tqdm(
        loader,
        desc=f"e{epoch + 1:02d}/{cfg.epochs} train",
        unit="b", leave=False, disable=not progress, dynamic_ncols=True,
    )
    # pAUC combined loss (src/losses.py; OFF unless cfg.pauc_enabled). The
    # beta schedule is resolved once per epoch, not per batch: None means
    # "warmup, plain BCE this epoch".
    pauc_beta = None
    if cfg.pauc_enabled:
        from src.losses import partial_auc_loss, pauc_beta_schedule
        pauc_beta = pauc_beta_schedule(
            epoch, target_beta=cfg.pauc_beta,
            warmup_epochs=cfg.pauc_warmup_epochs,
            anneal_epochs=cfg.pauc_anneal_epochs)

    for x, y, _fp in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        with autocast(device, cfg.precision):
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            if pauc_beta is not None:
                # partial_auc_loss runs fp32 internally regardless of this
                # autocast context (its own autocast(enabled=False) wrapper).
                pl = partial_auc_loss(
                    logits[y > 0.5], logits[y <= 0.5],
                    max_fpr=pauc_beta,
                    pos_fraction=cfg.pauc_pos_fraction,
                    margin=cfg.pauc_margin)
                loss = cfg.pauc_lambda * loss + (1.0 - cfg.pauc_lambda) * pl

        # With the scaler disabled (bf16/fp32) these three calls are pass-through:
        # scale() returns the loss untouched and step() forwards to the optimiser.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if ema_shadow is not None:
            _ema_update(ema_shadow, model, cfg.ema_decay)

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n_seen += bs
        bar.set_postfix_str(
            f"loss {total_loss / n_seen:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
            f"{n_seen / (time.perf_counter() - t0):.0f} img/s"
        )
    bar.close()

    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "train_loss": total_loss / max(1, n_seen),
        "train_seconds": time.perf_counter() - t0,
        "train_images": n_seen,
    }


@torch.no_grad()
def validate(
    model, loader, criterion, cfg: Config, device: str, epoch: int, progress: bool,
) -> Dict[str, Any]:
    model.eval()
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    fps_all: List[str] = []
    total_loss, n_seen = 0.0, 0
    t0 = time.perf_counter()

    bar = tqdm(
        loader, desc=f"e{epoch + 1:02d}/{cfg.epochs}  val ",
        unit="b", leave=False, disable=not progress, dynamic_ncols=True,
    )
    for x, y, fp in bar:
        x = x.to(device, non_blocking=True)
        yf = y.to(device, non_blocking=True).float()
        with autocast(device, cfg.precision):
            out = model(x).squeeze(-1)
        # loss in fp32 regardless of the autocast dtype: this number is read off
        # a curve and compared across runs, so it should not carry bf16 noise
        loss = criterion(out.float(), yf)

        total_loss += float(loss.item()) * x.shape[0]
        n_seen += x.shape[0]
        logits_all.append(out.detach().float().cpu().numpy().astype(np.float64))
        labels_all.append(y.numpy().astype(np.int64))
        fps_all.extend(list(fp))
    bar.close()

    if device == "cuda":
        torch.cuda.synchronize()
    logits = np.concatenate(logits_all) if logits_all else np.zeros(0)
    labels = np.concatenate(labels_all) if labels_all else np.zeros(0, dtype=np.int64)
    return {
        "logits": logits,
        "labels": labels,
        "filepaths": fps_all,
        "val_loss": total_loss / max(1, n_seen),
        "val_seconds": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(
    cfg: Config,
    resume: Optional[str] = None,
    max_epochs: Optional[int] = None,
    progress: bool = True,
    tag: str = "",
) -> Dict[str, Any]:
    """Train one fold. Returns the run summary dict (also written to disk).

    ``max_epochs`` caps the number of COMPLETED epochs for this invocation --
    an absolute cap, not a per-invocation budget. It leaves the schedule alone
    (``cfg.epochs`` still sets total_steps), so stopping at 2 of 30 and resuming
    is indistinguishable from never stopping. Used by the resume test to
    simulate a kill, and by the timing probes to price a few epochs of a long
    schedule without running it out.
    """
    run_name = split_name(cfg) + (f"_{tag}" if tag else "")
    run_dir = os.path.join(cfg.out_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    pred_dir = os.path.join(run_dir, "preds")
    for d in (ckpt_dir, pred_dir):
        os.makedirs(d, exist_ok=True)
    log_path = os.path.join(run_dir, LOG_NAME)

    # The `fold` column of the prediction dump. A CV run writes its real fold
    # (0..4); a LOCO run writes -centre, so a held-out-centre prediction can
    # never be mistaken for a CV fold by anything reading the parquet alone.
    pred_fold = cfg.fold if cfg.holdout_centre is None else -cfg.holdout_centre

    wall_t0 = time.perf_counter()
    settings = seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.to_yaml(os.path.join(run_dir, "config.yaml"))

    # --- data ---
    manifest_df = pd.read_csv(cfg.manifest)
    train_ds, val_ds = build_datasets(cfg, manifest_df)
    meta = val_metadata(manifest_df, val_ds.filepaths)
    cache_seconds = (train_ds.cache_build_seconds or 0.0) + (val_ds.cache_build_seconds or 0.0)

    # Two generators, deliberately. train_gen drives the sampler and is re-seeded
    # from (seed, epoch) before every epoch. loader_gen exists only so the
    # DataLoader has somewhere to draw its worker base seeds from -- a draw that
    # happens when workers are constructed, which with persistent_workers is
    # once per process and therefore at a different epoch in a resumed run. Kept
    # separate, that draw cannot perturb the batch order.
    train_gen = torch.Generator()
    loader_gen = torch.Generator()
    loader_gen.manual_seed(cfg.seed)
    seed_epoch(train_gen, cfg.seed, 0)
    train_sampler = make_sampler(train_ds, cfg, train_gen)
    train_loader = make_loader(train_ds, cfg, shuffle=False, sampler=train_sampler,
                               generator=loader_gen)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    # --- model / optimisation ---
    model = build_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()  # plain: imbalance is handled by the sampler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = make_scheduler(optimizer, total_steps, cfg.warmup_frac)
    scaler = make_scaler(cfg.precision, device)

    start_epoch, history = 0, []
    best = {"roc_auc": -float("inf"), "epoch": -1}

    # --- resume ---
    if resume:
        ckpt_path = os.path.join(ckpt_dir, CKPT_LAST) if resume == "auto" else resume
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"--resume: no checkpoint at {ckpt_path}")
        # to CPU, not to `device`: RNG and generator states are CPU ByteTensors
        # and torch refuses to restore them from CUDA. model.load_state_dict and
        # optimizer.load_state_dict both move their own tensors to the right
        # device, so nothing is lost by staging through the host.
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        check_resume_compatible(ck["config"], cfg.to_dict())
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
        best = dict(ck["best"])
        logger.info("resumed from %s at epoch %d", ckpt_path, start_epoch)
        append_jsonl(log_path, {"event": "resume", "from": ckpt_path,
                                "completed_epochs": start_epoch,
                                "time": time.time()})

    stop_at = cfg.epochs if max_epochs is None else min(cfg.epochs, max_epochs)

    counts = np.bincount(train_ds.labels(), minlength=2).tolist()

    # What the sampler ACTUALLY draws, not what its weights were meant to mean.
    # One epoch is drawn from a saved-and-restored copy of the generator state,
    # so this measurement cannot perturb the stream training then consumes.
    draw_counts = stratum_draw_counts(train_sampler, train_ds, train_gen)
    n_drawn = sum(draw_counts.values())
    logger.info(
        "sampler=%s -- realised draws for one epoch (%d): %s",
        cfg.sampler, n_drawn,
        ", ".join(f"{k} {v} ({100.0 * v / max(1, n_drawn):.1f}%)"
                  for k, v in draw_counts.items()),
    )

    header = {
        "event": "run_start",
        "time": time.time(),
        "run_dir": run_dir,
        "config": cfg.to_dict(),
        "seeding": settings,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "train_class_counts": counts,
        "val_class_counts": np.bincount(val_ds.labels(), minlength=2).tolist(),
        "sampler_stratum_draws": dict(draw_counts),
        "sampler_stratum_draws_note": (
            "realised (centre|class) counts over ONE drawn epoch, measured from "
            "a restored copy of the sampler generator; the training stream is "
            "unaffected"
        ),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": max(1, int(round(total_steps * cfg.warmup_frac))),
        "cache_build_seconds": cache_seconds,
        "cache_bytes": train_ds.cache_bytes() + val_ds.cache_bytes(),
        "start_epoch": start_epoch, "stop_at": stop_at,
        **param_counts(model),
    }
    append_jsonl(log_path, header)
    logger.info(
        "%s | %s | %d train (%d neg / %d pos) / %d val | %d steps/epoch | cache %.1f s",
        cfg.arch, cfg.precision, len(train_ds), counts[0], counts[1],
        len(val_ds), steps_per_epoch, cache_seconds,
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    monitor = GpuMonitor()
    monitor.start()

    # --- EMA / SWA state (passive; see the helpers' block comment). On resume
    # the shadow/sum come back from the checkpoint; a checkpoint written before
    # these features existed simply lacks the keys and the state re-initialises
    # from the current weights -- correct for EMA (it re-converges within ~1k
    # steps), and SWA restarts its window, both logged rather than silent.
    ema_shadow: Optional[Dict[str, torch.Tensor]] = None
    swa_state: Optional[Dict[str, Any]] = None
    swa_start_epoch = int(cfg.epochs * cfg.swa_start_frac)
    if cfg.ema_enabled:
        ema_shadow = _avg_state_init(model)
        if resume and "ema" in (locals().get("ck") or {}):
            for k, v in ck["ema"].items():
                if k in ema_shadow:
                    ema_shadow[k] = v.to(ema_shadow[k].device).float()
            logger.info("EMA shadow restored from checkpoint")
        elif resume:
            logger.warning("resume checkpoint has no EMA state; shadow "
                           "re-initialised from current weights at epoch %d",
                           start_epoch)
    if cfg.swa_enabled:
        swa_state = {"sum": {k: torch.zeros_like(v)
                             for k, v in _avg_state_init(model).items()},
                     "count": 0}
        if resume and "swa" in (locals().get("ck") or {}):
            swa_state["count"] = int(ck["swa"]["count"])
            for k, v in ck["swa"]["sum"].items():
                if k in swa_state["sum"]:
                    swa_state["sum"][k] = v.to(swa_state["sum"][k].device).float()
            logger.info("SWA accumulator restored (count=%d)", swa_state["count"])
        elif resume and start_epoch > swa_start_epoch:
            logger.warning("resume checkpoint has no SWA state and the SWA "
                           "window began at epoch %d; the average will cover "
                           "epochs %d.. only", swa_start_epoch, start_epoch)

    epoch_bar = tqdm(
        range(start_epoch, stop_at), desc="epochs", unit="ep",
        initial=0, total=stop_at - start_epoch,
        disable=not progress, dynamic_ncols=True,
    )
    try:
        for epoch in epoch_bar:
            # batch order is a pure function of (seed, epoch) -- see seed_epoch
            seed_epoch(train_gen, cfg.seed, epoch)
            # per-item augmentation is also a pure function of (seed, epoch, index)
            # -- see BarrettDataset.set_epoch for why this must go through shared
            # memory rather than a plain attribute
            train_ds.set_epoch(epoch)

            tr = train_one_epoch(model, train_loader, criterion, optimizer,
                                 scheduler, scaler, cfg, device, epoch, progress,
                                 ema_shadow=ema_shadow)
            if swa_state is not None and epoch >= swa_start_epoch:
                _swa_accumulate(swa_state, model)
            va = validate(model, val_loader, criterion, cfg, device, epoch, progress)

            # dump validation logits before computing anything from them, so the
            # metrics can always be recomputed from what is on disk
            pred_path = os.path.join(
                pred_dir, f"val_{split_name(cfg)}_e{epoch + 1:03d}.parquet")
            sub = meta.loc[va["filepaths"]]
            write_predictions(
                build_frame(
                    filepath=va["filepaths"],
                    centre=sub["centre"].tolist(),
                    class_label=sub["class_label"].tolist(),
                    label_int=va["labels"],
                    visibility=[None if pd.isna(v) else str(v)
                                for v in sub["visibility"]],
                    group_id_v2=sub["group_id_v2"].astype(str).tolist(),
                    repeat=cfg.repeat, fold=pred_fold, seed=cfg.seed,
                    logit=va["logits"],
                ),
                pred_path,
            )

            m = evaluate(va["labels"], va["logits"])
            util = monitor.drain()
            record = {
                "event": "epoch",
                "epoch": epoch + 1,
                "time": time.time(),
                "train_loss": tr["train_loss"],
                "val_loss": va["val_loss"],
                **m,
                "lr": scheduler.get_last_lr()[0],
                "train_seconds": tr["train_seconds"],
                "val_seconds": va["val_seconds"],
                "epoch_seconds": tr["train_seconds"] + va["val_seconds"],
                "gpu_util_mean": util["mean"],
                "gpu_util_p50": util["p50"],
                "gpu_util_samples": util["n"],
                "vram_peak_alloc_gb": (torch.cuda.max_memory_allocated() / 2 ** 30
                                       if device == "cuda" else 0.0),
                "vram_peak_reserved_gb": (torch.cuda.max_memory_reserved() / 2 ** 30
                                          if device == "cuda" else 0.0),
                # bf16 logits quantise; ties change AUC and wreck PPV at a
                # threshold. Track how many distinct values survive.
                "n_unique_logits": int(np.unique(va["logits"]).size),
                "n_val": int(va["logits"].size),
                "pred_path": pred_path,
            }
            history.append(record)

            improved = m["roc_auc"] > best["roc_auc"]
            if improved:
                best = {"roc_auc": m["roc_auc"], "epoch": epoch + 1}

            payload = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "train_generator": train_gen.get_state(),
                "loader_generator": loader_gen.get_state(),
                "rng": rng_state(),
                "history": history,
                "best": best,
                "config": cfg.to_dict(),
                "metrics": m,
                "selection": "last",
                "canonical": True,
            }
            if ema_shadow is not None:
                payload["ema"] = {k: v.cpu() for k, v in ema_shadow.items()}
            if swa_state is not None:
                payload["swa"] = {"count": swa_state["count"],
                                  "sum": {k: v.cpu()
                                          for k, v in swa_state["sum"].items()}}
            # checkpoint first, log second: the checkpoint carries `history`, so
            # a kill between the two loses nothing that a resume cannot replay.
            # This is the rolling RESUME checkpoint only -- see module docstring
            # -- and it is deleted below once the schedule completes; no
            # diagnostic best-epoch checkpoint is ever written (best-by-ROC-AUC
            # is tracked in `best` / summary.json, never as a file).
            save_checkpoint(os.path.join(ckpt_dir, CKPT_LAST), payload)
            append_jsonl(log_path, record)

            epoch_bar.set_postfix_str(
                f"auc {m['roc_auc']:.4f} | pauc {m['pauc_15_std']:.4f} | "
                f"ppv90 {m['ppv_at_90_recall']:.3f} | gpu {util['mean']:.0f}%"
            )
            tqdm.write(
                f"epoch {epoch + 1:3d}/{cfg.epochs}  "
                f"train {tr['train_loss']:.4f}  val {va['val_loss']:.4f}  "
                f"auc {m['roc_auc']:.4f}  pauc15 {m['pauc_15_std']:.4f}  "
                f"ppv90 {m['ppv_at_90_recall']:.3f}  "
                f"{record['epoch_seconds']:.1f}s  gpu {util['mean']:.0f}%"
            )
            _write_summary(run_dir, cfg, header, history, best, wall_t0,
                           monitor.overall(), device, completed=epoch + 1,
                           checkpoint_path=os.path.join(ckpt_dir, CKPT_LAST))

            # Checked after the first completed epoch only: by then the
            # steady-state footprint (activations + AdamW's exp_avg/exp_avg_sq)
            # is established, and every checkpoint/log/summary write for this
            # epoch has already landed on disk, so aborting here loses nothing
            # a resume cannot replay.
            if device == "cuda" and epoch == 0:
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                limit_bytes = VRAM_SPILL_FRACTION * total_bytes
                reserved_bytes = torch.cuda.max_memory_reserved()
                if reserved_bytes >= limit_bytes:
                    raise VramSpillError(
                        f"VRAM spill guard tripped after epoch 1: reserved "
                        f"{reserved_bytes / 2 ** 30:.2f} GiB >= "
                        f"{VRAM_SPILL_FRACTION:.0%} of "
                        f"{total_bytes / 2 ** 30:.2f} GiB physical VRAM, at "
                        f"batch_size={cfg.batch_size}. This driver does not "
                        f"raise OutOfMemoryError when it overflows -- it pages "
                        f"to system RAM over PCIe and runs roughly 40x slower "
                        f"with no error. Aborting now rather than silently "
                        f"paging through the remaining "
                        f"{cfg.epochs - (epoch + 1)} epochs. Lower batch_size "
                        f"or enable grad_checkpointing and re-run (checkpoint "
                        f"for epoch {epoch + 1} was saved; --resume will pick "
                        f"up from there once the config is fixed)."
                    )
    finally:
        epoch_bar.close()
        monitor.stop()

    completed = history[-1]["epoch"] if history else start_epoch
    # an incomplete run still needs its rolling resume checkpoint, so the only
    # checkpoint that exists for it is CKPT_LAST -- same file --resume reads
    checkpoint_path: Optional[str] = os.path.join(ckpt_dir, CKPT_LAST)
    # the canonical dump is the last epoch of a FULLY completed schedule; a
    # truncated run does not get to claim one
    if completed >= cfg.epochs and history:
        canonical = os.path.join(run_dir, f"val_{split_name(cfg)}.parquet")
        # Copy to a temp name, then durable_replace. This file is what
        # run_state() reads to decide a unit is DONE and can be skipped
        # forever, which makes it the single most damaging file in the repo to
        # get half-written -- a torn canonical parquet is a silently wrong
        # result, not a crash. The copy previously went straight to os.replace
        # with no fsync: atomic against a killed process, NOT against a host
        # reset, the same gap that corrupted a checkpoint on 2026-08-03.
        shutil.copyfile(history[-1]["pred_path"], canonical + ".tmp")
        durable_replace(canonical + ".tmp", canonical)
        logger.info("canonical predictions (last epoch): %s", canonical)

        # --- EMA / SWA variant outputs. The RAW canonical parquet above is
        # untouched; each variant gets its own validation pass and its own
        # parquet (val_<split>_<variant>.parquet), written only for a FULLY
        # completed schedule, same rule as the canonical. The live weights are
        # restored afterwards so the raw weights artefact below is exactly the
        # last epoch's weights, not an averaged impostor.
        variant_states: Dict[str, Dict[str, torch.Tensor]] = {}
        if ema_shadow is not None:
            variant_states["ema"] = ema_shadow
        if swa_state is not None and swa_state["count"] > 0:
            variant_states["swa"] = _swa_mean(swa_state)
        raw_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                     if variant_states else None)
        for vname, vstate in variant_states.items():
            merged = dict(model.state_dict())
            for k, v in vstate.items():
                merged[k] = v.to(merged[k].device, dtype=merged[k].dtype)
            model.load_state_dict(merged)
            vva = validate(model, val_loader, criterion, cfg, device,
                           cfg.epochs - 1, progress)
            vsub = meta.loc[vva["filepaths"]]
            vpath = os.path.join(run_dir, f"val_{split_name(cfg)}_{vname}.parquet")
            write_predictions(
                build_frame(
                    filepath=vva["filepaths"],
                    centre=vsub["centre"].tolist(),
                    class_label=vsub["class_label"].tolist(),
                    label_int=vva["labels"],
                    visibility=[None if pd.isna(v) else str(v)
                                for v in vsub["visibility"]],
                    group_id_v2=vsub["group_id_v2"].astype(str).tolist(),
                    repeat=cfg.repeat, fold=pred_fold, seed=cfg.seed,
                    logit=vva["logits"],
                ),
                vpath,
            )
            vm = evaluate(vva["labels"], vva["logits"])
            logger.info("%s variant: %s | auc %.4f pauc15 %.4f",
                        vname, vpath, vm["roc_auc"], vm["pauc_15_std"])
            if cfg.save_checkpoint:
                vweights = {
                    "epoch": completed,
                    "model": {k: v.detach().cpu().float()
                              for k, v in vstate.items()},
                    "config": cfg.to_dict(),
                    "metrics": vm,
                    "selection": vname,
                    "canonical": True,   # canonical FOR ITS VARIANT: full schedule
                    "variant": vname,
                }
                vck = (CKPT_WEIGHTS_EMA if vname == "ema" else CKPT_WEIGHTS_SWA)
                save_checkpoint(os.path.join(ckpt_dir, vck), vweights)
        if raw_state is not None:
            model.load_state_dict(raw_state)

        # Disposition of the rolling resume checkpoint now that the schedule is
        # done: a screening run (save_checkpoint=False, the default) keeps
        # nothing -- the parquets above are the whole output. A run explicitly
        # designated a final ensemble member keeps a slim fp32, weights-only
        # artefact instead. Either way CKPT_LAST itself is no longer needed --
        # resume only ever applies to a run that did not finish.
        if cfg.save_checkpoint:
            weights_payload = {
                "epoch": completed,
                "model": {k: v.detach().cpu().float()
                         for k, v in model.state_dict().items()},
                "config": cfg.to_dict(),
                "metrics": m,
                "selection": "last",
                "canonical": True,
            }
            checkpoint_path = os.path.join(ckpt_dir, CKPT_WEIGHTS)
            save_checkpoint(checkpoint_path, weights_payload)
        else:
            checkpoint_path = None
        last_path = os.path.join(ckpt_dir, CKPT_LAST)
        if os.path.exists(last_path):
            os.remove(last_path)

    summary = _write_summary(run_dir, cfg, header, history, best, wall_t0,
                             monitor.overall(), device, completed=completed,
                             checkpoint_path=checkpoint_path)
    append_jsonl(log_path, {"event": "run_end", "time": time.time(),
                            "completed_epochs": completed,
                            "wall_seconds": summary["wall_seconds"]})
    return summary


def _write_summary(run_dir, cfg, header, history, best, wall_t0, util, device,
                   completed, checkpoint_path) -> Dict[str, Any]:
    epoch_secs = [h["epoch_seconds"] for h in history]
    summary = {
        "run_dir": run_dir,
        "config": cfg.to_dict(),
        "header": header,
        "completed_epochs": completed,
        "wall_seconds": time.perf_counter() - wall_t0,
        "cache_build_seconds": header["cache_build_seconds"],
        "mean_epoch_seconds": float(np.mean(epoch_secs)) if epoch_secs else float("nan"),
        "median_epoch_seconds": float(np.median(epoch_secs)) if epoch_secs else float("nan"),
        "mean_train_seconds": (float(np.mean([h["train_seconds"] for h in history]))
                               if history else float("nan")),
        "mean_val_seconds": (float(np.mean([h["val_seconds"] for h in history]))
                             if history else float("nan")),
        "gpu_util_mean": util["mean"],
        "gpu_util_p50": util["p50"],
        "vram_peak_alloc_gb": (torch.cuda.max_memory_allocated() / 2 ** 30
                               if device == "cuda" else 0.0),
        "vram_peak_reserved_gb": (torch.cuda.max_memory_reserved() / 2 ** 30
                                  if device == "cuda" else 0.0),
        "canonical_checkpoint": checkpoint_path,
        "canonical_selection": "last epoch",
        "diagnostic_best_roc_auc": best,
        "diagnostic_note": BEST_WARNING,
        "history": history,
    }
    summary_path = os.path.join(run_dir, SUMMARY_NAME)
    tmp = summary_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    durable_replace(tmp, summary_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    overrides = {
        k: v for k, v in dict(
            repeat=args.repeat, fold=args.fold, seed=args.seed,
            holdout_centre=args.holdout_centre,
            epochs=args.epochs, batch_size=args.batch_size,
            image_size=args.image_size, cache_size=args.cache_size,
            precision=args.precision, sampler=args.sampler,
            num_workers=args.num_workers, arch=args.arch, lr=args.lr,
            out_dir=args.out_dir,
        ).items() if v is not None
    }
    if args.grad_checkpointing:
        overrides["grad_checkpointing"] = True
    if args.non_deterministic:
        overrides["deterministic"] = False
    return dataclasses.replace(cfg, **overrides)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train one (repeat, fold, seed). Canonical output is the LAST epoch."
    )
    ap.add_argument("--config", default=None, help="YAML config; defaults from src/config.py")
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--holdout-centre", type=int, default=None, choices=[1, 2],
                    help="leave-one-centre-out instead of (repeat, fold) CV")
    # overrides, so a timing probe does not need its own config file
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--cache-size", type=int, default=None)
    ap.add_argument("--precision", default=None, choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--sampler", default=None,
                    choices=["weighted", "balanced_centre_class", "none",
                             "random", "sequential"])
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--non-deterministic", action="store_true",
                    help="cudnn.benchmark on; faster, not bit-reproducible")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="stop once this many epochs are COMPLETE (schedule unchanged)")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="resume from checkpoints/last.pt, or an explicit path")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    prog = ap.add_mutually_exclusive_group()
    prog.add_argument("--progress", dest="progress", action="store_true", default=None)
    prog.add_argument("--no-progress", dest="progress", action="store_false")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    progress = sys.stderr.isatty() if args.progress is None else args.progress

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    summary = run(cfg, resume=args.resume, max_epochs=args.max_epochs,
                  progress=progress, tag=args.tag)

    print(f"\ncompleted {summary['completed_epochs']}/{cfg.epochs} epochs in "
          f"{summary['wall_seconds'] / 60:.1f} min "
          f"({summary['mean_epoch_seconds']:.1f} s/epoch, "
          f"GPU {summary['gpu_util_mean']:.0f}%)")
    print(f"canonical (LAST epoch): {summary['canonical_checkpoint']}")
    print(f"diagnostic best ROC-AUC: epoch {summary['diagnostic_best_roc_auc']['epoch']} "
          f"({summary['diagnostic_best_roc_auc']['roc_auc']:.4f}) -- do not select on this")


if __name__ == "__main__":
    main()
