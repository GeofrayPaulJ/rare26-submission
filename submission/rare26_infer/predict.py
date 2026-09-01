"""Inference pipeline: stack in, list of float64 probabilities out.

ORDER IS THE CONTRACT. The output JSON is a bare list of floats with no keys,
so position i must correspond to slice i of the input stack and nothing else.
The DataLoader is unshuffled, but that alone is not a guarantee worth resting
on -- workers complete out of order internally and a future batching change
could reorder silently -- so every item carries its own index and results are
scattered into a preallocated array by that index. A wrong permutation would
score like a broken model while looking perfectly well-formed.

PRECISION. The evaluation GPU is a T4 or A10G, neither of which has bf16, so
fp16 is the default and the path that gets tested. The classifier head still
runs fp32 (see model.py), and the sigmoid is taken in float64: at the observed
logit range of +/-8.5 an fp32 sigmoid is already losing resolution against
saturation, while float64 does not saturate until around logit 37. With 23,176
negatives, ties at the operating threshold inflate the false positive rate
directly, which is why the distinct-value assertion below is a hard failure and
not a warning.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:                     # keep the image runnable without tqdm
    tqdm = None

from .fov import FIT_QUALITY_FLOOR, centred_square, detect_crop_box
from .model import build_ensemble
from .preprocess import CACHE_SIZE, IMAGE_SIZE, preprocess
from .stack import find_stack_file, open_stack

logger = logging.getLogger("rare26")

# Fraction of the input count the distinct-probability count must exceed.
TIE_FRACTION = float(os.environ.get("RARE26_TIE_FRACTION", "0.99"))

# G4, 2026-08-12: hard wall-clock governor. The evaluation platform's own
# per-case limit is CONFIRMED at 600s (probe C, job ba39f6c6, "Time limit
# exceeded"). A killed job writes nothing at all (see reports/g3_timeout_semantics.md,
# reports/p6_probe_c_partial_output.md) -- there is no partial-credit path,
# so a self-imposed budget that finishes early and returns a valid, mostly-
# complete file is strictly better than trusting the platform's own kill.
# 540s = 600s with a 10% margin. This governor does not depend on knowing
# WHY per-batch cost grows (see reports/g1_scaling_defect.md,
# reports/g2_scaling_fix.md -- the accumulation was localised but not fixed
# this session); it only needs to observe that it does, and degrade before
# the observed rate would blow the budget.
TIME_BUDGET_SECONDS = float(os.environ.get("RARE26_TIME_BUDGET_SECONDS", "540"))

# G5, 2026-08-12: platform job 35c0face showed the governor firing off a
# projection built from batch 1 alone -- the cold-start batch, which every
# run on record (probe C, this job) runs 3-7x slower than the rate it
# settles to within a handful of batches. Projecting from that one batch
# bailed at 32/900 images having used only 126.7s of a 540s budget. Two
# guards: never project before MIN_BATCHES_FOR_PROJECTION batches have
# completed (so batch 1 can never be the sole input to a decision), and
# project from a TRAILING_WINDOW_BATCHES-batch trailing rate rather than
# the cumulative average (so an early slow patch doesn't linger in the
# estimate after the rate has already moved on). An unconditional hard
# stop (HARD_STOP_SECONDS, independently of any projection) is the
# backstop for the case the trailing-rate estimate is ever wrong in the
# dangerous direction.
#
# G6, 2026-08-12: platform job d12f343e confirmed G5 works (160/900
# scored, was 32; unique_prob_frac 1.0, was 0.0367) but still overshot the
# budget by ~2% (projected 551.8s vs a 540s budget) -- the trailing-2 rate
# at the moment of decision was formed from batches 3 and 4 (36.65s,
# 26.95s), which are STILL mid-decay from cold start on every run on
# record; the floor (~24s) isn't reached until batch 5+. Two more guards:
# MIN_BATCHES_FOR_PROJECTION raised 3 -> 5 (the decay runs through batch
# 4, confirmed on this job and every prior one), and the projection now
# takes the MINIMUM of a short (TRAILING_WINDOW_BATCHES) and a long
# (TRAILING_WINDOW_BATCHES_LONG) trailing window -- during decay the
# short window systematically overestimates (it's still weighted toward
# the slower recent-past batches), so the smaller of the two is the less
# alarmist, better-calibrated estimate. HARD_STOP_SECONDS is unchanged --
# it did not fire on d12f343e (187.3s < 480s) and remains the backstop
# for whenever a projection is wrong in the dangerous direction instead.
MIN_BATCHES_FOR_PROJECTION = 5
TRAILING_WINDOW_BATCHES = 2
TRAILING_WINDOW_BATCHES_LONG = 4
HARD_STOP_SECONDS = float(os.environ.get("RARE26_HARD_STOP_SECONDS", "480"))

# G5.2: neutral-filling every unscored slice with the SAME value (p=0.5)
# creates a large tied block, which inflates the false-positive count at
# the operating threshold under a ranking metric -- the exact failure mode
# the tie gate below exists to catch, self-inflicted by the governor's own
# previous fill. Distinct, strictly-decreasing, below-every-real-score
# values rank last with no ties instead. FILL_INCREMENT is deliberately
# far above float64 noise at the logit magnitudes this project observes
# (see reports/g5_governor_tuning.md for the precision check).
FILL_INCREMENT = 1e-6


def _fill_remaining_distinct(logits: np.ndarray, unseen_mask: np.ndarray) -> Tuple[float, float, int]:
    """Fill logits[unseen_mask] with distinct values, strictly below every
    scored logit, spaced by FILL_INCREMENT -- ranks last, in a defined
    order, with no ties. Returns (fill_floor, increment, n_filled)."""
    n_filled = int(unseen_mask.sum())
    scored_mask = np.isfinite(logits) & ~unseen_mask
    if scored_mask.any():
        fill_floor = float(logits[scored_mask].min()) - 1.0
    else:
        # Nothing was ever scored (a pathological, very-early hard stop) --
        # no real minimum to sit below. -10.0 sits well below any logit this
        # project has observed (range is roughly +/-8.5), documented rather
        # than silently arbitrary.
        fill_floor = -10.0
    unseen_indices = np.flatnonzero(unseen_mask)  # ascending, deterministic
    logits[unseen_indices] = fill_floor - np.arange(n_filled) * FILL_INCREMENT
    return fill_floor, FILL_INCREMENT, n_filled


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


class StackDataset(Dataset):
    """One stack slice -> one preprocessed CHW float32 tensor.

    The FOV detection is the expensive part (a connected-component labelling per
    frame), and it is pure CPU, so it lives here to be parallelised across
    DataLoader workers rather than serialised in the main process ahead of the
    GPU.
    """

    def __init__(self, stack, n: int) -> None:
        self.stack = stack
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        arr = self.stack.read(i)
        # Content hash of the raw slice, so the tie gate can tell a precision
        # collapse apart from a stack that simply contains the same frame
        # twice. ~1 ms on a 1 MB frame, inside the worker, against ~5 ms of FOV
        # detection -- cheap enough not to matter, and the gate is unsound
        # without it.
        digest = hashlib.blake2b(arr.tobytes(), digest_size=16).digest()
        # Debug-only, D4 (2026-08-10): force every image down the fallback
        # crop regardless of fit_quality, to quantify the fallback path's
        # own AUROC cost in isolation. Never set in the shipped image --
        # same convention as RARE26_MAX_MEMBERS above.
        if os.environ.get("RARE26_FORCE_FALLBACK"):
            h, w = arr.shape[:2]
            box, used_fallback, fit_quality, reason = (
                centred_square(w, h), True, 0.0, "RARE26_FORCE_FALLBACK debug override")
        else:
            box, used_fallback, fit_quality, reason = detect_crop_box(arr)
        chw = preprocess(arr, box, cache_size=CACHE_SIZE, image_size=IMAGE_SIZE)
        return (
            i,
            torch.from_numpy(chw),
            bool(used_fallback),
            float(fit_quality) if fit_quality is not None else float("nan"),
            reason,
            digest,
        )


def _collate(batch):
    idx = torch.tensor([b[0] for b in batch], dtype=torch.long)
    x = torch.stack([b[1] for b in batch])
    fb = [b[2] for b in batch]
    fq = [b[3] for b in batch]
    reasons = [b[4] for b in batch]
    digests = [b[5] for b in batch]
    return idx, x, fb, fq, reasons, digests


def _peak_host_bytes() -> Dict[str, float]:
    """Host memory, separating what is actually held from what is cached.

    ``memory.peak`` is the cgroup high-water mark and is the number the 32 GB
    cap is enforced against -- but on a run that streams a 22.8 GB file it is
    dominated by PAGE CACHE, not by anything the process is holding. Page cache
    is reclaimable: the kernel drops it under pressure rather than OOM-killing,
    so a peak near the limit is expected and harmless.

    What actually has to fit is anonymous memory -- ``anon`` from memory.stat,
    which is the model, the batches in flight and the worker processes. That is
    the figure the headroom claim rests on, so both are reported and never
    conflated.
    """
    out: Dict[str, float] = {}
    for path in ("/sys/fs/cgroup/memory.peak",
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            with open(path) as fh:
                out["cgroup_peak_gb"] = int(fh.read().strip()) / 2 ** 30
            break
        except (OSError, ValueError):
            continue
    try:
        with open("/sys/fs/cgroup/memory.current") as fh:
            out["cgroup_current_gb"] = int(fh.read().strip()) / 2 ** 30
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/memory.stat") as fh:
            stat = dict(
                (parts[0], int(parts[1]))
                for parts in (ln.split() for ln in fh) if len(parts) == 2
            )
        for key in ("anon", "file", "slab", "kernel_stack"):
            if key in stat:
                out[f"cgroup_{key}_gb"] = stat[key] / 2 ** 30
    except (OSError, ValueError):
        pass
    try:
        import resource
        # ru_maxrss is the per-process high-water RSS; children covers the
        # loader workers. Anonymous memory is the part that cannot be reclaimed.
        out["rusage_self_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        out["rusage_children_gb"] = (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20
        )
    except Exception:  # noqa: BLE001 -- diagnostics must never fail a run
        pass
    return out


def predict_stack(
    image_dir: str,
    weights_paths: Sequence[str],
    precision: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    dump_member_logits_path: Optional[str] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """Run the stack at ``image_dir`` through every member and return
    (logit-averaged probabilities, run stats).

    ENSEMBLE OF FOLDS, NOT OF SEEDS. ``weights_paths`` is the 5-fold,
    single-seed A4 ensemble (seed 0, folds 0-4) -- at training time each
    member was evaluated only on its own held-out fold, so no harness
    artefact contains a genuine "5-model average" reference logit for any
    image; every training image was in 4 of the 5 members' training sets.
    That is expected and is exactly the situation real (unseen) test images
    are in too: at inference every member scores every image and the logits
    are averaged. ``dump_member_logits_path``, when set, writes the
    (n_images, n_members) pre-average logit matrix to disk so a correctness
    check can instead verify each member's OWN held-out agreement with the
    harness (see submission/tools/check_ensemble_members.py) rather than
    against a reference that does not exist.
    """
    precision = precision or os.environ.get("RARE26_PRECISION", "fp16")
    batch_size = batch_size or _env_int("RARE26_BATCH_SIZE", 32)
    default_workers = min(8, (os.cpu_count() or 4))
    num_workers = (num_workers if num_workers is not None
                   else _env_int("RARE26_NUM_WORKERS", default_workers))

    stats: Dict[str, Any] = {}
    t_start = time.perf_counter()

    # --- open the stack (header only; no pixel data is read here) ---
    t0 = time.perf_counter()
    path = find_stack_file(image_dir)
    stack = open_stack(path)
    n = len(stack)
    stats["stack_file"] = os.path.basename(path)
    stats["n_images"] = n
    stats["open_seconds"] = time.perf_counter() - t0
    logger.info("stack: %s | %d slices | opened in %.2fs",
                stats["stack_file"], n, stats["open_seconds"])

    cap = _env_int("RARE26_MAX_IMAGES", 0)
    if cap and cap < n:
        n = cap
        logger.warning("RARE26_MAX_IMAGES=%d -- truncating (debug only)", n)
        stats["n_images"] = n

    if n == 0:
        raise ValueError(f"stack {path} contains no slices")

    # --- model(s) ---
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = build_ensemble(weights_paths, device=device)
    n_members = len(models)
    stats["model_load_seconds"] = time.perf_counter() - t0
    stats["n_members"] = n_members
    stats["device"] = device
    if device == "cuda":
        stats["gpu_name"] = torch.cuda.get_device_name(0)
        stats["torch_arch_list"] = torch.cuda.get_arch_list()
        torch.cuda.reset_peak_memory_stats()

    # fp16 on the eval hardware; bf16 is available locally but must never be the
    # default, because a T4 does not have it and would fall back or fail.
    if device == "cuda" and precision in ("fp16", "bf16"):
        amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext()
        precision = "fp32" if device == "cuda" else f"{precision}-cpu"
    stats["precision"] = precision
    logger.info("device=%s precision=%s batch=%d workers=%d members=%d",
                device, precision, batch_size, num_workers, n_members)

    loader = DataLoader(
        StackDataset(stack, n),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=_collate,
        # Workers each hold an open file handle and a couple of frames; keeping
        # them alive avoids re-parsing the TIFF page directory every epoch-like
        # pass, and prefetch is kept modest so queued batches cannot pile up
        # into the host memory budget.
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logits = np.full(n, np.nan, dtype=np.float64)
    member_logits = (
        np.full((n, n_members), np.nan, dtype=np.float64)
        if dump_member_logits_path else None
    )
    input_digests: set = set()
    n_fallback = 0
    fallback_indices: List[int] = []
    fallback_reasons: Dict[str, int] = {}
    fit_qualities: List[float] = []
    gpu_seconds = 0.0
    seen = 0

    # G4 governor state. active_models starts as the full ensemble; the
    # loop mutates the REFERENCE (not `models`, which stays the full list
    # for stats/manifest purposes) if it has to degrade. n_tta_views is a
    # forward-looking hook: this container does not run multi-view TTA
    # today (predict_stack does one forward pass per member per image), so
    # "drop TTA to identity" is a documented no-op here rather than a
    # simulated one -- a future build that adds TTA views must route them
    # through this same variable for the degrade step to do anything real.
    active_models = models
    n_tta_views = 1
    degradations: List[str] = []
    budget_exceeded_at_batch: Optional[int] = None
    images_filled_neutral = 0
    fill_floor_logit: Optional[float] = None
    fill_increment_logit: Optional[float] = None
    # G6.2: the two trailing-window projections that drove the last
    # degradation decision (or None if none was ever made), so the
    # BUDGET REPORT can show its own reasoning, not just its outcome.
    budget_projected_short_seconds: Optional[float] = None
    budget_projected_long_seconds: Optional[float] = None
    # G5.1: per-batch (not cumulative) wall time and size, so a projection
    # can be built from a trailing window rather than the running average --
    # a running average from batch 1 never fully sheds the cold-start batch's
    # weight; a trailing window does, once enough batches have passed it by.
    batch_wall_times: List[float] = []
    batch_sizes: List[int] = []

    t_loop = time.perf_counter()
    t_batch_start = t_loop
    t_last_eta = t_loop
    batch_iter = loader
    if tqdm is not None:
        batch_iter = tqdm(loader, desc=f"infer x{n_members}", unit="batch",
                          total=len(loader), file=sys.stderr)
    with torch.inference_mode():
        for bi, (idx, x, fb, fq, reasons, digests) in enumerate(batch_iter):
            input_digests.update(digests)
            for pos, (flag, reason) in enumerate(zip(fb, reasons)):
                if flag:
                    n_fallback += 1
                    fallback_indices.append(int(idx[pos]))
                    key = "fit_quality below floor" if "fit_quality" in reason else reason
                    fallback_reasons[key] = fallback_reasons.get(key, 0) + 1
            fit_qualities.extend([v for v in fq if not np.isnan(v)])

            x = x.to(device, non_blocking=True)
            tg = time.perf_counter()
            with autocast_ctx:
                # Logit-average, not probability-average: every member sees
                # the same input, and averaging pre-sigmoid keeps the
                # ensemble a single linear combination rather than a mixture,
                # which is what the pooled-OOF FPR@90R figures elsewhere in
                # this project were computed against.
                member_out = [m(x).squeeze(-1).float() for m in active_models]
                out = torch.stack(member_out, dim=0).mean(dim=0)
            if device == "cuda":
                torch.cuda.synchronize()
            gpu_seconds += time.perf_counter() - tg

            # The head already emits fp32; float64 from here so the sigmoid is
            # taken at full precision and the dump keeps every distinct value.
            logits[idx.numpy()] = out.cpu().numpy().astype(np.float64)
            if member_logits is not None and len(active_models) == n_members:
                stacked = torch.stack(member_out, dim=1)  # (batch, n_members)
                member_logits[idx.numpy()] = stacked.cpu().numpy().astype(np.float64)
            seen += int(x.shape[0])

            now = time.perf_counter()
            batch_wall_times.append(now - t_batch_start)
            batch_sizes.append(int(x.shape[0]))
            t_batch_start = now

            if bi % 50 == 0 or seen == n or now - t_last_eta >= 600:
                el = now - t_loop
                rate = seen / max(el, 1e-9)
                eta_s = (n - seen) / max(rate, 1e-9)
                logger.info("  %6d/%d slices | %.1f img/s | %.1fs elapsed | "
                            "ETA %.1f min",
                            seen, n, rate, el, eta_s / 60)
                t_last_eta = now

            # --- G5: hard stop, unconditional, checked every batch before
            # any projection is even consulted. Independent backstop for
            # the case a trailing-rate projection is ever wrong in the
            # dangerous direction. ---
            elapsed_since_start = now - t_start
            if elapsed_since_start > HARD_STOP_SECONDS and seen < n:
                budget_exceeded_at_batch = bi
                unseen_mask = ~np.isfinite(logits)
                fill_floor_logit, fill_increment_logit, images_filled_neutral = (
                    _fill_remaining_distinct(logits, unseen_mask)
                )
                logger.warning(
                    "==BUDGET DEGRADE== HARD STOP: elapsed %.1fs > hard-stop "
                    "threshold %.1fs -- stopping immediately regardless of "
                    "projection, at %d/%d slices scored, filling the "
                    "remaining %d with distinct rank-last values (floor=%.4f, "
                    "increment=%.1e)",
                    elapsed_since_start, HARD_STOP_SECONDS, seen, n,
                    images_filled_neutral, fill_floor_logit, fill_increment_logit,
                )
                degradations.append(
                    f"batch {bi}: HARD STOP at {elapsed_since_start:.1f}s, "
                    f"{images_filled_neutral} slices filled distinct below "
                    f"{fill_floor_logit:.4f}"
                )
                break

            # --- G5.1/G6.1: project completion, but never from fewer than
            # MIN_BATCHES_FOR_PROJECTION completed batches (so the cold-start
            # decay, confirmed to run through batch 4, can never be the sole
            # input to a decision). G6.2: compute both a short- and a
            # long-trailing-window projection and take the MINIMUM -- during
            # decay the short window is still weighted toward the slower
            # recent-past batches and systematically overestimates; the
            # longer window (more batches, more of the decay averaged
            # away) is the better-calibrated one whenever they disagree, and
            # taking the min rather than picking one window by fiat means
            # whichever window happens to be closer to the true floor at
            # this moment governs. Both figures are logged either way, so
            # the decision is auditable regardless of which one bound. ---
            if len(batch_wall_times) < MIN_BATCHES_FOR_PROJECTION:
                continue

            def _trailing_projection(window: int) -> float:
                w = batch_wall_times[-window:]
                s = batch_sizes[-window:]
                rate = sum(s) / max(sum(w), 1e-9)
                return elapsed_since_start + (n - seen) / max(rate, 1e-9)

            projected_short = _trailing_projection(TRAILING_WINDOW_BATCHES)
            projected_long = _trailing_projection(TRAILING_WINDOW_BATCHES_LONG)
            projected_total = min(projected_short, projected_long)
            budget_projected_short_seconds = projected_short
            budget_projected_long_seconds = projected_long
            if projected_total > TIME_BUDGET_SECONDS and seen < n:
                if n_tta_views > 1:
                    logger.warning(
                        "==BUDGET DEGRADE== step 1: dropping TTA views %d -> 1 "
                        "(identity) -- projected %.1fs (min of trailing-%d %.1fs, "
                        "trailing-%d %.1fs) > budget %.1fs at batch %d",
                        n_tta_views, projected_total, TRAILING_WINDOW_BATCHES,
                        projected_short, TRAILING_WINDOW_BATCHES_LONG, projected_long,
                        TIME_BUDGET_SECONDS, bi)
                    n_tta_views = 1
                    degradations.append(f"batch {bi}: TTA views -> identity")
                elif len(active_models) > 1:
                    logger.warning(
                        "==BUDGET DEGRADE== step 2: dropping ensemble %d -> 1 "
                        "member (first in manifest) -- projected %.1fs (min of "
                        "trailing-%d %.1fs, trailing-%d %.1fs) > budget %.1fs "
                        "at batch %d",
                        len(active_models), projected_total, TRAILING_WINDOW_BATCHES,
                        projected_short, TRAILING_WINDOW_BATCHES_LONG, projected_long,
                        TIME_BUDGET_SECONDS, bi)
                    active_models = active_models[:1]
                    degradations.append(f"batch {bi}: ensemble -> 1 member")
                else:
                    # Step 3: already at the cheapest configuration (1 member,
                    # identity view) and still projecting over budget. Stop
                    # scoring now and fill every unseen slice with distinct,
                    # rank-last values rather than run the clock out with
                    # nothing written at all -- see reports/g3_timeout_semantics.md:
                    # a platform-side kill leaves NO file; this always leaves one.
                    budget_exceeded_at_batch = bi
                    unseen_mask = ~np.isfinite(logits)
                    fill_floor_logit, fill_increment_logit, images_filled_neutral = (
                        _fill_remaining_distinct(logits, unseen_mask)
                    )
                    logger.warning(
                        "==BUDGET DEGRADE== step 3: already at 1 member/identity "
                        "and still projecting %.1fs (min of trailing-%d %.1fs, "
                        "trailing-%d %.1fs) > budget %.1fs -- stopping at %d/%d "
                        "slices scored, filling the remaining %d with distinct "
                        "rank-last values (floor=%.4f, increment=%.1e)",
                        projected_total, TRAILING_WINDOW_BATCHES, projected_short,
                        TRAILING_WINDOW_BATCHES_LONG, projected_long,
                        TIME_BUDGET_SECONDS, seen, n, images_filled_neutral,
                        fill_floor_logit, fill_increment_logit,
                    )
                    degradations.append(
                        f"batch {bi}: stopped early, {images_filled_neutral} "
                        f"slices filled distinct below {fill_floor_logit:.4f}"
                    )
                    break

    stats["inference_seconds"] = time.perf_counter() - t_loop
    stats["budget_seconds"] = TIME_BUDGET_SECONDS
    stats["hard_stop_seconds"] = HARD_STOP_SECONDS
    stats["budget_elapsed_seconds"] = time.perf_counter() - t_start
    stats["budget_degradations"] = degradations
    stats["budget_exceeded_at_batch"] = budget_exceeded_at_batch
    stats["images_scored"] = n - images_filled_neutral
    stats["images_filled_neutral"] = images_filled_neutral
    stats["fill_floor_logit"] = fill_floor_logit
    stats["fill_increment_logit"] = fill_increment_logit
    stats["budget_projected_short_seconds"] = budget_projected_short_seconds
    stats["budget_projected_long_seconds"] = budget_projected_long_seconds
    logger.info(
        "==BUDGET REPORT== budget=%.0fs hard_stop=%.0fs elapsed=%.1fs "
        "projected_short=%s projected_long=%s degradations=%s "
        "images_scored=%d images_filled_neutral=%d "
        "fill_floor_logit=%s fill_increment_logit=%s",
        TIME_BUDGET_SECONDS, HARD_STOP_SECONDS, stats["budget_elapsed_seconds"],
        f"{budget_projected_short_seconds:.1f}" if budget_projected_short_seconds is not None else "n/a",
        f"{budget_projected_long_seconds:.1f}" if budget_projected_long_seconds is not None else "n/a",
        degradations or "none", stats["images_scored"], images_filled_neutral,
        f"{fill_floor_logit:.4f}" if fill_floor_logit is not None else "n/a",
        f"{fill_increment_logit:.1e}" if fill_increment_logit is not None else "n/a",
    )
    stats["gpu_seconds"] = gpu_seconds
    stats["data_seconds"] = stats["inference_seconds"] - gpu_seconds
    stats["images_per_second"] = n / max(stats["inference_seconds"], 1e-9)

    if not np.isfinite(logits).all():
        n_bad = int((~np.isfinite(logits)).sum())
        raise AssertionError(f"{n_bad} slices produced non-finite logits")

    # --- float64 sigmoid ---
    probs = 1.0 / (1.0 + np.exp(-logits))
    if not np.isfinite(probs).all():
        raise AssertionError("non-finite probabilities after sigmoid")

    n_unique_logits = int(np.unique(logits).size)
    n_unique_probs = int(np.unique(probs).size)
    n_distinct_inputs = len(input_digests)
    stats.update(
        n_unique_logits=n_unique_logits,
        n_unique_probs=n_unique_probs,
        n_distinct_inputs=n_distinct_inputs,
        n_duplicate_inputs=n - n_distinct_inputs,
        unique_frac=n_unique_probs / n,
        logit_min=float(logits.min()), logit_max=float(logits.max()),
        prob_min=float(probs.min()), prob_max=float(probs.max()),
        n_fallback=n_fallback,
        fallback_frac=n_fallback / n,
        fallback_reasons=fallback_reasons,
        # Which slices took the fallback crop, so a parity check can separate
        # "the container preprocesses differently from the harness" (a bug)
        # from "the container deliberately crops these differently" (the point
        # of the fallback). Capped so a pathological run cannot bloat the dump.
        fallback_indices=sorted(fallback_indices)[:5000],
        fit_quality_floor=FIT_QUALITY_FLOOR,
        fit_quality_mean=(float(np.mean(fit_qualities)) if fit_qualities else float("nan")),
        fit_quality_min=(float(np.min(fit_qualities)) if fit_qualities else float("nan")),
    )
    if device == "cuda":
        stats["vram_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        stats["vram_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2 ** 30
    stats.update(_peak_host_bytes())
    stats["total_seconds"] = time.perf_counter() - t_start

    logger.info(
        "host memory: anon %.2f GiB (what must fit) | page cache %.2f GiB "
        "(reclaimable) | cgroup peak %.2f GiB",
        stats.get("cgroup_anon_gb", float("nan")),
        stats.get("cgroup_file_gb", float("nan")),
        stats.get("cgroup_peak_gb", float("nan")),
    )
    if device == "cuda":
        logger.info("VRAM: %.2f GiB allocated / %.2f GiB reserved",
                    stats["vram_peak_alloc_gb"], stats["vram_peak_reserved_gb"])
    logger.info("timing: %.1fs total | %.1fs inference (%.1fs GPU, %.1fs data) "
                "| %.1f img/s",
                stats["total_seconds"], stats["inference_seconds"],
                stats["gpu_seconds"], stats["data_seconds"],
                stats["images_per_second"])

    logger.info(
        "FOV fallback: %d/%d (%.2f%%) | reasons=%s",
        n_fallback, n, 100.0 * n_fallback / n, fallback_reasons or "{}",
    )
    logger.info(
        "logits: %d distinct / %d  range [%.4f, %.4f]",
        n_unique_logits, n, stats["logit_min"], stats["logit_max"],
    )

    # --- the tie gate, PRE-CHECK ONLY (informational) ---
    # 2026-08-10: the ASSERTING gate moved to inference.py, AFTER the JSON
    # file is written, reading the file back rather than trusting this
    # in-memory array -- D8 (reports/d8_serialization.md) found the
    # sigmoid -> list -> json.dumps -> write -> read round trip introduces
    # no measurable precision loss (max abs diff 2.8e-17 across 617 real
    # images), so this in-memory count and the file-read count should
    # always agree -- but "should always agree" is exactly the kind of
    # claim this project does not ship unverified. This block stays as a
    # cheap early log line; it does not raise. See interface_0_handler for
    # the real gate.
    #
    # The count is taken against DISTINCT INPUTS, not raw slice count. Two
    # byte-identical frames must produce the same probability -- that is
    # correctness, not a defect -- so charging those ties against the budget
    # would make the gate fire on a stack that merely repeats a frame.
    #
    # G4 originally special-cased budget-degraded fills here, because that
    # version filled every unscored slice with the SAME value (a real tied
    # block). G5.2 replaced that with distinct, strictly-decreasing rank-last
    # values -- filled slices no longer collide with each other OR with any
    # real score, so no carve-out is needed: they behave like ordinary
    # distinct probabilities as far as this gate is concerned. (`n_distinct_inputs`
    # can under-count on an early-stopped run, since `input_digests` only
    # collects from batches actually processed -- but that only makes
    # `required` SMALLER, i.e. more lenient, never a false failure.)
    tie_gate_denominator = n_distinct_inputs
    required = TIE_FRACTION * tie_gate_denominator
    stats["tie_gate_required"] = required
    stats["tie_gate_denominator"] = tie_gate_denominator
    stats["tie_gate_would_pass_in_memory"] = bool(n_unique_probs > required)
    if n_distinct_inputs < n:
        logger.info("input stack holds %d duplicate frame(s); the gate is scored "
                    "against %d distinct inputs",
                    n - n_distinct_inputs, n_distinct_inputs)
    if not stats["tie_gate_would_pass_in_memory"]:
        logger.warning(
            "in-memory tie pre-check would FAIL: %d distinct probabilities does "
            "not exceed %.0f%% of %d distinct inputs (%.1f); %d slices total. "
            "distinct logits=%d, logit range=[%.4f, %.4f], precision=%s. The "
            "real gate runs after the file is written -- this is advance warning.",
            n_unique_probs, TIE_FRACTION * 100, n_distinct_inputs, required, n,
            n_unique_logits, stats["logit_min"], stats["logit_max"], precision,
        )
    else:
        logger.info("in-memory tie pre-check passed: %d distinct probabilities > "
                    "%.1f required (%d distinct inputs)",
                    n_unique_probs, required, n_distinct_inputs)

    if member_logits is not None:
        if degradations:
            # The debug-only per-member dump and the G4 governor can only
            # coexist honestly by admitting the dump is incomplete: rows
            # scored under a degraded (fewer-than-n_members) batch, or
            # distinct-rank-last filled by step 3/hard stop, never got every
            # member's own logit.
            # Filling with the pooled logit (not a real per-member value)
            # keeps the array finite and shaped correctly for whatever reads
            # it back, and is logged loudly rather than silently patched.
            incomplete_mask = ~np.isfinite(member_logits).all(axis=1)
            n_incomplete = int(incomplete_mask.sum())
            member_logits[incomplete_mask] = logits[incomplete_mask, None]
            logger.warning(
                "per-member logit dump: %d/%d rows were scored under a "
                "budget-degraded batch (fewer members or distinct rank-last "
                "filled) -- those rows hold the POOLED logit repeated across all "
                "member columns, not each member's real output. "
                "Degradations: %s", n_incomplete, n, degradations,
            )
        if not np.isfinite(member_logits).all():
            raise AssertionError("non-finite per-member logits in diagnostic dump")
        np.save(dump_member_logits_path, member_logits)
        logger.info("per-member logits (%d x %d) dumped to %s",
                    n, n_members, dump_member_logits_path)

    return [float(p) for p in probs], stats


def write_stats(stats: Dict[str, Any], path: str) -> None:
    """Best-effort stats dump. /output is writable; never fail the run on this."""
    try:
        with open(path, "w") as fh:
            json.dump(stats, fh, indent=2, default=str)
    except OSError as exc:  # noqa: BLE001
        logger.warning("could not write stats to %s: %s", path, exc)
