"""Measure -- do not guess -- the largest batch size that fits in this GPU.

Sweeps the cross product image_size {384, 448} x precision {fp16, bf16} x
gradient_checkpointing {off, on} and binary-searches the batch size in each
cell, then prints a table with the peak memory and the throughput each cell
actually achieved.

A BATCH "FITS" IF ITS PEAK RESERVATION STAYED INSIDE PHYSICAL VRAM -- not if it
avoided raising OutOfMemoryError. On this host the CUDA allocator is allowed to
oversubscribe the device and back the overflow with host memory over PCIe, so an
oversized batch never crashes; it just gets ~40x slower. Measured on this card:
batch 37 reserved 14.6 GiB and ran at 65.5 img/s, while batch 78 "succeeded"
having reserved 29.9 GiB on a 15.9 GiB card, at 1.6 img/s. A probe that treats
absence-of-crash as success reports that 256 fits, and the training run built on
that number spends a week paging.

EVERY TRIAL RUNS IN A FRESH SUBPROCESS. Recovering from a CUDA OOM in-process
leaves the caching allocator fragmented, and a fragmented allocator fails at a
batch size a clean one would have accepted -- which silently biases every later
trial in the sweep downward. A subprocess per trial costs about eight seconds
and buys a number you can trust.

A trial is three *complete* optimiser steps, not a forward pass: AdamW does not
allocate its exp_avg / exp_avg_sq state until the first step, and that state is
two more copies of the parameters. A probe that only runs a forward pass
overestimates the batch that fits by a wide margin.

Settings match src/config.py defaults (deterministic=True, cudnn.benchmark off)
so the reported batch size is safe for the run you are about to launch. Turning
determinism off can change cuDNN's algorithm choice and its workspace, so
re-probe if you flip that flag.

Usage (inside the Prometheus container, from the repo root):
    python -m scripts.vram_probe
    python -m scripts.vram_probe --image-sizes 384 --precisions bf16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import Config, autocast  # noqa: E402
from src.model import build_model  # noqa: E402
from src.seeding import seed_everything  # noqa: E402

EXIT_FITS = 0
EXIT_OOM = 2

# One warmup then two timed steps. Three steps is the minimum that measures the
# real footprint: AdamW allocates exp_avg/exp_avg_sq during step 1, so peak
# memory is only meaningful from step 2 onward. More steps than this buys
# nothing but sweep time, and at batch 200+ a step is not cheap.
WARMUP_STEPS = 1
TIMED_STEPS = 2

# Fraction of total VRAM the predictor aims at when guessing where the ceiling
# is. Only ever used to place the search bracket -- never to report a number.
BUDGET_FRACTION = 0.95

# 2471 training images means batch 256 is already only ~10 steps per epoch.
# Anything above that is a memory curiosity, not a usable configuration.
DEFAULT_CAP = 256


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ---------------------------------------------------------------------------
# Child: one trial
# ---------------------------------------------------------------------------
def run_trial(image_size: int, precision: str, grad_ckpt: bool, batch_size: int,
              arch: str, deterministic: bool) -> Dict[str, float]:
    """Three full training steps at this batch size. Raises on OOM."""
    seed_everything(0, deterministic=deterministic)
    device = "cuda"
    cfg = Config(
        image_size=image_size,
        cache_size=max(image_size, 431),
        precision=precision,
        grad_checkpointing=grad_ckpt,
        batch_size=batch_size,
        arch=arch,
        pretrained=False,  # weights are irrelevant to footprint; skip the download
    )

    model = build_model(cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device, enabled=(precision == "fp16"))

    # float32 CHW, exactly what BarrettDataset hands the loader
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    y = torch.randint(0, 2, (batch_size,), device=device).float()

    torch.cuda.reset_peak_memory_stats()

    def step() -> None:
        with autocast(device, precision):
            loss = criterion(model(x).squeeze(-1), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    tw = time.perf_counter()
    for _ in range(WARMUP_STEPS):
        step()
    torch.cuda.synchronize()
    warm_dt = (time.perf_counter() - tw) / WARMUP_STEPS

    total_gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    budget_gib = BUDGET_FRACTION * total_gib
    if torch.cuda.max_memory_reserved() / 2 ** 30 > budget_gib:
        # Already spilled to host memory during warmup, and every subsequent step
        # would crawl at PCIe speed. The verdict cannot change, so stop paying for
        # it -- a spilled step at batch 43 costs 7.7 s against 0.5 s for one that
        # fits, and the sweep runs several of these while bisecting.
        return {
            "batch_size": batch_size,
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30,
            "seconds_per_step": warm_dt,
            "images_per_second": batch_size / warm_dt,
            "total_gib": total_gib,
            "fits": False,
        }

    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / TIMED_STEPS

    # "Did it raise OutOfMemoryError" is NOT a valid fit test on this machine.
    # Under the Windows WDDM driver the CUDA allocator is allowed to oversubscribe
    # physical VRAM and back the overflow with host memory, so an oversized batch
    # does not crash -- it quietly pages over PCIe and runs ~40x slower. Measured
    # here: batch 37 reserved 14.6 GiB at 65.5 img/s, batch 78 "succeeded" having
    # reserved 29.9 GiB, on a 15.9 GiB card, at 1.6 img/s.
    #
    # So a batch fits only if its peak reservation stayed inside the physical
    # device. That is what "fits in 16 GB VRAM" means, and it is the only signal
    # that separates a usable configuration from a catastrophically slow one.
    peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 30
    return {
        "batch_size": batch_size,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_reserved_gb": peak_reserved,
        "seconds_per_step": dt,
        "images_per_second": batch_size / dt,
        "total_gib": total_gib,
        "fits": bool(peak_reserved <= budget_gib),
    }


# ---------------------------------------------------------------------------
# Parent: binary search over a subprocess per trial
# ---------------------------------------------------------------------------
def trial_subprocess(image_size: int, precision: str, grad_ckpt: bool,
                     batch_size: int, arch: str, deterministic: bool,
                     verbose: bool = True) -> Optional[Dict[str, float]]:
    """Run one trial in a clean process. None means it OOMed."""
    cmd = [
        sys.executable, "-m", "scripts.vram_probe", "--trial",
        "--batch-size", str(batch_size),
        "--image-size", str(image_size),
        "--precision", precision,
        "--arch", arch,
    ]
    if grad_ckpt:
        cmd.append("--grad-checkpointing")
    if not deterministic:
        cmd.append("--non-deterministic")

    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode not in (EXIT_FITS, EXIT_OOM):
        raise RuntimeError(
            f"trial failed (exit {proc.returncode}) at bs={batch_size}:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )

    res = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__TRIAL__"):
            res = json.loads(line[len("__TRIAL__"):])
            break

    if res is None:  # a real CUDA OutOfMemoryError, no row produced
        if verbose:
            print(f"      bs={batch_size:<4d} OOM", flush=True)
        return None

    if verbose:
        verdict = "ok  " if res["fits"] else "SPILL"
        print(f"      bs={batch_size:<4d} {verdict} "
              f"{res['peak_reserved_gb']:5.2f} GiB reserved  "
              f"{res['images_per_second']:6.1f} img/s", flush=True)
    return res if res["fits"] else None


def search_max_batch(image_size: int, precision: str, grad_ckpt: bool, arch: str,
                     deterministic: bool, budget_gib: float,
                     cap: int = DEFAULT_CAP) -> Optional[Dict[str, float]]:
    """Find the largest batch that fits, verifying every candidate by running it.

    Activation memory is very close to linear in the batch size -- weights,
    gradients and AdamW state are fixed, everything else scales per image -- so
    two cheap trials at batch 4 and 8 pin the line, and the line says roughly
    where the ceiling is. That prediction is used ONLY to place the search
    bracket. Nothing derived from it is ever reported: the returned row is
    always a batch size that actually completed its optimiser steps.

    The alternative, doubling from 8 until something OOMs, spends most of its
    trials near the ceiling where each one costs a minute or more. Seeding the
    bracket cuts a checkpointed cell from ~13 trials to ~7.
    """
    def trial(bs: int):
        return trial_subprocess(image_size, precision, grad_ckpt, bs, arch, deterministic)

    anchors = {}
    for bs in (4, 8):
        res = trial(bs)
        if res is None:
            # cannot even fit 8; walk down and report whatever survives
            for small in (4, 2, 1):
                if small in anchors and anchors[small] is not None:
                    return anchors[small]
                res = anchors.get(small) or trial(small)
                if res is not None:
                    return res
            return None
        anchors[bs] = res

    slope = (anchors[8]["peak_reserved_gb"] - anchors[4]["peak_reserved_gb"]) / 4.0
    fixed = anchors[8]["peak_reserved_gb"] - 8 * slope
    if slope <= 0:  # measurement noise; fall back to a plain ramp from 8
        predicted = 16
    else:
        predicted = int((budget_gib - fixed) / slope)
    predicted = max(1, min(cap, predicted))
    print(f"      model: {fixed:.2f} GiB fixed + {slope:.4f} GiB/img "
          f"-> predict ~{predicted}", flush=True)

    best, lo, hi = anchors[8], 8, cap + 1  # lo fits, hi is the first known fail

    res = trial(predicted)
    if res is not None:
        best, lo = res, predicted
        # the model tends to under-predict slightly; climb until something fails
        bs = predicted
        while bs < cap:
            bs = min(cap, int(bs * 1.15) + 1)
            res = trial(bs)
            if res is None:
                hi = bs
                break
            best, lo = res, bs
        else:
            hi = cap + 1
    else:
        hi = predicted
        bs = predicted
        while bs > lo:
            bs = int(bs * 0.85)
            if bs <= lo:
                break
            res = trial(bs)
            if res is not None:
                best, lo = res, bs
                break
            hi = bs

    while hi - lo > 1:
        mid = (lo + hi) // 2
        res = trial(mid)
        if res is None:
            hi = mid
        else:
            best, lo = res, mid

    if lo >= cap:
        print(f"      note: hit the {cap} cap; the true ceiling is higher", flush=True)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--image-sizes", type=int, nargs="+", default=[384, 448])
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    # bf16 first: it is the precision the 384 run uses, so the cell that
    # unblocks the training launch is measured before anything else
    ap.add_argument("--precisions", nargs="+", default=["bf16", "fp16"])
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--non-deterministic", action="store_true")
    ap.add_argument("--arch", default=Config().arch)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.trial:
        try:
            res = run_trial(args.image_size, args.precision, args.grad_checkpointing,
                            args.batch_size, args.arch, not args.non_deterministic)
        except Exception as exc:  # noqa: BLE001 -- OOM is an expected outcome here
            if _is_oom(exc):
                sys.exit(EXIT_OOM)
            raise
        # always emit the row, even when it did not fit: the spilled measurements
        # are the evidence for where the ceiling is
        print("__TRIAL__" + json.dumps(res))
        sys.exit(EXIT_FITS if res["fits"] else EXIT_OOM)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; the probe measures real VRAM or nothing")

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    budget = (total / 2 ** 30) * BUDGET_FRACTION
    print(f"device      : {props.name}")
    print(f"total VRAM  : {total / 2 ** 30:.2f} GiB   free at start: {free / 2 ** 30:.2f} GiB")
    print(f"arch        : {args.arch}")
    print(f"determinism : {'on (cudnn.benchmark off)' if not args.non_deterministic else 'OFF'}")
    print(f"trial       : {WARMUP_STEPS} warmup + {TIMED_STEPS} timed full optimiser steps, "
          f"fresh process each")
    print(f"cap         : {args.cap} (batch above this is <10 steps/epoch on 2471 images)\n",
          flush=True)

    rows: List[Dict] = []
    t0 = time.perf_counter()
    for image_size in args.image_sizes:
        for precision in args.precisions:
            for grad_ckpt in (False, True):
                label = (f"{image_size} / {precision} / "
                         f"grad_ckpt {'on ' if grad_ckpt else 'off'}")
                print(f"  {label}", flush=True)
                res = search_max_batch(image_size, precision, grad_ckpt, args.arch,
                                       not args.non_deterministic, budget_gib=budget,
                                       cap=args.cap)
                row = {
                    "image_size": image_size,
                    "precision": precision,
                    "grad_checkpointing": grad_ckpt,
                    **(res or {"batch_size": 0, "peak_alloc_gb": float("nan"),
                               "peak_reserved_gb": float("nan"),
                               "seconds_per_step": float("nan"),
                               "images_per_second": float("nan")}),
                }
                rows.append(row)
                print(f"      -> max batch {row['batch_size']} "
                      f"({row['peak_reserved_gb']:.2f} GiB, "
                      f"{row['images_per_second']:.1f} img/s)\n", flush=True)

    print("=" * 86)
    print(f"{'img':>5} {'prec':>5} {'grad_ckpt':>10} {'max_bs':>7} "
          f"{'alloc GiB':>10} {'resvd GiB':>10} {'s/step':>8} {'img/s':>8}")
    print("-" * 86)
    for r in rows:
        print(f"{r['image_size']:>5} {r['precision']:>5} "
              f"{'on' if r['grad_checkpointing'] else 'off':>10} "
              f"{r['batch_size']:>7} {r['peak_alloc_gb']:>10.2f} "
              f"{r['peak_reserved_gb']:>10.2f} {r['seconds_per_step']:>8.3f} "
              f"{r['images_per_second']:>8.1f}")
    print("=" * 86)
    print(f"sweep took {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"\nmax_bs is the largest batch whose peak reservation stayed under "
          f"{BUDGET_FRACTION:.0%} of {total / 2 ** 30:.2f} GiB.")
    print("Leave headroom before using it. This driver does not OOM when you overshoot --")
    print("it pages to host RAM and gets ~40x slower, which looks like a mysteriously")
    print("slow run rather than a crash. The training loop also holds pinned batches in")
    print("flight and cuDNN's workspace choice shifts with the determinism flag.")

    out = args.json_out or os.path.join(REPO_ROOT, "runs", "vram_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"device": props.name, "total_gib": total / 2 ** 30,
                   "arch": args.arch, "rows": rows}, fh, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
