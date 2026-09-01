"""Build the local test fixtures: a full-scale synthetic stack, and a parity stack.

TWO MODES, TWO PURPOSES.

``synthetic`` builds the 25,000-frame stack that the memory and throughput
claims are measured against. The frames are real training images, cycled and
given a per-frame brightness/roll jitter. Random noise would have been easier
and would have proved nothing: noise has no circular field of view, so the FOV
detector would fall back on every frame and the measured cost would be of a
code path that never runs in production. Real frames also guarantee the logits
come out distinct, which is what the tie gate is there to check.

``parity`` writes the 617 validation images of repeat 0 / fold 0 at their native
resolution, in a recorded order, so the container's output can be compared
against the logits src/train.py dumped for exactly those images. Pages are
written at their true sizes rather than resized to a common shape -- resizing
would change the pixels and make the comparison meaningless.

Output is always BigTIFF: 25,000 frames of 637x512x3 is ~24.5 GB, well past
classic TIFF's 4 GB offset ceiling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
IMAGE_ROOT = os.path.join(REPO_ROOT, "00_source")

INPUT_SLUG = "stacked-barretts-esophagus-endoscopy-images"
INPUT_DIRNAME = "stacked-barretts-esophagus-endoscopy"

# The modal training resolution (1,401 of 3,095 images), so the synthetic stack
# is the size the real one is most likely to be.
MODAL_W, MODAL_H = 637, 512


def _load_rgb(rel: str) -> np.ndarray:
    return np.asarray(Image.open(os.path.join(IMAGE_ROOT, rel)).convert("RGB"))


def write_inputs_json(interface_dir: str) -> None:
    """The inputs.json the platform mounts, carrying the socket slug."""
    payload = [{
        "interface": {
            "slug": INPUT_SLUG,
            "kind": "Image",
            "relative_path": f"images/{INPUT_DIRNAME}",
        }
    }]
    with open(os.path.join(interface_dir, "inputs.json"), "w") as fh:
        json.dump(payload, fh, indent=4)


def build_synthetic(out_dir: str, n: int, pool_size: int, seed: int) -> None:
    df = pd.read_csv(MANIFEST)
    modal = df[(df["width"] == MODAL_W) & (df["height"] == MODAL_H)]
    if len(modal) == 0:
        raise SystemExit(f"no {MODAL_W}x{MODAL_H} images in the manifest")

    rng = np.random.default_rng(seed)
    pool_rows = modal["filepath"].tolist()[:pool_size]
    print(f"preloading {len(pool_rows)} real frames at {MODAL_W}x{MODAL_H} ...")
    pool = np.stack([_load_rgb(fp) for fp in
                     tqdm(pool_rows, desc="preload pool", unit="img",
                          file=sys.stderr)])
    print(f"pool resident: {pool.nbytes / 2 ** 30:.2f} GiB")
    print(f"each base frame is reused ~{n / max(1, len(pool)):.0f}x with independent jitter")

    img_dir = os.path.join(out_dir, "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, "stack.tif")

    print(f"writing {n} pages -> {path}")
    # Jitter is applied to the FOV interior only: scaling the letterbox black
    # would lift it above the detector's brightness threshold and change the
    # detected geometry, making the fixture unrepresentative.
    #
    # THE SENSOR NOISE TERM IS LOAD-BEARING. A first version used only a
    # brightness gain and a constant offset, and 25,000 frames drawn from 700
    # base images came back with 24,336 distinct logits (97.3%) -- enough to
    # trip the pipeline's own 99% tie gate. Only 43 of those ties were
    # byte-identical frames; the rest were distinct-but-near-identical frames
    # whose fp16 backbone features quantised onto the same value. A global gain
    # barely moves a normalised feature vector. Per-pixel noise makes each
    # frame genuinely different content, which is what 25,000 endoscopy frames
    # from twelve centres actually are -- the real 617-image validation stack
    # scores 616/617 (99.8%) distinct for exactly that reason.
    est_gb = n * MODAL_H * MODAL_W * 3 / 2 ** 30
    print(f"estimated size: {est_gb:.1f} GiB (bigtiff)")
    t_build = time.perf_counter()
    t_last_eta = t_build
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for i in tqdm(range(n), desc="write synthetic stack", unit="page",
                      file=sys.stderr):
            base = pool[i % len(pool)]
            frame = base.astype(np.float32)
            lit = frame.mean(axis=2) > 15
            frame[lit] *= float(rng.uniform(0.85, 1.15))
            # translate by a few pixels so the FOV geometry differs frame to
            # frame, as it does between real acquisitions
            dx, dy = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
            frame = np.roll(frame, (dy, dx), axis=(0, 1))
            frame += rng.normal(0.0, 2.0, size=frame.shape).astype(np.float32)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            tw.write(frame, photometric="rgb", compression=None, contiguous=False)
            # ETA line to STDOUT on a wall-clock cadence, independent of tqdm
            # (which goes to stderr and needs a TTY to be readable) -- so a
            # piped/teed log still shows progress on a detached overnight run.
            now = time.perf_counter()
            if now - t_last_eta >= 600 or (i + 1) == n or (i + 1) % 2500 == 0:
                rate = (i + 1) / max(now - t_build, 1e-9)
                eta_s = (n - i - 1) / max(rate, 1e-9)
                print(f"  {i + 1}/{n} pages "
                      f"({os.path.getsize(path) / 2 ** 30:.1f} GiB on disk, "
                      f"{rate:.0f} pages/s, ETA {eta_s / 60:.1f} min)", flush=True)
                t_last_eta = now

    write_inputs_json(out_dir)
    print(f"done: {os.path.getsize(path) / 2 ** 30:.2f} GiB")


def build_parity(out_dir: str, repeat: int, fold: int) -> None:
    df = pd.read_csv(MANIFEST)
    usable = df[df[f"fold_r{repeat}"] != -1]
    val = usable[usable[f"fold_r{repeat}"] == fold]
    fps = val["filepath"].tolist()
    print(f"parity stack: {len(fps)} validation images (r{repeat} f{fold})")

    img_dir = os.path.join(out_dir, "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, "stack.tif")

    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for fp in tqdm(fps, desc="write parity stack", unit="img", file=sys.stderr):
            tw.write(_load_rgb(fp), photometric="rgb", compression=None, contiguous=False)

    # The order sidecar is the whole point: the output JSON is positional, so
    # the comparison needs to know which filepath each position corresponds to.
    order_path = os.path.join(out_dir, "stack_order.csv")
    pd.DataFrame({"index": range(len(fps)), "filepath": fps}).to_csv(order_path, index=False)

    write_inputs_json(out_dir)
    print(f"written: {path} ({os.path.getsize(path) / 2 ** 20:.0f} MiB)")
    print(f"order   : {order_path}")


def build_parity_pooled(out_dir: str, repeat: int, folds: list) -> None:
    """All folds' held-out images in one stack, for the ensemble container.

    Each image appears exactly once (it is held out by exactly one fold under
    a k-fold split), tagged with which fold that is. This is what
    tools/check_ensemble_members.py needs: it can only score an image against
    the one member that had it held out, so the stack has to carry that
    mapping rather than just an order.
    """
    df = pd.read_csv(MANIFEST)
    frames = []
    for fold in folds:
        usable = df[df[f"fold_r{repeat}"] != -1]
        val = usable[usable[f"fold_r{repeat}"] == fold]
        frames.append(val.assign(_own_fold=fold))
    pooled = pd.concat(frames, ignore_index=True)
    fps = pooled["filepath"].tolist()
    print(f"parity-pooled stack: {len(fps)} validation images "
          f"across folds {folds} (r{repeat})")

    img_dir = os.path.join(out_dir, "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, "stack.tif")

    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for fp in tqdm(fps, desc="write parity-pooled stack", unit="img",
                       file=sys.stderr):
            tw.write(_load_rgb(fp), photometric="rgb", compression=None, contiguous=False)

    order_path = os.path.join(out_dir, "stack_order.csv")
    pd.DataFrame({"index": range(len(fps)), "filepath": fps,
                  "own_fold": pooled["_own_fold"].tolist()}).to_csv(order_path, index=False)

    write_inputs_json(out_dir)
    print(f"written: {path} ({os.path.getsize(path) / 2 ** 20:.0f} MiB)")
    print(f"order   : {order_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["synthetic", "parity", "parity-pooled"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--pool-size", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="parity-pooled only: which folds to include")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "synthetic":
        build_synthetic(args.out_dir, args.n, args.pool_size, args.seed)
    elif args.mode == "parity":
        build_parity(args.out_dir, args.repeat, args.fold)
    else:
        build_parity_pooled(args.out_dir, args.repeat, args.folds)


if __name__ == "__main__":
    main()
