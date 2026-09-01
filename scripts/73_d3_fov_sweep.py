"""D3 -- FOV robustness sweep. The FOV detector was tuned on two Dutch
centres (local fallback rate 0.65-1.4%). Synthesise 9 geometric variants of
200 known fold-0 images (real content, geometrically perturbed -- not noise,
so any fallback triggered is a genuine detector-robustness signal, not an
artefact of feeding the detector garbage), run the SUBMITTED container over
each variant stack, and read the container's own FOV-fallback rate and
resulting AUROC (vs the same 200 images' known labels).

Variants (each applied to the RAW loaded RGB image, before any container-side
processing -- the container's own fov.py runs on these pixels exactly as it
would on real input):
  a. letterbox_16x9   -- pillarboxed into a 16:9 canvas, black bars L/R
  b1. fov_radius_0.7x -- content shrunk 0.7x, more black border
  b2. fov_radius_1.3x -- content enlarged 1.3x, clipped at edges
  c. off_centre       -- content shifted 20% frame-width toward one corner
  d. square_no_mask   -- cropped tight to the manifest's own inscribed
                         square; NO black background at all
  e1. rescale_1920x1080
  e2. rescale_720x576  -- both distort aspect ratio (original is ~square)
  f1. aspect_4x3       -- centre-cropped to 4:3 (removes content, no padding)
  f2. aspect_16x9      -- centre-cropped to 16:9 (removes more)
  baseline             -- unmodified, control

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/73_d3_fov_sweep.py'
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile
from tqdm.auto import tqdm

REPO_ROOT = "/workspace/RARE26"
IMAGE_ROOT = os.path.join(REPO_ROOT, "00_source")
MANIFEST = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
INPUT_DIRNAME = "stacked-barretts-esophagus-endoscopy"
OUT_DIR = os.path.join(REPO_ROOT, "runs/submission_test/d3_fov")
N_IMAGES = 200
SEED = 20260810303


def load_rgb(rel: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(os.path.join(IMAGE_ROOT, rel)), cv2.COLOR_BGR2RGB)


def letterbox_16x9(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = int(round(h * 16 / 9))
    canvas = np.zeros((h, new_w, 3), dtype=img.dtype)
    x0 = (new_w - w) // 2
    canvas[:, x0:x0 + w] = img
    return canvas


def scale_content(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = int(round(h * factor)), int(round(w * factor))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros_like(img)
    if factor <= 1.0:
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
    else:
        y0, x0 = (nh - h) // 2, (nw - w) // 2
        canvas = resized[y0:y0 + h, x0:x0 + w]
    return canvas


def off_centre(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    dx, dy = int(0.20 * w), int(0.20 * h)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))


def square_no_mask(img: np.ndarray, il: int, it: int, ir: int, ib: int) -> np.ndarray:
    il, it, ir, ib = max(0, il), max(0, it), min(img.shape[1], ir), min(img.shape[0], ib)
    return img[it:ib, il:ir]


def rescale(img: np.ndarray, size) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def centre_crop_aspect(img: np.ndarray, aspect: float) -> np.ndarray:
    """aspect = width/height. Crops (never pads) to hit it exactly."""
    h, w = img.shape[:2]
    cur = w / h
    if cur > aspect:
        new_w = int(round(h * aspect))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    else:
        new_h = int(round(w / aspect))
        y0 = (h - new_h) // 2
        return img[y0:y0 + new_h, :]


def write_stack(images: list, out_dir: str) -> None:
    img_dir = os.path.join(out_dir, "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, "stack.tif")
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for im in images:
            tw.write(im, photometric="rgb", compression=None, contiguous=False)
    from make_test_stack import write_inputs_json  # noqa
    write_inputs_json(out_dir)


def main() -> int:
    sys.path.insert(0, os.path.join(REPO_ROOT, "submission", "tools"))

    df = pd.read_csv(MANIFEST)
    val = df[(df["fold_r0"] == 0) & df["inner_left"].notna()].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    sel = val.sample(n=N_IMAGES, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    print(f"[d3] loading {N_IMAGES} raw images...")
    raw = {}
    for _, row in tqdm(sel.iterrows(), total=len(sel), desc="load", file=sys.stderr):
        raw[row["filepath"]] = load_rgb(row["filepath"])

    variants = {
        "baseline": lambda fp, im: im,
        "a_letterbox_16x9": lambda fp, im: letterbox_16x9(im),
        "b1_fov_radius_0.7x": lambda fp, im: scale_content(im, 0.7),
        "b2_fov_radius_1.3x": lambda fp, im: scale_content(im, 1.3),
        "c_off_centre": lambda fp, im: off_centre(im),
        "d_square_no_mask": lambda fp, im: square_no_mask(
            im, int(sel.loc[sel.filepath == fp, "inner_left"].iloc[0]),
            int(sel.loc[sel.filepath == fp, "inner_top"].iloc[0]),
            int(sel.loc[sel.filepath == fp, "inner_right"].iloc[0]),
            int(sel.loc[sel.filepath == fp, "inner_bottom"].iloc[0])),
        "e1_rescale_1920x1080": lambda fp, im: rescale(im, (1920, 1080)),
        "e2_rescale_720x576": lambda fp, im: rescale(im, (720, 576)),
        "f1_aspect_4x3": lambda fp, im: centre_crop_aspect(im, 4 / 3),
        "f2_aspect_16x9": lambda fp, im: centre_crop_aspect(im, 16 / 9),
    }

    gt_path = os.path.join(OUT_DIR, "ground_truth.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    sel[["filepath", "class_label"]].assign(position=range(len(sel))).to_csv(gt_path, index=False)

    for name, fn in variants.items():
        vdir = os.path.join(OUT_DIR, name, "interface_0")
        images = [fn(fp, raw[fp]) for fp in sel["filepath"]]
        write_stack(images, vdir)
        print(f"[d3] variant '{name}': {len(images)} images written, "
              f"e.g. shape {images[0].shape}")

    print(f"[d3] ground truth: {gt_path}")
    print(f"[d3] variants: {list(variants.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
