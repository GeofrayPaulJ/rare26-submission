"""H2 -- build the three input stacks for the aspect-ratio reproduction test.

Fold 0's 617 held-out images (the exact set r0_f0_s0 scores 0.9358/0.1451
on) are already staged as a native-resolution TIFF at
runs/submission_test/parity/interface_0/images/.../stack.tif, built by
`make_test_stack.py parity` (repeat=0, fold=0 are its defaults). That IS
variant A -- reused, not rebuilt.

CORRECTION, flagged before any number is trusted (same pattern as L1's
pHash-threshold catch): the instruction asks for three ".mha stacks", but
a .mha volume requires every slice to share one (H, W) -- it is a true
N-D array, not a page list. Fold 0's 617 images span 40 distinct native
resolutions (292x263 .. 655x512), so a single ragged-shape .mha cannot
represent "as-is" at all. Variant A therefore stays a TIFF (tifffile
supports independent per-page shapes; `stack.py::open_stack` already reads
both formats through the identical `to_rgb_uint8 -> preprocess` path, so
the file container format is not part of what this test is measuring).
Variants B and C are both forced to a uniform 512x512, so THEY are built
as real .mha via SimpleITK, matching the instruction literally where the
format constraint allows it.

  A. as-is            -- reused TIFF, native per-image resolution, no change
  B. squash-to-square  -- cv2.resize each image directly to 512x512,
                          ignoring aspect ratio (distorts)
  C. centre-crop-square -- crop to a centred min(h,w) square first, THEN
                          resize that square to 512x512 (no distortion,
                          but discards the sides of a wide frame)

Also writes ground_truth.csv (label per position, aligned to A's existing
stack_order.csv) once, shared by all three variants since they are the
same 617 images in the same order.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/103_h2_build_variants.py'
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = "/workspace/RARE26"
MANIFEST = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
IMAGE_ROOT = os.path.join(REPO_ROOT, "00_source")
A_ORDER = os.path.join(REPO_ROOT, "runs/submission_test/parity/interface_0/stack_order.csv")
OUT_ROOT = os.path.join(REPO_ROOT, "runs/submission_test/h2_aspect_ratio")
INPUT_DIRNAME = "stacked-barretts-esophagus-endoscopy"
TARGET = 512


def _load_rgb(rel: str) -> np.ndarray:
    return np.asarray(Image.open(os.path.join(IMAGE_ROOT, rel)).convert("RGB"))


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    interp = cv2.INTER_AREA if (h <= ih and w <= iw) else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def squash_to_square(img: np.ndarray) -> np.ndarray:
    """B: force to 512x512 directly -- ignores aspect ratio."""
    return _resize(img, TARGET, TARGET)


def centre_crop_square(img: np.ndarray) -> np.ndarray:
    """C: centre square crop at native scale, THEN resize to 512x512."""
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = img[top:top + side, left:left + side]
    return _resize(cropped, TARGET, TARGET)


def write_inputs_json(interface_dir: str) -> None:
    import json
    payload = [{
        "interface": {
            "slug": "stacked-barretts-esophagus-endoscopy-images",
            "kind": "Image",
            "relative_path": f"images/{INPUT_DIRNAME}",
        }
    }]
    with open(os.path.join(interface_dir, "inputs.json"), "w") as fh:
        json.dump(payload, fh, indent=4)


def build_mha_variant(name: str, transform) -> None:
    order = pd.read_csv(A_ORDER)
    fps = order["filepath"].tolist()
    print(f"[h2] building variant {name}: {len(fps)} images -> {TARGET}x{TARGET} uniform .mha")

    frames = np.empty((len(fps), TARGET, TARGET, 3), dtype=np.uint8)
    for i, fp in enumerate(tqdm(fps, desc=name, unit="img", file=sys.stderr)):
        img = _load_rgb(fp)
        frames[i] = transform(img)

    out_dir = os.path.join(OUT_ROOT, name, "interface_0")
    img_dir = os.path.join(out_dir, "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    mha_path = os.path.join(img_dir, "stack.mha")

    sitk_img = sitk.GetImageFromArray(frames, isVector=True)
    sitk.WriteImage(sitk_img, mha_path)
    write_inputs_json(out_dir)
    order.to_csv(os.path.join(out_dir, "stack_order.csv"), index=False)
    print(f"[h2] wrote {mha_path} ({os.path.getsize(mha_path) / 2**20:.0f} MiB) "
          f"size={sitk_img.GetSize()} ncomp={sitk_img.GetNumberOfComponentsPerPixel()}")


def write_ground_truth() -> None:
    order = pd.read_csv(A_ORDER)
    manifest = pd.read_csv(MANIFEST)
    merged = order.merge(manifest[["filepath", "class_label"]], on="filepath", how="left")
    if merged["class_label"].isna().any():
        missing = merged[merged["class_label"].isna()]
        raise SystemExit(f"{len(missing)} stack images not found in {MANIFEST}")
    merged = merged.rename(columns={"index": "position"})
    gt_path = os.path.join(REPO_ROOT, "reports/h2_ground_truth.csv")
    merged[["filepath", "class_label", "position"]].to_csv(gt_path, index=False)
    n_pos = int((merged["class_label"] != "non-dysplastic").sum())
    print(f"[h2] wrote {gt_path}: {len(merged)} rows, {n_pos} positive (neoplasia)")


def main() -> int:
    os.makedirs(os.path.join(REPO_ROOT, "reports"), exist_ok=True)
    write_ground_truth()
    build_mha_variant("B_squash_square", squash_to_square)
    build_mha_variant("C_centre_crop_square", centre_crop_square)
    print("\n[h2] A (as-is) reuses runs/submission_test/parity/interface_0 -- not rebuilt.")
    print("[h2] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
