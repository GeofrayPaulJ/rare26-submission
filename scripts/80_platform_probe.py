"""P1 -- build the platform probe: 200 known-label RARE25 images, both
centres, >=30 positives, in a DELIBERATELY non-alphabetical slice order,
emitted as both .mha (GC's real ingestion format) and .tiff (what the
try-out page's own docs say inputs are: multi-page .tiff -- GC converts
on ingestion, so both are handed over rather than assuming which one the
try-out path actually exercises).

This is the reference build for the platform probe described in the
2026-08-11 instruction: run the CURRENTLY SUBMITTED container locally
over these same 200 images and record the scores, so a later platform
try-out run of the same file can be compared position-for-position
against a known-good local baseline.

Stratification: sample from every (centre, class_label) cell present in
the manifest, proportional to the smaller classes but with a floor that
guarantees >=30 positives (`neoplasia`) and >=1 image per centre per
class present. Order is then shuffled with a fixed seed distinct from
selection -- this is what "deliberately non-alphabetical" means here:
not merely unsorted, but explicitly permuted so a container that (bug)
silently re-sorted by filename before scoring would produce a
detectably wrong slice-to-score mapping.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/80_platform_probe.py'
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tifffile
from tqdm.auto import tqdm

REPO_ROOT = "/workspace/RARE26"
MANIFEST = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
IMAGE_ROOT = os.path.join(REPO_ROOT, "00_source")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "submission_test", "platform_probe")
MANIFEST_OUT = os.path.join(REPO_ROOT, "reports", "probe_manifest.csv")

INPUT_SLUG = "stacked-barretts-esophagus-endoscopy-images"
INPUT_DIRNAME = "stacked-barretts-esophagus-endoscopy"

N_TOTAL = 200
MIN_POSITIVES = 30
SELECT_SEED = 20260811   # which 200 images
ORDER_SEED = 8110226     # the non-alphabetical permutation, independent of selection
MODAL_W, MODAL_H = 637, 512


def stratified_select(df: pd.DataFrame) -> pd.DataFrame:
    """Pick N_TOTAL rows, >=MIN_POSITIVES positives, both centres represented
    in both classes where available, reproducibly."""
    rng = np.random.default_rng(SELECT_SEED)
    cells = df.groupby(["centre", "class_label"])

    # Positive floor: split MIN_POSITIVES across the two centres in
    # proportion to how many positives each centre actually has, so the
    # probe doesn't invent a class balance the real data doesn't have.
    pos = df[df["class_label"] == "neoplasia"]
    pos_by_centre = pos["centre"].value_counts(normalize=True)
    pos_quota = {}
    remaining = MIN_POSITIVES
    centres = sorted(pos_by_centre.index)
    for i, c in enumerate(centres):
        if i == len(centres) - 1:
            pos_quota[c] = remaining
        else:
            q = max(1, round(MIN_POSITIVES * pos_by_centre[c]))
            pos_quota[c] = q
            remaining -= q

    picked = []
    for c, q in pos_quota.items():
        cell = pos[pos["centre"] == c]
        n = min(q, len(cell))
        picked.append(cell.sample(n=n, random_state=rng.integers(0, 2**31 - 1)))
    n_pos_picked = sum(len(p) for p in picked)

    # Fill the remainder with negatives, proportional to each centre's share
    # of the negative pool, so both centres stay represented.
    n_remaining = N_TOTAL - n_pos_picked
    neg = df[df["class_label"] != "neoplasia"]
    neg_by_centre = neg["centre"].value_counts(normalize=True)
    neg_quota = {}
    remaining = n_remaining
    for i, c in enumerate(sorted(neg_by_centre.index)):
        if i == len(neg_by_centre) - 1:
            neg_quota[c] = remaining
        else:
            q = max(1, round(n_remaining * neg_by_centre[c]))
            neg_quota[c] = q
            remaining -= q

    for c, q in neg_quota.items():
        cell = neg[neg["centre"] == c]
        n = min(q, len(cell))
        picked.append(cell.sample(n=n, random_state=rng.integers(0, 2**31 - 1)))

    out = pd.concat(picked, ignore_index=True)
    if len(out) < N_TOTAL:
        # top up from whatever's left, uniformly, to hit exactly N_TOTAL
        used = set(out["filepath"])
        pool = df[~df["filepath"].isin(used)]
        extra = pool.sample(n=N_TOTAL - len(out), random_state=rng.integers(0, 2**31 - 1))
        out = pd.concat([out, extra], ignore_index=True)
    elif len(out) > N_TOTAL:
        out = out.sample(n=N_TOTAL, random_state=rng.integers(0, 2**31 - 1)).reset_index(drop=True)

    return out.reset_index(drop=True)


def non_alphabetical_order(df: pd.DataFrame) -> pd.DataFrame:
    """Permute rows so slice order != sorted(filepath). Verified, not assumed."""
    rng = np.random.default_rng(ORDER_SEED)
    perm = rng.permutation(len(df))
    shuffled = df.iloc[perm].reset_index(drop=True)
    alpha = df.sort_values("filepath").reset_index(drop=True)
    if (shuffled["filepath"].tolist() == alpha["filepath"].tolist()):
        # astronomically unlikely at n=200, but assert rather than assume
        shuffled = shuffled.iloc[::-1].reset_index(drop=True)
    return shuffled


def write_inputs_json(interface_dir: str) -> None:
    import json
    payload = [{
        "interface": {
            "slug": INPUT_SLUG,
            "kind": "Image",
            "relative_path": f"images/{INPUT_DIRNAME}",
        }
    }]
    os.makedirs(interface_dir, exist_ok=True)
    with open(os.path.join(interface_dir, "inputs.json"), "w") as fh:
        json.dump(payload, fh, indent=4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST)
    picked = stratified_select(df)
    ordered = non_alphabetical_order(picked)

    n_pos = int((ordered["class_label"] == "neoplasia").sum())
    n_by_centre = ordered["centre"].value_counts().to_dict()
    print(f"[probe] selected {len(ordered)} images, {n_pos} positive "
          f"(neoplasia), by centre: {n_by_centre}")
    assert len(ordered) == N_TOTAL, f"expected {N_TOTAL}, got {len(ordered)}"
    assert n_pos >= MIN_POSITIVES, f"expected >={MIN_POSITIVES} positives, got {n_pos}"
    assert ordered["centre"].nunique() >= 2, "expected both centres represented"
    alpha_order = ordered.sort_values("filepath")["filepath"].tolist()
    assert ordered["filepath"].tolist() != alpha_order, \
        "slice order must not equal alphabetical order"

    # --- load + resize every frame once, reused for both output formats ---
    frames = []
    for fp in tqdm(ordered["filepath"], desc="load+resize", unit="img", file=sys.stderr):
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(IMAGE_ROOT, fp)))
        img = cv2.resize(img, (MODAL_W, MODAL_H), interpolation=cv2.INTER_AREA)
        frames.append(img)
    stack_arr = np.stack(frames, axis=0)  # (Z, Y, X, 3) uint8
    print(f"[probe] numpy stack: shape={stack_arr.shape} dtype={stack_arr.dtype} "
          f"min={stack_arr.min()} max={stack_arr.max()} mean={stack_arr.mean():.3f}")

    # --- .mha: RGB uint8 vector image, GC's real ingestion format ---
    mha_dir = os.path.join(args.out_dir, "mha", "interface_0", "images", INPUT_DIRNAME)
    os.makedirs(mha_dir, exist_ok=True)
    mha_path = os.path.join(mha_dir, "stack.mha")
    sitk_img = sitk.GetImageFromArray(stack_arr, isVector=True)
    print(f"[probe] sitk image: size={sitk_img.GetSize()} "
          f"components={sitk_img.GetNumberOfComponentsPerPixel()} "
          f"pixel_type={sitk_img.GetPixelIDTypeAsString()}")
    sitk.WriteImage(sitk_img, mha_path)
    write_inputs_json(os.path.join(args.out_dir, "mha", "interface_0"))
    print(f"[probe] wrote {mha_path} ({os.path.getsize(mha_path) / 2**20:.1f} MiB)")

    # --- .tiff: multi-page, the format the try-out page's own docs name ---
    tiff_dir = os.path.join(args.out_dir, "tiff", "interface_0", "images", INPUT_DIRNAME)
    os.makedirs(tiff_dir, exist_ok=True)
    tiff_path = os.path.join(tiff_dir, "stack.tif")
    with tifffile.TiffWriter(tiff_path, bigtiff=True) as tw:
        for frame in tqdm(stack_arr, desc="write tiff", unit="page", file=sys.stderr):
            tw.write(frame, photometric="rgb", compression=None, contiguous=False)
    write_inputs_json(os.path.join(args.out_dir, "tiff", "interface_0"))
    print(f"[probe] wrote {tiff_path} ({os.path.getsize(tiff_path) / 2**20:.1f} MiB)")

    # --- manifest: exact filename-to-slice-index mapping, plus known labels ---
    os.makedirs(os.path.dirname(MANIFEST_OUT), exist_ok=True)
    manifest = ordered[["filepath", "centre", "class_label"]].copy()
    manifest.insert(0, "slice_index", range(len(manifest)))
    manifest["label_int"] = (manifest["class_label"] == "neoplasia").astype(int)
    manifest.to_csv(MANIFEST_OUT, index=False)
    print(f"[probe] manifest -> {MANIFEST_OUT} ({len(manifest)} rows)")

    print("\n[probe] FILE PATHS")
    print(f"  mha  : {mha_path}")
    print(f"  tiff : {tiff_path}")
    print(f"  manifest: {MANIFEST_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
