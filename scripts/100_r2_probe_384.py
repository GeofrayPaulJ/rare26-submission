"""R2 -- 384-image probe, the REAL case size per the organisers' 13
August forum answer (600s/batch of 384 images/.tiff file; each .tiff is
one case, one job). Same construction convention as every prior probe
in this project (P1, Probe C, V1): resize to 637x512 modal, RGB uint8
vector .mha, ALL available positives (maximises local evaluation power,
same stated rationale as Probe C/V1 -- this is NOT meant to mimic real
class prevalence, only real CASE SIZE) plus negatives stratified by
centre, non-alphabetical order, known labels for a local reference.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/100_r2_probe_384.py'
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm

REPO_ROOT = "/workspace/RARE26"
MANIFEST = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
IMAGE_ROOT = os.path.join(REPO_ROOT, "00_source")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "submission_test", "r2_probe_384")

INPUT_SLUG = "stacked-barretts-esophagus-endoscopy-images"
INPUT_DIRNAME = "stacked-barretts-esophagus-endoscopy"
MODAL_W, MODAL_H = 637, 512

N_TOTAL = 384
SELECT_SEED = 20260813001
ORDER_SEED = 8130226001


def write_inputs_json(interface_dir: str) -> None:
    payload = [{"interface": {"slug": INPUT_SLUG, "kind": "Image",
                              "relative_path": f"images/{INPUT_DIRNAME}"}}]
    os.makedirs(interface_dir, exist_ok=True)
    with open(os.path.join(interface_dir, "inputs.json"), "w") as fh:
        json.dump(payload, fh, indent=4)


def load_resize(fp: str) -> np.ndarray:
    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(IMAGE_ROOT, fp)))
    return cv2.resize(img, (MODAL_W, MODAL_H), interpolation=cv2.INTER_AREA)


def main() -> int:
    df = pd.read_csv(MANIFEST)
    pos = df[df["class_label"] == "neoplasia"]
    neg = df[df["class_label"] != "neoplasia"]
    n_neg_needed = N_TOTAL - len(pos)
    assert n_neg_needed > 0, f"N_TOTAL={N_TOTAL} too small for {len(pos)} positives"

    rng = np.random.default_rng(SELECT_SEED)
    neg_by_centre = neg["centre"].value_counts(normalize=True)
    quota, remaining = {}, n_neg_needed
    centres = sorted(neg_by_centre.index)
    for i, c in enumerate(centres):
        if i == len(centres) - 1:
            quota[c] = remaining
        else:
            q = max(1, round(n_neg_needed * neg_by_centre[c]))
            quota[c] = q
            remaining -= q

    picked_neg = []
    for c, q in quota.items():
        cell = neg[neg["centre"] == c]
        n = min(q, len(cell))
        picked_neg.append(cell.sample(n=n, random_state=rng.integers(0, 2**31 - 1)))
    picked = pd.concat([pos] + picked_neg, ignore_index=True)
    if len(picked) < N_TOTAL:
        used = set(picked["filepath"])
        pool = df[~df["filepath"].isin(used)]
        extra = pool.sample(n=N_TOTAL - len(picked), random_state=rng.integers(0, 2**31 - 1))
        picked = pd.concat([picked, extra], ignore_index=True)

    order_rng = np.random.default_rng(ORDER_SEED)
    perm = order_rng.permutation(len(picked))
    ordered = picked.iloc[perm].reset_index(drop=True)
    alpha = picked.sort_values("filepath")["filepath"].tolist()
    assert ordered["filepath"].tolist() != alpha, "must not equal alphabetical order"

    n_pos = int((ordered["class_label"] == "neoplasia").sum())
    n_by_centre = ordered["centre"].value_counts().to_dict()
    print(f"[r2] selected {len(ordered)} images, {n_pos} positive, "
          f"by centre: {n_by_centre}", flush=True)
    assert len(ordered) == N_TOTAL
    assert n_pos == len(pos), "expected ALL available positives"
    assert ordered["centre"].nunique() >= 2, "expected both centres represented"

    frames = [load_resize(fp) for fp in tqdm(
        ordered["filepath"], desc="R2 load+resize", unit="img", file=sys.stderr)]

    case_dir = os.path.join(OUT_DIR, "interface_0")
    img_dir = os.path.join(case_dir, "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    mha_path = os.path.join(img_dir, "stack.mha")
    stack_arr = np.stack(frames, axis=0)
    img = sitk.GetImageFromArray(stack_arr, isVector=True)
    sitk.WriteImage(img, mha_path)
    write_inputs_json(case_dir)
    print(f"[r2] wrote {mha_path} ({os.path.getsize(mha_path) / 2**20:.1f} MiB, "
          f"shape={stack_arr.shape})")

    # .tiff variant too -- R4 needs both for the .mha vs .tiff parity check.
    import tifffile
    tiff_dir = os.path.join(OUT_DIR, "interface_0_tiff", "images", INPUT_DIRNAME)
    os.makedirs(tiff_dir, exist_ok=True)
    tiff_path = os.path.join(tiff_dir, "stack.tif")
    tifffile.imwrite(tiff_path, stack_arr, bigtiff=True)
    write_inputs_json(os.path.join(OUT_DIR, "interface_0_tiff"))
    print(f"[r2] wrote {tiff_path} ({os.path.getsize(tiff_path) / 2**20:.1f} MiB)")

    manifest = ordered[["filepath", "centre", "class_label"]].copy()
    manifest.insert(0, "slice_index", range(len(manifest)))
    manifest["label_int"] = (manifest["class_label"] == "neoplasia").astype(int)
    out_manifest = os.path.join(REPO_ROOT, "reports", "r2_probe_384_manifest.csv")
    manifest.to_csv(out_manifest, index=False)
    print(f"[r2] manifest -> {out_manifest} ({len(manifest)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
