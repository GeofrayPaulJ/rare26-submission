"""P5 -- two more platform probes, free try-out jobs.

PROBE B -- the SAME 200 images and SAME order as `reports/probe_manifest.csv`
(scripts/80_platform_probe.py), split into 8 cases of 25 slices each
(contiguous chunks of the already-permuted 200-slice order -- the
permutation is already non-alphabetical, so chunking it preserves that
property within each case too). Tests cross-case ordering: each case's
JSON output can be reassembled against `reports/probe_manifest_b_cases.csv`
into the SAME global 200-slice order already scored once (P1's local
reference), so a platform vs local comparison is exactly as direct as P1's.

PROBE C -- 3,000 RARE25 images, single case, stratified: ALL 158
available positives (the entire manifest's neoplasia pool -- maximises
positive representation rather than sub-sampling it) plus 2,842
negatives drawn proportionally by centre, non-alphabetical order,
labels known. Tests scale (15x probe A/B), wall time, the 16 GB memory
ceiling, and whether FOV fallback/truncation behaviour changes above 200
images -- none of which the 200-image probes can show.

Both emitted as .mha only (RGB uint8 vector image) -- P1 already
established .mha is what the real try-out path exercises; .tiff is not
duplicated here to save build time, and can be added on request.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/84_platform_probe_p5.py'
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
PROBE_MANIFEST_A = os.path.join(REPO_ROOT, "reports", "probe_manifest.csv")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "submission_test", "platform_probe_p5")

INPUT_SLUG = "stacked-barretts-esophagus-endoscopy-images"
INPUT_DIRNAME = "stacked-barretts-esophagus-endoscopy"
MODAL_W, MODAL_H = 637, 512

N_CASE_B = 25
N_CASES_B = 8

N_TOTAL_C = 3000
SELECT_SEED_C = 20260811003
ORDER_SEED_C = 8110226003


def write_inputs_json(interface_dir: str) -> None:
    payload = [{"interface": {"slug": INPUT_SLUG, "kind": "Image",
                              "relative_path": f"images/{INPUT_DIRNAME}"}}]
    os.makedirs(interface_dir, exist_ok=True)
    with open(os.path.join(interface_dir, "inputs.json"), "w") as fh:
        json.dump(payload, fh, indent=4)


def load_resize(fp: str) -> np.ndarray:
    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(IMAGE_ROOT, fp)))
    return cv2.resize(img, (MODAL_W, MODAL_H), interpolation=cv2.INTER_AREA)


def write_mha(frames: list, out_path: str) -> None:
    stack_arr = np.stack(frames, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = sitk.GetImageFromArray(stack_arr, isVector=True)
    sitk.WriteImage(img, out_path)
    return stack_arr


# ---------------------------------------------------------------------------
# PROBE B -- 8 cases of 25 slices, from the EXISTING 200-image probe order
# ---------------------------------------------------------------------------
def build_probe_b() -> None:
    man = pd.read_csv(PROBE_MANIFEST_A).sort_values("slice_index").reset_index(drop=True)
    assert len(man) == 200, f"expected 200 rows in probe_manifest.csv, got {len(man)}"

    case_rows = []
    for case_id in range(N_CASES_B):
        lo, hi = case_id * N_CASE_B, (case_id + 1) * N_CASE_B
        chunk = man.iloc[lo:hi].reset_index(drop=True)
        frames = [load_resize(fp) for fp in tqdm(
            chunk["filepath"], desc=f"probe B case {case_id}", unit="img", file=sys.stderr)]

        case_dir = os.path.join(OUT_DIR, "b", f"case_{case_id}", "interface_0")
        img_dir = os.path.join(case_dir, "images", INPUT_DIRNAME)
        mha_path = os.path.join(img_dir, "stack.mha")
        write_mha(frames, mha_path)
        write_inputs_json(case_dir)

        for local_idx, row in chunk.iterrows():
            case_rows.append({
                "case_id": case_id, "local_slice_index": local_idx,
                "global_slice_index": row["slice_index"], "filepath": row["filepath"],
                "centre": row["centre"], "class_label": row["class_label"],
                "label_int": row["label_int"], "mha_path": mha_path,
            })
        print(f"[p5-b] case {case_id}: {len(chunk)} slices -> {mha_path}", flush=True)

    out_manifest = os.path.join(REPO_ROOT, "reports", "probe_manifest_b_cases.csv")
    pd.DataFrame(case_rows).to_csv(out_manifest, index=False)
    print(f"[p5-b] manifest -> {out_manifest} ({len(case_rows)} rows, "
         f"{N_CASES_B} cases x {N_CASE_B})")


# ---------------------------------------------------------------------------
# PROBE C -- 3,000 images, single case, ALL positives + stratified negatives
# ---------------------------------------------------------------------------
def build_probe_c() -> None:
    df = pd.read_csv(MANIFEST)
    pos = df[df["class_label"] == "neoplasia"]
    neg = df[df["class_label"] != "neoplasia"]
    n_neg_needed = N_TOTAL_C - len(pos)
    assert n_neg_needed > 0

    rng = np.random.default_rng(SELECT_SEED_C)
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
    if len(picked) < N_TOTAL_C:
        used = set(picked["filepath"])
        pool = df[~df["filepath"].isin(used)]
        extra = pool.sample(n=N_TOTAL_C - len(picked), random_state=rng.integers(0, 2**31 - 1))
        picked = pd.concat([picked, extra], ignore_index=True)

    order_rng = np.random.default_rng(ORDER_SEED_C)
    perm = order_rng.permutation(len(picked))
    ordered = picked.iloc[perm].reset_index(drop=True)
    alpha = picked.sort_values("filepath")["filepath"].tolist()
    assert ordered["filepath"].tolist() != alpha, "must not equal alphabetical order"

    n_pos = int((ordered["class_label"] == "neoplasia").sum())
    n_by_centre = ordered["centre"].value_counts().to_dict()
    print(f"[p5-c] selected {len(ordered)} images, {n_pos} positive, "
         f"by centre: {n_by_centre}", flush=True)
    assert len(ordered) == N_TOTAL_C
    assert n_pos == len(pos), "expected ALL available positives"

    frames = [load_resize(fp) for fp in tqdm(
        ordered["filepath"], desc="probe C load+resize", unit="img", file=sys.stderr)]

    case_dir = os.path.join(OUT_DIR, "c", "interface_0")
    img_dir = os.path.join(case_dir, "images", INPUT_DIRNAME)
    mha_path = os.path.join(img_dir, "stack.mha")
    stack_arr = write_mha(frames, mha_path)
    write_inputs_json(case_dir)
    print(f"[p5-c] wrote {mha_path} ({os.path.getsize(mha_path) / 2**20:.1f} MiB, "
         f"shape={stack_arr.shape})")

    manifest = ordered[["filepath", "centre", "class_label"]].copy()
    manifest.insert(0, "slice_index", range(len(manifest)))
    manifest["label_int"] = (manifest["class_label"] == "neoplasia").astype(int)
    out_manifest = os.path.join(REPO_ROOT, "reports", "probe_manifest_c.csv")
    manifest.to_csv(out_manifest, index=False)
    print(f"[p5-c] manifest -> {out_manifest} ({len(manifest)} rows)")


def main() -> int:
    print("=== PROBE B: 8 cases x 25 slices ===", flush=True)
    build_probe_b()
    print("\n=== PROBE C: 3,000 images, single case ===", flush=True)
    build_probe_c()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
