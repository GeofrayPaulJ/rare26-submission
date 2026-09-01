"""STEP 7 (2026-08-07) -- EVC lesion paste pipeline. CPU ONLY, NO TRAINING.

Extracts lesion regions from the EVC expert annotations (5 experts per
image; consensus = pixels marked by >= 3 of 5) on the 50 cancer-class EVC
images, and seamless-clones them onto RARE25 NEGATIVE frames:

  - COLOUR MATCH: the lesion patch's R channel is scaled so its rg_ratio
    (mean R / mean G, the manifest's own colour statistic) matches the
    destination frame's manifest rg_ratio before cloning -- EVC and RARE25
    scopes have different colour pipelines, and an unmatched paste reads as
    a sticker, not tissue.
  - INSCRIBED SQUARE RESPECTED: the paste target is sampled uniformly such
    that the whole lesion bounding box lands inside the destination's
    [inner_left..inner_right] x [inner_top..inner_bottom] inscribed square
    (precomputed in manifests/rare25_folds_v2.csv). A paste in the discarded
    annulus is wasted compute -- the training crop would throw it away.
  - EVC IMAGES NEVER ENTER TRAINING AS IMAGES. Only the mask-selected
    lesion pixels travel; no whole EVC frame is written anywhere a training
    manifest could pick it up. Outputs land in runs/evc_paste_preview/
    (preview PNGs + a JSON manifest of what went where) and
    reports/evc_paste_grid.png (40-tile grid) for HUMAN review at 09:00.
    Nothing is trained on tonight, per the brief.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/52_evc_paste.py'
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVC_ROOT = os.path.join(REPO_ROOT, "02_evc")
RARE_ROOT = os.path.join(REPO_ROOT, "00_source")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "evc_paste_preview")
GRID_PNG = os.path.join(REPO_ROOT, "reports", "evc_paste_grid.png")

CONSENSUS_MIN_EXPERTS = 3          # of 5 -- majority, not union/intersection
N_EXPERTS = 5
N_TILES = 40                       # human-review grid, 8 x 5
TILE = 320                         # px per grid tile
SCALE_RANGE = (0.20, 0.45)         # lesion max-dim as a fraction of inner square
SEED = 20260807
MIN_LESION_PX = 2000               # skip consensus masks too small to read


def consensus_mask(stem: str) -> np.ndarray | None:
    """>=3-of-5 expert consensus, uint8 {0,255}. None if any mask missing."""
    votes = None
    for e in range(1, N_EXPERTS + 1):
        p = os.path.join(EVC_ROOT, "annotations_bmp", f"{stem}_exp{e}.bmp")
        if not os.path.exists(p):
            return None
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        v = (m > 127).astype(np.uint8)
        votes = v if votes is None else votes + v
    return ((votes >= CONSENSUS_MIN_EXPERTS) * 255).astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """One lesion per paste: keep the largest connected component only."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == biggest) * 255).astype(np.uint8)


def rg_match(patch_bgr: np.ndarray, mask: np.ndarray, target_rg: float) -> np.ndarray:
    """Scale the patch's R channel so mean R / mean G over the lesion pixels
    equals the destination's manifest rg_ratio."""
    sel = mask > 0
    if not sel.any():
        return patch_bgr
    mean_g = float(patch_bgr[..., 1][sel].mean())
    mean_r = float(patch_bgr[..., 2][sel].mean())
    if mean_g < 1e-6 or mean_r < 1e-6 or not np.isfinite(target_rg):
        return patch_bgr
    scale = target_rg / (mean_r / mean_g)
    out = patch_bgr.astype(np.float32)
    out[..., 2] *= scale
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()

    inv = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "evc_inventory.csv"))
    cancer = inv[inv["class_label"] == "cancer"].reset_index(drop=True)

    rare = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))
    neg = rare[(rare["class_label"] == "non-neoplastic")
               & (rare["keep_for_training"] == True)  # noqa: E712
               & rare["inner_left"].notna()].reset_index(drop=True)
    if len(neg) == 0:
        # class label spelling differs between manifests; fall back explicitly
        labels = rare["class_label"].unique().tolist()
        neg_label = [l for l in labels if "non" in l.lower()][0]
        neg = rare[(rare["class_label"] == neg_label)
                   & rare["inner_left"].notna()].reset_index(drop=True)
    print(f"[evc-paste] {len(cancer)} cancer EVC images, "
          f"{len(neg)} RARE25 negative destinations")

    # --- extract lesions (consensus, largest component, bbox crop) ---
    lesions = []
    for _, row in tqdm(cancer.iterrows(), total=len(cancer),
                       desc="extract lesions", unit="img", file=sys.stderr):
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
        mask = consensus_mask(stem)
        if mask is None:
            continue
        mask = largest_component(mask)
        if int((mask > 0).sum()) < MIN_LESION_PX:
            continue
        img = cv2.imread(os.path.join(EVC_ROOT, row["filepath"]))
        if img is None or img.shape[:2] != mask.shape[:2]:
            continue
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        lesions.append({"stem": stem,
                        "patch": img[y0:y1, x0:x1],
                        "mask": mask[y0:y1, x0:x1]})
    print(f"[evc-paste] {len(lesions)} usable consensus lesions "
          f"(>= {CONSENSUS_MIN_EXPERTS}/5 experts, >= {MIN_LESION_PX}px)")
    if not lesions:
        print("[evc-paste] HALT: no usable lesions extracted")
        return 2

    # --- paste ---
    os.makedirs(OUT_DIR, exist_ok=True)
    dest_rows = neg.sample(n=N_TILES, random_state=SEED).reset_index(drop=True)
    records = []
    tiles = []
    for i, drow in enumerate(tqdm(dest_rows.itertuples(), total=N_TILES,
                                  desc="seamless-clone pastes", unit="paste",
                                  file=sys.stderr)):
        les = lesions[int(rng.integers(len(lesions)))]
        dst = cv2.imread(os.path.join(RARE_ROOT, drow.filepath))
        if dst is None:
            continue

        il, it = int(drow.inner_left), int(drow.inner_top)
        ir, ib = int(drow.inner_right), int(drow.inner_bottom)
        inner_w, inner_h = ir - il, ib - it
        if inner_w < 64 or inner_h < 64:
            continue

        # scale lesion so its max dimension is a fraction of the inner square
        frac = float(rng.uniform(*SCALE_RANGE))
        target_max = frac * min(inner_w, inner_h)
        ph, pw = les["patch"].shape[:2]
        s = target_max / max(ph, pw)
        nw, nh = max(8, int(pw * s)), max(8, int(ph * s))
        patch = cv2.resize(les["patch"], (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(les["mask"], (nw, nh), interpolation=cv2.INTER_NEAREST)

        patch = rg_match(patch, mask, float(drow.rg_ratio))

        # uniform placement with the WHOLE patch bbox inside the inner square
        if inner_w - nw < 2 or inner_h - nh < 2:
            continue
        cx = il + nw // 2 + int(rng.integers(0, inner_w - nw))
        cy = it + nh // 2 + int(rng.integers(0, inner_h - nh))

        try:
            out = cv2.seamlessClone(patch, dst, mask, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error as exc:
            print(f"[evc-paste] clone failed on {drow.filepath}: {exc}",
                  file=sys.stderr)
            continue

        name = f"paste_{i:02d}_{les['stem']}.png"
        cv2.imwrite(os.path.join(OUT_DIR, name), out)
        records.append({"tile": i, "output": name, "lesion": les["stem"],
                        "dest": drow.filepath, "dest_centre": str(drow.centre),
                        "scale_frac": round(frac, 3),
                        "centre_xy": [int(cx), int(cy)],
                        "dest_rg_ratio": float(drow.rg_ratio)})
        tiles.append(cv2.resize(out, (TILE, TILE), interpolation=cv2.INTER_AREA))

    if not tiles:
        print("[evc-paste] HALT: zero successful pastes")
        return 2

    # --- 40-tile grid (8 x 5) for human review ---
    cols, rows = 8, 5
    while len(tiles) < cols * rows:
        tiles.append(np.zeros((TILE, TILE, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols])
                      for r in range(rows)])
    os.makedirs(os.path.dirname(GRID_PNG), exist_ok=True)
    cv2.imwrite(GRID_PNG, grid)

    with open(os.path.join(OUT_DIR, "paste_manifest.json"), "w") as fh:
        json.dump({"seed": SEED, "consensus_min_experts": CONSENSUS_MIN_EXPERTS,
                   "n_lesions_usable": len(lesions), "pastes": records}, fh,
                  indent=2)

    print(f"[evc-paste] {len(records)} pastes written to {OUT_DIR}")
    print(f"[evc-paste] grid: {GRID_PNG} ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
