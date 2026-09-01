"""
11_evc_overlap.py -- Check whether EVC images overlap the RARE25 training set.

WHY THIS MATTERS: team Jmees-inc reportedly used EVC data in RARE25 and could not
definitively identify overlap with the challenge images. This script identifies it
definitively for the RARE25 *training* set (the only thing we have access to here).

CRITICAL METHOD CONSTRAINT: this reuses the exact same code that produced the
RARE25 hashes, rather than re-deriving a "similar" comparison, so distances are
directly comparable:
  - fit_fov_circle() and inner_square() are imported from 05_fov_crop.py (the
    centroid + area-derived-radius circle fit, validated on this data as more
    robust than cv2.minEnclosingCircle).
  - hash_one() is imported from 06_regroup.py -- the same in-memory
    inner-square-crop-then-imagehash.phash(hash_size=8) call used to build
    rare25_folds.csv. Its module-level SRC constant is pointed at 02_evc/ for
    the duration of this run (and restored after), so the exact same function
    object does the cropping/hashing for both datasets; nothing is
    reimplemented.
  - popcount64() (Hamming distance via SWAR bit-trick) is imported from
    03_near_duplicates.py, the same routine used for the RARE25 near-duplicate
    clustering.

For the RARE25 side, the sha256 and pHash values are read directly out of
manifests/rare25_folds.csv rather than recomputed -- they are already the
product of the identical pipeline, so reusing them is more faithful than
rederiving them (and guarantees rare25_folds.csv is only ever read, never
written).

EVC images are PNG (lossless), confirmed by 10_evc_inventory.py -- not JPEG, so
no extra pHash drift from lossy re-compression is expected here.

Read-only: 00_source and manifests/rare25_folds.csv are never modified; EVC
images are read from 02_evc/ (already extracted by 10_evc_inventory.py, never
re-written here). Deterministic and idempotent.

USAGE:
    python 11_evc_overlap.py
    python 11_evc_overlap.py --pair-threshold 8
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
MANIFESTS = ROOT / "manifests"
EVC_DIR = ROOT / "02_evc"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

fov05 = importlib.import_module("05_fov_crop")
regroup06 = importlib.import_module("06_regroup")
nd03 = importlib.import_module("03_near_duplicates")

SWEEP_THRESHOLDS = [2, 4, 6, 8, 10, 12]


def hash_evc_images(evc_df: pd.DataFrame, fov_threshold: int = 15) -> pd.DataFrame:
    """Computes FOV geometry + inner-square-cropped pHash for every EVC image,
    using the identical fit_fov_circle/inner_square (05) and hash_one (06)
    functions that produced the RARE25 hashes. Single-process (100 images is
    fast, and it keeps the SRC monkeypatch below safe -- ProcessPoolExecutor
    workers on Windows re-import modules fresh and would not see it)."""
    original_src = regroup06.SRC
    regroup06.SRC = EVC_DIR
    rows = []
    try:
        for _, row in evc_df.iterrows():
            rel_path = row["filepath"]
            img_path = EVC_DIR / rel_path
            with Image.open(img_path) as img:
                arr = np.asarray(img.convert("RGB"))
            h, w = arr.shape[:2]
            mask = arr.mean(axis=2) > fov_threshold

            fit = fov05.fit_fov_circle(mask)
            if fit is None:
                rows.append({"filepath": rel_path, "phash": None, "fov_error": "no FOV detected"})
                continue
            cx, cy, radius, fit_quality, area_pct = fit
            box = fov05.inner_square(cx, cy, radius, w, h)

            _, phash_hex, err = regroup06.hash_one((rel_path, box))
            rows.append({
                "filepath": rel_path,
                "fov_radius": round(radius, 2),
                "fit_quality": round(fit_quality, 4),
                "phash": phash_hex,
                "fov_error": err,
            })
    finally:
        regroup06.SRC = original_src

    return pd.DataFrame(rows)


def hamming_matrix(hex_a: list[str], hex_b: list[str]) -> np.ndarray:
    ints_a = np.array([int(h, 16) for h in hex_a], dtype=np.uint64)
    ints_b = np.array([int(h, 16) for h in hex_b], dtype=np.uint64)
    return nd03.popcount64(ints_a[:, None] ^ ints_b[None, :])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-threshold", type=int, default=8,
                     help="Max Hamming distance for the saved pair list (default 8)")
    ap.add_argument("--fov-threshold", type=int, default=15)
    args = ap.parse_args()

    rare_path = MANIFESTS / "rare25_folds.csv"
    evc_path = MANIFESTS / "evc_inventory.csv"
    if not rare_path.exists():
        raise SystemExit(f"Missing {rare_path}. Run the RARE25 pipeline (00-07) first.")
    if not evc_path.exists():
        raise SystemExit(f"Missing {evc_path}. Run 10_evc_inventory.py first.")

    rare_df = pd.read_csv(rare_path)  # read-only, never written back to
    evc_inv = pd.read_csv(evc_path)

    print(f"RARE25 training images: {len(rare_df)}   EVC images: {len(evc_inv)}")
    print("EVC image format (per 10_evc_inventory.py): all PNG (lossless) -- "
          "no lossy-recompression pHash drift expected versus RARE25's PNGs.")

    print("\nComputing EVC FOV geometry + inner-square-cropped pHash "
          "(reusing 05_fov_crop.fit_fov_circle/inner_square and 06_regroup.hash_one)...")
    evc_hashed = hash_evc_images(evc_inv, fov_threshold=args.fov_threshold)
    evc_df = evc_inv.merge(evc_hashed, on="filepath", how="left")

    n_fov_fail = int((evc_df["phash"].isna()).sum())
    if n_fov_fail:
        print(f"WARNING: {n_fov_fail} EVC image(s) failed FOV/hash computation:")
        for _, row in evc_df[evc_df["phash"].isna()].iterrows():
            print(f"  {row['filepath']}: {row.get('fov_error', 'unknown error')}")
    evc_df = evc_df[evc_df["phash"].notna()].reset_index(drop=True)

    # --- Exact sha256 matches ---
    print("\n=== Exact duplicate check (sha256) ===")
    rare_hash_map = rare_df.groupby("sha256_hash")[["filepath", "centre", "class_label"]].apply(
        lambda g: g.to_dict("records")
    )
    exact_hits = []
    for _, row in evc_df.iterrows():
        matches = rare_hash_map.get(row["sha256_hash"])
        if matches:
            for m in matches:
                exact_hits.append({
                    "evc_filepath": row["filepath"], "evc_class": row["class_label"],
                    "rare25_filepath": m["filepath"], "rare25_centre": m["centre"],
                    "rare25_class": m["class_label"],
                })
    if exact_hits:
        print(f"CRITICAL: {len(exact_hits)} exact sha256 match(es) -- literal duplicate files:")
        for h in exact_hits:
            print(f"  {h['evc_filepath']} [{h['evc_class']}]  ==  "
                  f"{h['rare25_filepath']} [{h['rare25_centre']}/{h['rare25_class']}]")
    else:
        print("No exact sha256 matches between EVC and RARE25 training images.")

    # --- pHash sweep ---
    print("\n=== pHash near-duplicate sweep ===")
    dist = hamming_matrix(evc_df["phash"].tolist(), rare_df["phash"].tolist())  # (n_evc, n_rare25)
    evc_class = evc_df["class_label"].to_numpy()
    rare_class = rare_df["class_label"].to_numpy()

    print(f"{'thresh':>7} {'evc matched':>12} {'rare25 matched':>15} {'class pairs (evc->rare25: count)'}")
    for t in SWEEP_THRESHOLDS:
        mask = dist <= t
        n_evc_matched = int(mask.any(axis=1).sum())
        n_rare_matched = int(mask.any(axis=0).sum())

        pair_counts: dict[tuple[str, str], int] = {}
        ei, ri = np.where(mask)
        for e, r in zip(ei, ri):
            key = (evc_class[e], rare_class[r])
            pair_counts[key] = pair_counts.get(key, 0) + 1
        pair_str = ", ".join(f"{a}->{b}:{c}" for (a, b), c in sorted(pair_counts.items()))
        print(f"{t:>7} {n_evc_matched:>12} {n_rare_matched:>15}   {pair_str}")

    frac_t4 = float((dist <= 4).any(axis=1).mean())
    n_evc_t4 = int((dist <= 4).any(axis=1).sum())
    print(f"\nAt distance <= 4 (near-certainly the same image): "
          f"{n_evc_t4}/{len(evc_df)} EVC images ({frac_t4:.1%}) match a RARE25 training image.")
    n_evc_5_8 = int(((dist <= 8) & ~(dist <= 4)).any(axis=1).sum())
    print(f"An additional {n_evc_5_8} EVC image(s) match only in the 5-8 distance band "
          f"(plausibly the same patient/scene, not necessarily the same captured frame).")

    # --- Full pair list at --pair-threshold ---
    t = args.pair_threshold
    mask = dist <= t
    ei, ri = np.where(mask)
    pairs = pd.DataFrame({
        "evc_filepath": evc_df["filepath"].to_numpy()[ei],
        "evc_class": evc_class[ei],
        "rare25_filepath": rare_df["filepath"].to_numpy()[ri],
        "rare25_centre": rare_df["centre"].to_numpy()[ri],
        "rare25_class": rare_class[ri],
        "hamming_distance": dist[ei, ri],
    }).sort_values(["hamming_distance", "evc_filepath"]).reset_index(drop=True)

    out_path = MANIFESTS / "evc_rare25_matches.csv"
    pairs.to_csv(out_path, index=False)
    print(f"\nFull pair list at distance <= {t}: {len(pairs)} pairs -> {out_path}")

    # --- Breakdown by RARE25 hospital and class ---
    print(f"\n=== Matched RARE25 images (distance <= {t}) by hospital x class ===")
    matched_rare_idx = sorted(set(ri))
    matched_rare = rare_df.iloc[matched_rare_idx]
    if len(matched_rare):
        print(pd.crosstab(matched_rare["centre"], matched_rare["class_label"]))
    else:
        print(f"(no RARE25 images matched at distance <= {t})")

    print("\nNote: overlap with the private/held-out test set is not measurable from "
          "this comparison and no claim is made about it -- this covers the RARE25 "
          "training set only.")


if __name__ == "__main__":
    main()
