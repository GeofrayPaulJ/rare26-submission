"""How often does the FOV fallback trigger, and on which images?

The detector was tuned on two Dutch centres. The test set comes from twelve
unseen ones, so the number that matters is not "does it work on the training
data" -- it was fitted there -- but how close the margin is on data the tuning
never saw. center_2's leave-one-centre-out holdout is the closest available
proxy: 816 images from a centre the fold splits deliberately keep separable.

A rate near zero means the geometry generalises and the fallback is genuinely a
safety net. A rate of several percent means the floor is doing real work and
the fallback path needs to be as trustworthy as the primary one.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "submission"))

from rare26_infer.fov import FIT_QUALITY_FLOOR, detect_crop_box  # noqa: E402

MANIFEST = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
IMAGE_ROOT = os.path.join(REPO_ROOT, "00_source")


def run(fps, label: str) -> dict:
    n_fb = 0
    reasons: dict = {}
    fqs = []
    for fp in fps:
        arr = np.asarray(Image.open(os.path.join(IMAGE_ROOT, fp)).convert("RGB"))
        _box, used_fallback, fq, reason = detect_crop_box(arr)
        if fq is not None:
            fqs.append(fq)
        if used_fallback:
            n_fb += 1
            key = "fit_quality below floor" if "fit_quality" in reason else reason
            reasons[key] = reasons.get(key, 0) + 1
    fqs_a = np.asarray(fqs)
    print(f"\n=== {label} ===")
    print(f"images            : {len(fps)}")
    print(f"fallback triggered: {n_fb}  ({100.0 * n_fb / max(1, len(fps)):.2f}%)")
    print(f"reasons           : {reasons or '{}'}")
    if fqs_a.size:
        print(f"fit_quality       : min {fqs_a.min():.4f} | p1 {np.percentile(fqs_a, 1):.4f} | "
              f"median {np.median(fqs_a):.4f} | mean {fqs_a.mean():.4f}")
    return {"n": len(fps), "n_fallback": n_fb, "reasons": reasons}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap images per group (0 = all)")
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST)
    print(f"fit_quality floor : {FIT_QUALITY_FLOOR} (1st percentile of training)")

    groups = {
        "center_2 HELD-OUT (holdout_center_2 == test)":
            df[df["holdout_center_2"] == "test"]["filepath"].tolist(),
        "center_1 (all)": df[df["centre"] == "center_1"]["filepath"].tolist(),
        "all training images": df["filepath"].tolist(),
    }
    for label, fps in groups.items():
        if args.limit:
            fps = fps[: args.limit]
        run(fps, label)


if __name__ == "__main__":
    main()
