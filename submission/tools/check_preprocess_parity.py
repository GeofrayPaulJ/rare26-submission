"""Prove the container's preprocessing is the training pipeline, bit for bit.

Comparing logits can only ever be an indirect argument about preprocessing,
because the arithmetic sits in between: two torch builds pick different cuDNN
kernels and accumulate bf16 in a different order, so a logit difference of 1e-2
tells you nothing about whether the pixels going in were the same.

This compares the PIXELS. It runs the container's ``preprocess`` and the
training loader's ``BarrettDataset.__getitem__`` over the same images and
requires the resulting CHW float32 tensors to be bit-identical. No GPU, no
autocast, no kernel selection -- just cv2 and numpy on both sides. If this
passes, any remaining logit difference is arithmetic and nothing else.

Fallback images are excluded and counted: the container deliberately crops them
differently, which is the entire point of the fallback, so requiring them to
match would be requiring the fallback not to work.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "submission"))

from PIL import Image  # noqa: E402

from rare26_infer.fov import detect_crop_box  # noqa: E402
from rare26_infer.preprocess import preprocess  # noqa: E402
from src.config import Config  # noqa: E402
from src.data import BarrettDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
    df = pd.read_csv(manifest)
    usable = df[df[f"fold_r{args.repeat}"] != -1]
    fps = usable[usable[f"fold_r{args.repeat}"] == args.fold]["filepath"].tolist()
    if args.limit:
        fps = fps[: args.limit]

    cfg = Config(manifest=manifest, image_root=os.path.join(REPO_ROOT, "00_source"),
                 redaction_stats=os.path.join(REPO_ROOT, "manifests", "redaction_check.csv"))
    # train=False: the eval path, which is what inference reproduces
    ds = BarrettDataset(fps, df, cfg, train=False, build_cache=True)

    n_exact = n_fallback = 0
    worst = 0.0
    worst_fp = ""
    for i, fp in enumerate(fps):
        harness_tensor = ds[i][0].numpy()

        arr = np.asarray(Image.open(os.path.join(cfg.image_root, fp)).convert("RGB"))
        box, used_fallback, _fq, _reason = detect_crop_box(arr)
        if used_fallback:
            n_fallback += 1
            continue
        container_tensor = preprocess(arr, box)

        if np.array_equal(container_tensor, harness_tensor):
            n_exact += 1
        else:
            d = float(np.abs(container_tensor - harness_tensor).max())
            if d > worst:
                worst, worst_fp = d, fp

    n_compared = len(fps) - n_fallback
    print(f"images                : {len(fps)}")
    print(f"FOV fallback (skipped): {n_fallback}")
    print(f"compared              : {n_compared}")
    print(f"bit-identical tensors : {n_exact}/{n_compared}")
    if n_exact != n_compared:
        print(f"worst mismatch        : {worst:.3e} on {worst_fp}")
        raise SystemExit(
            "\nFAIL: container preprocessing differs from the training pipeline."
        )
    print("\nPASS: container preprocessing is bit-identical to src/data.py "
          "(crop -> 431 -> 384 -> ImageNet normalise).")


if __name__ == "__main__":
    main()
