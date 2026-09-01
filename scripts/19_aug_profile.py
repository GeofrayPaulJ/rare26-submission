"""Per-transform CPU cost of the augmentation stack, at the real working size.

The data path currently has roughly 10x headroom over the GPU, so a stack that
costs a few milliseconds per image is free. JPEG re-encoding and per-subpixel
noise draws are the two candidates for not being free, and "which transform is
responsible" is a question worth being able to answer from a table rather than
a guess.

Two numbers per transform:

  cost_ms       time when it fires
  budget_ms     cost_ms x its probability -- its share of the MEAN per-image
                cost, which is the only figure that composes into an epoch

    python scripts/19_aug_profile.py [--n 64] [--out reports/aug_profile.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import augment as A  # noqa: E402
from src.config import AugConfig, Config  # noqa: E402
from src.data import _CROP_FUNCS, _decode, _resize_square  # noqa: E402

FULL_STACK = dict(photometric=True, optical=True, sensor=True, compression=True)


def load_frames(cfg: Config, manifest: pd.DataFrame, n: int) -> List[np.ndarray]:
    """Real frames, prepared exactly as training prepares them. Synthetic noise
    would misprice JPEG badly -- an incompressible image is the worst case for
    the encoder and not the case that will actually be paid."""
    usable = manifest[manifest["fold_r0"] != -1].sort_values("filepath")
    step = max(1, len(usable) // n)
    rows = usable.iloc[::step].head(n)
    frames = []
    for _, row in rows.iterrows():
        d = row.to_dict()
        img = _decode(os.path.join(REPO_ROOT, cfg.image_root, d["filepath"]))
        img = _CROP_FUNCS[cfg.crop_mode](img, d)
        frames.append(_resize_square(_resize_square(img, cfg.cache_size),
                                     cfg.image_size))
    return frames


def time_ms(fn: Callable[[np.ndarray, np.random.Generator], object],
            frames: List[np.ndarray], repeats: int = 3) -> float:
    """Median per-image milliseconds over `repeats` passes.

    Median, not mean: one page fault or one scheduler preemption in a 64-image
    pass would otherwise set the number, and the question here is what a
    transform costs typically, not what its worst frame cost once.
    """
    per_pass = []
    for r in range(repeats):
        rng = np.random.default_rng(1000 + r)
        t0 = time.perf_counter()
        for f in frames:
            fn(f, rng)
        per_pass.append((time.perf_counter() - t0) / len(frames) * 1e3)
    return float(statistics.median(per_pass))


def build_cases(aug: AugConfig, dist) -> Dict[str, tuple]:
    """(callable, probability) per transform, each forced to fire."""
    return {
        "hflip": (lambda f, r: A.horizontal_flip(f, r, 1.0), aug.hflip_p),
        "black_boxes": (lambda f, r: A.random_black_boxes(f, r, dist, 1.0),
                        aug.black_boxes_p),
        "gamma+white_balance": (
            lambda f, r: __import__("cv2").LUT(
                f, A.channel_gamma_wb_lut(
                    A.per_channel_gamma_draw(r, aug.gamma_m, aug.gamma_channel_m),
                    A.white_balance_draw(r, aug.white_balance_m,
                                         aug.white_balance_tint_frac))),
            aug.gamma_p),
        "hue_sat": (lambda f, r: A.hue_saturation(f, r, aug.hue_m, aug.sat_m),
                    aug.hue_sat_p),
        "vignette": (lambda f, r: A._apply_gain(
            f, A.vignette_field(*f.shape[:2], 0.4)), aug.vignette_p),
        "linear_gradient": (lambda f, r: A._apply_gain(
            f, A.linear_gradient_field(*f.shape[:2], 0.25, 1.0)),
            aug.linear_gradient_p),
        "radial_gradient": (lambda f, r: A._apply_gain(
            f, A.radial_gradient_field(*f.shape[:2], 0.25, 0.3, -0.2)),
            aug.radial_gradient_p),
        "specular": (lambda f, r: A.specular_highlights(f, r, aug.specular_m),
                     aug.specular_p),
        "distortion": (lambda f, r: A.barrel_distortion(f, 0.18),
                       aug.distortion_p),
        "motion_blur": (lambda f, r: __import__("cv2").filter2D(
            f, -1, A.motion_blur_kernel(13, 0.7)), aug.motion_blur_p),
        "defocus": (lambda f, r: __import__("cv2").filter2D(
            f, -1, A.defocus_kernel(8)), aug.defocus_p),
        "sharpen": (lambda f, r: A.unsharp_mask(f, 0.5, 1.4), aug.sharpen_p),
        "gaussian_noise": (lambda f, r: A.gaussian_noise(f, r, 12.0),
                           aug.gaussian_noise_p),
        "poisson_noise": (lambda f, r: A.poisson_noise(f, r, 60.0),
                          aug.poisson_noise_p),
        "downsample": (lambda f, r: A.downsample_cycle(f, 0.55),
                       aug.downsample_p),
        "jpeg": (lambda f, r: A.jpeg_recompress(f, 60), aug.jpeg_p),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs",
                                                     "convnext_b_384.yaml"))
    ap.add_argument("--n", type=int, default=64, help="frames per timing pass")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports",
                                                  "aug_profile.json"))
    args = ap.parse_args(argv)

    import cv2

    cv2.setNumThreads(0)  # match the DataLoader worker environment exactly

    cfg = Config.from_yaml(args.config)
    manifest = pd.read_csv(os.path.join(REPO_ROOT, cfg.manifest))
    frames = load_frames(cfg, manifest, args.n)
    dist = A.load_redaction_dist(os.path.join(REPO_ROOT, cfg.redaction_stats))
    aug_full = AugConfig(**FULL_STACK)
    aug_base = AugConfig()

    rows = []
    for name, (fn, p) in build_cases(aug_full, dist).items():
        cost = time_ms(fn, frames)
        rows.append({"transform": name, "cost_ms": cost, "p": p,
                     "budget_ms": cost * p})
    rows.sort(key=lambda r: -r["budget_ms"])

    end_base = time_ms(
        lambda f, r: A.apply_uint8(f, aug_base, r, True, dist), frames)
    end_full = time_ms(
        lambda f, r: A.apply_uint8(f, aug_full, r, True, dist), frames)
    # what the loader really pays per item: resize + augment + normalise
    item_base = time_ms(
        lambda f, r: A.apply(_resize_square(f, cfg.image_size), aug_base, r,
                             True, dist), frames)
    item_full = time_ms(
        lambda f, r: A.apply(_resize_square(f, cfg.image_size), aug_full, r,
                             True, dist), frames)

    print(f"{args.n} real frames at {cfg.image_size}x{cfg.image_size}, "
          f"cv2 threads = 0 (as in a DataLoader worker)\n")
    print(f"{'transform':22s} {'cost_ms':>9s} {'p':>6s} {'budget_ms':>10s}")
    print("-" * 51)
    for r in rows:
        print(f"{r['transform']:22s} {r['cost_ms']:9.3f} {r['p']:6.2f} "
              f"{r['budget_ms']:10.3f}")
    print("-" * 51)
    print(f"{'sum of budgets':22s} {'':>9s} {'':>6s} "
          f"{sum(r['budget_ms'] for r in rows):10.3f}")
    print()
    print(f"end-to-end augment only : baseline {end_base:6.3f} ms  ->  "
          f"full {end_full:6.3f} ms   ({end_full / max(1e-9, end_base):.1f}x)")
    print(f"full item (aug + norm)  : baseline {item_base:6.3f} ms  ->  "
          f"full {item_full:6.3f} ms   "
          f"(+{item_full - item_base:.2f} ms, "
          f"{100 * (item_full / max(1e-9, item_base) - 1):+.0f}%)")

    payload = {
        "n_frames": args.n, "image_size": cfg.image_size,
        "transforms": rows,
        "end_to_end_ms": {
            "augment_baseline": end_base, "augment_full": end_full,
            "item_baseline": item_base, "item_full": item_full,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
