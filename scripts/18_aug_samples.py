"""Render a 5x5 grid of one image under 25 draws of the full augmentation stack.

This exists to be LOOKED AT before any GPU time is spent. The magnitudes in
AugConfig sit at roughly 2-3x common library defaults, which is a deliberate
choice and also an easy one to get wrong; a picture answers "do these still
read as endoscopy" in a second, where a failed 13-hour sweep answers it
expensively and ambiguously.

Each tile is labelled with the transforms that actually fired, so a tile that
looks wrong names the knob to turn down instead of merely raising an alarm.

    python scripts/18_aug_samples.py [--filepath ...] [--out reports/aug_samples.png]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.augment import apply_uint8, load_redaction_dist  # noqa: E402
from src.config import AugConfig, Config  # noqa: E402
from src.data import _CROP_FUNCS, _decode, _resize_square  # noqa: E402

GRID = 5
FULL_STACK = dict(photometric=True, optical=True, sensor=True, compression=True)


def load_frame(cfg: Config, manifest: pd.DataFrame, filepath: str) -> np.ndarray:
    """Decode, crop and resize exactly as BarrettDataset does, so the grid shows
    what the model sees and not a differently-prepared cousin of it."""
    row = manifest.set_index("filepath").loc[filepath].to_dict()
    row["filepath"] = filepath
    img = _decode(os.path.join(REPO_ROOT, cfg.image_root, filepath))
    img = _CROP_FUNCS[cfg.crop_mode](img, row)
    img = _resize_square(img, cfg.cache_size)
    return _resize_square(img, cfg.image_size)


def pick_default_filepath(manifest: pd.DataFrame) -> str:
    """A well-fitted, unredacted training frame -- representative rather than
    cherry-picked, and deterministic so the grid is comparable between runs."""
    ok = manifest[
        (manifest["fold_r0"] != -1)
        & (~manifest["has_redaction"].astype(bool))
        & (manifest["fit_quality"] > manifest["fit_quality"].median())
    ].sort_values("filepath")
    if len(ok) == 0:
        ok = manifest[manifest["fold_r0"] != -1].sort_values("filepath")
    return str(ok.iloc[len(ok) // 2]["filepath"])


def wrap(names: List[str], width: int = 34) -> str:
    """Fold a fired-transform list onto as few short lines as possible."""
    if not names:
        return "(nothing fired)"
    lines, cur = [], ""
    for n in names:
        cand = n if not cur else f"{cur} {n}"
        if len(cand) > width:
            lines.append(cur)
            cur = n
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def render(
    frame: np.ndarray, aug: AugConfig, dist, seeds: List[int]
) -> Tuple[List[np.ndarray], List[List[str]]]:
    images, traces = [], []
    for s in seeds:
        trace: List[str] = []
        out = apply_uint8(
            frame, aug, np.random.default_rng(s), True, dist, trace=trace
        )
        images.append(out)
        traces.append(trace)
    return images, traces


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs",
                                                     "convnext_b_384.yaml"))
    ap.add_argument("--filepath", default=None, help="manifest filepath to render")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports",
                                                  "aug_samples.png"))
    ap.add_argument("--seed0", type=int, default=0,
                    help="first of the 25 consecutive augmentation seeds")
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    manifest = pd.read_csv(os.path.join(REPO_ROOT, cfg.manifest))
    filepath = args.filepath or pick_default_filepath(manifest)
    frame = load_frame(cfg, manifest, filepath)

    aug = AugConfig(**FULL_STACK)
    aug.validate()
    dist = load_redaction_dist(os.path.join(REPO_ROOT, cfg.redaction_stats))
    seeds = list(range(args.seed0, args.seed0 + GRID * GRID))
    images, traces = render(frame, aug, dist, seeds)

    fig = plt.figure(figsize=(17.5, 19.0), dpi=110)
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 4.6], hspace=0.10, top=0.945,
        bottom=0.012, left=0.012, right=0.988,
    )

    # --- reference row: the unaugmented source, at the same scale ---
    head = outer[0].subgridspec(1, 5, wspace=0.04)
    ax = fig.add_subplot(head[0, 2])
    ax.imshow(frame)
    ax.set_title("SOURCE (unaugmented)", fontsize=10, fontweight="bold", pad=5)
    ax.axis("off")

    # --- the 25 draws ---
    grid = outer[1].subgridspec(GRID, GRID, wspace=0.035, hspace=0.30)
    for k, (img, trace, seed) in enumerate(zip(images, traces, seeds)):
        ax = fig.add_subplot(grid[k // GRID, k % GRID])
        ax.imshow(img)
        ax.set_title(f"seed {seed}   |   {wrap(trace)}",
                     fontsize=6.4, loc="left", pad=2.5, linespacing=1.25)
        ax.axis("off")

    fig.suptitle(
        f"RARE26 augmentation stack -- 25 random draws of one frame\n"
        f"{filepath}   |   photometric + optical + sensor + compression, "
        f"shipped magnitudes (~2-3x library defaults)",
        fontsize=11.5, fontweight="bold",
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- how often each transform fired, as a cheap check on the probabilities
    counts: dict = {}
    for trace in traces:
        for name in trace:
            key = name.split("(")[0]
            counts[key] = counts.get(key, 0) + 1
    print(f"source frame : {filepath}  {frame.shape}")
    print(f"written      : {args.out}")
    print(f"fired counts over {len(seeds)} draws (expected = 25 x p):")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        p = getattr(aug, f"{name}_p", None)
        exp = f"  expected ~{25 * p:.1f}" if p is not None else ""
        print(f"  {name:18s} {n:3d}{exp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
