"""
09_imaging_mode.py -- Work out which images are white-light and which are narrow-band.

WHY: The endoscope shoots in two modes. White light (WLE) looks like normal pink
tissue. Narrow-band (NBI) filters the light and comes out green/cyan. They look
nothing alike, so a model trained mostly on one will be weak on the other.
You need to know the split, and your folds should be balanced across it.

HOW: NBI has almost no red channel relative to green. Measuring the red-to-green
ratio inside the FOV separates the two modes cleanly, with no model required.

The script picks the split point automatically (Otsu's method on the ratio) rather
than using a hardcoded number, then prints the histogram so you can eyeball
whether the separation is genuinely clean or whether you need to intervene.

USAGE:
    python 09_imaging_mode.py
    python 09_imaging_mode.py --manual-threshold 0.75
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "00_source"
MANIFESTS = ROOT / "manifests"


def measure_one(args):
    rel_path, cx, cy, radius = args
    try:
        img = Image.open(SRC / rel_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32)
        h, w = arr.shape[:2]

        # Sample only well inside the FOV -- avoids the dark vignette edge,
        # which would drag the colour statistics toward black.
        yy, xx = np.ogrid[:h, :w]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * 0.8) ** 2

        # Ignore near-black and blown-out specular pixels.
        lum = arr.mean(axis=2)
        usable = inside & (lum > 20) & (lum < 245)
        if usable.sum() < 100:
            usable = inside

        r = arr[..., 0][usable].mean()
        g = arr[..., 1][usable].mean()
        b = arr[..., 2][usable].mean()

        return {
            "filepath": rel_path,
            "mean_r": round(float(r), 2),
            "mean_g": round(float(g), 2),
            "mean_b": round(float(b), 2),
            "rg_ratio": round(float(r / g), 4) if g > 0 else None,
            "bg_ratio": round(float(b / g), 4) if g > 0 else None,
            "mode_error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"filepath": rel_path, "rg_ratio": None, "mode_error": str(exc)}


def otsu_threshold(values, bins=256):
    """Find the split point that best separates two populations."""
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    w0 = np.cumsum(hist)
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)

    cumsum = np.cumsum(hist * centres)
    m0 = np.divide(cumsum, w0, out=np.zeros_like(cumsum), where=w0 > 0)
    m1 = np.divide(cumsum[-1] - cumsum, w1, out=np.zeros_like(cumsum), where=w1 > 0)

    between = w0 * w1 * (m0 - m1) ** 2
    between[~valid] = -1
    return float(centres[np.argmax(between)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-threshold", type=float, default=None,
                    help="Override the automatic split point on red/green ratio.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = MANIFESTS / "rare25_folds.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run 07_folds.py first.")

    df = pd.read_csv(src)
    needed = {"fov_centre_x", "fov_centre_y", "fov_radius"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"Missing FOV columns {needed - set(df.columns)}. "
                         f"Re-run 05_fov_crop.py and 07_folds.py.")

    jobs = list(zip(df.filepath, df.fov_centre_x, df.fov_centre_y, df.fov_radius))
    print(f"Measuring colour in {len(jobs)} images...")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in tqdm(pool.map(measure_one, jobs, chunksize=32), total=len(jobs)):
            results.append(res)

    colour = pd.DataFrame(results)
    df = df.merge(colour, on="filepath", how="left")

    ratios = df["rg_ratio"].dropna().values
    thresh = args.manual_threshold if args.manual_threshold else otsu_threshold(ratios)
    df["imaging_mode"] = np.where(df["rg_ratio"] < thresh, "NBI", "WLE")

    print(f"\nSplit point on red/green ratio: {thresh:.4f}"
          f"{' (manual)' if args.manual_threshold else ' (automatic)'}")

    # Histogram so you can see whether the separation is actually clean.
    print("\nRed/green ratio distribution:")
    hist, edges = np.histogram(ratios, bins=30)
    peak = hist.max()
    for count, lo in zip(hist, edges[:-1]):
        bar = "#" * int(40 * count / peak) if peak else ""
        marker = " <-- split" if lo <= thresh < lo + (edges[1] - edges[0]) else ""
        print(f"  {lo:6.3f} {bar:<40s} {count:5d}{marker}")

    print("\n=== MODE COUNTS ===")
    print(df.imaging_mode.value_counts().to_string())
    print("\n=== MODE x CLASS ===")
    ct = pd.crosstab(df.imaging_mode, df.class_label)
    ct["pos_rate_%"] = (ct.neoplasia / ct.sum(axis=1) * 100).round(2)
    print(ct)
    print("\n=== MODE x HOSPITAL ===")
    print(pd.crosstab(df.imaging_mode, df.centre))
    print("\n=== MODE BALANCE ACROSS FOLDS (repeat 0) ===")
    print(pd.crosstab(df.fold_r0, df.imaging_mode))

    nbi_pos = ct.loc["NBI", "neoplasia"] if "NBI" in ct.index else 0
    if nbi_pos < 10:
        print(f"\n*** Only {nbi_pos} positive NBI images. Too few to train or "
              f"validate on separately -- treat NBI as an augmentation target, "
              f"not a separate model.")

    dst = MANIFESTS / "rare25_folds_with_mode.csv"
    df.to_csv(dst, index=False)
    print(f"\nWritten: {dst}")


if __name__ == "__main__":
    main()
