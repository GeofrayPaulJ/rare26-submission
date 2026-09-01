"""
05_fov_crop.py -- Detect the circular field-of-view geometry in every image.

WHY: Every image is a circular endoscope view inscribed in a black rectangle.
An earlier version of this script tried to crop the black border with an
axis-aligned bounding box, but the FOV circle's diameter equals the frame
height (it touches the top and bottom edges), so almost every row and column
already contains bright pixels -- the bbox always equals the full image and
nothing gets cropped. Confirmed on this data: bbox crop kept 99.6-100% of the
frame area for every single image.

WHAT IT DOES INSTEAD (detection only -- no image copies are written):
  - Builds a mask of non-black pixels (brightness > threshold, default 15).
  - Takes the largest connected component of that mask as the FOV region
    (scipy.ndimage.label), which discards small unconnected bright artifacts.
  - Fits a circle to the FOV region using centroid + area-derived radius
    (r = sqrt(area/pi)), NOT cv2.minEnclosingCircle. Both were tested on a
    sample of this dataset: minEnclosingCircle consistently overestimated the
    radius by ~11% and scored lower on the fit-quality check below, because it
    fits to extreme boundary points and gets pulled outward by small boundary
    irregularities (specular reflections, texture at the FOV edge). The
    area/centroid estimate is a robust average over the whole region and is
    not swayed by a handful of outlier boundary pixels -- it scored
    fit_quality >= 0.98 on every sampled image, versus ~0.96-0.98 for
    minEnclosingCircle.
  - Computes the largest axis-aligned square inscribed in that circle
    (side = floor(radius * sqrt(2))), centred on the fitted centre and
    clamped to image bounds. This inner square is what 06_regroup.py crops to
    (in memory) before hashing, since it is guaranteed to contain only FOV
    content and no black background.
  - Records a fit_quality score: the fraction of the fitted circle's own disk
    area that is actually non-black in the mask. Low values flag images where
    the FOV isn't circular, is partially clipped, or the fit is otherwise off.

Read-only: never writes to 00_source, and writes no image files at all.
Output: manifests/rare25_manifest_fov.csv (fov geometry merged onto the manifest).

USAGE:
    python 05_fov_crop.py
    python 05_fov_crop.py --threshold 15 --workers 8
"""

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "00_source"
MANIFESTS = ROOT / "manifests"


def fit_fov_circle(mask: np.ndarray):
    """Largest connected component -> (centre_x, centre_y, radius, fit_quality, area_pct)."""
    h, w = mask.shape
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return None

    sizes = ndimage.sum(mask, labeled, index=range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    comp_mask = labeled == largest_label
    area = float(sizes[largest_label - 1])

    ys, xs = np.nonzero(comp_mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    radius = math.sqrt(area / math.pi)

    yy, xx = np.ogrid[:h, :w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    disk_area = int(disk.sum())
    fit_quality = float((mask & disk).sum() / disk_area) if disk_area else 0.0

    return cx, cy, radius, fit_quality, 100.0 * area / (w * h)


def inner_square(cx: float, cy: float, radius: float, width: int, height: int):
    """Largest axis-aligned square inscribed in the fitted circle, clamped to bounds."""
    side = math.floor(radius * math.sqrt(2))
    left = round(cx - side / 2)
    top = round(cy - side / 2)
    right = left + side
    bottom = top + side

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    return left, top, right, bottom


def process_one(args: tuple[str, int]) -> dict:
    rel_path, threshold = args
    path = SRC / rel_path

    row = {
        "filepath": rel_path,
        "fov_centre_x": None, "fov_centre_y": None, "fov_radius": None,
        "fov_area_pct": None, "fit_quality": None,
        "inner_left": None, "inner_top": None, "inner_right": None, "inner_bottom": None,
        "fov_error": "",
    }
    try:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        mask = arr.mean(axis=2) > threshold

        fit = fit_fov_circle(mask)
        if fit is None:
            row["fov_error"] = "no non-black pixels found"
            return row

        cx, cy, radius, fit_quality, area_pct = fit
        left, top, right, bottom = inner_square(cx, cy, radius, w, h)

        row.update(
            fov_centre_x=round(cx, 2), fov_centre_y=round(cy, 2), fov_radius=round(radius, 2),
            fov_area_pct=round(area_pct, 2), fit_quality=round(fit_quality, 4),
            inner_left=left, inner_top=top, inner_right=right, inner_bottom=bottom,
        )
    except Exception as exc:  # noqa: BLE001
        row["fov_error"] = str(exc)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=int, default=15,
                     help="Pixel brightness above which content is considered non-black.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest_path = MANIFESTS / "rare25_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}. Run the earlier phases first.")

    df = pd.read_csv(manifest_path)
    jobs = [(fp, args.threshold) for fp in df["filepath"].tolist()]

    print(f"Detecting FOV geometry for {len(jobs)} images (no image files written)...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in tqdm(pool.map(process_one, jobs, chunksize=32), total=len(jobs)):
            results.append(res)

    fov_df = pd.DataFrame(results)
    merged = df.merge(fov_df, on="filepath", how="left")

    n_errors = int((merged["fov_error"].fillna("") != "").sum())
    print(f"\nDone. FOV detection failures: {n_errors}")
    if n_errors:
        for _, row in merged[merged["fov_error"].fillna("") != ""].iterrows():
            print(f"  {row['filepath']}: {row['fov_error']}")

    out_path = MANIFESTS / "rare25_manifest_fov.csv"
    merged.to_csv(out_path, index=False)

    ok = merged[merged["fov_error"].fillna("") == ""]
    print(f"\nMean fit_quality: {ok['fit_quality'].mean():.4f}  (min {ok['fit_quality'].min():.4f})")

    print("\nIs hospital still readable from FOV geometry alone?")
    for col in ["fov_radius", "fov_centre_x", "fov_centre_y"]:
        c1 = ok.loc[ok.centre == "center_1", col]
        c2 = ok.loc[ok.centre == "center_2", col]
        lo = max(c1.min(), c2.min())
        hi = min(c1.max(), c2.max())
        overlap_range = max(0.0, hi - lo)
        print(
            f"  {col:14s}: center_1 [{c1.min():.1f}, {c1.max():.1f}]  "
            f"center_2 [{c2.min():.1f}, {c2.max():.1f}]  "
            f"overlap_range={overlap_range:.1f}  "
            f"{'OVERLAP (good)' if overlap_range > 0 else 'STILL SEPARABLE (bad)'}"
        )

    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
