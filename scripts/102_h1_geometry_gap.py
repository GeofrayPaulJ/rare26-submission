"""H1 -- measure the geometry gap between the organisers' fixture and the
training-derived probes this project has always used.

L2 found the organisers' own test fixture (example_batch_0_15.tiff) is
natively 512x512, square. Every training image and every synthetic probe
this project has built assumes the 637x512 modal aspect ratio instead. The
FOV detector finds a circle either way -- 0% fallback on the fixture proved
that much already -- but a circle that touches the top and bottom of a
637x512 frame and one that touches the top and bottom of a 512x512 frame are
not necessarily cropped the same way once inscribed-square-then-clamp runs.
This script measures that gap directly, no fallback logic invoked -- raw
geometry only.

Organiser fixture: runs/submission_test/l2_original_tiff/example_batch_0_15.tiff
  (16 slices, read fresh -- no precomputed geometry exists for it).
200 RARE25 training images: the same 200-image set D3 already built
  (runs/submission_test/d3_fov/ground_truth.csv), geometry read from the
  existing manifests/rare25_manifest_fov.csv (05_fov_crop.py's own output --
  not recomputed, to stay byte-identical with what training/inference use).

fit_fov_circle / inner_square are VERBATIM from scripts/05_fov_crop.py
(same copy submission/rare26_infer/fov.py ships).

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/102_h1_geometry_gap.py'
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import tifffile

REPO_ROOT = "/workspace/RARE26"
FIXTURE = os.path.join(REPO_ROOT, "runs/submission_test/l2_original_tiff/example_batch_0_15.tiff")
D3_GT = os.path.join(REPO_ROOT, "runs/submission_test/d3_fov/ground_truth.csv")
FOV_MANIFEST = os.path.join(REPO_ROOT, "manifests/rare25_manifest_fov.csv")
THRESHOLD = 15
FIT_QUALITY_FLOOR = 0.9040  # submission/rare26_infer/fov.py -- 1st pct over training


# --- verbatim from scripts/05_fov_crop.py / submission/rare26_infer/fov.py ---
def fit_fov_circle(mask: np.ndarray):
    from scipy import ndimage
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


def inner_square(cx, cy, radius, width, height):
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
# --- end verbatim ---


def circle_area_survival(cx, cy, radius, box, w, h) -> float:
    """Fraction of the fitted circle's own pixel area that lies inside box."""
    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    circle_area = int(circle_mask.sum())
    if circle_area == 0:
        return float("nan")
    left, top, right, bottom = box
    crop_mask = np.zeros((h, w), dtype=bool)
    crop_mask[top:bottom, left:right] = True
    inter = int((circle_mask & crop_mask).sum())
    return inter / circle_area


def measure_fixture() -> list[dict]:
    rows = []
    with tifffile.TiffFile(FIXTURE) as tif:
        for i, page in enumerate(tif.pages):
            arr = np.asarray(page.asarray())
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            h, w = arr.shape[:2]
            mask = arr.mean(axis=2) > THRESHOLD
            fit = fit_fov_circle(mask)
            row = {"source": "organiser_fixture", "id": f"slice_{i}", "width": w, "height": h,
                   "aspect_ratio": round(w / h, 4)}
            if fit is None:
                row.update(fov_error="no non-black pixels found")
                rows.append(row)
                continue
            cx, cy, radius, fit_quality, area_pct = fit
            box = inner_square(cx, cy, radius, w, h)
            left, top, right, bottom = box
            crop_frac_of_frame = ((right - left) * (bottom - top)) / (w * h)
            circle_survival = circle_area_survival(cx, cy, radius, box, w, h)
            row.update(
                fov_centre_x=round(cx, 2), fov_centre_y=round(cy, 2), fov_radius=round(radius, 2),
                fit_quality=round(fit_quality, 4), fov_area_pct=round(area_pct, 2),
                would_fallback=bool(fit_quality < FIT_QUALITY_FLOOR),
                inner_left=left, inner_top=top, inner_right=right, inner_bottom=bottom,
                crop_w=right - left, crop_h=bottom - top,
                crop_frac_of_frame=round(crop_frac_of_frame, 4),
                circle_area_survival_frac=round(circle_survival, 4),
            )
            rows.append(row)
    return rows


def measure_training_200() -> list[dict]:
    gt = pd.read_csv(D3_GT)
    fov = pd.read_csv(FOV_MANIFEST)
    merged = gt.merge(fov, on="filepath", how="left")
    missing = merged["fov_centre_x"].isna().sum()
    if missing:
        print(f"WARNING: {missing}/{len(merged)} of the 200 have no precomputed FOV geometry")

    rows = []
    for _, r in merged.iterrows():
        row = {"source": "rare25_training", "id": r["filepath"], "width": int(r["width"]),
               "height": int(r["height"]), "aspect_ratio": round(r["width"] / r["height"], 4)}
        if pd.isna(r["fov_centre_x"]):
            row.update(fov_error=str(r.get("fov_error", "missing")))
            rows.append(row)
            continue
        w, h = int(r["width"]), int(r["height"])
        cx, cy, radius = float(r["fov_centre_x"]), float(r["fov_centre_y"]), float(r["fov_radius"])
        fit_quality = float(r["fit_quality"])
        box = (int(r["inner_left"]), int(r["inner_top"]), int(r["inner_right"]), int(r["inner_bottom"]))
        left, top, right, bottom = box
        crop_frac_of_frame = ((right - left) * (bottom - top)) / (w * h)
        circle_survival = circle_area_survival(cx, cy, radius, box, w, h)
        row.update(
            fov_centre_x=round(cx, 2), fov_centre_y=round(cy, 2), fov_radius=round(radius, 2),
            fit_quality=round(fit_quality, 4), fov_area_pct=round(float(r["fov_area_pct"]), 2),
            would_fallback=bool(fit_quality < FIT_QUALITY_FLOOR),
            inner_left=left, inner_top=top, inner_right=right, inner_bottom=bottom,
            crop_w=right - left, crop_h=bottom - top,
            crop_frac_of_frame=round(crop_frac_of_frame, 4),
            circle_area_survival_frac=round(circle_survival, 4),
        )
        rows.append(row)
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    df = pd.DataFrame([r for r in rows if "fov_error" not in r])
    n_err = len(rows) - len(df)
    out = {"label": label, "n": len(rows), "n_fov_errors": n_err}
    for col in ["aspect_ratio", "fit_quality", "crop_frac_of_frame", "circle_area_survival_frac"]:
        if col in df.columns and len(df):
            out[col] = {
                "mean": round(float(df[col].mean()), 4),
                "median": round(float(df[col].median()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            }
    if "would_fallback" in df.columns and len(df):
        out["n_would_fallback"] = int(df["would_fallback"].sum())
    return out


def main() -> int:
    print("[h1] measuring organiser fixture (16 slices)...")
    fixture_rows = measure_fixture()
    print("[h1] measuring 200 RARE25 training images (from manifests/rare25_manifest_fov.csv)...")
    training_rows = measure_training_200()

    all_rows = fixture_rows + training_rows
    raw_path = os.path.join(REPO_ROOT, "reports/h1_geometry_gap_raw.csv")
    pd.DataFrame(all_rows).to_csv(raw_path, index=False)
    print(f"[h1] wrote {raw_path}")

    fixture_summary = summarize(fixture_rows, "organiser_fixture (16 slices, 512x512)")
    training_summary = summarize(training_rows, "rare25_training (200 images, modal 637x512-ish)")

    with open(os.path.join(REPO_ROOT, "reports/h1_geometry_gap_summary.json"), "w") as fh:
        json.dump({"organiser_fixture": fixture_summary, "rare25_training": training_summary}, fh, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(fixture_summary, indent=2))
    print(json.dumps(training_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
