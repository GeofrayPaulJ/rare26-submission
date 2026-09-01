"""
13_artifact_followup.py -- Two follow-ups to 12_compression_check.py.

PART 1 ablates the AUC=0.700 class-prediction result from 12_compression_check.py by
refitting the identical logistic regression (same cv_auc/fold_r0 splits, imported from
12_compression_check.py -- not reimplemented) on three feature subsets, to separate
"compression artifact" indicators from "texture" indicators:
  A. blockiness_ratio, dct_comb_score            (compression-specific)
  B. unique_colour_frac, high_freq_energy_ratio  (texture/appearance)
  C. all four (control -- should reproduce 0.700)

PART 2 detects the black rectangular de-identification overlays seen in the Part-4
montage of 12_compression_check.py. These sit INSIDE the tissue FOV (unlike the
circular vignette) and, per the RARE25 paper, are a de-identification artifact absent
from the test data -- so anything a model learns from them is worthless at test time,
and they also corrupt any downstream pipeline that assumes the FOV is clean tissue
(the FOV circle fit, pHash patient-grouping, lesion-paste augmentation).

Detection reuses the already-fitted FOV geometry from the manifest (fov_centre_x/y,
fov_radius) -- no re-fitting. Connected components of near-black pixels
(brightness <= 10) *inside* that circle are found via scipy.ndimage.label (the same
library call used for the FOV fit itself in 05_fov_crop.py); components are kept only
if they are approximately rectangular (fill ratio against their own bounding box >=
--fill-ratio, default 0.9) and large enough (area >= --min-area, default 200px) --
this is what separates man-made rectangular overlays from organic dark shapes (lumen
openings, shadows), which are not fill-ratio >= 0.9 against their own bbox.

Read-only: 00_source and manifests/rare25_folds.csv are never modified; reads only
manifests/compression_check.csv and rare25_folds.csv. Deterministic (fixed
random_state everywhere), idempotent.

USAGE:
    python 13_artifact_followup.py
    python 13_artifact_followup.py --min-area 200 --fill-ratio 0.9 --black-thresh 10
"""

from __future__ import annotations

import argparse
import importlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
MANIFESTS = ROOT / "manifests"
SRC = ROOT / "00_source"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

cc12 = importlib.import_module("12_compression_check")  # reuse cv_auc(), not reimplemented

FEATURE_SETS = {
    "A_compression_specific": ["blockiness_ratio", "dct_comb_score"],
    "B_texture": ["unique_colour_frac", "high_freq_energy_ratio"],
    "C_all_four_control": cc12.INDICATORS,
}


# ============================= PART 1 =============================

def run_ablation(df: pd.DataFrame) -> dict[str, np.ndarray]:
    y = (df["class_label"] == "neoplasia").astype(int).to_numpy()
    results = {}
    for name, features in FEATURE_SETS.items():
        aucs = cc12.cv_auc(df, features, y, fold_col="fold_r0")
        results[name] = aucs
    return results


# ============================= PART 2 =============================

def detect_redactions(gray: np.ndarray, cx: float, cy: float, radius: float,
                       min_area: int, fill_ratio_thresh: float, black_thresh: int) -> list[dict]:
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    near_black = (gray <= black_thresh) & disk

    labeled, n = ndimage.label(near_black)
    if n == 0:
        return []
    sizes = ndimage.sum(near_black, labeled, index=range(1, n + 1))
    objects = ndimage.find_objects(labeled)

    boxes = []
    for lbl in range(1, n + 1):
        area = sizes[lbl - 1]
        if area < min_area:
            continue
        sl = objects[lbl - 1]
        y0, y1 = sl[0].start, sl[0].stop - 1
        x0, x1 = sl[1].start, sl[1].stop - 1
        bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
        fill_ratio = area / bbox_area
        if fill_ratio >= fill_ratio_thresh:
            boxes.append({
                "area": int(area), "fill_ratio": float(fill_ratio),
                "left": int(x0), "top": int(y0), "right": int(x1), "bottom": int(y1),
            })
    return boxes


def process_one(args: tuple) -> dict:
    filepath, cx, cy, radius, min_area, fill_ratio_thresh, black_thresh = args
    row = {
        "filepath": filepath, "has_redaction": False, "redaction_count": 0,
        "redaction_total_area": 0, "redaction_area_pct": 0.0,
        "largest_left": None, "largest_top": None, "largest_right": None, "largest_bottom": None,
        "largest_area": 0, "redaction_error": "",
    }
    try:
        with Image.open(SRC / filepath) as img:
            gray = np.asarray(img.convert("L")).astype(np.float64)
        boxes = detect_redactions(gray, cx, cy, radius, min_area, fill_ratio_thresh, black_thresh)
        if boxes:
            fov_area = np.pi * radius ** 2
            total_area = sum(b["area"] for b in boxes)
            largest = max(boxes, key=lambda b: b["area"])
            row.update(
                has_redaction=True, redaction_count=len(boxes),
                redaction_total_area=int(total_area),
                redaction_area_pct=100.0 * total_area / fov_area,
                largest_left=largest["left"], largest_top=largest["top"],
                largest_right=largest["right"], largest_bottom=largest["bottom"],
                largest_area=largest["area"],
            )
    except Exception as exc:  # noqa: BLE001
        row["redaction_error"] = str(exc)
    return row


def build_redaction_montage(df: pd.DataFrame, out_path: Path, patch_size: int = 220) -> None:
    subset = df[df["has_redaction"]].sort_values("largest_area", ascending=False).head(16).reset_index(drop=True)
    if len(subset) == 0:
        return
    cols, rows = 4, 4
    label_h, pad, header_h = 16, 4, 22
    cell_w, cell_h = patch_size + pad, patch_size + label_h + pad
    canvas = Image.new("RGB", (cols * cell_w + pad, header_h + rows * cell_h + pad), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 4), "16 largest detected redaction boxes", fill=(255, 255, 255), font=font)

    for idx, row in subset.iterrows():
        r, c = divmod(idx, cols)
        with Image.open(SRC / row["filepath"]) as img:
            img = img.convert("RGB")
            w, h = img.size
            bcx = (row["largest_left"] + row["largest_right"]) / 2
            bcy = (row["largest_top"] + row["largest_bottom"]) / 2
            left = int(np.clip(bcx - patch_size / 2, 0, max(0, w - patch_size)))
            top = int(np.clip(bcy - patch_size / 2, 0, max(0, h - patch_size)))
            patch = img.crop((left, top, left + patch_size, top + patch_size))
        x0 = pad + c * cell_w
        y0 = header_h + pad + r * cell_h
        canvas.paste(patch, (x0, y0))
        draw.text((x0, y0 + patch_size + 1), f"area={int(row['largest_area'])}px", fill=(255, 255, 0), font=font)

    canvas.save(out_path)


def chi_and_fisher(df: pd.DataFrame, group_col: str) -> None:
    ct = pd.crosstab(df["has_redaction"], df[group_col])
    print(f"\nCrosstab: has_redaction x {group_col}")
    print(ct.to_string())
    if ct.shape == (2, 2):
        chi2, p_chi, dof, _ = chi2_contingency(ct)
        _, p_fisher = fisher_exact(ct)
        print(f"  chi-square: chi2={chi2:.3f}, p={p_chi:.4g}    Fisher's exact: p={p_fisher:.4g}")
    else:
        chi2, p_chi, dof, _ = chi2_contingency(ct)
        print(f"  chi-square: chi2={chi2:.3f}, dof={dof}, p={p_chi:.4g}  "
              f"(Fisher's exact skipped -- table is not 2x2)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-area", type=int, default=200, help="Minimum connected-component area (px)")
    ap.add_argument("--fill-ratio", type=float, default=0.9, help="Min fill ratio vs own bounding box")
    ap.add_argument("--black-thresh", type=int, default=10, help="Brightness <= this counts as near-black")
    args = ap.parse_args()

    cc_path = MANIFESTS / "compression_check.csv"
    if not cc_path.exists():
        raise SystemExit(f"Missing {cc_path}. Run 12_compression_check.py first.")
    df = pd.read_csv(cc_path)  # read-only, never written back to
    clean = df.dropna(subset=cc12.INDICATORS).reset_index(drop=True)
    print(f"Loaded {len(df)} rows ({len(clean)} usable for the indicator-based analyses).")

    # ================= PART 1: ablation =================
    print("\n" + "=" * 70)
    print("PART 1 -- ablating the AUC=0.700 class-prediction result")
    print("=" * 70)
    ablation = run_ablation(clean)
    for name, features in FEATURE_SETS.items():
        aucs = ablation[name]
        print(f"\n{name}  features={features}")
        print(f"  fold AUCs: {np.round(aucs, 4).tolist()}")
        print(f"  mean AUC = {aucs.mean():.4f}  (std={aucs.std():.4f}, "
              f"range=[{aucs.min():.4f}, {aucs.max():.4f}])")

    auc_a = ablation["A_compression_specific"].mean()
    auc_b = ablation["B_texture"].mean()
    auc_c = ablation["C_all_four_control"].mean()
    print(f"\nControl (C) reproduced {auc_c:.4f} (12_compression_check.py reported 0.700).")
    print("\nInterpretation:")
    if auc_a > 0.6:
        print(f"  Compression-specific features (A) alone reach AUC={auc_a:.3f} -- MEANINGFULLY "
              f"ABOVE 0.5. Compression artifacts carry class information; this is a real shortcut.")
    else:
        print(f"  Compression-specific features (A) alone reach AUC={auc_a:.3f} -- near chance.")
    if auc_b > 0.6:
        print(f"  Texture features (B) alone reach AUC={auc_b:.3f} -- most of the signal in (C) is "
              f"explained by texture/appearance, not compression artifacts. Neoplasia legitimately "
              f"looks more heterogeneous than flat normal mucosa -- if (A) is near chance, this is "
              f"real biology, not a leak.")
    else:
        print(f"  Texture features (B) alone reach AUC={auc_b:.3f}.")

    # ================= PART 2: redaction detection =================
    print("\n" + "=" * 70)
    print("PART 2 -- detecting black rectangular de-identification overlays")
    print("=" * 70)
    jobs = [
        (row.filepath, row.fov_centre_x, row.fov_centre_y, row.fov_radius,
         args.min_area, args.fill_ratio, args.black_thresh)
        for row in df.itertuples()
    ]
    print(f"Scanning {len(jobs)} images for rectangular near-black regions inside the fitted FOV "
          f"(area>={args.min_area}px, fill_ratio>={args.fill_ratio}, brightness<={args.black_thresh})...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in tqdm(pool.map(process_one, jobs, chunksize=32), total=len(jobs)):
            results.append(res)
    red_df = pd.DataFrame(results)
    full = df.merge(red_df, on="filepath", how="left")

    n_errors = int((full["redaction_error"].fillna("") != "").sum())
    if n_errors:
        print(f"WARNING: {n_errors} image(s) failed redaction scanning.")

    n_affected = int(full["has_redaction"].sum())
    pct_affected = 100.0 * n_affected / len(full)
    print(f"\nImages with at least one detected redaction box: {n_affected} / {len(full)} ({pct_affected:.2f}%)")
    if n_affected:
        affected = full[full["has_redaction"]]
        print(f"  redaction_count: mean={affected['redaction_count'].mean():.2f}, "
              f"max={affected['redaction_count'].max()}")
        print(f"  redaction_area_pct of FOV: mean={affected['redaction_area_pct'].mean():.3f}%, "
              f"median={affected['redaction_area_pct'].median():.3f}%, "
              f"max={affected['redaction_area_pct'].max():.3f}%")

    print("\n--- Cross-tabs ---")
    chi_and_fisher(full, "class_label")
    chi_and_fisher(full, "centre")

    print("\n--- Fold distribution (should NOT cluster) ---")
    chi_and_fisher(full, "fold_r0")

    print("\n--- Refit Part 1 logistic regression with redaction features added ---")
    clean_full = full.dropna(subset=cc12.INDICATORS).reset_index(drop=True)
    y_class = (clean_full["class_label"] == "neoplasia").astype(int).to_numpy()
    features_plus = cc12.INDICATORS + ["has_redaction", "redaction_area_pct"]
    clean_full["has_redaction"] = clean_full["has_redaction"].astype(int)
    aucs_plus = cc12.cv_auc(clean_full, features_plus, y_class, fold_col="fold_r0")
    print(f"Features = {features_plus}")
    print(f"  fold AUCs: {np.round(aucs_plus, 4).tolist()}")
    print(f"  mean AUC = {aucs_plus.mean():.4f}  (std={aucs_plus.std():.4f})")
    print(f"  (Part 1 control C was {auc_c:.4f} without redaction features)")
    if aucs_plus.mean() - auc_c > 0.03:
        print(f"  -> AUC jumped materially ({auc_c:.3f} -> {aucs_plus.mean():.3f}): redaction "
              f"presence itself predicts class label. This is a genuine shortcut, and per the "
              f"RARE25 paper these overlays are absent from test data -- worthless there.")
    else:
        print(f"  -> No material jump. Redaction presence/area does not add class-predictive signal "
              f"beyond the four indicators.")

    print("\n--- Do redacted images disproportionately fall into larger pHash groups? ---")
    group_sizes = full.groupby("group_id").size()
    full["group_size"] = full["group_id"].map(group_sizes)
    red_sizes = full.loc[full["has_redaction"], "group_size"]
    clean_sizes = full.loc[~full["has_redaction"], "group_size"]
    print(f"  group_size for redacted images:     mean={red_sizes.mean():.2f}, median={red_sizes.median():.1f}")
    print(f"  group_size for non-redacted images: mean={clean_sizes.mean():.2f}, median={clean_sizes.median():.1f}")
    if len(red_sizes) >= 2 and len(clean_sizes) >= 2:
        stat, p = mannwhitneyu(red_sizes, clean_sizes, alternative="two-sided")
        print(f"  Mann-Whitney U: stat={stat:.1f}, p={p:.4g}")
    top_groups = group_sizes.sort_values(ascending=False).head(10).index
    overall_rate = full["has_redaction"].mean()
    top_rate = full.loc[full["group_id"].isin(top_groups), "has_redaction"].mean()
    print(f"  overall redaction rate: {overall_rate:.4f}   "
          f"redaction rate within the 10 largest pHash groups: {top_rate:.4f}"
          + ("  <-- ELEVATED, boxes may be linking unrelated images" if top_rate > 2 * overall_rate else ""))

    # ================= montage =================
    montage_path = MANIFESTS / "redaction_check_top16.png"
    if n_affected:
        build_redaction_montage(full, montage_path)
        print(f"\nSaved: {montage_path}")
    else:
        print("\nNo redactions detected -- skipping montage.")

    # ================= save =================
    out_csv = MANIFESTS / "redaction_check.csv"
    full.to_csv(out_csv, index=False)
    print(f"\nWritten: {out_csv}")

    # ================= summary =================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Compression-specific features alone predict class: "
          f"{'YES' if auc_a > 0.6 else 'NO'} (AUC={auc_a:.3f})")
    print(f"Texture features alone predict class: {'YES' if auc_b > 0.6 else 'NO'} (AUC={auc_b:.3f})")
    print(f"Images with detected redaction overlays: {n_affected}/{len(full)} ({pct_affected:.2f}%)")
    print(f"Redaction presence adds class-predictive signal beyond the 4 indicators: "
          f"{'YES' if aucs_plus.mean() - auc_c > 0.03 else 'NO'} "
          f"({auc_c:.3f} -> {aucs_plus.mean():.3f})")


if __name__ == "__main__":
    main()
