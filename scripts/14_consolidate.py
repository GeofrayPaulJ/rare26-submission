"""
14_consolidate.py -- Final data-preparation cleanup: four review jobs, then one
canonical manifest.

JOB 1: resolves the 5 exact-duplicate groups from manifests/exact_duplicates.csv
(sha256 collisions found in Phase 2, never acted on) -- checks they each landed in a
single pHash group_id and don't span folds, assigns keep_for_training (first file per
group by filepath = True, rest = False), and saves a review montage.

JOB 2: the 12 pHash groups containing both class labels are the highest-value leak
catch in the whole pipeline (same patient with/without a lesion in frame). Lists them,
flags any spanning both hospitals (implausible for one patient -- signals a false
grouping from pHash chaining rather than genuine same-patient redundancy), confirms
fold intactness, and saves a per-group review montage.

JOB 3: montages the 28 images with fit_quality < 0.9 from 05_fov_crop.py, with the
fitted circle drawn as an overlay, so the fit failure mode is visible.

JOB 4: merges rare25_folds.csv + compression_check.csv (4 continuous indicators) +
redaction_check.csv (redaction fields) + rare25_folds_with_mode.csv (rg_ratio,
mean_r/g/b -- keeping these as a valid hospital-confound diagnostic while dropping
imaging_mode, which 09_imaging_mode.py showed was spurious: the R/G-ratio histogram is
unimodal and never drops below 1.0 anywhere in the dataset, so there is no real
white-light/NBI split to encode) + keep_for_training from Job 1, into ONE canonical
manifest, asserting 3095 rows and zero nulls anywhere in the result. Writes
manifests/rare25_canonical.{csv,parquet} and manifests/DATA_README.md.

Read-only: 00_source is never modified; no existing manifest is overwritten (only new
files are written: review_duplicates.png, review_mixed/*.png, review_fov_lowfit_*.png,
rare25_canonical.{csv,parquet}, DATA_README.md). Deterministic and idempotent.

USAGE:
    python 14_consolidate.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "manifests"
SRC = ROOT / "00_source"

FOLD_COLS = [f"fold_r{i}" for i in range(10)]
FONT = ImageFont.load_default()


# ============================= montage helpers =============================

def _load_thumbnail(abs_path: Path, target_w: int) -> Image.Image:
    img = Image.open(abs_path).convert("RGB")
    w, h = img.size
    scale = target_w / w
    return img.resize((target_w, max(1, int(h * scale))), Image.LANCZOS)


def _load_annotated_thumbnail(abs_path: Path, cx: float, cy: float, radius: float,
                               target_w: int, outline=(0, 255, 255), width=4) -> Image.Image:
    img = Image.open(abs_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=outline, width=width)
    w, h = img.size
    scale = target_w / w
    return img.resize((target_w, max(1, int(h * scale))), Image.LANCZOS)


def choose_grid(n: int, max_tiles: int = 48) -> tuple[int, int, int]:
    """Returns (cols, thumb_width, n_shown) -- caps huge groups to a sane sheet size."""
    shown = min(n, max_tiles)
    if shown <= 4:
        return shown, 240, shown
    if shown <= 12:
        return 4, 200, shown
    if shown <= 24:
        return 6, 150, shown
    return 8, 110, shown


def build_grid_montage(tiles: list[dict], cols: int, thumb_w: int, out_path: Path, title: str = "") -> None:
    """tiles: list of {'abs_path', 'caption': [lines], optional 'circle': (cx,cy,r)}"""
    line_h = 13
    max_lines = max((len(t["caption"]) for t in tiles), default=2)
    label_h, pad, header_h = max_lines * line_h + 4, 4, 22
    cell_w, cell_h = thumb_w + pad, int(thumb_w * 0.85) + label_h + pad
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (cols * cell_w + pad, header_h + rows * cell_h + pad), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 4), title, fill=(255, 255, 255), font=FONT)

    for idx, t in enumerate(tiles):
        r, c = divmod(idx, cols)
        if "circle" in t:
            cx, cy, rad = t["circle"]
            thumb = _load_annotated_thumbnail(t["abs_path"], cx, cy, rad, thumb_w)
        else:
            thumb = _load_thumbnail(t["abs_path"], thumb_w)
        x0 = pad + c * cell_w
        y0 = header_h + pad + r * cell_h
        canvas.paste(thumb, (x0, y0))
        for li, line in enumerate(t["caption"]):
            draw.text((x0, y0 + thumb.size[1] + 2 + li * line_h), line, fill=(255, 255, 0), font=FONT)

    canvas.save(out_path)


# ============================= JOB 1 =============================

def job1_resolve_duplicates(folds: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("JOB 1 -- resolving the 5 exact-duplicate groups")
    print("=" * 70)

    dup_path = MANIFESTS / "exact_duplicates.csv"
    dup = pd.read_csv(dup_path)
    merged = dup.merge(folds[["filepath", "group_id"] + FOLD_COLS], on="filepath", how="left")

    print(f"\n{dup['duplicate_group_id'].nunique()} duplicate groups, {len(dup)} files.")

    print("\n--- pHash group_id consistency (each duplicate group must land in ONE group_id) ---")
    grouping_failures = []
    for gid, sub in merged.groupby("duplicate_group_id"):
        distinct = sub["group_id"].nunique()
        status = "OK" if distinct == 1 else "*** GROUPING FAILURE ***"
        print(f"  duplicate_group {gid}: pHash group_id(s)={sorted(sub['group_id'].unique().tolist())}  {status}")
        if distinct != 1:
            grouping_failures.append(gid)
    if grouping_failures:
        print(f"\n*** LOUD WARNING: duplicate group(s) {grouping_failures} span multiple pHash "
              f"group_ids. Identical files MUST group together by construction -- this indicates "
              f"a bug in the grouping pipeline (06_regroup.py) and needs investigation. ***")
    else:
        print("\nAll 5 duplicate groups land in a single pHash group_id each. OK.")

    print("\n--- Fold-spanning check (a duplicate group must not split across folds in any repeat) ---")
    fold_failures = []
    for gid, sub in merged.groupby("duplicate_group_id"):
        bad = [c for c in FOLD_COLS if sub[c].nunique() > 1]
        if bad:
            fold_failures.append((gid, bad))
            print(f"  duplicate_group {gid}: *** SPANS FOLDS in {bad} ***")
        else:
            print(f"  duplicate_group {gid}: intact in every repeat. OK.")
    if not fold_failures:
        print("\nNo duplicate group spans folds in any repeat. OK.")

    # keep_for_training: first file per group (sorted by filepath) = True, rest = False.
    keep = pd.DataFrame({"filepath": folds["filepath"], "keep_for_training": True})
    keep = keep.set_index("filepath")
    for gid, sub in dup.groupby("duplicate_group_id"):
        ordered = sub.sort_values("filepath")["filepath"].tolist()
        for fp in ordered[1:]:
            keep.loc[fp, "keep_for_training"] = False
    keep = keep.reset_index()

    n_false = int((~keep["keep_for_training"]).sum())
    print(f"\nkeep_for_training: {n_false} row(s) marked False (redundant duplicates), "
          f"{len(keep) - n_false} marked True.")
    for fp in keep.loc[~keep["keep_for_training"], "filepath"]:
        print(f"  False: {fp}")

    # Montage: all 12 duplicate files, 4x3 grid, labelled sha256 prefix + filepath.
    tiles = []
    for _, row in dup.sort_values(["duplicate_group_id", "filepath"]).iterrows():
        tiles.append({
            "abs_path": SRC / row["filepath"],
            "caption": [f"grp{row['duplicate_group_id']} {row['sha256_hash'][:10]}...",
                        f"{row['centre']}/.../{Path(row['filepath']).name[:18]}.."],
        })
    out_path = MANIFESTS / "review_duplicates.png"
    build_grid_montage(tiles, cols=4, thumb_w=200, out_path=out_path,
                        title="12 exact-duplicate files (5 sha256 groups)")
    print(f"\nSaved: {out_path}")

    return keep


# ============================= JOB 2 =============================

def job2_inspect_mixed_groups(folds: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("JOB 2 -- inspecting the 12 mixed-class pHash groups")
    print("=" * 70)

    class_counts = folds.groupby("group_id")["class_label"].nunique()
    mixed_ids = sorted(class_counts[class_counts > 1].index.tolist())
    print(f"\n{len(mixed_ids)} mixed-class groups found "
          f"({'matches' if len(mixed_ids) == 12 else 'DOES NOT MATCH'} the expected 12).")

    out_dir = MANIFESTS / "review_mixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    both_hospital_ids = []
    fold_split_ids = []

    for gid in mixed_ids:
        sub = folds[folds["group_id"] == gid].sort_values("filepath")
        n = len(sub)
        class_breakdown = sub["class_label"].value_counts().to_dict()
        hospitals = sorted(sub["centre"].unique().tolist())
        spans_both = len(hospitals) > 1
        if spans_both:
            both_hospital_ids.append(gid)

        bad_repeats = [c for c in FOLD_COLS if sub[c].nunique() > 1]
        if bad_repeats:
            fold_split_ids.append((gid, bad_repeats))

        print(f"\ngroup_id={gid}  n={n}  class={class_breakdown}  hospitals={hospitals}"
              + ("  *** SPANS BOTH HOSPITALS -- implausible for one patient, likely false grouping ***"
                 if spans_both else "")
              + (f"  *** SPLIT ACROSS FOLDS: {bad_repeats} ***" if bad_repeats else ""))
        filepaths = sub["filepath"].tolist()
        for fp in filepaths[:10]:
            print(f"    {fp}")
        if len(filepaths) > 10:
            print(f"    ... and {len(filepaths) - 10} more (full list in the montage / canonical manifest)")

        cols, thumb_w, n_shown = choose_grid(n)
        shown = sub.head(n_shown)
        tiles = [
            {"abs_path": SRC / row["filepath"],
             "caption": [f"{row['class_label']} / {row['centre']}",
                         f"{Path(row['filepath']).name[:20]}.."]}
            for _, row in shown.iterrows()
        ]
        title = f"group {gid}  n={n}" + (f" (showing {n_shown})" if n_shown < n else "")
        if spans_both:
            title += "  -- SPANS BOTH HOSPITALS"
        build_grid_montage(tiles, cols=cols, thumb_w=thumb_w, out_path=out_dir / f"group_{gid}.png", title=title)

    print(f"\n--- Summary ---")
    print(f"Groups spanning both hospitals: {len(both_hospital_ids)} / {len(mixed_ids)}  {both_hospital_ids}")
    if both_hospital_ids:
        print("*** These groupings are almost certainly pHash chaining artifacts, not genuine "
              "same-patient photos -- a single patient cannot appear at two hospitals in this "
              "dataset. Treat these specific groups' pairing as unreliable. ***")
    if fold_split_ids:
        print(f"*** LEAK: {len(fold_split_ids)} mixed group(s) split across folds: {fold_split_ids} ***")
    else:
        print("All 12 mixed groups are intact within every fold repeat. OK -- the leak these "
              "groupings exist to prevent is not occurring.")
    print(f"\nSaved {len(mixed_ids)} montage(s) -> {out_dir}\\group_<id>.png")


# ============================= JOB 3 =============================

def job3_montage_low_fit(folds: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("JOB 3 -- montaging low fit_quality (<0.9) FOV fits")
    print("=" * 70)

    low = folds[folds["fit_quality"] < 0.9].sort_values("fit_quality")
    print(f"\n{len(low)} image(s) with fit_quality < 0.9.")

    rows_per_sheet = 16
    n_sheets = math.ceil(len(low) / rows_per_sheet)
    for sheet_idx in range(n_sheets):
        chunk = low.iloc[sheet_idx * rows_per_sheet: (sheet_idx + 1) * rows_per_sheet]
        tiles = []
        for _, row in chunk.iterrows():
            tiles.append({
                "abs_path": SRC / row["filepath"],
                "circle": (row["fov_centre_x"], row["fov_centre_y"], row["fov_radius"]),
                "caption": [
                    f"fq={row['fit_quality']:.3f} r={row['fov_radius']:.0f}px",
                    f"{row['centre']}/.../{Path(row['filepath']).name[:16]}..",
                ],
            })
        out_path = MANIFESTS / f"review_fov_lowfit_{sheet_idx + 1:02d}.png"
        build_grid_montage(tiles, cols=4, thumb_w=220, out_path=out_path,
                            title=f"Low fit_quality FOV fits, sheet {sheet_idx + 1}/{n_sheets}")
        print(f"Saved: {out_path}  ({len(chunk)} images)")


# ============================= JOB 4 =============================

BASE_COLS = (
    ["filepath", "centre", "class_label", "sha256_hash", "width", "height", "color_mode",
     "file_size_bytes", "image_format"]
    + ["fov_centre_x", "fov_centre_y", "fov_radius", "fov_area_pct", "fit_quality",
       "inner_left", "inner_top", "inner_right", "inner_bottom"]
    + ["group_id"] + FOLD_COLS + ["holdout_center_1", "holdout_center_2"]
)
COMPRESSION_COLS = ["filepath", "blockiness_ratio", "unique_colour_frac", "high_freq_energy_ratio", "dct_comb_score"]
REDACTION_COLS = ["filepath", "has_redaction", "redaction_count", "redaction_area_pct",
                   "largest_left", "largest_top", "largest_right", "largest_bottom"]
MODE_COLS = ["filepath", "mean_r", "mean_g", "mean_b", "rg_ratio"]

COLUMN_DOCS = {
    "filepath": ("00_locate_and_verify.py / 01_build_manifest.py", "Path to the image, relative to 00_source/."),
    "centre": ("01_build_manifest.py", "Source hospital subfolder: center_1 or center_2."),
    "class_label": ("01_build_manifest.py", "neoplasia or non-dysplastic, inferred from folder structure."),
    "sha256_hash": ("01_build_manifest.py", "Full-file SHA-256, used for exact-duplicate detection."),
    "width": ("01_build_manifest.py", "Image width in pixels."),
    "height": ("01_build_manifest.py", "Image height in pixels."),
    "color_mode": ("01_build_manifest.py", "PIL colour mode (all RGB in this dataset)."),
    "file_size_bytes": ("01_build_manifest.py", "Raw PNG file size. DIAGNOSTIC / CONFOUND: near-perfectly "
                         "separates hospital (center_1 max=554,475B, center_2 min=722,621B -- disjoint ranges)."),
    "image_format": ("01_build_manifest.py", "PIL-detected format (PNG for all images)."),
    "fov_centre_x": ("05_fov_crop.py", "Fitted field-of-view circle centre, x (raw image pixel coords)."),
    "fov_centre_y": ("05_fov_crop.py", "Fitted field-of-view circle centre, y (raw image pixel coords)."),
    "fov_radius": ("05_fov_crop.py", "Fitted field-of-view circle radius in pixels (centroid + "
                    "area-derived: r=sqrt(area/pi), chosen over cv2.minEnclosingCircle as more robust "
                    "to boundary noise on this data -- see script docstring)."),
    "fov_area_pct": ("05_fov_crop.py", "FOV connected-component area as a percentage of the full frame."),
    "fit_quality": ("05_fov_crop.py", "Fraction of the fitted circle's own disk area that is actually "
                     "non-black. DIAGNOSTIC: <0.9 flags a poor/partial FOV fit -- see review_fov_lowfit_*.png "
                     "(28 such images)."),
    "inner_left": ("05_fov_crop.py", "Left edge of the largest axis-aligned square inscribed in the FOV circle."),
    "inner_top": ("05_fov_crop.py", "Top edge of the inscribed square."),
    "inner_right": ("05_fov_crop.py", "Right edge of the inscribed square."),
    "inner_bottom": ("05_fov_crop.py", "Bottom edge of the inscribed square -- this box is what 06_regroup.py "
                      "crops to (in memory) before hashing."),
    "group_id": ("06_regroup.py", "Near-duplicate / probable-same-patient group, via mutual-KNN pHash "
                  "clustering on the inner-square crop (threshold=12, chosen as the largest threshold "
                  "keeping the biggest group under 20% of the data). THIS is the column CV folds are "
                  "grouped on -- there is no real patient ID in this dataset's public layout. "
                  "See review_mixed/ for the 12 groups spanning both class labels, 8 of which "
                  "implausibly span both hospitals and are likely false groupings from pHash chaining, "
                  "not genuine same-patient photos."),
    **{f"fold_r{i}": ("07_folds.py", f"Validation-fold index (0-4) for repeat {i} of grouped, "
                        f"class-stratified 5-fold CV. Train on rows where this != k, validate where == k.")
       for i in range(10)},
    "holdout_center_1": ("07_folds.py", "'test' if centre==center_1 else 'train' -- hospital-holdout split."),
    "holdout_center_2": ("07_folds.py", "'test' if centre==center_2 else 'train' -- hospital-holdout split."),
    "blockiness_ratio": ("12_compression_check.py", "DIAGNOSTIC, not a training feature. Mean |pixel diff| "
                          "across 8x8 JPEG-grid boundaries (aligned to the raw image's absolute pixel grid) "
                          "over the same at non-boundary positions, inside the FOV. ~1.0 = no evidence of "
                          "block quantisation. Confirmed via 13_artifact_followup.py Part 1 that this "
                          "indicator alone does NOT predict class (AUC=0.563, near chance)."),
    "unique_colour_frac": ("12_compression_check.py", "DIAGNOSTIC. Distinct RGB values inside the FOV as a "
                            "fraction of sampled pixels. Part of the texture signal that DOES predict class "
                            "(AUC=0.700 combined with high_freq_energy_ratio) -- attributed to genuine tissue "
                            "heterogeneity (neoplasia looks more varied than flat mucosa), not a compression "
                            "artifact, since the compression-specific indicators alone do not predict class."),
    "high_freq_energy_ratio": ("12_compression_check.py", "DIAGNOSTIC. FFT power fraction above 0.5x Nyquist "
                                "inside the FOV. See unique_colour_frac note -- same texture-signal finding."),
    "dct_comb_score": ("12_compression_check.py", "DIAGNOSTIC, not a training feature. Comb-likeness of the "
                        "per-block DCT AC(0,1) coefficient histogram (evidence of quantisation). Confirmed "
                        "not class-predictive on its own (grouped with blockiness_ratio, AUC=0.563)."),
    "has_redaction": ("13_artifact_followup.py", "Whether a rectangular near-black de-identification overlay "
                       "was detected inside the FOV. IMPORTANT CONFOUND: present in 24.7% of images overall, "
                       "but 54.4% of center_2 images vs only 14.1% of center_1 (chi2 p=~1e-116) -- a near-"
                       "diagnostic hospital fingerprint. Also significantly class-associated (35.4% of "
                       "neoplasia vs 24.1% of non-dysplastic, p=0.0018), though adding it to the class "
                       "logistic regression gave no material AUC lift over the 4 continuous indicators alone. "
                       "Per the RARE25 paper these overlays are absent from test data."),
    "redaction_count": ("13_artifact_followup.py", "Number of detected redaction boxes in the image (0 if none)."),
    "redaction_area_pct": ("13_artifact_followup.py", "Total redaction box area as a percentage of the FOV area."),
    "largest_left": ("13_artifact_followup.py", "Bounding-box left edge of the largest detected redaction box "
                      "(raw image pixel coords). NULL when has_redaction is False (no box detected) -- this "
                      "is expected, not a missing-data problem; see the null-check note in this README."),
    "largest_top": ("13_artifact_followup.py", "Bounding-box top edge of the largest detected redaction box. "
                     "NULL when has_redaction is False."),
    "largest_right": ("13_artifact_followup.py", "Bounding-box right edge of the largest detected redaction box. "
                       "NULL when has_redaction is False."),
    "largest_bottom": ("13_artifact_followup.py", "Bounding-box bottom edge of the largest detected redaction box. "
                        "NULL when has_redaction is False."),
    "mean_r": ("09_imaging_mode.py", "DIAGNOSTIC. Mean red channel inside 0.8x FOV radius, excluding near-"
                "black/blown-out pixels."),
    "mean_g": ("09_imaging_mode.py", "DIAGNOSTIC. Mean green channel, same sampling as mean_r."),
    "mean_b": ("09_imaging_mode.py", "DIAGNOSTIC. Mean blue channel, same sampling as mean_r."),
    "rg_ratio": ("09_imaging_mode.py", "DIAGNOSTIC / CONFOUND. mean_r/mean_g. KEPT deliberately as a hospital-"
                  "confound diagnostic (center_1 mean=1.94 vs center_2 mean=1.76, overlapping but "
                  "distinct -- consistent with per-hospital colour calibration/white-balance differences). "
                  "The imaging_mode column derived from this in 09_imaging_mode.py was found SPURIOUS and is "
                  "deliberately excluded from this file -- see the note below."),
    "keep_for_training": ("14_consolidate.py (Job 1)", "False for 7 of the 12 files involved in the 5 exact-"
                            "duplicate (sha256) groups found by 02_duplicate_audit.py (the redundant copies; "
                            "one file per group, chosen by sorted filepath, keeps True). True for all other "
                            "3088 images. No files were deleted from disk -- filter training code on this "
                            "column instead."),
}


def job4_consolidate(folds: pd.DataFrame, keep_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("JOB 4 -- consolidating into one canonical manifest")
    print("=" * 70)

    compression = pd.read_csv(MANIFESTS / "compression_check.csv")[COMPRESSION_COLS]
    redaction = pd.read_csv(MANIFESTS / "redaction_check.csv")[REDACTION_COLS]
    mode = pd.read_csv(MANIFESTS / "rare25_folds_with_mode.csv")[MODE_COLS]

    base = folds[BASE_COLS].copy()
    canonical = (
        base
        .merge(compression, on="filepath", how="left")
        .merge(redaction, on="filepath", how="left")
        .merge(mode, on="filepath", how="left")
        .merge(keep_df, on="filepath", how="left")
    )

    print(f"\nRow count: {len(canonical)}")
    assert len(canonical) == 3095, f"FAIL: expected 3095 rows, got {len(canonical)}"
    print("Row count assertion: OK (3095)")

    # largest_left/top/right/bottom are legitimately null when has_redaction is False (no box ->
    # no bbox) -- that is correct "not applicable" semantics, not a join failure. Every other
    # column, including filepath (the join key) and has_redaction/redaction_count/redaction_area_pct
    # themselves, must be fully populated for every row; a null anywhere else means a merge dropped
    # or failed to match a row.
    expected_null_cols = {"largest_left", "largest_top", "largest_right", "largest_bottom"}
    null_counts = canonical.isna().sum()
    unexpected = null_counts[(null_counts > 0) & (~null_counts.index.isin(expected_null_cols))]
    if len(unexpected):
        print("\n*** FAIL: unexpected nulls found after merge -- a join key mismatch is likely: ***")
        print(unexpected.to_string())
        raise SystemExit("Aborting: canonical manifest has unexpected nulls. See columns listed above.")

    n_no_box = int((~canonical["has_redaction"]).sum())
    bbox_nulls = null_counts[null_counts.index.isin(expected_null_cols)]
    assert (bbox_nulls == n_no_box).all(), (
        f"FAIL: largest_* null count ({bbox_nulls.tolist()}) does not match "
        f"has_redaction==False count ({n_no_box}) -- bbox nulls should exactly track no-redaction rows."
    )
    print(f"Null-check assertion: OK. Zero unexpected nulls (join keys and all non-bbox columns fully "
          f"populated for all 3095 rows). largest_left/top/right/bottom are null for exactly the "
          f"{n_no_box} rows with has_redaction=False, as expected.")

    csv_path = MANIFESTS / "rare25_canonical.csv"
    parquet_path = MANIFESTS / "rare25_canonical.parquet"
    canonical.to_csv(csv_path, index=False)
    canonical.to_parquet(parquet_path, index=False)
    print(f"\nWritten: {csv_path}")
    print(f"Written: {parquet_path}")
    print(f"Columns ({len(canonical.columns)}): {list(canonical.columns)}")

    write_data_readme(canonical)
    return canonical


def write_data_readme(canonical: pd.DataFrame) -> None:
    lines = []
    a = lines.append
    a("# RARE25 canonical manifest -- data dictionary\n")
    a(f"`rare25_canonical.csv` / `.parquet` -- {len(canonical)} rows, one per training image. "
      f"Built by `14_consolidate.py` from `rare25_folds.csv`, `compression_check.csv`, "
      f"`redaction_check.csv`, and `rare25_folds_with_mode.csv`. `00_source/` and every existing "
      f"manifest are read-only inputs; nothing upstream was modified to build this file.\n")
    n_no_box = int((~canonical["has_redaction"]).sum())
    a(f"**Null check**: every column is fully populated for all {len(canonical)} rows EXCEPT "
      f"`largest_left`/`largest_top`/`largest_right`/`largest_bottom`, which are null for exactly "
      f"the {n_no_box} rows where `has_redaction` is False (no box detected -> no bbox to report). "
      f"That is expected \"not applicable\" semantics, not missing data -- the build script asserts "
      f"this null count matches `has_redaction==False` exactly and fails loudly on any other null.\n")

    a("## Columns\n")
    a("| column | produced by | description |")
    a("|---|---|---|")
    for col in canonical.columns:
        script, desc = COLUMN_DOCS.get(col, ("?", "undocumented"))
        desc_escaped = desc.replace("|", "\\|")
        a(f"| `{col}` | {script} | {desc_escaped} |")

    a("\n## Diagnostics vs. training features\n")
    a("Columns marked DIAGNOSTIC above are measurements produced to audit the dataset for shortcuts "
      "and confounds -- they are not intended as model input features. Specifically:\n")
    a("- `blockiness_ratio`, `unique_colour_frac`, `high_freq_energy_ratio`, `dct_comb_score` -- the "
      "four compression/texture indicators from `12_compression_check.py`. Together they predict "
      "class label at AUC=0.700 (grouped 5-fold CV), but `13_artifact_followup.py` showed this is "
      "driven entirely by the two texture indicators (AUC=0.700 alone) while the two compression-"
      "specific indicators are near chance (AUC=0.563) -- i.e. this reflects genuine tissue "
      "heterogeneity in neoplasia, not a compression-artifact shortcut. Safe to leave out of a "
      "classifier; useful for auditing new data splits.")
    a("- `mean_r`, `mean_g`, `mean_b`, `rg_ratio` -- retained specifically as a hospital-confound "
      "diagnostic (see Known Confounds below), not because they carry validated clinical signal.")
    a("- `has_redaction`, `redaction_count`, `redaction_area_pct`, `largest_*` -- de-identification "
      "artifact detection, not tissue signal. `has_redaction` is significantly associated with both "
      "class and (much more strongly) hospital -- see below.")

    a("\n## Known confounds (verified, not assumed)\n")
    a("Hospital (`centre`) is recoverable from several columns that have nothing to do with tissue "
      "appearance -- any model given raw access to these can learn \"which hospital\" instead of "
      "\"is there cancer\":\n")
    a("- **`file_size_bytes`** -- near-perfectly separates hospital: center_1 max=554,475B vs "
      "center_2 min=722,621B (disjoint ranges, verified on the full 3095-row set).")
    a("- **`width`/`height`** -- differ in mean (center_1 width mean=632, center_2 mean=617) with "
      "partial range overlap.")
    a("- **`rg_ratio`/`mean_r`/`mean_g`/`mean_b`** -- differ in mean by hospital (rg_ratio "
      "center_1=1.94 vs center_2=1.76), consistent with per-hospital colour calibration/white "
      "balance, not a clinical difference.")
    a("- **`has_redaction`** -- 54.4% of center_2 images carry a redaction overlay vs 14.1% of "
      "center_1 (chi-square p~1e-116). This is the strongest hospital fingerprint of the four.")
    a("\n`group_id` deserves its own caution: 12 groups mix both class labels (see "
      "`manifests/review_mixed/`), and 8 of those 12 implausibly span both hospitals -- a single "
      "patient cannot be treated at two centres in this dataset, so those 8 groupings are most "
      "likely pHash chaining artifacts, not genuine same-patient photos. One group (`group_id` "
      "with 262 members) is almost certainly dominated by chaining rather than true redundancy -- "
      "see `03_near_duplicates.py`/`06_regroup.py` percolation analysis in earlier session notes.\n")

    a("## imaging_mode: tested and found spurious -- do not re-derive it\n")
    a("`09_imaging_mode.py` split images into WLE/NBI using Otsu's method on `rg_ratio`. That split "
      "is **not real**: the R/G-ratio histogram is unimodal (one smooth hump), and R/G ratio never "
      "drops below 1.0 anywhere in the 3095-image dataset (min=1.16), whereas genuine narrow-band "
      "imaging has the red channel almost entirely filtered out (R/G well below 1, typically "
      "0.3-0.7). Visual inspection of the single lowest-R/G image (the one the script was most "
      "confident was \"NBI\") confirmed it is plainly pink/tan white-light tissue. The mode split "
      "correlated far more with hospital than with anything resembling an optical filter change. "
      "**`imaging_mode` is deliberately excluded from this canonical file.** `rg_ratio`/`mean_r/g/b` "
      "are kept, but only as the hospital-confound diagnostic described above -- not as a modality "
      "label. If you find yourself wanting a WLE/NBI split again, re-read this note first.\n")

    (MANIFESTS / "DATA_README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {MANIFESTS / 'DATA_README.md'}")


# ============================= main =============================

def main() -> None:
    folds_path = MANIFESTS / "rare25_folds.csv"
    if not folds_path.exists():
        raise SystemExit(f"Missing {folds_path}. Run the earlier pipeline first.")
    folds = pd.read_csv(folds_path)  # read-only, never written back to

    keep_df = job1_resolve_duplicates(folds)
    job2_inspect_mixed_groups(folds)
    job3_montage_low_fit(folds)
    canonical = job4_consolidate(folds, keep_df)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY -- what still needs human eyes")
    print("=" * 70)
    print("1. manifests/review_duplicates.png -- confirm the 5 sha256-duplicate groups are true "
          "duplicates, not hash collisions, before trusting keep_for_training=False on those 7 rows.")
    print("2. manifests/review_mixed/group_*.png -- 12 groups, ESPECIALLY the 8 spanning both "
          "hospitals (implausible for one patient -- likely false pHash-chaining groupings). Decide "
          "whether to break those 8 groups apart before the next fold rebuild.")
    print("3. manifests/review_fov_lowfit_01.png / _02.png -- 28 images with a questionable FOV "
          "circle fit; check whether any need the fit redone or should be excluded.")
    print("4. rare25_canonical.csv/.parquet -- ready to use, but has_redaction/rg_ratio/"
          "file_size_bytes are hospital confounds; do not feed them (or width/height) to a "
          "classifier without deliberate handling.")
    print("5. manifests/DATA_README.md -- written; keep it updated if any of the above review items "
          "change the grouping or fold assignment.")


if __name__ == "__main__":
    main()
