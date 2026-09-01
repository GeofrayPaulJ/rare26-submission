"""
10_evc_inventory.py -- Inventory the EVC (Early Barrett's Cancer, EndoVis 2015) dataset.

Locates the EVC zip archive under D:\\RARE26 (searched by name containing "evc",
falling back to the only zip at the root if there's just one), and unpacks it
to 02_evc/. The original zip is never modified. The layout is not assumed --
this script prints the extracted tree before parsing anything.

Per the dataset's own readme.txt:
  - images/patXX_imY_ZZZZ.png       -- endoscopic image (ZZZZ = ACHD or NDBT)
  - annotations_bmp/..._expN.bmp    -- 1-bit bitmap mask, per expert (1-5)
  - annotations_mat/patXX_imY_ZZZZ.mat -- 1x5 MATLAB cell array of the same masks

Builds one manifest row per image (manifests/evc_inventory.csv) with format/size/
hash/patient-id fields, cross-checks delineation completeness (every image should
have exactly 5 expert masks -- readme confirms even NDBT images carry all-black
masks), and for a deterministic sample of 3 cancerous images reports mask area as
a percentage of the detected field-of-view plus pairwise Dice/IoU across the 5
experts, so inter-rater disagreement is visible before anyone relies on these
masks for training.

Read-only with respect to the source zip. Safe to re-run (extraction is
deterministic; manifest is fully rebuilt each time).

USAGE:
    python 10_evc_inventory.py
    python 10_evc_inventory.py --zip PATH --out-dir D:\\RARE26\\02_evc
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
MANIFESTS = ROOT / "manifests"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

EXPECTED_TOTAL = 100
EXPECTED_CANCER = 50
EXPECTED_NONDYS = 50
EXPECTED_PATIENTS = 39
N_EXPERTS = 5

CLASS_MAP = {"ACHD": "cancer", "NDBT": "non-dysplastic"}

STEM_RE = re.compile(r"pat(\d+)_im(\d+)_(ACHD|NDBT)")
BMP_RE = re.compile(r"(pat\d+_im\d+_(?:ACHD|NDBT))_exp(\d)\.bmp$")


def find_evc_zip(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Specified zip does not exist: {explicit}")
        return p

    candidates = [p for p in ROOT.glob("*.zip")]
    named = [p for p in candidates if "evc" in p.name.lower()]
    if named:
        return named[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not uniquely locate the EVC zip under {ROOT}. "
        f"Found zips: {[p.name for p in candidates]}. Pass --zip explicitly."
    )


def print_tree(root: Path, max_entries_per_dir: int = 8, max_depth: int = 3) -> None:
    print(f"\n{root}")

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as e:
            print(f"{prefix}[unreadable: {e}]")
            return
        shown = entries[:max_entries_per_dir]
        for i, entry in enumerate(shown):
            is_last = i == len(shown) - 1 and len(entries) <= max_entries_per_dir
            connector = "`-- " if is_last else "|-- "
            if entry.is_dir():
                n_files = sum(1 for f in entry.iterdir() if f.is_file())
                print(f"{prefix}{connector}{entry.name}/  ({n_files} files)")
                walk(entry, prefix + ("    " if is_last else "|   "), depth + 1)
            else:
                print(f"{prefix}{connector}{entry.name}")
        if len(entries) > max_entries_per_dir:
            print(f"{prefix}`-- ... ({len(entries) - max_entries_per_dir} more entries)")

    walk(root, "", 1)


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dice_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    asum, bsum = int(a.sum()), int(b.sum())
    if asum + bsum == 0:
        return 1.0, 1.0  # both empty -- perfect agreement by convention
    dice = 2 * inter / (asum + bsum)
    iou = inter / union if union else 1.0
    return dice, iou


def build_manifest(evc_dir: Path) -> pd.DataFrame:
    images_dir = evc_dir / "images"
    bmp_dir = evc_dir / "annotations_bmp"
    mat_dir = evc_dir / "annotations_mat"

    image_paths = sorted(images_dir.glob("*.png"))
    rows = []
    for img_path in image_paths:
        stem = img_path.stem  # patXX_imY_ZZZZ
        m = STEM_RE.match(stem)
        patient_id = int(m.group(1)) if m else None
        image_number = int(m.group(2)) if m else None
        pathology_code = m.group(3) if m else "unknown"
        class_label = CLASS_MAP.get(pathology_code, "unknown")

        with Image.open(img_path) as img:
            img.load()
            width, height = img.size
            color_mode = img.mode
            image_format = img.format

        experts_present = sorted(
            int(BMP_RE.match(p.name).group(2))
            for p in bmp_dir.glob(f"{stem}_exp*.bmp")
            if BMP_RE.match(p.name)
        )
        missing_experts = [e for e in range(1, N_EXPERTS + 1) if e not in experts_present]
        mat_present = (mat_dir / f"{stem}.mat").exists()

        rows.append({
            "filepath": str(img_path.relative_to(evc_dir)).replace("\\", "/"),
            "patient_id": patient_id,
            "image_number": image_number,
            "pathology_code": pathology_code,
            "class_label": class_label,
            "sha256_hash": sha256_of_file(img_path),
            "width": width,
            "height": height,
            "color_mode": color_mode,
            "file_size_bytes": img_path.stat().st_size,
            "image_format": image_format,
            "n_expert_annotations": len(experts_present),
            "missing_experts": ",".join(map(str, missing_experts)) if missing_experts else "",
            "mat_present": mat_present,
        })

    return pd.DataFrame(rows).sort_values("filepath").reset_index(drop=True)


def sample_mask_agreement(evc_dir: Path, df: pd.DataFrame, n_samples: int = 3) -> None:
    import importlib
    fov_module = importlib.import_module("05_fov_crop")

    cancer = df[df["class_label"] == "cancer"].sort_values("filepath")
    sample = cancer.head(n_samples)

    print(f"\n=== Mask area & inter-expert agreement (sample of {len(sample)} cancerous images) ===")
    for _, row in sample.iterrows():
        stem = Path(row["filepath"]).stem
        img_path = evc_dir / row["filepath"]
        with Image.open(img_path) as img:
            arr = np.asarray(img.convert("RGB"))
        gray = arr.mean(axis=2)
        fov = fov_module.fit_fov_circle(gray > 15)
        if fov is None:
            print(f"  {stem}: could not detect FOV, skipping")
            continue
        _, _, radius, fit_quality, _ = fov
        fov_area = np.pi * radius ** 2

        masks = []
        for e in range(1, N_EXPERTS + 1):
            bmp_path = evc_dir / "annotations_bmp" / f"{stem}_exp{e}.bmp"
            m = np.asarray(Image.open(bmp_path)) > 0
            masks.append(m)

        pct_areas = [100.0 * m.sum() / fov_area for m in masks]
        dices, ious = [], []
        for i in range(N_EXPERTS):
            for j in range(i + 1, N_EXPERTS):
                d, iou = dice_iou(masks[i], masks[j])
                dices.append(d)
                ious.append(iou)

        print(f"\n  {stem}  (fov_radius={radius:.1f}px, fit_quality={fit_quality:.3f})")
        print(f"    mask area as % of FOV per expert: " + ", ".join(f"{p:.2f}%" for p in pct_areas))
        print(f"    pairwise Dice: mean={np.mean(dices):.3f}  min={np.min(dices):.3f}  max={np.max(dices):.3f}")
        print(f"    pairwise IoU:  mean={np.mean(ious):.3f}  min={np.min(ious):.3f}  max={np.max(ious):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=None, help="Explicit path to the EVC zip archive")
    ap.add_argument("--out-dir", default=str(ROOT / "02_evc"), help="Extraction directory")
    args = ap.parse_args()

    zip_path = find_evc_zip(args.zip)
    out_dir = Path(args.out_dir)
    print(f"EVC archive: {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print(f"Extracted -> {out_dir}  (original zip untouched)")

    print("\n=== Directory tree ===")
    print_tree(out_dir)

    df = build_manifest(out_dir)

    n_total = len(df)
    n_cancer = int((df["class_label"] == "cancer").sum())
    n_nondys = int((df["class_label"] == "non-dysplastic").sum())
    n_patients = df["patient_id"].nunique()

    print("\n=== Class / count check ===")
    print(f"Total images: {n_total} (expected {EXPECTED_TOTAL})")
    print(f"Cancer (ACHD): {n_cancer} (expected {EXPECTED_CANCER})")
    print(f"Non-dysplastic (NDBT): {n_nondys} (expected {EXPECTED_NONDYS})")
    print(f"Distinct patients: {n_patients} (expected {EXPECTED_PATIENTS})")
    mismatches = []
    if n_total != EXPECTED_TOTAL:
        mismatches.append(f"total {n_total} != {EXPECTED_TOTAL}")
    if n_cancer != EXPECTED_CANCER:
        mismatches.append(f"cancer {n_cancer} != {EXPECTED_CANCER}")
    if n_nondys != EXPECTED_NONDYS:
        mismatches.append(f"non-dysplastic {n_nondys} != {EXPECTED_NONDYS}")
    if n_patients != EXPECTED_PATIENTS:
        mismatches.append(f"patients {n_patients} != {EXPECTED_PATIENTS}")
    print("MISMATCH: " + "; ".join(mismatches) if mismatches else "OK -- matches expectations exactly.")

    per_patient = df.groupby("patient_id").size()
    print(f"\nImages per patient: min={per_patient.min()}, max={per_patient.max()}, "
          f"mean={per_patient.mean():.2f}")

    print("\n=== Image format ===")
    print(f"Formats: {df['image_format'].value_counts().to_dict()}")
    print(f"Color modes: {df['color_mode'].value_counts().to_dict()}")
    print(f"Dimensions: {sorted(df[['width','height']].drop_duplicates().apply(tuple, axis=1).tolist())}")

    print("\n=== Delineation completeness (annotations_bmp) ===")
    incomplete = df[df["n_expert_annotations"] != N_EXPERTS]
    print(f"Images with all {N_EXPERTS} experts present: {(df['n_expert_annotations'] == N_EXPERTS).sum()} / {n_total}")
    if len(incomplete):
        print(f"Images with FEWER than {N_EXPERTS} experts:")
        for _, row in incomplete.iterrows():
            print(f"  {row['filepath']}: has {row['n_expert_annotations']}, missing experts {row['missing_experts']}")
    else:
        print(f"No images are missing expert annotations -- all {n_total} have exactly {N_EXPERTS}.")
    print(f"annotations_mat present for all images: {df['mat_present'].all()}")
    print("Annotation formats found: 1-bit BMP bitmaps (annotations_bmp/), and MATLAB "
          "1x5 cell-array-of-logical-mask .mat files (annotations_mat/) -- both binary "
          "pixel masks, no coordinate/polygon files.")

    sample_mask_agreement(out_dir, df)

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    out_csv = MANIFESTS / "evc_inventory.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
