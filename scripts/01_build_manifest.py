"""Phase 1: build the image manifest.

Walks every image file under the dataset root and, for each one, records
filepath, centre, class_label, sha256 hash, dimensions, color mode, file
size, and image format. Average/difference perceptual hashes are computed
in the same pass (one file open per image, reused by Phase 3) and are kept
as extra columns; a cluster_id column is added later by Phase 3.

Read-only: original dataset files are never modified or moved.

Usage: python 01_build_manifest.py [--root PATH] [--workers N] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import iter_image_files, locate_dataset_root, relative_centre_and_class

MANIFEST_COLUMNS = [
    "filepath", "centre", "class_label", "sha256_hash", "width", "height",
    "color_mode", "file_size_bytes", "image_format", "ahash", "dhash", "load_error",
]


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def process_one(args: tuple[str, str]) -> dict:
    root_str, rel_str = args
    root = Path(root_str)
    filepath = root / rel_str
    centre, class_label = relative_centre_and_class(filepath, root)

    row = dict.fromkeys(MANIFEST_COLUMNS)
    row.update(filepath=rel_str.replace("\\", "/"), centre=centre, class_label=class_label, load_error="")

    try:
        row["file_size_bytes"] = filepath.stat().st_size
        row["sha256_hash"] = _sha256_of_file(filepath)
    except OSError as e:
        row["load_error"] = f"file read error: {e}"
        return row

    try:
        with Image.open(filepath) as img:
            img.load()
            row["width"], row["height"] = img.size
            row["color_mode"] = img.mode
            row["image_format"] = img.format
            row["ahash"] = str(imagehash.average_hash(img))
            row["dhash"] = str(imagehash.dhash(img))
    except Exception as e:  # noqa: BLE001 - any decode failure must be captured, not fatal
        row["load_error"] = f"image decode error: {type(e).__name__}: {e}"

    return row


def build_manifest(root: Path, workers: int | None = None) -> pd.DataFrame:
    files = sorted(iter_image_files(root))
    rel_files = [str(p.relative_to(root)) for p in files]
    print(f"Found {len(rel_files)} image files under {root}")

    tasks = [(str(root), rel) for rel in rel_files]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Hashing/reading images"):
            rows.append(fut.result())

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df = df.sort_values("filepath").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--workers", type=int, default=None, help="Process pool size (default: os.cpu_count())")
    ap.add_argument("--out-dir", default=None, help="Output dir for manifest (default: <repo>/manifests)")
    args = ap.parse_args()

    root = locate_dataset_root(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_manifest(root, workers=args.workers)

    n_errors = int((df["load_error"] != "").sum())
    print(f"Processed {len(df)} images, {n_errors} failed to load/parse.")
    if n_errors:
        print("Failed files:")
        for fp in df.loc[df["load_error"] != "", "filepath"]:
            print(f"  {fp}")

    csv_path = out_dir / "rare25_manifest.csv"
    parquet_path = out_dir / "rare25_manifest.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved manifest -> {csv_path}")
    print(f"Saved manifest -> {parquet_path}")


if __name__ == "__main__":
    main()
