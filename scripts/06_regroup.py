"""
06_regroup.py -- Work out which images are probably the same patient.

WHY: The dataset has no patient IDs. Multiple photos of one patient are almost
certainly in there. If a patient's photos end up in both training and testing,
your scores will be inflated and you will not know it.

WHY THE FIRST ATTEMPT FAILED: aHash/dHash compare overall image structure. Every
image here is a bright blob on black, so they mostly compared borders -- which
are identical everywhere -- and chained 92% of the dataset into one group.

WHY THE SECOND ATTEMPT (cropping to an axis-aligned bbox) DIDN'T ACTUALLY FIX
IT: the FOV is a circle whose diameter equals the frame height, so the bbox of
non-black pixels is always ~the full frame -- nothing was really cropped, so
that run's grouping was still riding on vignette/background geometry.

WHAT'S DIFFERENT HERE:
  1. Reads FOV circle geometry from manifests/rare25_manifest_fov.csv (written
     by 05_fov_crop.py) and crops each image, IN MEMORY ONLY, to the largest
     axis-aligned square inscribed in that circle (inner_left/top/right/bottom).
     This square is guaranteed to contain only FOV/tissue content, no black
     background -- unlike the old bbox crop. Nothing is written to disk; images
     are loaded straight from 00_source.
  2. Uses pHash, which compares frequency content rather than raw layout.
  3. Requires MUTUAL nearest-neighbour agreement before linking two images,
     which is what stops one long chain swallowing the dataset.
  4. Refuses to hand you a degenerate answer -- it warns loudly if any group
     exceeds a sane share of the data.

USAGE:
    python 06_regroup.py
    python 06_regroup.py --threshold 10 --sweep
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "00_source"
MANIFESTS = ROOT / "manifests"


def hash_one(args):
    rel_path, box = args
    left, top, right, bottom = box
    try:
        img = Image.open(SRC / rel_path).convert("RGB")
        cropped = img.crop((left, top, right, bottom))
        return rel_path, str(imagehash.phash(cropped, hash_size=8)), ""
    except Exception as exc:  # noqa: BLE001
        return rel_path, None, str(exc)


def hex_to_bits(h):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster(bits, threshold, mutual_k=5):
    """Link pairs under `threshold` bits apart, but only if each is within the
    other's k nearest neighbours. That mutual requirement is what prevents one
    long chain from absorbing everything."""
    n = len(bits)
    dist = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        dist[i] = (bits[i] != bits).sum(axis=1)
    np.fill_diagonal(dist, 999)

    knn = np.argsort(dist, axis=1)[:, :mutual_k]
    knn_sets = [set(row) for row in knn]

    uf = UnionFind(n)
    for i in range(n):
        for j in knn[i]:
            if dist[i, j] <= threshold and i in knn_sets[j]:
                uf.union(i, int(j))

    roots = [uf.find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots]), dist


def summarise(labels, df, tag):
    sizes = pd.Series(labels).value_counts()
    pos = df[df.class_label == "neoplasia"]
    pos_groups = pos["group_id"].nunique()
    largest_pct = 100 * sizes.iloc[0] / len(labels)
    print(f"\n[{tag}] {len(labels)} images -> {len(sizes)} groups")
    print(f"       largest group: {sizes.iloc[0]} ({largest_pct:.1f}%)")
    print(f"       positive images: {len(pos)} -> {pos_groups} groups")
    print(f"       positive group sizes: {pos['group_id'].value_counts().head(8).tolist()}")
    if largest_pct > 20:
        print("       *** WARNING: largest group exceeds 20% of the data.")
        print("       *** This grouping is NOT usable for cross-validation folds.")
        print("       *** Lower --threshold and re-run.")
    else:
        print("       OK -- usable for grouped cross-validation.")
    return largest_pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--mutual-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sweep", action="store_true",
                     help="Try several thresholds and print a comparison table.")
    args = ap.parse_args()

    src = MANIFESTS / "rare25_manifest_fov.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run 05_fov_crop.py first.")

    df = pd.read_csv(src)
    missing_box = df[["inner_left", "inner_top", "inner_right", "inner_bottom"]].isna().any(axis=1)
    if missing_box.any():
        print(f"Skipping {missing_box.sum()} image(s) with no FOV geometry (failed detection).")
        df = df[~missing_box].reset_index(drop=True)

    boxes = df[["inner_left", "inner_top", "inner_right", "inner_bottom"]].astype(int).values
    jobs = list(zip(df["filepath"].tolist(), [tuple(b) for b in boxes]))

    print(f"Hashing {len(jobs)} images, cropped in-memory to their inner FOV square...")
    out = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rel, h, err in tqdm(pool.map(hash_one, jobs, chunksize=32), total=len(jobs)):
            out[rel] = h
            if err:
                print(f"  failed: {rel}: {err}")

    df["phash"] = df["filepath"].map(out)
    df = df[df["phash"].notna()].reset_index(drop=True)
    bits = np.stack([hex_to_bits(h) for h in df["phash"]])

    if args.sweep:
        print("\nThreshold sweep:")
        print(f"{'thresh':>7} {'groups':>8} {'largest':>9} {'largest%':>9} {'pos groups':>11}")
        for t in [4, 6, 8, 10, 12, 14]:
            labels, _ = cluster(bits, t, args.mutual_k)
            tmp = df.copy()
            tmp["group_id"] = labels
            sizes = pd.Series(labels).value_counts()
            pg = tmp[tmp.class_label == "neoplasia"]["group_id"].nunique()
            print(f"{t:>7} {len(sizes):>8} {sizes.iloc[0]:>9} "
                  f"{100*sizes.iloc[0]/len(labels):>8.1f}% {pg:>11}")
        print()

    labels, _ = cluster(bits, args.threshold, args.mutual_k)
    df["group_id"] = labels
    summarise(labels, df, f"threshold={args.threshold}")

    dst = MANIFESTS / "rare25_manifest_grouped.csv"
    df.to_csv(dst, index=False)
    print(f"\nWritten: {dst}")
    print("The 'group_id' column is what folds must be split on.")


if __name__ == "__main__":
    main()
