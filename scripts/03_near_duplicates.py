"""Phase 3: near-duplicate clustering via perceptual hashing, used as a proxy
for patient/video grouping (no patient/video identifiers exist in this dataset's
public layout).

For every pair of images, computes Hamming distance on both average-hash (aHash)
and difference-hash (dHash) -- reusing the hashes already computed in Phase 1 so
no image is reopened. Two images are linked (unioned into the same cluster) if
EITHER distance is <= --threshold (default 8). Cluster assignment uses a
union-find over the resulting graph.

Adds a cluster_id column back into the manifest (csv + parquet) and reports
cluster-size distributions overall and restricted to positive (neoplasia) images.

Usage: python 03_near_duplicates.py [--manifest PATH] [--threshold N] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorized bit-count (SWAR algorithm) for an array of uint64 values."""
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def cluster_by_hash(df: pd.DataFrame, threshold: int) -> pd.Series:
    """Union-find clustering; returns a cluster_id Series aligned to df.index."""
    n = len(df)
    uf = UnionFind(n)

    valid = df["ahash"].notna() & df["dhash"].notna()
    valid_idx = np.flatnonzero(valid.to_numpy())

    if len(valid_idx) > 1:
        ahash_ints = np.array([int(h, 16) for h in df.loc[valid, "ahash"]], dtype=np.uint64)
        dhash_ints = np.array([int(h, 16) for h in df.loc[valid, "dhash"]], dtype=np.uint64)

        a_dist = popcount64(ahash_ints[:, None] ^ ahash_ints[None, :])
        d_dist = popcount64(dhash_ints[:, None] ^ dhash_ints[None, :])
        edge_mask = (a_dist <= threshold) | (d_dist <= threshold)
        np.fill_diagonal(edge_mask, False)

        ii, jj = np.where(np.triu(edge_mask, k=1))
        for local_i, local_j in zip(ii, jj):
            uf.union(int(valid_idx[local_i]), int(valid_idx[local_j]))

    roots = [uf.find(i) for i in range(n)]
    # Remap arbitrary root ids -> sequential cluster_id (1..k), ordered by first appearance.
    remap: dict[int, int] = {}
    cluster_ids = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap) + 1
        cluster_ids.append(remap[r])
    return pd.Series(cluster_ids, index=df.index, name="cluster_id")


def size_histogram(cluster_ids: pd.Series) -> Counter:
    sizes = cluster_ids.value_counts()
    return Counter(sizes.values.tolist())


def print_histogram(hist: Counter, label: str) -> None:
    print(f"\n{label} cluster-size distribution (size -> #clusters of that size):")
    for size in sorted(hist):
        print(f"  size {size:3d}: {hist[size]:4d} cluster(s)")


def main() -> None:
    manifests_dir = Path(__file__).resolve().parent.parent / "manifests"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(manifests_dir / "rare25_manifest.parquet"))
    ap.add_argument("--threshold", type=int, default=8, help="Max Hamming distance (either hash) to link two images")
    ap.add_argument("--out-dir", default=str(manifests_dir))
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    df = pd.read_parquet(manifest_path) if manifest_path.suffix == ".parquet" else pd.read_csv(manifest_path)
    df["load_error"] = df["load_error"].fillna("")

    print(f"Clustering {len(df)} images at Hamming threshold <= {args.threshold} (aHash OR dHash)...")
    cluster_ids = cluster_by_hash(df, args.threshold)
    df["cluster_id"] = cluster_ids

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_sizes = df["cluster_id"].value_counts()
    df["cluster_size"] = df["cluster_id"].map(cluster_sizes)

    cluster_cols = ["filepath", "centre", "class_label", "cluster_id", "cluster_size", "ahash", "dhash"]
    clusters_path = out_dir / "near_duplicate_clusters.csv"
    df[cluster_cols].sort_values(["cluster_id", "filepath"]).to_csv(clusters_path, index=False)
    print(f"Saved -> {clusters_path}")

    # Write cluster_id back into the main manifest (csv + parquet), per spec.
    manifest_cols_out = [c for c in df.columns if c != "cluster_size"]
    csv_out = manifests_dir / "rare25_manifest.csv"
    parquet_out = manifests_dir / "rare25_manifest.parquet"
    df[manifest_cols_out].to_csv(csv_out, index=False)
    df[manifest_cols_out].to_parquet(parquet_out, index=False)
    print(f"Updated manifest with cluster_id -> {csv_out}")
    print(f"Updated manifest with cluster_id -> {parquet_out}")

    n_images = len(df)
    n_clusters = df["cluster_id"].nunique()
    print(f"\n=== Overall clustering summary ===")
    print(f"Images: {n_images}  ->  Clusters: {n_clusters}  (reduction factor {n_images / n_clusters:.2f}x)")
    print_histogram(size_histogram(df["cluster_id"]), "Overall")

    largest_id = cluster_sizes.idxmax()
    largest_size = int(cluster_sizes.max())
    if largest_size >= 0.2 * n_images:
        giant = df[df["cluster_id"] == largest_id]
        n_centres = giant["centre"].nunique()
        n_classes = giant["class_label"].nunique()
        print(
            f"\nWARNING: cluster {largest_id} contains {largest_size} images "
            f"({largest_size / n_images:.0%} of the dataset), spanning {n_centres} centre(s) "
            f"and {n_classes} class label(s)."
        )
        print(
            "This is consistent with union-find 'chaining' through a dense similarity graph "
            "(a well-known artifact of naive aHash/dHash at loose thresholds) rather than genuine "
            "same-video/same-patient redundancy -- images in this cluster are not necessarily "
            "mutually similar, only connected via a chain of pairwise near-duplicate links. "
            "Re-run with a lower --threshold (e.g. 2-4) to check for a sharp jump in max cluster "
            "size, and treat the resulting cluster/effective-N+ counts as an upper bound on "
            "redundancy, not a precise patient/video count. See inventory_report.md for details."
        )

    pos = df[df["class_label"] == "neoplasia"]
    if len(pos):
        pos_cluster_ids = pos["cluster_id"]
        n_pos_images = len(pos)
        n_pos_clusters = pos_cluster_ids.nunique()
        print(f"\n=== Positive-class (neoplasia) clustering summary ===")
        print(f"Positive images: {n_pos_images}  ->  Positive-containing clusters: {n_pos_clusters}")
        print_histogram(size_histogram(pos_cluster_ids), "Positive-only")
        print(
            f"\nEffective N+ for cross-validation planning: {n_pos_clusters} "
            f"(vs. nominal {n_pos_images} positive images)."
        )
        if n_pos_clusters < n_pos_images:
            print(
                f"IMPLICATION: {n_pos_images} positive images collapse into only {n_pos_clusters} "
                f"near-duplicate clusters ({n_pos_images / n_pos_clusters:.2f}x redundancy). "
                f"Any CV split that does not group by cluster_id will leak near-duplicate frames "
                f"of the same lesion across train/val folds, inflating apparent performance. "
                f"Treat {n_pos_clusters} as the true positive sample size for power/CV-fold planning, "
                f"not {n_pos_images}."
            )
    else:
        print("\nNo positive (neoplasia) images found -- skipping positive-only summary.")


if __name__ == "__main__":
    main()
