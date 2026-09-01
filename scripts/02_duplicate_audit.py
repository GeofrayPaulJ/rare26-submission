"""Phase 2: exact duplicate audit, using sha256_hash from the manifest.

Reports:
  - every hash shared by 2+ files (exact duplicate groups)
  - CRITICAL: any hash that appears under BOTH class labels (neo AND ndbe) --
    this is a labelling error, not a benign duplicate, and is flagged separately.

Usage: python 02_duplicate_audit.py [--manifest PATH] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["load_error"] = df["load_error"].fillna("")
    return df


def audit_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (duplicates_df, cross_class_df)."""
    valid = df[df["sha256_hash"].notna() & (df["load_error"] == "")].copy()

    hash_counts = valid.groupby("sha256_hash")["filepath"].transform("count")
    dup_mask = hash_counts > 1
    dup_df = valid.loc[dup_mask].sort_values(["sha256_hash", "filepath"]).copy()

    group_sizes = dup_df.groupby("sha256_hash")["filepath"].transform("count")
    dup_df["duplicate_group_size"] = group_sizes
    group_id_map = {h: i for i, h in enumerate(sorted(dup_df["sha256_hash"].unique()), start=1)}
    dup_df["duplicate_group_id"] = dup_df["sha256_hash"].map(group_id_map)

    classes_per_hash = dup_df.groupby("sha256_hash")["class_label"].nunique()
    cross_class_hashes = classes_per_hash[classes_per_hash > 1].index
    cross_df = dup_df[dup_df["sha256_hash"].isin(cross_class_hashes)].copy()
    cross_df["is_cross_class"] = True
    dup_df["is_cross_class"] = dup_df["sha256_hash"].isin(cross_class_hashes)

    cols = ["duplicate_group_id", "sha256_hash", "duplicate_group_size", "is_cross_class",
            "filepath", "centre", "class_label"]
    return dup_df[cols], cross_df[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    manifests_dir = Path(__file__).resolve().parent.parent / "manifests"
    ap.add_argument("--manifest", default=str(manifests_dir / "rare25_manifest.parquet"))
    ap.add_argument("--out-dir", default=str(manifests_dir))
    args = ap.parse_args()

    df = load_manifest(Path(args.manifest))
    dup_df, cross_df = audit_duplicates(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dup_path = out_dir / "exact_duplicates.csv"
    dup_df.to_csv(dup_path, index=False)

    n_groups = dup_df["duplicate_group_id"].nunique()
    n_files = len(dup_df)
    n_cross_groups = cross_df["sha256_hash"].nunique()

    print(f"Exact duplicate audit ({len(df)} images scanned)")
    print(f"  Duplicate groups: {n_groups}  ({n_files} files total involved)")
    print(f"  Saved -> {dup_path}")

    if n_cross_groups:
        print()
        print("=" * 70)
        print(f"CRITICAL: {n_cross_groups} hash(es) appear under BOTH class labels")
        print("(same image content filed as both neoplasia AND non-dysplastic)")
        print("=" * 70)
        for h, grp in cross_df.groupby("sha256_hash"):
            print(f"\n  sha256={h}")
            for _, row in grp.sort_values("filepath").iterrows():
                print(f"    [{row['class_label']:16s}] {row['filepath']}")
    else:
        print("\nNo cross-class hash collisions found (no labelling errors of this kind detected).")

    if n_groups:
        print("\nAll duplicate groups (same-class duplicates are usually benign re-crops/exports):")
        for h, grp in dup_df.groupby("sha256_hash"):
            classes = sorted(grp["class_label"].unique())
            flag = " <-- CROSS-CLASS" if len(classes) > 1 else ""
            print(f"  {h[:16]}...  x{len(grp)}  classes={classes}{flag}")


if __name__ == "__main__":
    main()
