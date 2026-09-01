"""Phase 4: summary report.

Reads the artifacts produced by phases 0-3 (manifest, exact_duplicates.csv,
near_duplicate_clusters.csv) and writes a single markdown summary to
./manifests/inventory_report.md (also printed to stdout).

Usage: python 04_report.py [--manifests-dir DIR] [--threshold N]
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import EXPECTED_NEGATIVE, EXPECTED_POSITIVE, EXPECTED_TOTAL

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
nd = importlib.import_module("03_near_duplicates")  # module name starts with a digit


def threshold_sensitivity(df: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    """Re-cluster at several thresholds (cheap: hashes are already computed) to
    show how sharply cluster counts collapse -- diagnostic for hash-chaining."""
    ahash_ints = np.array([int(h, 16) for h in df["ahash"]], dtype=np.uint64)
    dhash_ints = np.array([int(h, 16) for h in df["dhash"]], dtype=np.uint64)
    a_dist = nd.popcount64(ahash_ints[:, None] ^ ahash_ints[None, :])
    d_dist = nd.popcount64(dhash_ints[:, None] ^ dhash_ints[None, :])
    pos_mask = (df["class_label"] == "neoplasia").to_numpy()

    rows = []
    for thr in thresholds:
        edge_mask = (a_dist <= thr) | (d_dist <= thr)
        np.fill_diagonal(edge_mask, False)
        uf = nd.UnionFind(len(df))
        ii, jj = np.where(np.triu(edge_mask, k=1))
        for i, j in zip(ii, jj):
            uf.union(int(i), int(j))
        roots = np.array([uf.find(i) for i in range(len(df))])
        sizes = pd.Series(roots).value_counts()
        rows.append({
            "threshold": thr,
            "n_clusters": len(sizes),
            "largest_cluster": int(sizes.max()),
            "largest_pct": sizes.max() / len(df),
            "positive_clusters": len(set(roots[pos_mask])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    manifests_dir_default = Path(__file__).resolve().parent.parent / "manifests"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", default=str(manifests_dir_default))
    ap.add_argument("--threshold", type=int, default=8, help="Primary clustering threshold used in Phase 3 (for labeling)")
    args = ap.parse_args()
    mdir = Path(args.manifests_dir)

    df = pd.read_parquet(mdir / "rare25_manifest.parquet")
    df["load_error"] = df["load_error"].fillna("")
    dup_df = pd.read_csv(mdir / "exact_duplicates.csv")
    clusters_df = pd.read_csv(mdir / "near_duplicate_clusters.csv")

    total = len(df)
    class_counts = df["class_label"].value_counts()
    centre_counts = df["centre"].value_counts()
    centre_class = df.groupby(["centre", "class_label"]).size()

    pos = class_counts.get("neoplasia", 0)
    neg = class_counts.get("non-dysplastic", 0)
    mismatches = []
    if total != EXPECTED_TOTAL:
        mismatches.append(f"Total images: expected **{EXPECTED_TOTAL}**, found **{total}** (diff {total - EXPECTED_TOTAL:+d})")
    if pos != EXPECTED_POSITIVE:
        mismatches.append(f"Positive/neoplasia: expected **{EXPECTED_POSITIVE}**, found **{pos}** (diff {pos - EXPECTED_POSITIVE:+d})")
    if neg != EXPECTED_NEGATIVE:
        mismatches.append(f"Negative/non-dysplastic: expected **{EXPECTED_NEGATIVE}**, found **{neg}** (diff {neg - EXPECTED_NEGATIVE:+d})")

    cross_df = dup_df[dup_df["is_cross_class"]] if "is_cross_class" in dup_df.columns else dup_df.iloc[0:0]
    n_dup_groups = dup_df["duplicate_group_id"].nunique() if len(dup_df) else 0
    n_dup_files = len(dup_df)
    n_cross_groups = cross_df["sha256_hash"].nunique() if len(cross_df) else 0

    n_clusters = clusters_df["cluster_id"].nunique()
    cluster_sizes = clusters_df.groupby("cluster_id").size()
    largest_cluster_id = cluster_sizes.idxmax()
    largest_cluster_size = int(cluster_sizes.max())

    pos_clusters_df = clusters_df[clusters_df["class_label"] == "neoplasia"]
    n_pos_images = len(pos_clusters_df)
    n_pos_clusters = pos_clusters_df["cluster_id"].nunique()

    failed = df[df["load_error"] != ""]

    sens = threshold_sensitivity(df, sorted(set([2, 4, 6, args.threshold])))

    lines: list[str] = []
    a = lines.append

    a("# RARE25/RARE26 Barrett's Esophagus Dataset -- Inventory Report\n")

    a("## 1. Dataset totals\n")
    a(f"- **Total images:** {total}")
    a(f"- **By class:** neoplasia = {pos}, non-dysplastic = {neg}"
      + (f", unknown = {class_counts.get('unknown', 0)}" if class_counts.get("unknown", 0) else ""))
    a("- **By centre:**")
    for centre, count in centre_counts.items():
        a(f"  - {centre}: {count}")
    a("- **By centre x class:**")
    for (centre, label), count in centre_class.items():
        a(f"  - {centre} / {label}: {count}")
    a("")

    a("## 2. Expected-count check (Phase 0)\n")
    if mismatches:
        a("**MISMATCH DETECTED:**\n")
        for m in mismatches:
            a(f"- {m}")
    else:
        a(f"OK -- counts match expectations exactly (total={EXPECTED_TOTAL}, positive={EXPECTED_POSITIVE}, negative={EXPECTED_NEGATIVE}).")
    a("")

    a("## 3. Exact duplicates (Phase 2)\n")
    a(f"- **Duplicate groups (same sha256):** {n_dup_groups} groups, {n_dup_files} files involved")
    a(f"- **Cross-class duplicates (same image filed under BOTH neo and ndbe): {n_cross_groups}**")
    if n_cross_groups:
        a("\n> **CRITICAL -- labelling error, not a benign duplicate:**\n")
        for h, grp in cross_df.groupby("sha256_hash"):
            a(f"> - sha256 `{h[:16]}...`:")
            for _, row in grp.sort_values("filepath").iterrows():
                a(f">   - `[{row['class_label']}]` {row['filepath']}")
    else:
        a("- No hash appears under both class labels -- no duplicate-driven labelling errors detected.")
    if n_dup_groups:
        a("\nAll duplicate groups (see `exact_duplicates.csv` for full file list):\n")
        a("| sha256 (prefix) | files | class(es) | cross-class? |")
        a("|---|---|---|---|")
        for h, grp in dup_df.groupby("sha256_hash"):
            classes = sorted(grp["class_label"].unique())
            flag = "**YES**" if len(classes) > 1 else "no"
            a(f"| `{h[:16]}...` | {len(grp)} | {', '.join(classes)} | {flag} |")
    a("")

    a("## 4. Near-duplicate clustering (Phase 3)\n")
    a(f"- **Threshold used:** Hamming distance <= {args.threshold} on aHash OR dHash")
    a(f"- **Overall:** {total} images -> **{n_clusters} clusters** ({total / n_clusters:.2f}x reduction)")
    a(f"- **Positive (neoplasia) only:** {n_pos_images} images -> **{n_pos_clusters} clusters**")
    a(f"- **Effective N+ for cross-validation planning: {n_pos_clusters}** (vs. nominal {n_pos_images} positive images)")
    if n_pos_clusters < n_pos_images:
        a(
            f"\n> **Implication:** {n_pos_images} positive images collapse into only {n_pos_clusters} "
            f"near-duplicate clusters ({n_pos_images / n_pos_clusters:.2f}x redundancy). A CV split that "
            f"does not group folds by `cluster_id` will leak near-duplicate frames of the same lesion "
            f"across train/validation, inflating apparent performance. **Use {n_pos_clusters}, not "
            f"{n_pos_images}, as the effective positive sample size when planning folds/power.**"
        )

    if largest_cluster_size >= 0.2 * total:
        giant_rows = clusters_df[clusters_df["cluster_id"] == largest_cluster_id]
        n_centres_giant = giant_rows["centre"].nunique()
        n_classes_giant = giant_rows["class_label"].nunique()
        a(
            f"\n### Caveat: dominant cluster looks like hash-chaining, not genuine redundancy\n\n"
            f"Cluster `{largest_cluster_id}` alone contains **{largest_cluster_size} images "
            f"({largest_cluster_size / total:.0%} of the dataset)**, spanning {n_centres_giant} centre(s) "
            f"and {n_classes_giant} class label(s). Because this dataset has no patient/video "
            f"identifiers, and independently-sourced clinical centres would not share patients, a "
            f"single cluster covering the large majority of *both* centres is inconsistent with genuine "
            f"same-patient/same-video redundancy. It is far more consistent with **union-find chaining**: "
            f"aHash/dHash are naive global-structure hashes, and at threshold={args.threshold} the "
            f"pairwise-similarity graph is dense enough to percolate into one giant connected component "
            f"-- images in the cluster are linked via a *chain* of near-duplicate pairs, not because "
            f"every image in it resembles every other.\n\n"
            f"Threshold-sensitivity sweep (same aHash/dHash values, varying only the cutoff) confirms a "
            f"sharp percolation transition:\n\n"
            f"| threshold | # clusters | largest cluster | largest as % of dataset | positive-only clusters |\n"
            f"|---|---|---|---|---|"
        )
        for _, row in sens.iterrows():
            a(f"| {int(row['threshold'])} | {int(row['n_clusters'])} | {int(row['largest_cluster'])} | {row['largest_pct']:.0%} | {int(row['positive_clusters'])} |")
        a(
            f"\nAt threshold=2 there is essentially no collapsing (positive clusters = {sens.loc[sens.threshold==2,'positive_clusters'].values[0] if 2 in sens.threshold.values else 'n/a'} "
            f"of 158), while by threshold={args.threshold} it drops to {n_pos_clusters}. **Treat the "
            f"threshold={args.threshold} effective-N+ of {n_pos_clusters} as an upper bound on redundancy "
            f"(most conservative for CV grouping), not a literal patient/video count.** For CV-fold "
            f"grouping specifically, grouping by `cluster_id` at threshold={args.threshold} is still the "
            f"*safer* choice (over-grouping only costs statistical power, whereas under-grouping risks "
            f"train/val leakage) -- but a lower threshold (2-4) should be used if the goal is estimating "
            f"the true number of distinct lesions/patients rather than building leak-proof folds."
        )
    a("")

    a("## 5. Images that failed to load/parse\n")
    if len(failed):
        a(f"**{len(failed)} image(s) failed:**\n")
        for _, row in failed.iterrows():
            a(f"- `{row['filepath']}`: {row['load_error']}")
    else:
        a("None -- all images loaded and parsed successfully.")
    a("")

    a("## 6. Output artifacts\n")
    a("- `manifests/rare25_manifest.csv` / `.parquet` -- one row per image (includes `cluster_id`)")
    a("- `manifests/exact_duplicates.csv` -- exact-duplicate groups, cross-class flagged")
    a("- `manifests/near_duplicate_clusters.csv` -- perceptual-hash cluster assignments")
    a("- `manifests/inventory_report.md` -- this report")

    report_text = "\n".join(lines)
    out_path = mdir / "inventory_report.md"
    out_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n\n(Report saved -> {out_path})")


if __name__ == "__main__":
    main()
