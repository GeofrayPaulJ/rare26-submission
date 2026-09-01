"""
07_folds.py -- Decide which images go in which fold, once, permanently.

WHY: Every experiment from here on must use the same splits, or you cannot
compare results between them. This writes them to disk so they never change.

WHAT IT PRODUCES:
  1. 5 folds x 10 repeats, grouped so no patient appears on both sides.
  2. A separate hospital-holdout split -- train on one hospital, test on the
     other. This is the honest test of whether your model generalises, because
     the real test set comes from twelve hospitals you have never seen.

USAGE:
    python 07_folds.py
    python 07_folds.py --folds 5 --repeats 10
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "manifests"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    src = MANIFESTS / "rare25_manifest_grouped.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run 06_regroup.py first.")

    df = pd.read_csv(src)
    y = (df["class_label"] == "neoplasia").astype(int).values
    groups = df["group_id"].values

    largest_pct = 100 * pd.Series(groups).value_counts().iloc[0] / len(df)
    if largest_pct > 20:
        raise SystemExit(
            f"STOP: largest group is {largest_pct:.1f}% of the data.\n"
            f"Folds built on this would be meaningless. Re-run 06_regroup.py "
            f"with a lower --threshold first."
        )

    print(f"{len(df)} images, {y.sum()} positive, {df.group_id.nunique()} groups")

    # --- Repeated grouped stratified CV ---
    for rep in range(args.repeats):
        skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=rep)
        col = f"fold_r{rep}"
        df[col] = -1
        for k, (_, val_idx) in enumerate(skf.split(df, y, groups)):
            df.loc[df.index[val_idx], col] = k

    # --- Sanity: positives per fold, first repeat ---
    print(f"\nPositives per fold (repeat 0):")
    check = df[df.class_label == "neoplasia"]["fold_r0"].value_counts().sort_index()
    print(check.to_string())
    if check.min() < 5:
        print("  *** WARNING: a fold has fewer than 5 positives. Results from it "
              "will be extremely noisy.")

    # --- Leakage check ---
    leaks = 0
    for rep in range(args.repeats):
        col = f"fold_r{rep}"
        per_group = df.groupby("group_id")[col].nunique()
        leaks += (per_group > 1).sum()
    print(f"\nGroups split across folds (should be 0): {leaks}")

    # --- Hospital holdout ---
    df["holdout_center_1"] = (df["centre"] == "center_1").map({True: "test", False: "train"})
    df["holdout_center_2"] = (df["centre"] == "center_2").map({True: "test", False: "train"})
    print("\nHospital-holdout splits added: holdout_center_1, holdout_center_2")

    dst = MANIFESTS / "rare25_folds.csv"
    df.to_csv(dst, index=False)
    print(f"\nWritten: {dst}")
    print("Freeze this file. Every experiment reads its splits from here.")


if __name__ == "__main__":
    main()
