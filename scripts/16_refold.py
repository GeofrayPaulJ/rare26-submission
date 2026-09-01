"""
16_refold.py -- Repair the cross-validation folds.

WHY: 07_folds.py produced folds with two defects.

DEFECT A -- cross-centre groups. 44 perceptual-hash groups (438 images)
chain images from both center_1 and center_2 into one "patient" group. A
patient cannot attend two hospitals -- these are hash-chain collapses, not
real clusters. Fix: split every group_id on `centre` to make group_id_v2.

DEFECT B -- repeats don't re-partition. sklearn's StratifiedGroupKFold sorts
groups by the std of their per-class counts before greedy bin-packing. For a
single dominant group (like the old group_id=1158, 262 images) that sort
puts it first *regardless of random_state*, and the greedy packer always
drops the first group processed into fold 0 when all folds are still empty.
So the biggest group lands in fold 0 in every one of the ten repeats, and
10.1% of the dataset never moves. shuffle=True in StratifiedGroupKFold only
reshuffles *ties* in that sort -- it can't touch the dominant group's
position. Fix: bypass sklearn's internal sort. Reimplement the same greedy
bin-packing (same fold-selection rule sklearn uses) but process groups in an
order that is shuffled fresh, with seed = 1000 + repeat_index, for every
repeat.

INPUT
    manifests/rare25_canonical.csv
    manifests/positive_review.csv

OUTPUT
    manifests/rare25_folds_v2.csv

Does not modify rare25_canonical.csv or rare25_folds.csv -- those stay as
the audit trail of the broken run.

USAGE
    python 16_refold.py
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "manifests"

N_FOLDS = 5
N_REPEATS = 10
SEED_BASE = 1000


def stratified_group_kfold_shuffled(y, groups, n_splits, seed):
    """Greedy stratified group k-fold, sklearn-compatible fold-selection rule,
    but with the group processing order shuffled by `seed` instead of sorted
    by class-count std. That's the whole fix for defect B: sklearn's own sort
    puts the most-skewed (usually largest) group first on every call, so the
    greedy packer -- which always prefers an empty fold, and ties break to
    the lowest index -- drops it in fold 0 every single time. Shuffling the
    processing order means a different group goes first each repeat.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)

    classes, y_inv = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    y_cnt = np.bincount(y_inv, minlength=n_classes).astype(float)

    uniq_groups, groups_inv = np.unique(groups, return_inverse=True)
    n_groups = len(uniq_groups)

    y_counts_per_group = np.zeros((n_groups, n_classes))
    for c_idx, g_idx in zip(y_inv, groups_inv):
        y_counts_per_group[g_idx, c_idx] += 1

    rng = np.random.RandomState(seed)
    order = np.arange(n_groups)
    rng.shuffle(order)

    y_counts_per_fold = np.zeros((n_splits, n_classes))
    fold_of_group = np.empty(n_groups, dtype=int)

    for group_idx in order:
        group_y_counts = y_counts_per_group[group_idx]
        best_fold, min_eval, min_samples = None, np.inf, np.inf
        for f in range(n_splits):
            y_counts_per_fold[f] += group_y_counts
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[f] -= group_y_counts
            fold_eval = std_per_class.mean()
            samples_in_fold = y_counts_per_fold[f].sum()
            better = fold_eval < min_eval or (
                np.isclose(fold_eval, min_eval) and samples_in_fold < min_samples
            )
            if better:
                min_eval, min_samples, best_fold = fold_eval, samples_in_fold, f
        y_counts_per_fold[best_fold] += group_y_counts
        fold_of_group[group_idx] = best_fold

    return fold_of_group[groups_inv]


def pinned_count(df, fold_cols):
    """Images whose fold assignment is identical across every repeat."""
    return int((df[fold_cols].nunique(axis=1) == 1).sum())


def mean_pairwise_agreement(df, fold_cols):
    agree = [
        (df[a] == df[b]).mean()
        for a, b in itertools.combinations(fold_cols, 2)
    ]
    return float(np.mean(agree))


def main():
    canonical_path = MANIFESTS / "rare25_canonical.csv"
    review_path = MANIFESTS / "positive_review.csv"
    out_path = MANIFESTS / "rare25_folds_v2.csv"

    df = pd.read_csv(canonical_path)
    print(f"Loaded {canonical_path.name}: {len(df)} rows, {len(df.columns)} cols")

    fold_cols = [f"fold_r{i}" for i in range(N_REPEATS)]

    # ---- "before" snapshot, from the broken columns already in canonical ----
    before_groups = df["group_id"].nunique()
    before_largest = int(df["group_id"].value_counts().iloc[0])
    before_pinned = pinned_count(df, fold_cols)
    before_agreement = mean_pairwise_agreement(df, fold_cols)

    # =====================================================================
    # ADD visibility (left join on filepath)
    # =====================================================================
    review = pd.read_csv(review_path)
    print(f"Loaded {review_path.name}: {len(review)} rows")
    df = df.merge(review[["filepath", "visibility"]], on="filepath", how="left")

    vis_counts = df["visibility"].value_counts()
    expected_vis = {"obvious": 85, "moderate": 45, "would_have_missed": 28}
    print("\nvisibility counts:")
    print(vis_counts.to_string())
    for label, expected in expected_vis.items():
        got = int(vis_counts.get(label, 0))
        if got != expected:
            raise SystemExit(
                f"HALT: visibility='{label}' count is {got}, expected {expected}."
            )
    print("visibility counts match expected 85 / 45 / 28 -- OK")

    # =====================================================================
    # DEFECT A -- split every group on centre
    # =====================================================================
    df["group_id_v2"] = df["group_id"].astype(str) + "_" + df["centre"].astype(str)

    after_groups = df["group_id_v2"].nunique()
    after_largest = int(df["group_id_v2"].value_counts().iloc[0])

    print(f"\ngroup_id  -> group_id_v2 split:")
    print(f"  group count:   {before_groups} -> {after_groups} (expected 2562 -> 2606)")
    print(f"  largest group: {before_largest} -> {after_largest} (expected 262 -> 166)")

    if after_groups != 2606:
        raise SystemExit(
            f"HALT: expected 2606 groups after splitting on centre, got {after_groups}."
        )
    if after_largest != 166:
        raise SystemExit(
            f"HALT: expected largest group_id_v2 to be 166, got {after_largest}."
        )
    print("  group_id_v2 split verified -- OK")

    # =====================================================================
    # EXCLUSION -- regenerate folds over keep_for_training == True only
    # =====================================================================
    n_excluded = int((~df["keep_for_training"]).sum())
    n_kept = int(df["keep_for_training"].sum())
    print(f"\nkeep_for_training: {n_kept} kept, {n_excluded} excluded (fold = -1)")
    if n_kept != 3088:
        raise SystemExit(f"HALT: expected 3088 rows with keep_for_training==True, got {n_kept}.")

    for col in fold_cols:
        df[col] = -1

    kept_idx = df.index[df["keep_for_training"]]
    kept = df.loc[kept_idx]
    y = (kept["class_label"] == "neoplasia").astype(int).values
    groups = kept["group_id_v2"].values

    # =====================================================================
    # DEFECT B -- shuffle group order per repeat, seed = 1000 + repeat_index
    # =====================================================================
    for rep in range(N_REPEATS):
        seed = SEED_BASE + rep
        row_fold = stratified_group_kfold_shuffled(y, groups, N_FOLDS, seed)
        df.loc[kept_idx, f"fold_r{rep}"] = row_fold

    # holdout_center_1 / holdout_center_2 -- untouched, carried straight through
    assert "holdout_center_1" in df.columns and "holdout_center_2" in df.columns

    # =====================================================================
    # VERIFICATION
    # =====================================================================
    kept = df.loc[kept_idx]  # refresh with fold values filled in
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # 1. zero groups spanning more than one fold, every repeat
    check1 = True
    for rep in range(N_REPEATS):
        col = f"fold_r{rep}"
        leaks = int((kept.groupby("group_id_v2")[col].nunique() > 1).sum())
        if leaks != 0:
            check1 = False
            print(f"  repeat {rep}: {leaks} leaking groups")
    print(f"[{'PASS' if check1 else 'FAIL'}] 1. No group_id_v2 spans more than one fold, in any repeat")

    # 2. positives per fold within +/-2 of 158/5, every repeat
    target_pos = 158 / N_FOLDS
    check2 = True
    pos = kept[kept["class_label"] == "neoplasia"]
    for rep in range(N_REPEATS):
        counts = pos[f"fold_r{rep}"].value_counts().reindex(range(N_FOLDS), fill_value=0)
        if (counts - target_pos).abs().max() > 2:
            check2 = False
            print(f"  repeat {rep}: positive counts per fold {counts.tolist()}")
    print(f"[{'PASS' if check2 else 'FAIL'}] 2. Positives per fold within +/-2 of {target_pos:.1f}, in every repeat")

    # 3. total images per fold within +/-5% of 3088/5, every repeat
    target_total = n_kept / N_FOLDS
    tol = 0.05 * target_total
    check3 = True
    for rep in range(N_REPEATS):
        counts = kept[f"fold_r{rep}"].value_counts().reindex(range(N_FOLDS), fill_value=0)
        if (counts - target_total).abs().max() > tol:
            check3 = False
            print(f"  repeat {rep}: total counts per fold {counts.tolist()}")
    print(f"[{'PASS' if check3 else 'FAIL'}] 3. Total images per fold within +/-5% of {target_total:.1f}, in every repeat")

    # 4. images identical across all ten repeats, must be < 50 (kept rows only --
    #    the 7 excluded rows are pinned at -1 by definition and aren't part of CV)
    after_pinned = pinned_count(kept, fold_cols)
    check4 = after_pinned < 50
    print(f"[{'PASS' if check4 else 'FAIL'}] 4. Images with identical fold across all repeats: {after_pinned} (< 50 required, was 313)")

    # 5. group_id_v2 "1158_center_1" appears in >= 4 distinct folds across repeats
    target_group = "1158_center_1"
    if target_group in kept["group_id_v2"].values:
        row = kept[kept["group_id_v2"] == target_group].iloc[0]
        folds_seen = sorted(int(row[c]) for c in fold_cols)
        distinct = sorted(set(folds_seen))
        check5 = len(distinct) >= 4
        print(f"[{'PASS' if check5 else 'FAIL'}] 5. group_id_v2='{target_group}' appears in {len(distinct)} distinct folds: {distinct}")
        print(f"       per-repeat: {dict(zip(fold_cols, folds_seen))}")
    else:
        check5 = False
        print(f"[FAIL] 5. group_id_v2='{target_group}' not found")

    all_pass = check1 and check2 and check3 and check4 and check5
    if not all_pass:
        raise SystemExit("\nHALT: one or more verification checks failed.")

    # =====================================================================
    # BEFORE / AFTER TABLE
    # =====================================================================
    after_agreement = mean_pairwise_agreement(kept, fold_cols)

    print("\n" + "=" * 70)
    print("BEFORE / AFTER")
    print("=" * 70)
    rows = [
        ("group count", before_groups, after_groups),
        ("largest group", before_largest, after_largest),
        ("pinned images (identical fold, all 10 repeats)", before_pinned, after_pinned),
        ("mean pairwise fold agreement across repeats (chance = 0.20)",
         f"{before_agreement:.3f}", f"{after_agreement:.3f}"),
    ]
    width = max(len(r[0]) for r in rows)
    print(f"{'metric':<{width}}  {'before':>10}  {'after':>10}")
    for name, b, a in rows:
        print(f"{name:<{width}}  {b!s:>10}  {a!s:>10}")

    # =====================================================================
    # WRITE OUTPUT
    # =====================================================================
    df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
