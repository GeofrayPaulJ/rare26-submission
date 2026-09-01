"""L1 -- grouping stringency sweep. NO retraining: filters EXISTING
checkpoints' held-out predictions down to the subset that remains
leak-free under progressively more conservative identity-grouping, and
recomputes AUROC/FPR@90R on the survivors.

CORRECTION FOUND DURING SETUP: the actual production grouping threshold
(the one that produced manifests/rare25_folds_v2.csv, the shipping
splits) is pHash Hamming <=12, NOT <=8 as assumed in the instruction --
confirmed by reproducing rare25_canonical.csv's group_id exactly
(2562 groups, largest 262) only at threshold=12 (checked 4,6,8,10,12,14,16).
Levels below are renumbered to reflect this; the reasoning (sweep from
baseline toward stricter) is unchanged, only the starting point.

deploy_a4_full_s{0..4} are NOT usable at any level -- S1V already
established these are full-data checkpoints with no held-out set by
construction, so there is no held-out prediction to filter. Uses
a4_checkpointed (runs/a4_checkpointed/r0_f{0..4}_s0), the actual 5-fold
CV checkpoints whose logits are the shipping ensemble's own members.

Primary target: FOLD 0 alone (r0_f0_s0, 617 held-out images) -- this is
the exact checkpoint/number `e2_checkpoint_identity.md` recorded as
AUROC 0.9358, the number under trial. Pooled-across-5-folds reported
as a secondary robustness check.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/101_l1_grouping_stringency.py'
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CANONICAL = "manifests/rare25_folds_v2.csv"  # the CORRECTED fold_r0 the checkpoints actually
                                              # trained under (rare25_canonical.csv's own
                                              # fold_r0 is the pre-refold BROKEN assignment,
                                              # deliberately preserved as an audit trail by
                                              # 16_refold.py -- using it here was the bug that
                                              # produced spurious "538/617 excluded" at L0)
GROUPED = "manifests/rare25_manifest_grouped.csv"  # has phash
FOLD_TEMPLATE = "runs/a4_checkpointed/r0_f{f}_s0/val_r0_f{f}_s0.parquet"


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


def phash_cluster(bits: np.ndarray, threshold: int, mutual_k: int = 5) -> np.ndarray:
    """Verbatim port of 06_regroup.py's cluster() -- mutual-kNN + union-find."""
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
    return np.array([remap[r] for r in roots])


def fpr_at_recall(labels: np.ndarray, probs: np.ndarray, recall: float = 0.90) -> float:
    order = np.argsort(-probs)
    lab = labels[order]
    n_pos = lab.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(lab)
    need = min(np.searchsorted(tp, np.ceil(recall * n_pos), side="left"), len(lab) - 1)
    fp = (np.arange(len(lab)) + 1 - tp)[need]
    n_neg = len(lab) - n_pos
    return float(fp / max(n_neg, 1))


def evaluate_level(name: str, group_series: pd.Series, canon: pd.DataFrame,
                    fold_vals: dict) -> dict:
    """group_series: Series indexed like canon (same row order), giving the
    new group label for every kept-for-training image. fold_vals: dict
    fold_idx -> val_df (from the parquet, has filepath/label_int/logit)."""
    canon = canon.copy()
    canon["_new_group"] = group_series.values
    fp_to_group = dict(zip(canon["filepath"], canon["_new_group"]))
    fp_to_fold = dict(zip(canon["filepath"], canon["fold_r0"]))

    results = {}
    pooled_frames = []
    for f, val_df in fold_vals.items():
        train_fps = set(canon.loc[(canon["fold_r0"] != f) & (canon["fold_r0"] != -1), "filepath"])
        train_groups = set(fp_to_group[fp] for fp in train_fps if fp in fp_to_group)

        v = val_df.copy()
        v["_new_group"] = v["filepath"].map(fp_to_group)
        v["_leaky"] = v["_new_group"].isin(train_groups)
        clean = v[~v["_leaky"]]

        probs = 1 / (1 + np.exp(-clean["logit"].to_numpy()))
        labels = clean["label_int"].to_numpy()
        n_total, n_clean = len(v), len(clean)
        n_pos_clean = int(labels.sum())
        auroc = roc_auc_score(labels, probs) if n_pos_clean > 0 and n_pos_clean < n_clean else float("nan")
        fpr90 = fpr_at_recall(labels, probs) if n_pos_clean > 0 else float("nan")
        results[f] = {
            "n_total": n_total, "n_clean": n_clean, "n_excluded": n_total - n_clean,
            "n_pos_clean": n_pos_clean, "auroc": auroc, "fpr90": fpr90,
        }
        pooled_frames.append(clean)

    pooled = pd.concat(pooled_frames, ignore_index=True)
    pooled_probs = 1 / (1 + np.exp(-pooled["logit"].to_numpy()))
    pooled_labels = pooled["label_int"].to_numpy()
    pooled_n_pos = int(pooled_labels.sum())
    if len(pooled) == 0 or pooled_n_pos == 0 or pooled_n_pos == len(pooled):
        pooled_auroc = float("nan")
        pooled_fpr90 = float("nan")
    else:
        pooled_auroc = roc_auc_score(pooled_labels, pooled_probs)
        pooled_fpr90 = fpr_at_recall(pooled_labels, pooled_probs)

    fold0 = results[0]
    print(f"[{name}] fold0: n={fold0['n_clean']}/{fold0['n_total']} "
          f"(excluded {fold0['n_excluded']}) AUROC={fold0['auroc']:.4f} "
          f"FPR90={fold0['fpr90']:.4f}", flush=True)
    print(f"[{name}] pooled(5 folds): n={len(pooled)}/{sum(r['n_total'] for r in results.values())} "
          f"AUROC={pooled_auroc:.4f} FPR90={pooled_fpr90:.4f}", flush=True)

    return {
        "level": name,
        "fold0": fold0,
        "pooled": {"n": len(pooled), "n_total": sum(r['n_total'] for r in results.values()),
                   "auroc": pooled_auroc, "fpr90": pooled_fpr90},
        "per_fold": results,
    }


def main() -> int:
    canon = pd.read_csv(CANONICAL)
    kept = canon[canon["keep_for_training"]].reset_index(drop=True)
    print(f"canonical: {len(canon)} rows, kept_for_training: {len(kept)}", flush=True)

    grouped = pd.read_csv(GROUPED)
    # align grouped (has phash) to kept, by filepath
    grouped_idx = grouped.set_index("filepath")
    bits_by_fp = {fp: hex_to_bits(h) for fp, h in zip(grouped["filepath"], grouped["phash"])}

    fold_vals = {}
    for f in range(5):
        fold_vals[f] = pd.read_parquet(FOLD_TEMPLATE.format(f=f))
    print(f"loaded {sum(len(v) for v in fold_vals.values())} held-out predictions across 5 folds\n",
          flush=True)

    all_results = []

    # --- Level 0: baseline, ACTUAL production grouping (group_id_v2, threshold=12) ---
    # Sanity check only -- should reproduce known figures with ~0 exclusions.
    baseline_group = kept["filepath"].map(dict(zip(canon["filepath"], canon["group_id"].astype(str) + "_" + canon["centre"])))
    r = evaluate_level("L0 baseline (threshold=12, group_id_v2)", baseline_group, kept, fold_vals)
    all_results.append(r)

    # --- pHash sweep: 13, 14, 16 (12 is baseline; the instruction's "8" and "16"
    # relabelled given the threshold=12 correction above) ---
    bits_kept = np.stack([bits_by_fp[fp] for fp in kept["filepath"]])
    for t in (13, 14, 16):
        labels_arr = phash_cluster(bits_kept, t, mutual_k=5)
        sizes = pd.Series(labels_arr).value_counts()
        largest_pct = 100 * sizes.iloc[0] / len(labels_arr)
        centres = kept["centre"].to_numpy()
        group_v2 = pd.Series([f"{labels_arr[i]}_{centres[i]}" for i in range(len(labels_arr))])
        print(f"\n--- pHash threshold={t}: {len(sizes)} groups, largest={sizes.iloc[0]} "
              f"({largest_pct:.1f}%) {'** DEGENERATE (>20%) **' if largest_pct > 20 else 'usable'} ---",
              flush=True)
        r = evaluate_level(f"L_phash{t}", group_v2, kept, fold_vals)
        r["largest_pct"] = largest_pct
        r["degenerate"] = largest_pct > 20
        all_results.append(r)

    # --- Level 3: agglomerative clustering on manifest features ---
    feat_cols_present = ["width", "height", "file_size_bytes", "rg_ratio"]
    feat = kept[feat_cols_present].copy()
    feat["has_redaction"] = kept["has_redaction"].astype(int)
    feat["redaction_count"] = kept["redaction_count"].fillna(0)
    feat["redaction_area_pct"] = kept["redaction_area_pct"].fillna(0)
    for c in ["largest_left", "largest_top", "largest_right", "largest_bottom"]:
        feat[c] = kept[c].fillna(-1)
    X = StandardScaler().fit_transform(feat.to_numpy())

    for dist_thresh, tag in [(3.0, "coarse"), (5.0, "coarser")]:
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh, linkage="ward")
        labels_arr = agg.fit_predict(X)
        sizes = pd.Series(labels_arr).value_counts()
        largest_pct = 100 * sizes.iloc[0] / len(labels_arr)
        centres = kept["centre"].to_numpy()
        group_v2 = pd.Series([f"agg{labels_arr[i]}_{centres[i]}" for i in range(len(labels_arr))])
        print(f"\n--- L3 agglomerative ({tag}, dist_threshold={dist_thresh}): "
              f"{len(sizes)} groups, largest={sizes.iloc[0]} ({largest_pct:.1f}%) "
              f"{'** DEGENERATE (>20%) **' if largest_pct > 20 else 'usable'} ---", flush=True)
        r = evaluate_level(f"L3_agglomerative_{tag}", group_v2, kept, fold_vals)
        r["largest_pct"] = largest_pct
        r["degenerate"] = largest_pct > 20
        r["n_clusters"] = len(sizes)
        all_results.append(r)

    # --- Level 4: resolution-cluster x centre, the coarsest defensible proxy ---
    res_group = kept["width"].astype(str) + "x" + kept["height"].astype(str) + "_" + kept["centre"]
    sizes = res_group.value_counts()
    largest_pct = 100 * sizes.iloc[0] / len(res_group)
    print(f"\n--- L4 resolution x centre: {len(sizes)} groups, largest={sizes.iloc[0]} "
          f"({largest_pct:.1f}%) {'** DEGENERATE (>20%) **' if largest_pct > 20 else 'usable'} ---",
          flush=True)
    r = evaluate_level("L4_resolution_x_centre", res_group, kept, fold_vals)
    r["largest_pct"] = largest_pct
    r["degenerate"] = largest_pct > 20
    r["n_clusters"] = len(sizes)
    all_results.append(r)

    with open("reports/l1_grouping_stringency_raw.json", "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print("\nwrote reports/l1_grouping_stringency_raw.json", flush=True)

    print("\n=== SUMMARY CURVE (fold0 alone, the 0.9358 checkpoint) ===")
    for r in all_results:
        f0 = r["fold0"]
        print(f"  {r['level']:40s} n={f0['n_clean']:4d}/{f0['n_total']:4d}  "
              f"AUROC={f0['auroc']:.4f}  FPR90={f0['fpr90']:.4f}")

    print("\n=== SUMMARY CURVE (pooled, 5 folds) ===")
    for r in all_results:
        p = r["pooled"]
        print(f"  {r['level']:40s} n={p['n']:4d}/{p['n_total']:4d}  "
              f"AUROC={p['auroc']:.4f}  FPR90={p['fpr90']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
