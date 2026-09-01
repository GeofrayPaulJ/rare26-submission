"""D8 -- output serialisation precision. Reads the ACTUAL written JSON files
(not in-memory arrays), and reproduces the exact sigmoid -> [float(p) for p
in probs] -> json.dumps -> write -> read -> json.loads pipeline
(submission/rare26_infer/predict.py:328-329,428, submission/inference.py:65)
on D1's Group A logits, to test whether serialisation ALONE can reproduce
the leaderboard collapse (FPR@90R ~0.83-0.85 despite very different AUROC
across two sets -- the signature of a tied block at the decision
threshold).

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/75_d8_serialization.py'
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = "/workspace/RARE26"


def fpr_at_recall(y_true, y_score, recall=0.90):
    order = np.argsort(-np.asarray(y_score))
    y_true = np.asarray(y_true)[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    recall_arr = tp / n_pos
    idx = min(np.searchsorted(recall_arr, recall), len(y_true) - 1)
    return fp[idx] / n_neg if n_neg > 0 else float("nan")


def analyze_file(path: str, label: str) -> dict:
    with open(path) as fh:
        raw = fh.read()
    vals = json.loads(raw)
    arr = np.array(vals, dtype=np.float64)
    n = len(arr)
    n_distinct = len(set(vals))
    below = {t: int((arr < t).sum()) for t in (1e-4, 1e-6, 1e-8)}
    out = {"label": label, "path": path, "n": n, "n_distinct_in_file": n_distinct,
           "pct_distinct": round(100 * n_distinct / n, 3),
           "below_1e-4": below[1e-4], "below_1e-6": below[1e-6], "below_1e-8": below[1e-8]}
    print(json.dumps(out, indent=2))
    return out


def main() -> int:
    print("=== D8.1 -- distinct-value counts, read from the ACTUAL WRITTEN FILES ===")
    files = {
        "D1 full-pool (3088)": "runs/submission_test/d1_fullpool/output/stacked-neoplastic-lesion-likelihoods.json",
        "D2 shuffled (617)": "runs/submission_test/d2_order/output/stacked-neoplastic-lesion-likelihoods.json",
        "D6 case-0 (772)": "runs/submission_test/d6_multicase/output/stacked-neoplastic-lesion-likelihoods.json",
        "D3 baseline (200)": "runs/submission_test/d3_fov/baseline/output/stacked-neoplastic-lesion-likelihoods.json",
    }
    file_stats = []
    for label, rel in files.items():
        p = os.path.join(REPO_ROOT, rel)
        if os.path.exists(p):
            file_stats.append(analyze_file(p, label))

    print("\n=== D8.2 -- distinct count restricted to the bottom decile of scores ===")
    d1_path = os.path.join(REPO_ROOT, files["D1 full-pool (3088)"])
    with open(d1_path) as fh:
        probs = np.array(json.loads(fh.read()), dtype=np.float64)
    order = json.load(open(os.path.join(REPO_ROOT, "reports/d1_fullpool_parity.json")))
    decile_cut = np.percentile(probs, 10)
    bottom = probs[probs <= decile_cut]
    print(f"bottom-decile cutoff (10th pctile of D1 probs): {decile_cut:.6e}")
    print(f"n in bottom decile: {len(bottom)}, n distinct in bottom decile: {len(set(bottom.tolist()))}, "
          f"pct distinct: {100 * len(set(bottom.tolist())) / len(bottom):.2f}%")

    print("\n=== D8.4 -- round-trip reproduction test (D1 Group A) ===")
    stack_order = pd.read_csv(os.path.join(
        REPO_ROOT, "runs/submission_test/parity_pooled/interface_0/stack_order.csv"))
    with open(d1_path) as fh:
        d1_probs = np.array(json.loads(fh.read()), dtype=np.float64)
    stack_order = stack_order.copy()
    stack_order["container_prob"] = d1_probs
    group_a = stack_order[stack_order["own_fold"] == 0].reset_index(drop=True)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests/rare25_folds_v2.csv"))
    lbl = manifest[["filepath", "class_label"]].copy()
    lbl["y_true"] = (lbl["class_label"] != "non-dysplastic").astype(int)
    ga = group_a.merge(lbl[["filepath", "y_true"]], on="filepath", how="left")

    # reconstruct logits from the already-served, full-precision probabilities
    p = ga["container_prob"].values
    logits = np.log(p / (1 - p))

    # --- EXACT container pipeline: float64 sigmoid -> [float(p) for p in probs] -> json.dumps -> write -> read ---
    probs2 = 1.0 / (1.0 + np.exp(-logits))
    py_list = [float(x) for x in probs2]
    tmp_path = os.path.join(REPO_ROOT, "runs/submission_test/d8_roundtrip_test.json")
    with open(tmp_path, "w") as fh:
        fh.write(json.dumps(py_list, indent=4))
    with open(tmp_path) as fh:
        roundtripped = np.array(json.loads(fh.read()), dtype=np.float64)

    auc_before = roc_auc_score(ga["y_true"], p)
    auc_after = roc_auc_score(ga["y_true"], roundtripped)
    fpr_before = fpr_at_recall(ga["y_true"].values, p)
    fpr_after = fpr_at_recall(ga["y_true"].values, roundtripped)
    max_abs_diff = float(np.max(np.abs(p - roundtripped)))
    n_distinct_before = len(set(p.tolist()))
    n_distinct_after = len(set(roundtripped.tolist()))

    result = {
        "n": len(ga), "AUROC_before_roundtrip": round(float(auc_before), 4),
        "AUROC_after_roundtrip": round(float(auc_after), 4),
        "FPR@90R_before_roundtrip": round(float(fpr_before), 4),
        "FPR@90R_after_roundtrip": round(float(fpr_after), 4),
        "max_abs_value_diff": max_abs_diff,
        "n_distinct_before": n_distinct_before, "n_distinct_after": n_distinct_after,
        "REPRODUCES_LEADERBOARD_COLLAPSE": bool(auc_after < 0.65 and fpr_after > 0.80),
    }
    print(json.dumps(result, indent=2))

    with open(os.path.join(REPO_ROOT, "reports/d8_serialization.json"), "w") as fh:
        json.dump({"file_stats": file_stats, "roundtrip_test": result}, fh, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
