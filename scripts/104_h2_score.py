"""H2 -- score AUROC / FPR@90R for each container output against ground truth.

Ground truth: reports/h2_ground_truth.csv (written by 103_h2_build_variants.py,
617 rows, aligned to A's stack_order.csv position -- B and C reuse the exact
same order, so one ground-truth file covers all variants and both container
images).

Usage:
    python scripts/104_h2_score.py --output-json <path/to/stacked-...json> --label "A_native/ensemble"
    python scripts/104_h2_score.py --summary   # print/write the accumulated table
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = "/workspace/RARE26"
GT_PATH = os.path.join(REPO_ROOT, "reports/h2_ground_truth.csv")
RESULTS_PATH = os.path.join(REPO_ROOT, "reports/h2_aspect_ratio_reproduction_results.json")


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


def score_one(output_json: str, label: str) -> dict:
    gt = pd.read_csv(GT_PATH).sort_values("position").reset_index(drop=True)
    y = (gt["class_label"] != "non-dysplastic").astype(int).to_numpy()

    with open(output_json) as fh:
        probs = np.asarray(json.load(fh), dtype=np.float64)
    if len(probs) != len(y):
        raise SystemExit(f"length mismatch: {len(probs)} probs vs {len(y)} ground-truth rows")

    auroc = float(roc_auc_score(y, probs))
    fpr90 = fpr_at_recall(y, probs)

    stats_path = os.path.join(os.path.dirname(output_json), "rare26_run_stats.json")
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as fh:
            stats = json.load(fh)

    result = {
        "label": label,
        "n": len(probs),
        "n_pos": int(y.sum()),
        "auroc": round(auroc, 4),
        "fpr90": round(fpr90, 4),
        "n_members": stats.get("n_members"),
        "device": stats.get("device"),
        "precision": stats.get("precision"),
        "n_fallback": stats.get("n_fallback"),
        "fallback_frac": stats.get("fallback_frac"),
        "n_unique_probs": stats.get("n_unique_probs"),
    }
    print(json.dumps(result, indent=2))

    all_results = []
    if os.path.isfile(RESULTS_PATH):
        with open(RESULTS_PATH) as fh:
            all_results = json.load(fh)
    all_results = [r for r in all_results if r["label"] != label]
    all_results.append(result)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[h2-score] appended to {RESULTS_PATH}")
    return result


def print_summary() -> None:
    if not os.path.isfile(RESULTS_PATH):
        raise SystemExit(f"no results yet: {RESULTS_PATH}")
    with open(RESULTS_PATH) as fh:
        results = json.load(fh)
    header = "{:32s} {:>6s} {:>6s} {:>8s} {:>8s} {:>10s}".format(
        "label", "n", "n_pos", "AUROC", "FPR@90R", "fallback%")
    print(header)
    for r in results:
        fb = f"{100*r['fallback_frac']:.2f}" if r.get("fallback_frac") is not None else "?"
        print("{:32s} {:6d} {:6d} {:8.4f} {:8.4f} {:>10s}".format(
            r["label"], r["n"], r["n_pos"], r["auroc"], r["fpr90"], fb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json")
    ap.add_argument("--label")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        print_summary()
        return 0
    if not args.output_json or not args.label:
        raise SystemExit("--output-json and --label required unless --summary")
    score_one(args.output_json, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
