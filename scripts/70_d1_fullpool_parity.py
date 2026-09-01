"""D1 -- full-pool container test. Runs the SUBMITTED single-member
container's scores (already produced by do_test_run.sh over the
parity-pooled 3088-image stack) against ground truth and against the
harness's own pooled-OOF parquets.

Splits into two groups because they answer different questions:
  GROUP A (n=617, own_fold==0): the images fold-0's model (the one the
    container ships) was actually held out on. Container vs harness here
    is a fair same-model comparison -- this is what JOB F's 617-image
    parity check already covers.
  GROUP B (n=2471, own_fold!=0): images that were TRAINING data for the
    shipped model. Container score here vs each image's own properly-held-out
    harness model (a DIFFERENT model per image) is not a same-model parity
    check -- it answers a different question: does the container, when fed
    images/geometries outside its own held-out set, still produce sane
    scores, or does something break for a subset of them.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/70_d1_fullpool_parity.py'
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = "/workspace/RARE26"
STACK_DIR = os.path.join(REPO_ROOT, "runs/submission_test/parity_pooled/interface_0")
OUT_DIR = os.path.join(REPO_ROOT, "runs/submission_test/d1_fullpool/output")


def fpr_at_recall(y_true, y_score, recall=0.90):
    order = np.argsort(-y_score)
    y_true = np.asarray(y_true)[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    recall_arr = tp / n_pos
    idx = np.searchsorted(recall_arr, recall)
    idx = min(idx, len(y_true) - 1)
    return fp[idx] / n_neg if n_neg > 0 else float("nan")


def main() -> int:
    order = pd.read_csv(os.path.join(STACK_DIR, "stack_order.csv"))
    with open(os.path.join(OUT_DIR, "stacked-neoplastic-lesion-likelihoods.json")) as fh:
        probs = json.load(fh)
    assert len(probs) == len(order), f"{len(probs)} probs vs {len(order)} order rows"
    order = order.copy()
    order["container_prob"] = probs
    order["container_logit"] = np.log(np.array(probs) / (1 - np.array(probs) + 1e-12) + 1e-12)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests/rare25_folds_v2.csv"))
    lbl = manifest[["filepath", "class_label"]].copy()
    lbl["y_true"] = (lbl["class_label"] != "non-dysplastic").astype(int)
    df = order.merge(lbl[["filepath", "y_true"]], on="filepath", how="left")
    assert df["y_true"].isna().sum() == 0, "unmatched filepaths against manifest"

    harness = []
    for k in range(5):
        p = os.path.join(REPO_ROOT, f"runs/a4_checkpointed/r0_f{k}_s0/val_r0_f{k}_s0.parquet")
        h = pd.read_parquet(p)[["filepath", "logit", "label_int"]]
        harness.append(h)
    harness = pd.concat(harness, ignore_index=True)
    harness = harness.rename(columns={"logit": "harness_logit", "label_int": "harness_label"})
    df = df.merge(harness, on="filepath", how="left")
    assert df["harness_logit"].isna().sum() == 0, "missing harness OOF logit for some image"
    assert (df["y_true"] == df["harness_label"]).all(), "label mismatch manifest vs harness parquet"

    def block(sub: pd.DataFrame, name: str) -> dict:
        auc = roc_auc_score(sub["y_true"], sub["container_prob"])
        fpr90 = fpr_at_recall(sub["y_true"].values, sub["container_prob"].values)
        rho, _ = spearmanr(sub["container_logit"], sub["harness_logit"])
        delta = (sub["container_logit"] - sub["harness_logit"]).abs()
        harness_auc = roc_auc_score(sub["y_true"], sub["harness_logit"])
        return {"n": len(sub), "container_AUROC": round(float(auc), 4),
                "harness_AUROC": round(float(harness_auc), 4),
                "container_FPR@90R": round(float(fpr90), 4),
                "spearman_rho_vs_harness": round(float(rho), 5),
                "logit_delta_mean": round(float(delta.mean()), 4),
                "logit_delta_max": round(float(delta.max()), 4)}

    group_a = df[df["own_fold"] == 0]
    group_b = df[df["own_fold"] != 0]

    result = {
        "ALL_3088": block(df, "all"),
        "GROUP_A_own_fold0_same_model_617": block(group_a, "A"),
        "GROUP_B_other_folds_in_sample_2471": block(group_b, "B"),
    }

    print(json.dumps(result, indent=2))
    with open(os.path.join(REPO_ROOT, "reports/d1_fullpool_parity.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    # worst offenders for follow-up (D2-D5)
    df["abs_delta_from_harness_rank"] = (df["container_prob"].rank() - df["harness_logit"].rank()).abs()
    worst = df.sort_values("abs_delta_from_harness_rank", ascending=False).head(20)
    worst[["filepath", "own_fold", "y_true", "container_prob", "harness_logit"]].to_csv(
        os.path.join(REPO_ROOT, "reports/d1_worst_rank_disagreements.csv"), index=False)
    print("worst rank-disagreement images written to reports/d1_worst_rank_disagreements.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
