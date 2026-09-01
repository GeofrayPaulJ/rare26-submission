"""Remediation item A4 (2026-08-07): does fp16 reorder images near the
threshold relative to the bf16 training harness?

Scores the pooled-OOF set (all 5 folds' held-out images, 3088 total) through
the submission container's actual fp16 forward path and compares against the
bf16 logits src/train.py dumped for the same images. Each image is scored by
its OWN held-out fold's member only (the classic pooled-OOF convention used
throughout this project's reports -- see reports/a4.md, reports/gastronet.md),
extracted from the ensemble container's per-member diagnostic dump. This
isolates precision (fp16 container vs bf16 harness) from the
ensemble-averaging question, which is a separate concern: at real (unseen)
test time every member scores every image and the logits are averaged, but
there is no harness reference for that quantity (see predict_stack's
docstring in submission/rare26_infer/predict.py for why).

IMPORTANT CONTEXT FOR THE FPR@90R FIGURES BELOW. This script's FPR@90R uses
ONLY seed 0 per fold (5 units total -- the shipped container's actual
composition), not the 5-seed x 5-fold (25-unit) pooling behind the
frequently-cited 0.0280 headline in reports/gastronet.md. Single-seed pooled
FPR@90R running substantially higher (worse) than the multi-seed-pooled
figure is an already-established pattern in this project -- see
reports/magnitude_sweep.md's headline table, "1-seed med" column (e.g. A0:
pooled k=5 0.0427 vs 1-seed med 0.1061) -- so an absolute FPR@90R around 0.12
here is expected, not a bug, and is NOT comparable to 0.0280 that appears
elsewhere. What IS comparable, and is the actual point of this script, is the
DELTA between the fp16 and bf16 columns computed on the identical single-seed
set.

Prerequisites (run once, from repo root):
    python submission/tools/make_test_stack.py parity-pooled \\
      --out-dir runs/submission_test/parity_pooled/interface_0
    docker run ... --env RARE26_DUMP_MEMBER_LOGITS=/output/member_logits.npy \\
      rare26-convnext-base-ensemble   # (see submission/do_test_run.sh)

    python scripts/47_a4_fp16_vs_bf16.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_curve

TARGET_RECALL = 0.90
PARITY_POOLED_DIR = "runs/submission_test/parity_pooled"
MANIFEST = "submission/resources/ensemble/manifest.txt"
REFERENCE_DIR = "runs/a4_checkpointed"
OUT_JSON = "reports/a4_fp16_vs_bf16_rank_agreement.json"


def fpr_at_recall(y_true, y_score, recall: float = TARGET_RECALL) -> float:
    fpr, tpr, _ = roc_curve(y_true, np.asarray(y_score))
    return float(np.interp(recall, tpr, fpr))


def fold_of(name: str) -> int:
    stem = name.split(".")[0]
    part = next(p for p in stem.split("_") if p.startswith("f") and p[1:].isdigit())
    return int(part[1:])


def main() -> None:
    member_logits = np.load(f"{PARITY_POOLED_DIR}/output/member_logits.npy")
    order = pd.read_csv(f"{PARITY_POOLED_DIR}/interface_0/stack_order.csv")
    with open(MANIFEST) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    fold_of_col = [fold_of(n) for n in names]

    ref_by_fold = {
        fold: pd.read_parquet(
            f"{REFERENCE_DIR}/r0_f{fold}_s0/val_r0_f{fold}_s0.parquet"
        ).set_index("filepath")
        for fold in sorted(set(fold_of_col))
    }

    with open(f"{PARITY_POOLED_DIR}/output/rare26_run_stats.json") as f:
        stats = json.load(f)
    fallback_idx = set(stats.get("fallback_indices", []))

    rows = []
    for pos, fp in enumerate(order["filepath"]):
        own_fold = int(order["own_fold"].iloc[pos])
        ref = ref_by_fold[own_fold]
        if fp not in ref.index:
            continue
        col = fold_of_col.index(own_fold)
        rows.append((
            fp, own_fold, float(member_logits[pos, col]),
            float(ref.loc[fp, "logit"]), int(ref.loc[fp, "label_int"]),
            pos in fallback_idx,
        ))

    df = pd.DataFrame(rows, columns=["filepath", "fold", "fp16_logit",
                                      "bf16_logit", "label", "is_fallback"])
    print(f"total images matched: {len(df)}")
    print(f"fallback-path images (excluded from the sharp precision comparison): "
          f"{df['is_fallback'].sum()}")

    def block(sub: pd.DataFrame, label: str) -> dict:
        rho, rho_p = spearmanr(sub["fp16_logit"], sub["bf16_logit"])
        tau, tau_p = kendalltau(sub["fp16_logit"], sub["bf16_logit"])
        fpr_fp16 = fpr_at_recall(sub["label"], sub["fp16_logit"])
        fpr_bf16 = fpr_at_recall(sub["label"], sub["bf16_logit"])
        d_logit = (sub["fp16_logit"] - sub["bf16_logit"]).abs()
        print(f"\n--- {label} (n={len(sub)}) ---")
        print(f"Spearman rho : {rho:.6f}  (p={rho_p:.3e})")
        print(f"Kendall tau  : {tau:.6f}  (p={tau_p:.3e})")
        print(f"FPR@90R fp16 (container): {fpr_fp16:.4f}")
        print(f"FPR@90R bf16 (harness)  : {fpr_bf16:.4f}")
        print(f"delta (fp16 - bf16)     : {fpr_fp16 - fpr_bf16:+.4f}")
        print(f"max |logit diff|  : {d_logit.max():.4f}")
        print(f"mean |logit diff| : {d_logit.mean():.4f}")
        return {
            "n": len(sub), "spearman_rho": rho, "spearman_p": rho_p,
            "kendall_tau": tau, "kendall_p": tau_p,
            "fpr_at_90_recall_fp16_container": fpr_fp16,
            "fpr_at_90_recall_bf16_harness": fpr_bf16,
            "delta_fpr_at_90_recall": fpr_fp16 - fpr_bf16,
            "max_abs_logit_diff": float(d_logit.max()),
            "mean_abs_logit_diff": float(d_logit.mean()),
        }

    detector_block = block(df[~df["is_fallback"]], "DETECTOR PATH")
    all_block = block(df, f"ALL {len(df)} IMAGES (incl. {df['is_fallback'].sum()} fallback)")

    out = {
        "n_total": len(df), "n_fallback": int(df["is_fallback"].sum()),
        "detector_path": detector_block, "all_images": all_block,
        "note": ("Single-seed (5-unit) pooled OOF, not the 5-seed x 5-fold "
                 "25-unit pooling behind the 0.0280 headline in "
                 "reports/gastronet.md -- see module docstring."),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten: {OUT_JSON}")


if __name__ == "__main__":
    main()
