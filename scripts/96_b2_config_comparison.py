"""B2 -- sanity comparison, new (full_s0) vs old (fallback r0_f0_s0)
single-checkpoint configuration, on the SAME 617-image fold-0 reference
stack F3 already built (runs/submission_test/f3_battery/baseline).

METHODOLOGICAL CAVEAT, stated up front and in the report: this is NOT a
clean apples-to-apples comparison. The OLD checkpoint (r0_f0_s0) never
trained on these 617 images -- they are its own held-out fold, a genuine
OOF evaluation. The NEW checkpoint (deploy_a4_full_s0) trains on ALL
3,088 images with no holdout by construction (S1V) -- these 617 images
were IN its training set. So the new checkpoint's number here is
train-set performance, not OOF, and is expected to look optimistic
relative to the old checkpoint's genuine OOF number. This script exists
to catch anything grossly broken (e.g. the new checkpoint scoring at
chance, which WOULD be alarming even given the caveat), not to make a
real accuracy claim -- V1's platform upload is the only clean evidence
for that.

CPU only (CUDA_VISIBLE_DEVICES=), zero GPU.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/96_b2_config_comparison.py'
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/RARE26/submission")

from rare26_infer.predict import predict_stack  # noqa: E402

REPO_ROOT = "/workspace/RARE26"
STACK_PATH = os.path.join(
    REPO_ROOT, "runs", "submission_test", "f3_battery", "baseline", "interface_0",
    "images", "stacked-barretts-esophagus-endoscopy",
)
MANIFEST = os.path.join(REPO_ROOT, "reports", "f3_reference_manifest.csv")

CONFIGS = {
    "old_fallback_r0f0s0": [
        os.path.join(REPO_ROOT, "submission", "resources", "fallback", "a4_r0_f0_s0.pth")
    ],
    "new_full_s0": [
        os.path.join(REPO_ROOT, "submission", "resources", "full_s0_single",
                      "a4full_r0_f_all_s0.pth")
    ],
}


def fpr_at_recall(labels, scores, recall=0.90):
    order = np.argsort(-scores)
    labels = np.asarray(labels)[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    tp = np.cumsum(labels)
    need = np.searchsorted(tp, np.ceil(recall * n_pos), side="left")
    need = min(need, len(labels) - 1)
    fp = (np.arange(len(labels)) + 1 - tp)[need]
    return float(fp / max(n_neg, 1))


def main() -> int:
    man = pd.read_csv(MANIFEST).sort_values("slice_index").reset_index(drop=True)
    labels = man["label_int"].to_numpy()
    print(f"[b2] {len(man)} images, {int(labels.sum())} positive", flush=True)

    results = {}
    for name, weights in CONFIGS.items():
        print(f"\n[b2] scoring config={name} weights={weights}", flush=True)
        probs, stats = predict_stack(STACK_PATH, weights, num_workers=8, batch_size=32)
        probs = np.asarray(probs)
        auroc = roc_auc_score(labels, probs)
        fpr90 = fpr_at_recall(labels, probs, 0.90)
        results[name] = {
            "auroc": auroc, "fpr_at_90r": fpr90,
            "n_fallback": stats["n_fallback"], "fallback_frac": stats["fallback_frac"],
            "logit_min": stats["logit_min"], "logit_max": stats["logit_max"],
            "total_seconds": stats["total_seconds"],
        }
        print(f"[b2]   AUROC={auroc:.4f}  FPR@90R={fpr90:.4f}  "
              f"fallback={stats['fallback_frac']:.3f}  "
              f"logit range=[{stats['logit_min']:.2f}, {stats['logit_max']:.2f}]  "
              f"{stats['total_seconds']:.1f}s", flush=True)

    out_path = os.path.join(REPO_ROOT, "reports", "b2_config_comparison.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[b2] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
