"""N1 -- compute per-arm/per-cohort metrics, the pooled cross-cohort gate,
and apply the pre-registered acceptance rule exactly as written in
reports/n1_pre_registration.md. No re-litigation of the rule after seeing
numbers.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/112_n1_metrics.py'
"""
from __future__ import annotations

import json
import sys
from importlib import import_module

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
official_score = import_module("08_score").official_score

FPR_FLOOR = 0.0898
PPV_FLOOR = 0.0518
ARMS = ["N1-0", "N1-a", "N1-b", "N1-c"]


def fpr_at_recall(labels, probs, recall=0.90):
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


def ppv_at_recall(labels, probs, recall=0.90):
    order = np.argsort(-probs)
    lab = labels[order]
    n_pos = lab.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(lab)
    need = min(np.searchsorted(tp, np.ceil(recall * n_pos), side="left"), len(lab) - 1)
    tp_at = tp[need]
    n_at = need + 1
    return float(tp_at / n_at)


def main() -> int:
    with open("reports/n1_raw_logits.json") as fh:
        data = json.load(fh)

    r25_labels = np.array(data["rare25"]["labels"])
    evc_labels = np.array(data["evc"]["labels"])

    report = {}
    for arm in ARMS:
        r25_logit = np.array(data["rare25"]["logits_by_arm"][arm])
        evc_logit = np.array(data["evc"]["logits_by_arm"][arm])
        r25_prob = 1 / (1 + np.exp(-r25_logit))
        evc_prob = 1 / (1 + np.exp(-evc_logit))

        r25_metrics = {
            "auroc": round(float(roc_auc_score(r25_labels, r25_prob)), 4),
            "fpr90": round(fpr_at_recall(r25_labels, r25_prob), 4),
            "ppv90": round(ppv_at_recall(r25_labels, r25_prob), 4),
        }

        evc_pos_logit = evc_logit[evc_labels == 1]
        evc_auroc = round(float(roc_auc_score(evc_labels, evc_prob)), 4)
        evc_metrics = {
            "auroc": evc_auroc,
            "pos_logit_median": round(float(np.median(evc_pos_logit)), 4),
            "frac_pos_low_mode": round(float((evc_pos_logit < 0).mean()), 4),
        }

        pooled_labels = np.r_[r25_labels, evc_labels]
        pooled_probs = np.r_[r25_prob, evc_prob]
        gate = official_score(pooled_labels, pooled_probs, n_iterations=1000, imbalance_ratio=100, seed=20260821)
        gate_metrics = {"pooled_auroc": round(float(gate["AUROC"]), 4), "pooled_ppv90": round(float(gate["Score"]), 4)}

        report[arm] = {"rare25": r25_metrics, "evc": evc_metrics, "gate": gate_metrics}
        print(f"{arm}: RARE25={r25_metrics}  EVC={evc_metrics}  GATE={gate_metrics}", flush=True)

    control = report["N1-0"]
    verdicts = {}
    for arm in ["N1-a", "N1-b", "N1-c"]:
        d_ppv = report[arm]["gate"]["pooled_ppv90"] - control["gate"]["pooled_ppv90"]
        d_fpr = report[arm]["rare25"]["fpr90"] - control["rare25"]["fpr90"]
        cond_i = d_ppv > PPV_FLOOR
        cond_ii = d_fpr <= FPR_FLOOR
        verdicts[arm] = {
            "delta_pooled_ppv90_vs_control": round(d_ppv, 4), "condition_i_pass": bool(cond_i),
            "delta_rare25_fpr90_vs_control": round(d_fpr, 4), "condition_ii_pass": bool(cond_ii),
            "ACCEPT": bool(cond_i and cond_ii),
        }
    any_pass = any(v["ACCEPT"] for v in verdicts.values())

    print("\n=== ACCEPTANCE ===")
    print(json.dumps(verdicts, indent=2))
    print(f"\nANY ARM PASSES: {any_pass}")

    out = {"per_arm": report, "verdicts": verdicts, "any_pass": any_pass,
           "fpr_floor": FPR_FLOOR, "ppv_floor": PPV_FLOOR}
    with open("reports/n1_metrics_raw.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote reports/n1_metrics_raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
