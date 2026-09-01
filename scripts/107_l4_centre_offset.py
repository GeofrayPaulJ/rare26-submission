"""L4 -- centre-offset heterogeneity.

JOB A: from the pooled-OOF parquets of the shipping checkpoint family
(runs/a4_checkpointed/r0_f{0-4}_s0, seed 0, standard grouped 5-fold CV --
each fold's ONE held-out model scores validation images from BOTH
centres, unlike LOCO where a model never sees its held-out centre at
all): per-centre logit mean/SD, the centre_1-vs-centre_2 mean offset for
positives and negatives separately, within-centre and pooled AUROC/
FPR@90R/PPV@90R.

JOB B: synthetic k-centre simulation. Partition the same pooled-OOF
scores into k synthetic centres (k=2,4,8,12), add a per-centre logit
offset ~ N(0, SD) at SD = observed (JOB A), 1.5x, 2x. 1000 draws each.
Also a bisection search at k=12 for the SD that brings median pooled
AUROC to 0.6466.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/107_l4_centre_offset.py'
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20260820)
FOLD_TEMPLATE = "runs/a4_checkpointed/r0_f{f}_s0/val_r0_f{f}_s0.parquet"
TARGET_AUROC = 0.6466
N_DRAWS = 1000


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


def ppv_at_recall(labels: np.ndarray, probs: np.ndarray, recall: float = 0.90) -> float:
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


def load_pooled() -> pd.DataFrame:
    frames = []
    for f in range(5):
        df = pd.read_parquet(FOLD_TEMPLATE.format(f=f))
        frames.append(df[["filepath", "centre", "label_int", "logit"]])
    pooled = pd.concat(frames, ignore_index=True)
    pooled["prob"] = 1 / (1 + np.exp(-pooled["logit"]))
    return pooled


def job_a(pooled: pd.DataFrame) -> dict:
    out = {}
    for cls, label_val in [("positives", 1), ("negatives", 0)]:
        sub = pooled[pooled["label_int"] == label_val]
        c1 = sub[sub["centre"] == "center_1"]["logit"]
        c2 = sub[sub["centre"] == "center_2"]["logit"]
        out[cls] = {
            "centre_1": {"n": len(c1), "logit_mean": round(float(c1.mean()), 4), "logit_sd": round(float(c1.std()), 4)},
            "centre_2": {"n": len(c2), "logit_mean": round(float(c2.mean()), 4), "logit_sd": round(float(c2.std()), 4)},
            "offset_c1_minus_c2": round(float(c1.mean() - c2.mean()), 4),
        }

    all_c1 = pooled[pooled["centre"] == "center_1"]["logit"]
    all_c2 = pooled[pooled["centre"] == "center_2"]["logit"]
    out["overall_class_agnostic"] = {
        "centre_1": {"logit_mean": round(float(all_c1.mean()), 4), "logit_sd": round(float(all_c1.std()), 4)},
        "centre_2": {"logit_mean": round(float(all_c2.mean()), 4), "logit_sd": round(float(all_c2.std()), 4)},
        "offset_c1_minus_c2": round(float(all_c1.mean() - all_c2.mean()), 4),
    }

    def metrics(sub: pd.DataFrame) -> dict:
        y, p = sub["label_int"].to_numpy(), sub["prob"].to_numpy()
        return {
            "n": len(sub), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
            "auroc": round(float(roc_auc_score(y, p)), 4),
            "fpr90": round(fpr_at_recall(y, p), 4),
            "ppv90": round(ppv_at_recall(y, p), 4),
        }

    out["within_centre_1"] = metrics(pooled[pooled["centre"] == "center_1"])
    out["within_centre_2"] = metrics(pooled[pooled["centre"] == "center_2"])
    out["pooled"] = metrics(pooled)
    out["pooled_minus_within_gap"] = {
        "auroc": round(out["pooled"]["auroc"] - min(out["within_centre_1"]["auroc"], out["within_centre_2"]["auroc"]), 4),
        "auroc_vs_worse_within": "pooled - worse-within",
        "fpr90": round(out["pooled"]["fpr90"] - max(out["within_centre_1"]["fpr90"], out["within_centre_2"]["fpr90"]), 4),
        "ppv90": round(out["pooled"]["ppv90"] - min(out["within_centre_1"]["ppv90"], out["within_centre_2"]["ppv90"]), 4),
    }
    return out


def simulate_once(logits: np.ndarray, labels: np.ndarray, k: int, sd: float, rng: np.random.Generator) -> tuple:
    n = len(logits)
    centre_id = rng.integers(0, k, n)
    offsets = rng.normal(0.0, sd, k)
    shifted = logits + offsets[centre_id]
    probs = 1 / (1 + np.exp(-shifted))
    auc = roc_auc_score(labels, probs)
    ppv = ppv_at_recall(labels, probs)
    return auc, ppv


def sweep(logits: np.ndarray, labels: np.ndarray, ks: list, sds: dict, n_draws: int = N_DRAWS) -> dict:
    results = {}
    for sd_label, sd in sds.items():
        results[sd_label] = {}
        for k in ks:
            aucs, ppvs = [], []
            for _ in range(n_draws):
                a, p = simulate_once(logits, labels, k, sd, RNG)
                aucs.append(a)
                ppvs.append(p)
            results[sd_label][f"k={k}"] = {
                "sd": round(sd, 4),
                "median_auroc": round(float(np.median(aucs)), 4),
                "median_ppv90": round(float(np.median(ppvs)), 4),
                "auroc_p10_p90": [round(float(np.percentile(aucs, 10)), 4), round(float(np.percentile(aucs, 90)), 4)],
            }
    return results


def find_sd_for_target(logits: np.ndarray, labels: np.ndarray, k: int, target: float,
                        sd_start: float = 1.0, n_draws: int = 400, max_expand: int = 20, tol: int = 40) -> dict:
    """Bisection on median pooled AUROC as a function of SD (non-increasing in SD),
    at fixed k. First expands sd_hi geometrically until median_auroc drops to/below
    target (or gives up and reports unreachable within max_expand doublings), THEN
    bisects the bracket."""
    trace = []

    def median_auroc_at(sd: float) -> float:
        aucs = [simulate_once(logits, labels, k, sd, RNG)[0] for _ in range(n_draws)]
        return float(np.median(aucs))

    lo, m_lo = 0.0, median_auroc_at(0.0)
    trace.append({"sd": 0.0, "median_auroc": round(m_lo, 4)})

    hi = sd_start
    m_hi = median_auroc_at(hi)
    trace.append({"sd": round(hi, 4), "median_auroc": round(m_hi, 4)})
    expansions = 0
    while m_hi > target and expansions < max_expand:
        hi *= 2
        m_hi = median_auroc_at(hi)
        trace.append({"sd": round(hi, 4), "median_auroc": round(m_hi, 4)})
        expansions += 1

    if m_hi > target:
        return {"k": k, "target_auroc": target, "sd_found": None,
                "sd_hi_tested": round(hi, 4), "median_auroc_at_sd_hi": round(m_hi, 4),
                "note": f"UNREACHABLE: median AUROC still {m_hi:.4f} > target at SD={hi:.2f} "
                        f"after {expansions} doublings", "trace": trace}

    # bracket confirmed: m_lo > target > m_hi -- bisect
    for _ in range(tol):
        mid = (lo + hi) / 2
        m_mid = median_auroc_at(mid)
        trace.append({"sd": round(mid, 4), "median_auroc": round(m_mid, 4)})
        if m_mid > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.005:
            break
    sd_found = (lo + hi) / 2
    return {"k": k, "target_auroc": target, "sd_found": round(sd_found, 4),
            "sd_hi_tested": round(hi, 4), "median_auroc_at_sd_hi": round(m_hi, 4), "trace": trace}


def main() -> int:
    pooled = load_pooled()
    a = job_a(pooled)

    print("=== JOB A ===")
    print(json.dumps(a, indent=2))

    logits = pooled["logit"].to_numpy()
    labels = pooled["label_int"].to_numpy()

    observed_sd_pos = abs(a["positives"]["offset_c1_minus_c2"])
    observed_sd_neg = abs(a["negatives"]["offset_c1_minus_c2"])
    observed_sd_overall = abs(a["overall_class_agnostic"]["offset_c1_minus_c2"])
    print(f"\nobserved offsets: positives={observed_sd_pos:.4f}  negatives={observed_sd_neg:.4f}  "
          f"overall(class-agnostic)={observed_sd_overall:.4f}")

    sds = {
        "1.0x_observed": observed_sd_overall,
        "1.5x_observed": observed_sd_overall * 1.5,
        "2.0x_observed": observed_sd_overall * 2.0,
    }
    b_sweep = sweep(logits, labels, [2, 4, 8, 12], sds)
    print("\n=== JOB B sweep ===")
    print(json.dumps(b_sweep, indent=2))

    b_search = find_sd_for_target(logits, labels, k=12, target=TARGET_AUROC, sd_start=1.0)
    print("\n=== JOB B: SD required at k=12 for median AUROC = 0.6466 ===")
    print(json.dumps({k: v for k, v in b_search.items() if k != "trace"}, indent=2))
    print(f"as a multiple of observed overall offset ({observed_sd_overall:.4f}): "
          f"{(b_search['sd_found'] / observed_sd_overall) if b_search['sd_found'] else 'UNREACHABLE'}")

    out = {
        "job_a": a,
        "observed_offsets": {"positives": observed_sd_pos, "negatives": observed_sd_neg, "overall": observed_sd_overall},
        "job_b_sweep": b_sweep,
        "job_b_sd_search_k12": b_search,
    }
    with open("reports/l4_centre_offset_raw.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote reports/l4_centre_offset_raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
