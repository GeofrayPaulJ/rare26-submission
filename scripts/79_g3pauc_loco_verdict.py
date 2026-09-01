"""G3+pAUC LOCO verdict, per reports/g3_component_pre_registration.md.

ACCEPT iff G3+pAUC beats the G3 baseline (runs/g3_rn50_gastronet_loco) on
BOTH n=1 LOCO directions, paired BY SEED, with >= 4/5 seeds agreeing in
sign per direction. Same structural test as pAUC-P1 vs A4-pinned
(scripts/64_paucloco_chain.py's convention, reports/pauc_p1_loco.md's
table format). Zero GPU -- reads canonical prediction parquets only.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = "/workspace/RARE26"
ARM_DIR = os.path.join(REPO_ROOT, "runs/g3_pauc_loco")
BASE_DIR = os.path.join(REPO_ROOT, "runs/g3_rn50_gastronet_loco")
OUT_MD = os.path.join(REPO_ROOT, "reports/g3_pauc_loco.md")
OUT_JSON = os.path.join(REPO_ROOT, "reports/g3_pauc_loco.json")


def fpr_at_recall(y_true, y_score, recall=0.90):
    order = np.argsort(-np.asarray(y_score))
    y = np.asarray(y_true)[order]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    idx = min(np.searchsorted(tp / n_pos, recall), len(y) - 1)
    return fp[idx] / n_neg


def unit_fpr(run_dir: str, unit: str) -> float:
    # canonical parquet: val_<unit>.parquet at the unit dir root, or the
    # last-epoch preds file -- match run_cv.py's canonical naming
    p = os.path.join(run_dir, unit, f"val_{unit}.parquet")
    if not os.path.exists(p):
        cands = sorted(
            f for f in os.listdir(os.path.join(run_dir, unit, "preds"))
            if f.endswith(".parquet"))
        p = os.path.join(run_dir, unit, "preds", cands[-1])
    df = pd.read_parquet(p)
    return float(fpr_at_recall(df["label_int"].values, df["logit"].values))


def main() -> int:
    result = {"directions": {}}
    accepts = []
    lines = ["# G3+pAUC LOCO -- verdict, per reports/g3_component_pre_registration.md",
             "",
             "Computed 2026-08-10 by `scripts/79_g3pauc_loco_verdict.py`, zero GPU, "
             "from canonical LOCO parquets. Arm: `runs/g3_pauc_loco` "
             "(`configs/g3_pauc.yaml`). Comparator: G3 baseline "
             "`runs/g3_rn50_gastronet_loco`, paired BY SEED. "
             "ACCEPT iff the arm beats baseline on BOTH directions, >=4/5 seeds "
             "agreeing in sign per direction.", ""]

    for centre in (1, 2):
        rows = []
        for seed in range(5):
            unit = f"loco_c{centre}_s{seed}"
            arm = unit_fpr(ARM_DIR, unit)
            base = unit_fpr(BASE_DIR, unit)
            rows.append({"seed": seed, "arm": arm, "base": base,
                         "delta": base - arm})  # positive = arm better
        arm_med = float(np.median([r["arm"] for r in rows]))
        base_med = float(np.median([r["base"] for r in rows]))
        delta_med = float(np.median([r["delta"] for r in rows]))
        n_agree = sum(1 for r in rows if r["delta"] > 0)
        beats = delta_med > 0 and n_agree >= 4
        accepts.append(beats)
        result["directions"][f"holdout_center_{centre}"] = {
            "rows": rows, "arm_median": arm_med, "base_median": base_med,
            "delta_median": delta_med, "n_seeds_arm_better": n_agree,
            "beats": beats}

        lines += [f"## holdout_center_{centre}", "",
                  f"G3+pAUC median {arm_med:.4f}, G3 baseline median {base_med:.4f}. "
                  f"Paired delta (baseline minus arm, positive = arm better): "
                  f"median {delta_med:+.4f}, {n_agree}/5 seeds arm-better -- "
                  f"{'BEATS' if beats else 'does not beat'} baseline on this direction.",
                  "", "| seed | G3+pAUC | G3 baseline | delta |", "|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['seed']} | {r['arm']:.4f} | {r['base']:.4f} | {r['delta']:+.4f} |")
        lines.append("")

    verdict = all(accepts)
    result["ACCEPT"] = verdict
    lines.insert(3, f"## VERDICT: **{'ACCEPT' if verdict else 'REJECT'}**\n")

    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                      if isinstance(v, dict) else v
                      for k, v in result["directions"].items()}, indent=2))
    print("ACCEPT:", verdict)
    print("written:", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
