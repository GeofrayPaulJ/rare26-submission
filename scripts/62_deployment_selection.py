"""JOB A (2026-08-08) -- SELECTION, not gating. NO TRAINING, NO GPU.

Every arm rejected at n=1 (reports/single_checkpoint_reframe.md). That is a
finding about the GATE, not about the arms: at n=1 the across-seed IQR runs
2-4x the k=5 LOO-4 bar the gates were calibrated against (e.g. A0's
single-seed range 0.0898 vs its old k=5 LOO-4 bar 0.0143), so the gate
rejects everything, including whatever would actually help. This script
reframes the question as SELECTION among that rejected field -- ranking,
not accept/reject -- and writes ONE table, reports/deployment_selection.md.

DATA SOURCES, per arm:
  A0/A1/A2/A3/A4-pinned/A4-corrected/G0/G1/G2/G3 -- reports/
    single_checkpoint_reframe.json (pooled single_seed + per_fold_tail +
    LOCO n1). A4-pinned's LOCO now exists (runs/a4_pinned085_swa_loco,
    the "raw" variant -- see scripts/56's LOCO_ARMS comment for the
    passivity argument) and was merged into that report before this
    script ran, not duplicated here.
  EMA / SWA -- reports/swa_arm.json (pooled: seed-0 ONLY, no across-seed
    spread -- reported honestly as such) + reports/swaloco_n1.json
    (LOCO: full 5-seed n=1 median/IQR, both directions).
  pAUC-P1 -- reports/pauc_p1.json (pooled single_seed, 5 seeds) + a fresh
    per-fold-tail computation against runs/pauc_p1 (reusing scripts/56's
    per_fold_tail(), zero GPU). NO LOCO YET (JOB C, not run) -- shown but
    excluded from the LOCO-based ranking, not given a false rank.

RANKING: primary key = worst of the two n=1 LOCO direction medians
(max(c1, c2), lower is better); tiebreak = pooled per-fold tail (max,
lower is better). Arms without LOCO data are listed but not ranked.

NO ACCEPT/REJECT LANGUAGE in the output -- this is a ranking, not a gate.

    python scripts/62_deployment_selection.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt(x, p=4):
    return "n/a" if x is None else f"{x:.{p}f}"


def main() -> int:
    REFR = json.load(open(os.path.join(REPORT_DIR, "single_checkpoint_reframe.json")))
    SWA = json.load(open(os.path.join(REPORT_DIR, "swa_arm.json")))
    SWALOCO = json.load(open(os.path.join(REPORT_DIR, "swaloco_n1.json")))
    PAUCP1 = json.load(open(os.path.join(REPORT_DIR, "pauc_p1.json")))

    import pandas as pd
    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))
    S56 = _load("_reframe", "56_single_checkpoint_reframe.py")
    pauc_p1_tail = S56.per_fold_tail("runs/pauc_p1", manifest)

    rows = []

    # --- the ten reframe arms ---
    for key, _ in S56.POOLED_ARMS:
        arm = REFR["pooled"].get(key)
        if arm is None:
            continue
        ss = arm["single_seed"]["fpr_at_90_recall"]
        auc = arm["single_seed"]["centre_auc_negatives"]["median"]
        tail = arm["per_fold_tail"]["max"]
        larm = REFR["loco"].get(key)
        loco1 = loco2 = None
        if larm is not None:
            loco1 = larm["n1"]["1"]["median"] if "1" in larm["n1"] else larm["n1"][1]["median"]
            loco2 = larm["n1"]["2"]["median"] if "2" in larm["n1"] else larm["n1"][2]["median"]
        rows.append({"arm": key, "pooled_median": ss["median"], "pooled_iqr": ss["iqr"],
                    "loco1": loco1, "loco2": loco2, "tail": tail,
                    "auc_dist": abs(auc - 0.5), "has_loco": loco1 is not None})

    # --- EMA / SWA (last night's swa arm) ---
    for v in ("ema", "swa"):
        pooled_v = SWA["pooled"][v]["fpr_at_90_recall"]
        auc_v = SWA["pooled"][v]["centre_auc_negatives"]
        tail_v = SWA["per_fold"][v]["fpr_max"]
        loco1 = SWALOCO["per_direction"]["1"]["fpr"][v]["median"]
        loco2 = SWALOCO["per_direction"]["2"]["fpr"][v]["median"]
        rows.append({"arm": v.upper(), "pooled_median": pooled_v, "pooled_iqr": None,
                    "loco1": loco1, "loco2": loco2, "tail": tail_v,
                    "auc_dist": abs(auc_v - 0.5), "has_loco": True,
                    "pooled_note": "seed-0 only, no across-seed spread"})

    # --- pAUC-P1 ---
    ss = PAUCP1["pauc_p1"]["single_seed"]["fpr_at_90_recall"]
    auc = PAUCP1["pauc_p1"]["single_seed"]["centre_auc_negatives"]["median"]
    rows.append({"arm": "pAUC-P1", "pooled_median": ss["median"], "pooled_iqr": ss["iqr"],
                "loco1": None, "loco2": None, "tail": pauc_p1_tail["max"],
                "auc_dist": abs(auc - 0.5), "has_loco": False,
                "loco_note": "not yet run -- JOB C"})

    ranked = [r for r in rows if r["has_loco"]]
    unranked = [r for r in rows if not r["has_loco"]]
    for r in ranked:
        r["worst_loco"] = max(r["loco1"], r["loco2"])
    ranked.sort(key=lambda r: (r["worst_loco"], r["tail"]))

    L = []
    A = L.append
    A("# Deployment selection -- ranking, not gating")
    A("")
    A(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
     f"`scripts/62_deployment_selection.py`. Every arm rejects at n=1 "
     f"(`reports/single_checkpoint_reframe.md`) -- the across-seed IQR at "
     f"n=1 runs 2-4x the k=5 LOO-4 bar every gate this project has used was "
     f"calibrated against, so the gate rejects everything, including "
     f"whatever would help most. This is a SELECTION among that rejected "
     f"field, ranked, not an accept/reject verdict on any of them.")
    A("")
    A("Ranked on the WORST of the two n=1 LOCO direction medians (lower "
     "is better), tie broken by the pooled per-fold tail (max, lower is "
     "better). Arms without LOCO data are listed below the ranked table, "
     "not given a false rank.")
    A("")
    A("| rank | arm | n=1 pooled FPR@90R (median, IQR) | n=1 LOCO c1 | "
     "n=1 LOCO c2 | worst LOCO | pooled per-fold tail (max) | "
     "centre-AUC(neg) dist. from null |")
    A("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranked, 1):
        pooled_cell = (f"{fmt(r['pooled_median'])} ({fmt(r['pooled_iqr'])})"
                       if r["pooled_iqr"] is not None
                       else f"{fmt(r['pooled_median'])} ({r.get('pooled_note', 'n/a')})")
        A(f"| {i} | {r['arm']} | {pooled_cell} | {fmt(r['loco1'])} | "
         f"{fmt(r['loco2'])} | {fmt(r['worst_loco'])} | {fmt(r['tail'])} | "
         f"{fmt(r['auc_dist'])} |")
    A("")

    if unranked:
        A("## Not ranked (no LOCO data)")
        A("")
        A("| arm | n=1 pooled FPR@90R (median, IQR) | pooled per-fold tail (max) | "
         "centre-AUC(neg) dist. from null | why |")
        A("|---|---|---|---|---|")
        for r in unranked:
            pooled_cell = (f"{fmt(r['pooled_median'])} ({fmt(r['pooled_iqr'])})"
                           if r["pooled_iqr"] is not None else fmt(r["pooled_median"]))
            A(f"| {r['arm']} | {pooled_cell} | {fmt(r['tail'])} | "
             f"{fmt(r['auc_dist'])} | {r.get('loco_note', 'no LOCO run')} |")
        A("")

    if len(ranked) >= 3:
        top3 = ranked[:3]
        A("## Top three, and the margin between them")
        A("")
        for i, r in enumerate(top3, 1):
            A(f"{i}. **{r['arm']}** -- worst-direction LOCO {fmt(r['worst_loco'])}, "
             f"tail {fmt(r['tail'])}")
        m12 = top3[1]["worst_loco"] - top3[0]["worst_loco"]
        m23 = top3[2]["worst_loco"] - top3[1]["worst_loco"]
        A("")
        A(f"Margin 1st->2nd (worst-direction LOCO): {m12:+.4f}. "
         f"Margin 2nd->3rd: {m23:+.4f}.")
        A("")

    with open(os.path.join(REPORT_DIR, "deployment_selection.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "deployment_selection.json"), "w") as fh:
        json.dump({"ranked": ranked, "unranked": unranked}, fh, indent=2, default=float)
    print(f"written: {os.path.join(REPORT_DIR, 'deployment_selection.md')}")
    print(f"top-ranked arm: {ranked[0]['arm']}" if ranked else "no ranked arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
