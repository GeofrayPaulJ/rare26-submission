"""JOB C (2026-08-08) -- pAUC-P1 LOCO. tmux session `paucloco`.

configs/a4_pauc_smoke.yaml (fixed src/losses.py combined loss), mode=loco,
both directions x seeds 0-4 (10 units), out_dir runs/pauc_p1_loco.

Applies reports/pauc_p1_loco_pre_registration.md's decision EXACTLY:
n=1 LOCO FPR@90R per seed, paired against A4-pinned's own n=1 LOCO
(runs/a4_pinned085_swa_loco's "raw" variant, per-seed, not median) --
ship iff BOTH directions have a positive median paired delta (A4-pinned
minus pAUC-P1, i.e. pAUC-P1 lower/better) with >=4/5 seeds agreeing in
sign. Result recorded either way, per the pre-registration.

    python scripts/64_paucloco_chain.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")

PID_FILE = os.path.join(LOG_DIR, "paucloco_chain.pid")
SENTINEL_DONE = os.path.join(LOG_DIR, "paucloco_chain_done")
OUT_DIR = "runs/pauc_p1_loco"
COMPARATOR_DIR = "runs/a4_pinned085_swa_loco"   # A4-pinned's own n=1 LOCO ("raw" variant)
SEEDS = (0, 1, 2, 3, 4)
CENTRES = (1, 2)


def log(msg: str) -> None:
    print(f"[paucloco-chain {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body or time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


def run_arm() -> int:
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_cv.py"),
          "--config", "configs/a4_pauc_smoke.yaml", "--mode", "loco",
          "--centres", "1,2", "--seeds", "0,1,2,3,4",
          "--out-dir", OUT_DIR]
    log(f"RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def analyse() -> dict:
    import pandas as pd
    for p in (REPO_ROOT, SCRIPTS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    from src.evaluate import metric_block

    per_direction = {}
    for centre in CENTRES:
        pauc_vals, comp_vals = {}, {}
        for seed in SEEDS:
            unit = f"loco_c{centre}_s{seed}"
            p_pauc = os.path.join(REPO_ROOT, OUT_DIR, unit, f"val_{unit}.parquet")
            p_comp = os.path.join(REPO_ROOT, COMPARATOR_DIR, unit, f"val_{unit}.parquet")
            for label, path, store in (("pauc", p_pauc, pauc_vals), ("comp", p_comp, comp_vals)):
                df = pd.read_parquet(path)
                m = metric_block(df["label_int"].to_numpy(), df["logit"].to_numpy())
                store[seed] = m["fpr_at_90_recall"]

        deltas = [comp_vals[s] - pauc_vals[s] for s in SEEDS]  # positive = pAUC-P1 better
        median = float(np.median(deltas))
        same_sign = sum(1 for d in deltas if (d > 0) == (median > 0))
        per_direction[centre] = {
            "pauc_p1": pauc_vals, "a4_pinned": comp_vals,
            "pauc_p1_median": float(np.median(list(pauc_vals.values()))),
            "a4_pinned_median": float(np.median(list(comp_vals.values()))),
            "paired_deltas": deltas, "median_delta": median,
            "n_same_sign": same_sign,
            "beats_a4_pinned": median > 0 and same_sign >= 4,
        }

    ship = all(per_direction[c]["beats_a4_pinned"] for c in CENTRES)
    return {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "per_direction": per_direction, "ship_decision": ship}


def render(r: dict) -> None:
    L = []
    A = L.append
    A("# pAUC-P1 LOCO -- JOB C result, per reports/pauc_p1_loco_pre_registration.md")
    A("")
    A(f"Generated {r['generated_utc']}. `configs/a4_pauc_smoke.yaml`, "
     f"`mode=loco`, both directions x seeds 0-4 (10 units), `{OUT_DIR}`. "
     f"Comparator: A4-pinned's own n=1 LOCO (`{COMPARATOR_DIR}`, raw "
     f"variant), paired by seed.")
    A("")
    A(f"## SHIP DECISION: **{'SHIP' if r['ship_decision'] else 'DO NOT SHIP'}**")
    A("")
    A("Per the pre-registration: ship iff pAUC-P1 beats A4-pinned on BOTH "
     "n=1 LOCO directions, paired by seed, >=4/5 seeds agreeing.")
    A("")
    for centre, d in r["per_direction"].items():
        A(f"## holdout_center_{centre}")
        A("")
        A(f"pAUC-P1 median {d['pauc_p1_median']:.4f}, A4-pinned median "
         f"{d['a4_pinned_median']:.4f}. Paired delta (A4-pinned minus "
         f"pAUC-P1, positive = pAUC-P1 better): median "
         f"{d['median_delta']:+.4f}, {d['n_same_sign']}/5 seeds agree -- "
         f"{'BEATS' if d['beats_a4_pinned'] else 'does not beat'} A4-pinned "
         f"on this direction.")
        A("")
        A("| seed | pAUC-P1 | A4-pinned | delta |")
        A("|---|---|---|---|")
        for s in SEEDS:
            A(f"| {s} | {d['pauc_p1'][s]:.4f} | {d['a4_pinned'][s]:.4f} | "
             f"{d['a4_pinned'][s] - d['pauc_p1'][s]:+.4f} |")
        A("")
    A("## Interpretation for the paper")
    A("")
    A(f"This is evidence on whether pAUC-P1's near-null centre-AUC "
     f"confound (0.5187, vs every other arm's 0.02-0.20 distance) is "
     f"load-bearing for held-out-centre generalisation: "
     f"**{'the hypothesis holds -- the pooled cost partly reflects a shortcut invalid on unseen centres, and the gap closes enough to beat the comparator on LOCO' if r['ship_decision'] else 'the hypothesis does NOT hold on this evidence -- the pooled cost is not primarily a confound-removal cost, since it does not close enough on held-out-centre data to beat A4-pinned'}.**")
    A("")

    with open(os.path.join(REPORT_DIR, "pauc_p1_loco.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "pauc_p1_loco.json"), "w") as fh:
        json.dump(r, fh, indent=2, default=float)
    log(f"written: reports/pauc_p1_loco.md -- {'SHIP' if r['ship_decision'] else 'DO NOT SHIP'}")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    touch(PID_FILE, f"{os.getpid()}\n")
    log("pAUC-P1 LOCO (JOB C) chain starting.")
    try:
        if not os.path.exists(SENTINEL_DONE):
            rc = run_arm()
            log(f"run_cv.py exited {rc}")
            if rc != 0:
                log("not a clean 10/10 -- re-run this script to resume.")
                return 3
            touch(SENTINEL_DONE)
        r = analyse()
        render(r)
        return 0
    finally:
        try:
            with open(PID_FILE) as fh:
                if int(fh.read().strip()) == os.getpid():
                    os.remove(PID_FILE)
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
