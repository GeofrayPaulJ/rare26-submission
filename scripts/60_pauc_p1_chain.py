"""STEP 8 BRANCH B, stage 5 (2026-08-07/08) -- pAUC pooled arm P1. tmux
session `pauc` (reused after the smoke's session goes idle).

configs/a4_pauc_smoke.yaml's aug/arch/loss settings, run as a FULL pooled
arm: 5 folds x 5 seeds = 25 units, out_dir runs/pauc_p1, save_checkpoint
false (screening -- matches every other pooled sweep arm's convention).
GATED: only invoked by scripts/59_full_data_ema.py's launcher if
reports/pauc_smoke.json shows a clean smoke (no divergence, no gradient
collapse). NO LOCO tonight, per the brief -- pooled-OOF only.

Reports single-checkpoint (n=1, single_seed median/IQR) AND k=5 pooled
FPR@90R against A4-corrected (the standing reference), same structure as
scripts/56's reframe, since this arm's deployment-relevant figure is n=1
regardless of what k=5 shows.

    python scripts/60_pauc_p1_chain.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")

PID_FILE = os.path.join(LOG_DIR, "pauc_p1_chain.pid")
SENTINEL_DONE = os.path.join(LOG_DIR, "pauc_p1_chain_done")
OUT_DIR = "runs/pauc_p1"


def log(msg: str) -> None:
    print(f"[pauc-p1-chain {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body or time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


def run_arm() -> int:
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_cv.py"),
          "--config", "configs/a4_pauc_smoke.yaml", "--mode", "cv",
          "--repeats", "0", "--folds", "0,1,2,3,4", "--seeds", "0,1,2,3,4",
          "--out-dir", OUT_DIR]
    log(f"RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def analyse() -> dict:
    import pandas as pd
    for p in (REPO_ROOT, SCRIPTS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "_ms", os.path.join(SCRIPTS_DIR, "26_magnitude_sweep.py"))
    MS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MS)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))
    a = MS.analyse_pooled(OUT_DIR, manifest)

    with open(os.path.join(REPO_ROOT, "reports", "gastronet.json")) as fh:
        gj = json.load(fh)
    a4_ref = gj["reference_a4"]

    return {"pauc_p1": a, "a4_corrected_reference": a4_ref}


def render(r: dict) -> None:
    L = []
    A = L.append
    A("# pAUC P1 -- full pooled arm (25 units)")
    A("")
    A(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. "
     f"`configs/a4_pauc_smoke.yaml` (fixed src/losses.py combined loss), "
     f"5 folds x 5 seeds, `{OUT_DIR}`. No LOCO tonight. Reported against "
     f"BOTH k=5 (the historical decision unit) and n=1 single-checkpoint "
     f"(median/IQR across seeds -- the deployment-relevant unit per "
     f"tonight's reframe) A4-corrected reference.")
    A("")
    p = r["pauc_p1"]
    a4 = r["a4_corrected_reference"]
    A("| metric | pAUC-P1 k=5 | A4-corrected k=5 | pAUC-P1 n=1 median | "
     "A4-corrected n=1 median |")
    A("|---|---|---|---|---|")
    for f in ("fpr_at_90_recall", "fpr_prior_equalised", "roc_auc",
             "pauc_15_std", "centre_auc_negatives"):
        A(f"| {f} | {p['k5'][f]:.4f} | {a4['k5'][f]:.4f} | "
         f"{p['single_seed'][f]['median']:.4f} | "
         f"{a4['single_seed'][f]['median']:.4f} |")
    A("")

    with open(os.path.join(REPORT_DIR, "pauc_p1.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "pauc_p1.json"), "w") as fh:
        json.dump(r, fh, indent=2, default=float)
    log("written: reports/pauc_p1.md")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    touch(PID_FILE, f"{os.getpid()}\n")
    log("pAUC P1 (pooled, 25 units) chain starting.")
    try:
        if not os.path.exists(SENTINEL_DONE):
            rc = run_arm()
            log(f"run_cv.py exited {rc}")
            if rc != 0:
                log("not a clean 25/25 -- re-run this script to resume.")
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
