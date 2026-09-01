"""STEP 8 BRANCH B, stage 2 (2026-08-07) -- pAUC smoke. tmux session `pauc`.

configs/a4_pauc_smoke.yaml (combined loss lambda*BCE + (1-lambda)*pAUC with
the fixed src/losses.py: squared hinge, two-sided restriction, BCE warmup
epochs 0-4, beta annealed 1.0 -> 0.15 over epochs 5-9), 1 fold x 3 seeds.

The question is NOT "is it better" (3 units cannot answer that) -- it is
"does the fixed loss TRAIN": no divergence (finite losses throughout, no
val-AUC collapse after the warmup handover at epoch 5) and no gradient
collapse (train loss keeps moving after the handover). Report the epoch
curve; reports/pauc_smoke.{md,json}.

    python scripts/54_pauc_chain.py
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

PID_FILE = os.path.join(LOG_DIR, "pauc_chain.pid")
SENTINEL_DONE = os.path.join(LOG_DIR, "pauc_chain_done")
OUT_DIR = "runs/pauc_smoke"
SEEDS = (0, 1, 2)
WARMUP_EPOCH = 5   # first epoch (0-based) on the combined loss


def log(msg: str) -> None:
    print(f"[pauc-chain {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body or time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


def run_arm() -> int:
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_cv.py"),
          "--config", "configs/a4_pauc_smoke.yaml", "--mode", "cv",
          "--repeats", "0", "--folds", "0", "--seeds", "0,1,2",
          "--out-dir", OUT_DIR]
    log(f"RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def analyse() -> dict:
    out = {"units": {}, "diverged": False, "gradient_collapsed": False}
    for s in SEEDS:
        unit = f"r0_f0_s{s}"
        with open(os.path.join(REPO_ROOT, OUT_DIR, unit, "summary.json")) as fh:
            summary = json.load(fh)
        hist = summary["history"]
        rows = [{"epoch": h["epoch"], "train_loss": h["train_loss"],
                 "val_loss": h["val_loss"], "roc_auc": h["roc_auc"],
                 "pauc_15_std": h["pauc_15_std"],
                 "n_unique_logits": h["n_unique_logits"]} for h in hist]
        tl = np.array([r["train_loss"] for r in rows])
        auc = np.array([r["roc_auc"] for r in rows])

        finite = bool(np.isfinite(tl).all() and np.isfinite(auc).all())
        # divergence: any non-finite loss, or val AUC collapsing to chance
        # after the warmup handover
        post = auc[WARMUP_EPOCH:]
        diverged = (not finite) or bool((post < 0.6).any())
        # gradient collapse: the combined-loss train curve frozen after the
        # handover (successive deltas all ~0)
        post_tl = tl[WARMUP_EPOCH:]
        deltas = np.abs(np.diff(post_tl))
        collapsed = bool(len(deltas) > 3 and (deltas < 1e-6).all())

        out["units"][unit] = {"rows": rows, "finite": finite,
                              "diverged": diverged,
                              "gradient_collapsed": collapsed,
                              "final_auc": float(auc[-1]),
                              "final_pauc15": float(rows[-1]["pauc_15_std"])}
        out["diverged"] |= diverged
        out["gradient_collapsed"] |= collapsed
    return out


def render(r: dict) -> None:
    L = []
    A = L.append
    A("# pAUC smoke -- does the fixed loss train?")
    A("")
    A(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. "
     f"`configs/a4_pauc_smoke.yaml`, 1 fold x 3 seeds, `{OUT_DIR}`. Combined "
     f"loss 0.3*BCE + 0.7*pAUC (squared hinge, two-sided restriction); BCE "
     f"warmup epochs 1-5, beta annealed 1.0 -> 0.15 over epochs 6-10 "
     f"(1-based). NOTE: the train-loss LEVEL steps at the epoch-6 handover "
     f"by construction (the loss definition changes); read continuity of the "
     f"VAL curves across it, not the raw train-loss level.")
    A("")
    A(f"**Divergence: {'DETECTED -- see per-unit' if r['diverged'] else 'none'}. "
     f"Gradient collapse: {'DETECTED' if r['gradient_collapsed'] else 'none'}.**")
    A("")
    for unit, u in r["units"].items():
        A(f"## {unit} (final ROC-AUC {u['final_auc']:.4f}, "
         f"pAUC15 {u['final_pauc15']:.4f})")
        A("")
        A("| epoch | train_loss | val_loss | ROC-AUC | pAUC15 | distinct logits |")
        A("|---|---|---|---|---|---|")
        for row in u["rows"]:
            A(f"| {row['epoch']} | {row['train_loss']:.4f} | "
             f"{row['val_loss']:.4f} | {row['roc_auc']:.4f} | "
             f"{row['pauc_15_std']:.4f} | {row['n_unique_logits']} |")
        A("")

    with open(os.path.join(REPORT_DIR, "pauc_smoke.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "pauc_smoke.json"), "w") as fh:
        json.dump(r, fh, indent=2, default=float)
    log("written: reports/pauc_smoke.md")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    touch(PID_FILE, f"{os.getpid()}\n")
    log("pAUC smoke chain starting.")
    exit_code = 0
    try:
        if not os.path.exists(SENTINEL_DONE):
            rc = run_arm()
            log(f"run_cv.py exited {rc}")
            if rc != 0:
                log("not a clean 3/3. Per the brief, this is an INDEPENDENT "
                   "stage -- falling through to swaloco (step 3) rather "
                   "than halting the chain. Re-run this script later to "
                   "resume/complete the smoke itself.")
                exit_code = 3
            else:
                touch(SENTINEL_DONE)
        if os.path.exists(SENTINEL_DONE):
            r = analyse()
            render(r)
            log(f"diverged={r['diverged']} collapsed={r['gradient_collapsed']}")
        # Chain into step 3 (swaloco) regardless of this stage's own outcome.
        subprocess.run(["bash", os.path.join(SCRIPTS_DIR, "run_swaloco_chain.sh")],
                       cwd=REPO_ROOT)
        return exit_code
    finally:
        try:
            with open(PID_FILE) as fh:
                if int(fh.read().strip()) == os.getpid():
                    os.remove(PID_FILE)
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
