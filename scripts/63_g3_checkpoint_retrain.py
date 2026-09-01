"""JOB D prerequisite (2026-08-08) -- G3 single-checkpoint retrain. tmux
session `g3ckpt`.

G3 ranked #1 in reports/deployment_selection.md but has ZERO retained
checkpoints -- it ran screening-only (save_checkpoint: false) all night,
same situation A4 was in before its own checkpointed retrain. This runs
ONLY r0_f0_s0 (single member, matching JOB D's "single member" instruction
and this project's established single-checkpoint convention -- A4-pinned's
own fallback is r0_f0_s0 too, not a cherry-picked "best" seed) with
configs/g3_checkpointed.yaml, VERBATIM except this script does not gate on
that config's own header precondition ("runs only if G3 passed Rule 2") --
JOB A's ranking explicitly supersedes that old k=5-era gate for this
purpose; the ranking IS the new authorization.

Logits asserted bit-identical against the existing runs/g3_rn50_gastronet/
r0_f0_s0 canonical parquet (same seed, same data order, same everything
except what gets written to disk) -- a mismatch is a HALT, not a warning,
per this project's standing convention for every checkpointed retrain.

Chains into scripts/run_paucloco_chain.sh (JOB C) on completion regardless
of outcome (independent GPU stage).

    python scripts/63_g3_checkpoint_retrain.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")

PID_FILE = os.path.join(LOG_DIR, "g3ckpt_chain.pid")
SENTINEL_DONE = os.path.join(LOG_DIR, "g3ckpt_chain_done")
OUT_DIR = "runs/gastronet_g3"
REFERENCE_DIR = "runs/g3_rn50_gastronet"
UNIT = "r0_f0_s0"


def log(msg: str) -> None:
    print(f"[g3ckpt-chain {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body or time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


def run_arm() -> int:
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_cv.py"),
          "--config", "configs/g3_checkpointed.yaml", "--mode", "cv",
          "--repeats", "0", "--folds", "0", "--seeds", "0",
          "--out-dir", OUT_DIR]
    log(f"RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def verify_bit_identity() -> dict:
    import numpy as np
    import pandas as pd
    new_p = os.path.join(REPO_ROOT, OUT_DIR, UNIT, f"val_{UNIT}.parquet")
    ref_p = os.path.join(REPO_ROOT, REFERENCE_DIR, UNIT, f"val_{UNIT}.parquet")
    if not os.path.exists(ref_p):
        return {"checked": False, "reason": f"{ref_p} not found"}
    a = pd.read_parquet(new_p).sort_values("filepath")["logit"].to_numpy()
    b = pd.read_parquet(ref_p).sort_values("filepath")["logit"].to_numpy()
    identical = bool(a.shape == b.shape and np.array_equal(a, b))
    return {"checked": True, "identical": identical, "n": int(a.shape[0])}


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    touch(PID_FILE, f"{os.getpid()}\n")
    log("G3 single-checkpoint retrain starting (unblocks JOB D).")
    exit_code = 0
    try:
        if not os.path.exists(SENTINEL_DONE):
            rc = run_arm()
            log(f"run_cv.py exited {rc}")
            if rc != 0:
                log("not clean -- re-run this script to resume. Falling "
                   "through to JOB C regardless (independent stage).")
                exit_code = 3
            else:
                bi = verify_bit_identity()
                log(f"bit-identity: {bi}")
                if bi.get("checked") and not bi.get("identical", True):
                    log("HALT-WORTHY: bit-identity mismatch -- the retrain "
                       "does not reproduce runs/g3_rn50_gastronet's own "
                       "canonical logits. NOT touching JOB D automatically; "
                       "human must investigate before this checkpoint is "
                       "trusted for a container.")
                    with open(os.path.join(REPORT_DIR, "g3_checkpoint_bit_identity.json"), "w") as fh:
                        json.dump(bi, fh, indent=2)
                    exit_code = 4
                else:
                    with open(os.path.join(REPORT_DIR, "g3_checkpoint_bit_identity.json"), "w") as fh:
                        json.dump(bi, fh, indent=2)
                    touch(SENTINEL_DONE)
                    log("bit-identity PASSED (or unverifiable-but-not-contradicted); "
                       "checkpoint ready for JOB D.")
        subprocess.run(["bash", os.path.join(SCRIPTS_DIR, "run_paucloco_chain.sh")],
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
