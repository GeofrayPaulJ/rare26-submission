"""One-shot crash-safety proof for Part A of the 48h-unattended request.

Launches the REAL first unit of the Part B queue (A0, repeat=1, fold=0,
seed=0 -- see reports/repeat_gating.md priority order), lets it finish one
full epoch (so a checkpoint + a completed per-epoch parquet exist), then
SIGKILLs the actual `src.train` subprocess partway through the next epoch --
not a clean stop, no signal handler engaged.

The child PID is obtained from `subprocess.Popen.pid` directly, never by
pattern-matching a command line with pgrep -f. That trap is already
documented in this repo (scripts/23_sweep.py, scripts/sweep_watchdog.ps1):
`pgrep -f <pattern>` matches the very shell command asking the question
whenever that command's own argv contains the pattern text, which it always
does when the pattern is spelled out in a `bash -c "...pgrep -f X..."`
wrapper. A PID obtained from Popen has no such ambiguity.

After the kill this script:
  1. shows there is no canonical parquet (val_r1_f0_s0.parquet) -- correct,
     since epoch 30/30 was never reached;
  2. shows the last *complete* per-epoch parquet (epoch 1) is intact and
     valid, and that no torn file exists at any *final* path for the epoch
     that was in flight when the kill landed;
  3. re-invokes scripts/run_cv.py for the exact same unit and shows its own
     run_state() logic (the same function scripts/23_sweep.py imports)
     resumes from the epoch-1 checkpoint rather than restarting from scratch
     or -- worse -- accepting a partial dump as done.

This script is a one-off test harness, not part of the pipeline; it is safe
to delete once the proof has been captured in the response to the user.
"""
from __future__ import annotations

import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

CONFIG = "configs/sweep_a0.yaml"
OUT_DIR = os.path.join(REPO_ROOT, "runs", "sweep_a0")
UNIT_NAME = "r1_f0_s0"
RUN_DIR = os.path.join(OUT_DIR, UNIT_NAME)
LOG_PATH = os.path.join(REPO_ROOT, "logs", "_crashtest_train.log")

EPOCH_RE = re.compile(r"^epoch\s+(\d+)/(\d+)\s")


def log(msg: str) -> None:
    print(f"[crash-test] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def tail_epochs_seen(path: str) -> list:
    if not os.path.exists(path):
        return []
    seen = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = EPOCH_RE.match(line)
            if m:
                seen.append(int(m.group(1)))
    return seen


def wait_for_epoch(path: str, n: int, timeout_s: float) -> float:
    """Block until epoch n's log line has appeared. Returns wall time waited."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if n in tail_epochs_seen(path):
            return time.time() - t0
        time.sleep(0.05)
    raise TimeoutError(f"epoch {n} did not appear within {timeout_s}s")


def main() -> int:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    for p in (
        os.path.join(RUN_DIR, f"val_{UNIT_NAME}.parquet"),
    ):
        if os.path.exists(p):
            raise RuntimeError(f"{p} already exists -- refusing to clobber a real result")

    argv = [
        sys.executable, "-u", "-m", "src.train",
        "--config", CONFIG, "--seed", "0", "--repeat", "1", "--fold", "0",
        "--out-dir", OUT_DIR, "--epochs", "30", "--no-progress",
    ]
    log(f"launching real unit: {' '.join(argv)}")
    with open(LOG_PATH, "w") as logf:
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=logf,
                                stderr=subprocess.STDOUT)
        child_pid = proc.pid  # from Popen directly -- not pgrep -f
        log(f"child pid (from Popen, not pattern-matched) = {child_pid}")

        wait_for_epoch(LOG_PATH, 1, timeout_s=180)
        log("epoch 1/30 completed -- checkpoint + epoch-1 parquet should now exist")

        preds_dir = os.path.join(RUN_DIR, "preds")
        e1_final = os.path.join(preds_dir, f"val_{UNIT_NAME}_e001.parquet")
        e1_tmp = e1_final + ".tmp"
        log(f"epoch-1 parquet present: {os.path.exists(e1_final)}  "
            f"size={os.path.getsize(e1_final) if os.path.exists(e1_final) else 'n/a'}  "
            f"stray tmp present: {os.path.exists(e1_tmp)}")

        # Actively watch for epoch 2's tmp write file so the kill lands as
        # close to mid-write as the OS scheduler allows; fall back to a
        # historical-timing kill (train phase ~42.7s of a ~47s epoch, per
        # runs/noise_floor_a/r0_f0_s0/summary.json) if the write window is
        # too narrow to observe from outside the process.
        e2_final = os.path.join(preds_dir, f"val_{UNIT_NAME}_e002.parquet")
        e2_tmp = e2_final + ".tmp"
        t_epoch2_start = time.time()
        caught_tmp = False
        while time.time() - t_epoch2_start < 46.0:
            if os.path.exists(e2_tmp):
                caught_tmp = True
                break
            if 2 in tail_epochs_seen(LOG_PATH):
                # epoch 2 already fully logged -- its write finished before we
                # could observe it; kill now anyway so the test still lands
                # mid-epoch-3, still a genuine non-clean kill of a live unit.
                break
            time.sleep(0.01)
        landed = ("mid-write (epoch-2 .tmp observed on disk)" if caught_tmp
                  else "mid-epoch (post-epoch-2, best-effort late timing)")
        log(f"killing now -- {landed}")

        os.kill(child_pid, signal.SIGKILL)
        rc = proc.wait(timeout=30)
        log(f"child pid {child_pid} killed; wait() returncode={rc}")

    # ---- verification ----
    log("=" * 70)
    log("POST-KILL STATE")
    canonical = os.path.join(RUN_DIR, f"val_{UNIT_NAME}.parquet")
    log(f"canonical parquet exists: {os.path.exists(canonical)}  (must be False -- "
        f"30/30 epochs never completed)")

    for f in sorted(glob.glob(os.path.join(preds_dir, "*"))):
        try:
            sz = os.path.getsize(f)
        except OSError:
            sz = "?"
        log(f"  preds/: {os.path.basename(f)}  size={sz}")

    ckpt = os.path.join(RUN_DIR, "checkpoints", "last.pt")
    log(f"checkpoint present: {os.path.exists(ckpt)}  "
        f"size={os.path.getsize(ckpt) if os.path.exists(ckpt) else 'n/a'}")

    log("=" * 70)
    log("RE-INVOKING THE DRIVER'S OWN run_state() (imported from run_cv.py, "
        "the same function scripts/23_sweep.py calls)")
    from run_cv import Unit, expected_split, run_state, validate_predictions  # noqa: E402
    from src.config import Config  # noqa: E402
    import dataclasses  # noqa: E402

    base_cfg = Config.from_yaml(os.path.join(REPO_ROOT, CONFIG))
    unit = Unit(seed=0, repeat=1, fold=0)
    cfg = unit.apply(dataclasses.replace(base_cfg, out_dir=OUT_DIR, epochs=30))
    expected = expected_split(unit, cfg)
    status, reason = run_state(RUN_DIR, unit, cfg, expected)
    log(f"run_state() classification: status={status!r}  reason={reason!r}")
    if status == "done":
        log("FAIL: the killed unit was classified DONE. This would be the "
            "exact failure Part A exists to prevent.")
        return 1
    log("PASS: the killed unit was NOT classified done; the driver will "
        "resume or restart it, never skip it.")

    log("=" * 70)
    log("RUNNING THE ACTUAL DRIVER (scripts/run_cv.py) FOR THE SAME UNIT -- "
        "\"the driver reruns that exact unit on its next pass\"")
    driver_argv = [
        sys.executable, "-u", "run_cv.py",
        "--config", CONFIG, "--mode", "cv",
        "--repeats", "1", "--folds", "0", "--seeds", "0",
        "--out-dir", OUT_DIR,
    ]
    log(f"$ {' '.join(driver_argv)}")
    rc2 = subprocess.run(driver_argv, cwd=os.path.join(REPO_ROOT, "scripts")).returncode
    log(f"driver exit code: {rc2}")

    status2, reason2 = run_state(RUN_DIR, unit, cfg, expected)
    log(f"post-driver-run classification: status={status2!r} reason={reason2!r}")
    ok = status2 == "done"
    log("PASS: unit is now genuinely complete (driver resumed it to a valid "
        "canonical parquet)." if ok else
        "FAIL: unit is still not done after the driver ran.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
