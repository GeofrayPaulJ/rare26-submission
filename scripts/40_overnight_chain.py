"""Overnight chain, 2026-08-05. Detached in tmux, resumable via sentinel files
in logs/ (same idiom as scripts/run_sweep.sh -- re-running this script skips
any stage whose sentinel already exists, and run_cv.py's own per-unit
resumability covers a kill mid-stage).

ORDER (fixed, do not reorder -- 5b/5c need an idle card or the numbers are
worthless):

  0. Wait for the current sweep (g3_loco + g0_rn50_imagenet) to reach its
     sweep_stage_complete sentinel.
  1. scripts/32_g3_loco.py, scripts/33_g0_control.py -- insert report sections.
     The G3 Rule 2 verdict is printed immediately and loudly (not buried).
  3. Overnight training, branch on JOB 5a (checkpoint inventory) and the G3
     Rule 2 verdict from step 1:
       a. A4 has no checkpoints (confirmed by JOB 5a) -> retrain A4 with
          save_checkpoint: true, 25 units, via run_cv.py against
          configs/sweep_a4_checkpointed.yaml. ALWAYS FIRST. Then assert bit
          identity against runs/sweep_a4. A mismatch HALTS the chain here --
          it does not proceed to G3.
       b. If G3 passed Rule 2: retrain G3 with save_checkpoint: true, 25
          units, via configs/g3_checkpointed.yaml -> runs/gastronet_g3.
          Assert bit identity against runs/g3_rn50_gastronet. A mismatch
          HALTS the chain here.
       c. If G3 passed: G3 repeats 1-2 (50 units), via
          configs/g3_rn50_gastronet.yaml, repeats=1,2, out_dir
          runs/g3_rn50_gastronet (same dir the repeat-0 units already live
          in). NO TIME-BUDGET SKIP (2026-08-05 amendment): run_cv.py is
          resumable and skips already-validated units, so falling through
          and running until stopped costs nothing. Runs to 50/50 or until
          externally stopped; only "G3 did not pass" skips this stage.
  2. JOB 5b/5c: confirm GPU idle, then scripts/36_throughput_bench.py.
     MOVED HERE 2026-08-05 (was originally step 2, before 3a) -- it needs to
     sit at the ACTUAL idle boundary between the two run_cv.py-heavy phases
     (3a/3b/3c, and 4 below), not merely "early". Whenever G3 fails Rule 2
     (collapsing 3b/3c to no-ops, as happened on the first run), the true
     idle boundary is immediately after 3a -- putting the bench before 3a
     instead just ran it once, however long ago the chain started, with no
     mechanism to retry if it failed. Also fixed the same day: a nonzero
     exit here no longer marks the stage done (see stage2_throughput_bench's
     own docstring) -- that silent-mark-done-on-failure bug is what orphaned
     the first attempt (a ViT-B/14 img_size crash) with no retry path.
  4. TERMINAL STAGE, runs in EITHER branch (G3 passed or failed) -- the only
     thing that skips it is a HALT earlier in the chain (3a or 3b): A4
     checkpointed retrain at repeats 1 and 2 (50 units), save_checkpoint:
     true, same config/out_dir as stage 3a (configs/sweep_a4_checkpointed.yaml
     -> runs/a4_checkpointed), bit-identity asserted against runs/sweep_a4's
     existing repeat-1/repeat-2 parquets. Added 2026-08-05 specifically so
     the card does not sit idle after a G3 Rule-2 failure -- A4 across all
     three repeats, checkpointed, is the ensemble configuration intended for
     actual deployment. Same resumable/skip-validated logic; runs until
     50/50 or stopped.
     Skip any unit whose parquet already exists and validates -- run_cv.py
     already does this natively, nothing extra needed here.

    python scripts/40_overnight_chain.py
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

SENTINEL_REPORTS = os.path.join(LOG_DIR, "chain_stage1_reports_done")
SENTINEL_BENCH = os.path.join(LOG_DIR, "chain_stage2_bench_done")
SENTINEL_A4_DONE = os.path.join(LOG_DIR, "chain_stage3a_a4_done")
SENTINEL_A4_HALT = os.path.join(LOG_DIR, "chain_stage3a_a4_HALT")
SENTINEL_G3_DONE = os.path.join(LOG_DIR, "chain_stage3b_g3_done")
SENTINEL_G3_HALT = os.path.join(LOG_DIR, "chain_stage3b_g3_HALT")
SENTINEL_G3_REPEATS_DONE = os.path.join(LOG_DIR, "chain_stage3c_g3_repeats_done")
# G3-did-not-pass-Rule-2 is the ONLY thing that skips stage 3c -- there is no
# time-budget skip (2026-08-05 amendment: run_cv.py is resumable and banks
# progress, so falling through and running until stopped costs nothing).
SENTINEL_G3_REPEATS_SKIPPED = os.path.join(LOG_DIR, "chain_stage3c_g3_repeats_SKIPPED")
SENTINEL_A4_REPEATS_DONE = os.path.join(LOG_DIR, "chain_stage4_a4_repeats_done")
# Checked by scripts/sweep_watchdog.ps1 (host side) to decide when it is
# finally safe to stop polling -- it also globs chain_stage*_HALT for the
# halt case, which the existing per-stage HALT sentinels already satisfy.
CHAIN_COMPLETE = os.path.join(LOG_DIR, "chain_complete")

G3_REPEATS_UNITS = 50
A4_REPEATS_UNITS = 50

PID_FILE = os.path.join(LOG_DIR, "chain.pid")


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[chain {ts}] {msg}", flush=True)


def run(cmd, **kwargs) -> subprocess.CompletedProcess:
    log(f"RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT, **kwargs)


def touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body or time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


def write_pid() -> None:
    """Same PID-file convention as scripts/23_sweep.py: content is the pid,
    ownership-checked on cleanup so a second/stale invocation can never
    delete a live driver's file. This is what lets scripts/run_chain.sh (the
    entrypoint-hook-invoked wrapper) tell "already running" from "dead,
    relaunch me" without a pgrep self-match trap."""
    touch(PID_FILE, f"{os.getpid()}\n")


def cleanup_pid() -> None:
    try:
        with open(PID_FILE) as fh:
            owner = int(fh.read().strip())
    except (OSError, ValueError):
        return
    if owner != os.getpid():
        return
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def wait_for_gastronet_queue() -> None:
    """Stage 0: wait for the currently-running g3_loco + g0 sweep to settle."""
    status_path = os.path.join(LOG_DIR, "sweep_status.json")
    done_path = os.path.join(LOG_DIR, "sweep_stage_complete")
    halt_path = os.path.join(LOG_DIR, "sweep_halt.json")
    log("Stage 0: waiting for the GPU sweep (g3_loco + g0) to settle...")
    while True:
        if os.path.exists(halt_path):
            with open(halt_path) as fh:
                halt = json.load(fh)
            log(f"HALT: the main sweep halted deliberately before finishing: "
               f"{halt}. The overnight chain will NOT start GPU work on top "
               f"of a halted, possibly-broken tree. Stopping here.")
            sys.exit(3)
        if os.path.exists(done_path):
            log("Main sweep reached sweep_stage_complete. Proceeding.")
            return
        if os.path.exists(status_path):
            with open(status_path) as fh:
                st = json.load(fh)
            log(f"  ...still running: {st.get('units_completed')}/"
               f"{st.get('units_total')} units, stage={st.get('current_stage')}")
        time.sleep(120)


def stage1_reports() -> None:
    if os.path.exists(SENTINEL_REPORTS):
        log("Stage 1 already done (sentinel present); skipping.")
        return
    log("Stage 1: inserting G3-LOCO and G0-control report sections.")
    r1 = run([sys.executable, "-u", os.path.join(SCRIPTS_DIR, "32_g3_loco.py")])
    r2 = run([sys.executable, "-u", os.path.join(SCRIPTS_DIR, "33_g0_control.py")])
    if r1.returncode != 0 or r2.returncode != 0:
        log(f"WARNING: report scripts returned nonzero (32={r1.returncode}, "
           f"33={r2.returncode}). Continuing anyway -- these are reporting "
           f"steps, not GPU safety gates, but check reports/gastronet.md.")
    touch(SENTINEL_REPORTS)


def gpu_idle_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    vals = [int(x.strip()) for x in out.stdout.strip().splitlines() if x.strip()]
    return max(vals) if vals else 0


def stage2_throughput_bench() -> None:
    """JOB 5b/5c. Positioned between stage 3a and stage 4 (2026-08-05
    amendment) -- that is the actual idle boundary between the two GPU-heavy
    run_cv.py invocations, not a guess at a "natural" gap that may never
    arrive (G3's stages 3b/3c are no-ops whenever G3 fails Rule 2, which
    collapses that boundary to immediately after 3a).

    BUG FIXED 2026-08-05: this used to touch(SENTINEL_BENCH) unconditionally,
    including when 36_throughput_bench.py exited nonzero -- which is exactly
    what orphaned the first attempt (a ViT-B/14 img_size crash) with no retry.
    The sentinel is now written ONLY on a clean (rc==0) run; a failed attempt
    leaves it absent so the next chain invocation retries instead of
    silently treating a crash as done.
    """
    if os.path.exists(SENTINEL_BENCH):
        log("Stage 2 already done (sentinel present); skipping.")
        return
    log("Stage 2: confirming GPU idle before throughput bench...")
    for attempt in range(30):
        used = gpu_idle_mib()
        if used == 0:
            break
        log(f"  GPU still shows {used} MiB used; waiting 20s (attempt {attempt+1}/30)...")
        time.sleep(20)
    else:
        log("GPU never went idle after 10 minutes of waiting. Not running "
           "5b/5c under load -- skipping for now; NOT marking done, so the "
           "next chain invocation (or the next call to this stage) retries.")
        return
    r = run([sys.executable, "-u", os.path.join(SCRIPTS_DIR, "36_throughput_bench.py")])
    if r.returncode != 0:
        log(f"throughput bench exited {r.returncode} -- NOT marking stage 2 "
           f"done, so it is retried rather than silently orphaned. Check "
           f"reports/throughput_bench.md and the traceback above.")
        return
    log("Stage 2: throughput bench completed cleanly.")
    touch(SENTINEL_BENCH)


def run_cv(config: str, out_dir: str, repeats: str = "0",
          label: str = "") -> int:
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_cv.py"),
          "--config", config, "--mode", "cv", "--repeats", repeats,
          "--out-dir", out_dir]
    log(f"Launching run_cv.py for {label or config} -> {out_dir} "
       f"(repeats={repeats})")
    r = run(cmd)
    return r.returncode


def bit_identity(new_dir: str, reference_dir: str, label: str,
                 repeats: str = "0") -> bool:
    r = run([sys.executable, "-u",
            os.path.join(SCRIPTS_DIR, "39_bit_identity_check.py"),
            "--new", new_dir, "--reference", reference_dir,
            "--mode", "cv", "--repeats", repeats, "--label", label])
    return r.returncode == 0


def read_a4_checkpoint_status() -> bool:
    """True if A4 (runs/sweep_a4) currently has ANY weights_fp32.pt on disk."""
    a4_dir = os.path.join(REPO_ROOT, "runs", "sweep_a4")
    if not os.path.isdir(a4_dir):
        return False
    for root, _dirs, files in os.walk(a4_dir):
        if "weights_fp32.pt" in files:
            return True
    return False


def read_g3_rule2_verdict() -> bool:
    path = os.path.join(REPORT_DIR, "g3_loco_rule2.json")
    if not os.path.exists(path):
        log("No reports/g3_loco_rule2.json found -- treating G3 as NOT "
           "passed (conservative default).")
        return False
    with open(path) as fh:
        v = json.load(fh)
    return bool(v.get("accept", False))


def stage3a_retrain_a4() -> bool:
    """Returns True if this branch completed cleanly (or was already done);
    False if it HALTed (mismatch), which must stop the whole chain."""
    if os.path.exists(SENTINEL_A4_HALT):
        log("Stage 3a previously HALTed (sentinel present). NOT retrying "
           "automatically -- a bit-identity mismatch is a finding that needs "
           "a human look, not a silent retry. Chain stops here.")
        return False
    if os.path.exists(SENTINEL_A4_DONE):
        log("Stage 3a (A4 checkpointed retrain) already done; skipping.")
        return True

    has_ckpt = read_a4_checkpoint_status()
    log(f"Stage 3a: A4 currently has checkpoints on disk: {has_ckpt}")
    if has_ckpt:
        log("A4 already has weights_fp32.pt -- nothing to retrain. Marking done.")
        touch(SENTINEL_A4_DONE)
        return True

    log("Stage 3a: A4 has NO checkpoints. Retraining, config verbatim + "
       "save_checkpoint: true, 25 units. THIS IS FIRST IN EVERY BRANCH.")
    rc = run_cv("configs/sweep_a4_checkpointed.yaml", "runs/a4_checkpointed",
               repeats="0", label="A4 checkpointed")
    if rc != 0:
        log(f"A4 checkpointed retrain (run_cv.py) exited {rc}. Checking bit "
           f"identity anyway on whatever units completed, but this needs "
           f"attention regardless of the outcome below.")

    ok = bit_identity("runs/a4_checkpointed", "runs/sweep_a4", "A4")
    if not ok:
        log("HALT: A4 checkpointed retrain is NOT bit-identical to "
           "runs/sweep_a4. This is a finding, not proceeding to G3. See "
           "reports/bit_identity_a4.json.")
        touch(SENTINEL_A4_HALT)
        return False

    log("A4 checkpointed retrain: bit-identical. Reproducibility confirmed. "
       "Stage 3a done.")
    touch(SENTINEL_A4_DONE)
    return True


def stage3b_retrain_g3(g3_passed: bool) -> bool:
    if not g3_passed:
        log("Stage 3b: G3 did NOT pass Rule 2 -- skipping the checkpointed "
           "retrain per the branch rules. Not spending GPU-hours on an arm "
           "that didn't earn the slot.")
        return True
    if os.path.exists(SENTINEL_G3_HALT):
        log("Stage 3b previously HALTed (sentinel present). Chain stops here.")
        return False
    if os.path.exists(SENTINEL_G3_DONE):
        log("Stage 3b (G3 checkpointed retrain) already done; skipping.")
        return True

    log("Stage 3b: G3 PASSED Rule 2. Retraining, config verbatim + "
       "save_checkpoint: true, 25 units.")
    rc = run_cv("configs/g3_checkpointed.yaml", "runs/gastronet_g3",
               repeats="0", label="G3 checkpointed")
    if rc != 0:
        log(f"G3 checkpointed retrain (run_cv.py) exited {rc}. Checking bit "
           f"identity anyway.")

    ok = bit_identity("runs/gastronet_g3", "runs/g3_rn50_gastronet", "G3")
    if not ok:
        log("HALT: G3 checkpointed retrain is NOT bit-identical to "
           "runs/g3_rn50_gastronet. Finding, not proceeding to G3 repeats. "
           "See reports/bit_identity_g3.json.")
        touch(SENTINEL_G3_HALT)
        return False

    log("G3 checkpointed retrain: bit-identical. Stage 3b done.")
    touch(SENTINEL_G3_DONE)
    return True


def count_units_landed(out_dir: str, unit_prefix_repeats=(1, 2)) -> dict:
    """How many (repeat, fold, seed) units in out_dir's run_cv.py index are
    ok/skipped (i.e. landed, whether from this invocation or a prior one),
    restricted to the given repeats. Reads the index run_cv.py itself
    maintains -- no separate bookkeeping to drift out of sync with it."""
    index_path = os.path.join(REPO_ROOT, out_dir, "run_index.json")
    if not os.path.exists(index_path):
        return {"landed": 0, "total": None, "detail": "no index yet"}
    with open(index_path) as fh:
        idx = json.load(fh)
    units = idx.get("units", {})
    relevant = {k: v for k, v in units.items()
               if any(k.startswith(f"r{r}_") for r in unit_prefix_repeats)}
    landed = sum(1 for v in relevant.values() if v.get("status") in ("ok", "skipped"))
    return {"landed": landed, "total": len(relevant), "detail": relevant}


def stage3c_g3_repeats(g3_passed: bool) -> None:
    """G3 repeats 1-2 (50 units). NO TIME-BUDGET SKIP: per explicit
    instruction (2026-08-05), the 09:00 IST deadline does not gate this
    stage. run_cv.py is resumable and skips already-validated units, so a
    partial run banks progress and costs nothing -- this falls through and
    runs until it finishes 50/50 or is stopped from outside. Whoever stops
    it (or checks later) reads how many units landed via
    count_units_landed() / runs/g3_rn50_gastronet/run_index.json, not from
    this function having decided a cutoff itself.
    """
    if not g3_passed:
        log("Stage 3c: G3 did not pass Rule 2 -- skipping repeats 1-2. "
           "(This is the only condition that skips this stage; there is no "
           "time-budget skip.)")
        touch(SENTINEL_G3_REPEATS_SKIPPED, "G3 did not pass Rule 2\n")
        return
    if os.path.exists(SENTINEL_G3_REPEATS_DONE):
        log("Stage 3c already done (50/50); skipping.")
        return

    log("Stage 3c: G3 passed. Running G3 repeats 1-2 (50 units), falling "
       "through with NO time-budget skip -- runs until complete or stopped. "
       "run_cv.py resumes from wherever it was left if interrupted.")
    rc = run_cv("configs/g3_rn50_gastronet.yaml", "runs/g3_rn50_gastronet",
               repeats="1,2", label="G3 repeats 1-2")
    landed = count_units_landed("runs/g3_rn50_gastronet")
    log(f"Stage 3c: {landed['landed']}/{landed['total']} repeat-1/2 units "
       f"landed (ok or already-valid) so far.")
    if rc != 0:
        log(f"G3 repeats 1-2 (run_cv.py) exited {rc} (nonzero -- could be a "
           f"stop signal or a genuine failure; the landed-unit count above "
           f"is accurate either way since it reads run_cv.py's own index). "
           f"Not marking this stage done; re-running the chain will resume "
           f"from where it left off.")
        return
    log("Stage 3c: all 50 repeat-1/2 units complete.")
    touch(SENTINEL_G3_REPEATS_DONE)


def stage4_a4_repeats() -> None:
    """TERMINAL STAGE. Added 2026-08-05 specifically so the card does not sit
    idle after a G3 Rule-2 failure: runs in EITHER branch (G3 passed or
    failed). Same config/out_dir as stage 3a, repeats 1-2 instead of 0 --
    together with stage 3a this gives A4 checkpointed weights across all
    three repeats, the configuration intended for actual deployment. NO
    TIME-BUDGET SKIP, same as stage 3c: falls through and runs to 50/50 or
    until stopped from outside.

    Only called from main() after stage 3a/3b have both returned "ok" (no
    HALT) -- a bit-identity break earlier in the chain is a reproducibility
    finding serious enough that this function deliberately does not run
    past it uninvited, even though A4-repeats-1-2 is not itself dependent on
    G3. See main()'s own control flow for where that gate lives.
    """
    if os.path.exists(SENTINEL_A4_REPEATS_DONE):
        log("Stage 4 already done (50/50); skipping.")
        return

    log("Stage 4 (terminal, either branch): A4 checkpointed retrain at "
       "repeats 1-2 (50 units) -- this is the deployment ensemble "
       "configuration. Falling through with NO time-budget skip.")
    rc = run_cv("configs/sweep_a4_checkpointed.yaml", "runs/a4_checkpointed",
               repeats="1,2", label="A4 repeats 1-2 checkpointed")
    landed = count_units_landed("runs/a4_checkpointed", unit_prefix_repeats=(1, 2))
    log(f"Stage 4: {landed['landed']}/{landed['total']} repeat-1/2 A4 units "
       f"landed (ok or already-valid) so far.")
    if rc != 0:
        log(f"A4 repeats 1-2 (run_cv.py) exited {rc} -- could be a stop "
           f"signal or a genuine failure; landed-unit count above is "
           f"accurate either way. Not running the bit-identity check or "
           f"marking done until a full 50/50 pass.")
        return

    ok = bit_identity("runs/a4_checkpointed", "runs/sweep_a4",
                      "A4_repeats1-2", repeats="1,2")
    if not ok:
        log("FINDING: A4 repeats 1-2 checkpointed retrain is NOT "
           "bit-identical to the existing runs/sweep_a4 repeat-1/repeat-2 "
           "predictions. Reported, not auto-retried. See "
           "reports/bit_identity_a4_repeats1-2.json. NOT marking stage 4 "
           "done -- re-running the chain will re-attempt the bit-identity "
           "check (units themselves are already on disk and will be "
           "skipped by run_cv.py).")
        return

    log("Stage 4: all 50 repeat-1/2 A4 units complete and bit-identical. "
       "A4 now has checkpointed weights across all three repeats.")
    touch(SENTINEL_A4_REPEATS_DONE)


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    write_pid()
    log("Overnight chain starting.")

    try:
        wait_for_gastronet_queue()
        stage1_reports()

        a4_ok = stage3a_retrain_a4()
        if not a4_ok:
            log("Chain stopping after stage 3a HALT. Bench/3b/3c/4 not attempted.")
            return 3

        g3_passed = read_g3_rule2_verdict()
        log(f"G3 Rule 2 verdict read for branching: passed={g3_passed}")

        g3_ok = stage3b_retrain_g3(g3_passed)
        if not g3_ok:
            log("Chain stopping after stage 3b HALT. Bench/3c/4 not attempted.")
            return 3

        stage3c_g3_repeats(g3_passed)

        # 2026-08-05: moved here (between 3a/3b/3c and 4) from its original
        # spot before 3a. This IS the idle boundary between the two GPU-heavy
        # run_cv.py invocations -- placing it before 3a instead just meant it
        # ran once, at the very start, however long ago that was, rather than
        # at the point actually adjacent to stage 4.
        stage2_throughput_bench()

        # Terminal stage: runs in EITHER branch (g3_passed True or False),
        # since only a HALT above (already returned) skips it.
        stage4_a4_repeats()

        log("Overnight chain complete.")
        touch(CHAIN_COMPLETE)
        return 0
    finally:
        cleanup_pid()


if __name__ == "__main__":
    raise SystemExit(main())
