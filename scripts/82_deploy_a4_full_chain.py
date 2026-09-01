"""R1 -- sequential driver for Stage 1: A4-corrected full-data deployment
checkpoints, seeds 0 through 4, one GPU, one seed at a time.

Seed 0 outranks 1-4 (it is the 15 August submission artefact); this
driver still runs 1-4 after it rather than stopping, since the queue
instruction says "seeds 0 through 4," not "seed 0 only."

Each seed reuses `scripts/81_deploy_a4_full.py`'s `train_full_data()`
directly (not a subprocess call) so the gpu_hold_kill check inside its
epoch loop, the resume-from-last.pt logic, and the bit-identity assertion
before epoch 1 all apply unchanged, per seed. A gpu_hold_kill stop on any
seed halts the WHOLE chain immediately -- it does not skip ahead to the
next seed, because the sentinel means "get off the GPU now," not "get
off this seed only."

PID file / done sentinel follow this project's standing watchdog
convention (see scripts/sweep_watchdog.ps1's per-driver PID/cmdline
check) so this driver can be wired into the watchdog the same way every
other chain already is.

    python scripts/82_deploy_a4_full_chain.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO_ROOT, "logs")
SEEDS = [0, 1, 2, 3, 4]
PID_FILE = os.path.join(LOG_DIR, "deploy_a4_full_chain.pid")
DONE_SENTINEL = os.path.join(LOG_DIR, "deploy_a4_full_chain_done")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def log(msg: str) -> None:
    print(f"[deploy-a4-full-chain {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def _load_unit_module():
    spec = importlib.util.spec_from_file_location(
        "_deploy_a4_full_unit", os.path.join(REPO_ROOT, "scripts", "81_deploy_a4_full.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(PID_FILE, "w") as fh:
        fh.write(f"{os.getpid()}\n")
    log(f"chain starting -- seeds {SEEDS}")

    unit = _load_unit_module()
    per_seed_seconds = {}
    exit_code = 0
    try:
        for seed in SEEDS:
            weights_path = os.path.join(REPO_ROOT, "runs", f"deploy_a4_full_s{seed}",
                                        "checkpoints", "weights_fp32.pt")
            if os.path.exists(weights_path):
                log(f"seed {seed}: weights already present, skipping (skip-validated).")
                per_seed_seconds[seed] = 0.0
                continue

            log(f"seed {seed}: starting")
            t0 = time.perf_counter()
            summary = unit.train_full_data(seed)
            dt = time.perf_counter() - t0
            per_seed_seconds[seed] = dt

            if summary.get("stopped"):
                log(f"seed {seed}: STOPPED by gpu_hold_kill at epoch "
                   f"{summary['epochs_completed']}/30 -- halting the WHOLE chain, "
                   f"not just this seed. Re-run this script to resume seed {seed} "
                   f"and continue the remaining seeds after it.")
                exit_code = 4
                break

            log(f"seed {seed}: done, {summary['epochs_completed']} epochs, "
               f"{dt / 3600:.2f} h, final_train_loss={summary.get('final_train_loss')}")

            completed = [s for s in SEEDS if s <= seed]
            remaining = [s for s in SEEDS if s > seed]
            if remaining:
                avg = sum(per_seed_seconds[s] for s in completed if per_seed_seconds[s] > 0) / \
                     max(1, sum(1 for s in completed if per_seed_seconds[s] > 0))
                log(f"queue ETA after seed {seed}: {avg / 3600:.2f} h/seed avg -> "
                   f"~{len(remaining) * avg / 3600:.2f} h remaining for seeds {remaining}")

        else:
            with open(DONE_SENTINEL, "w") as fh:
                fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
            log(f"chain complete: all seeds {SEEDS} done, "
               f"{sum(per_seed_seconds.values()) / 3600:.2f} h total")

        return exit_code
    except Exception as exc:  # noqa: BLE001
        log(f"FAILED: {exc!r} -- re-run the same command to resume")
        return 3
    finally:
        try:
            with open(PID_FILE) as fh:
                if int(fh.read().strip()) == os.getpid():
                    os.remove(PID_FILE)
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
