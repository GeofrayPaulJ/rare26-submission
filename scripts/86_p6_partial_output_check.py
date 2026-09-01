"""P6 -- empirical confirmation of G3's static-read finding: does killing
the production entrypoint mid-run leave ANY file under /output?

G3 (reports/g3_timeout_semantics.md) found by reading predict.py/inference.py
that write_json_file/write_stats fire exactly once, after predict_stack()
fully returns, with no per-batch/per-case flush. This script empirically
confirms that by running the real interface_0_handler-equivalent path
(rare26_infer.predict.predict_stack against probe C's actual stack.mha,
CPU only -- CUDA hidden, so the run is deliberately slow enough to still be
mid-loop) in a child process, SIGKILLing it after a fixed delay, then
checking the output directory for anything at all.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/86_p6_partial_output_check.py'
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

REPO_ROOT = "/workspace/RARE26"
IMAGE_DIR = os.path.join(
    REPO_ROOT, "runs", "submission_test", "platform_probe_p5", "c", "interface_0",
    "images", "stacked-barretts-esophagus-endoscopy",
)
WEIGHTS_DIR = os.path.join(REPO_ROOT, "submission", "resources", "ensemble")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "submission_test", "p6_kill_test")
KILL_AFTER_S = 45

CHILD_SCRIPT = f"""
import os, sys, json
sys.path.insert(0, {os.path.join(REPO_ROOT, "submission")!r})
os.environ.setdefault("RARE26_NUM_WORKERS", "8")
os.environ.setdefault("RARE26_BATCH_SIZE", "32")
from rare26_infer.predict import predict_stack, write_stats

weights = sorted(
    os.path.join({WEIGHTS_DIR!r}, f) for f in os.listdir({WEIGHTS_DIR!r}) if f.endswith(".pth")
)
probs, stats = predict_stack({IMAGE_DIR!r}, weights)

# Mirrors inference.py's write order exactly: json first, stats second.
os.makedirs({OUT_DIR!r}, exist_ok=True)
with open(os.path.join({OUT_DIR!r}, "stacked-neoplastic-lesion-likelihoods.json"), "w") as fh:
    fh.write(json.dumps(probs, indent=4))
write_stats(stats, os.path.join({OUT_DIR!r}, "rare26_run_stats.json"))
print("CHILD COMPLETED NORMALLY -- should not happen, kill fired too late", flush=True)
"""


def main() -> int:
    if os.path.isdir(OUT_DIR):
        for f in os.listdir(OUT_DIR):
            os.remove(os.path.join(OUT_DIR, f))
    os.makedirs(OUT_DIR, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # zero GPU, per instruction

    print(f"[p6] launching child, will SIGKILL after {KILL_AFTER_S}s", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD_SCRIPT],
        cwd=REPO_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    t0 = time.perf_counter()
    time.sleep(KILL_AFTER_S)
    still_running = proc.poll() is None
    if still_running:
        proc.send_signal(signal.SIGKILL)
        print(f"[p6] sent SIGKILL at t={time.perf_counter() - t0:.1f}s "
              f"(child was still running -- confirms mid-run kill)", flush=True)
    else:
        print(f"[p6] child had ALREADY EXITED before the kill (code={proc.returncode}) "
              f"-- rerun with a shorter KILL_AFTER_S to test a genuine mid-run kill",
              flush=True)

    out, _ = proc.communicate(timeout=15)
    print("---- child stdout/stderr (tail) ----")
    print("\n".join(out.splitlines()[-30:]))
    print("-------------------------------------")

    contents = sorted(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else []
    print(f"[p6] /output contents after kill: {contents!r}")
    if contents:
        print("[p6] RESULT: FILE(S) WRITTEN despite the kill -- P6 finding is "
              "PARTIAL/INCREMENTAL OUTPUT EXISTS, contradicting the static read.")
        return 1
    print("[p6] RESULT: EMPTY -- confirms G3's static-code finding: a mid-run "
          "kill leaves NOTHING on disk. No partial output is ever scorable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
