"""JOB 1 (2026-08-07) -- A4 LOCO under CORRECTED code. Detached in tmux
(session rare26_a4_loco_chain), resumable via run_cv.py's own native
skip-validated logic and a PID file (same pattern as
scripts/44_pinned_arm_chain.py / scripts/run_pinned_chain.sh).

WHY THIS RUN EXISTS. Every A4 LOCO figure on record (0.1605 / 0.1011, from
runs/sweep_a2_loco) was trained under OLD-FORMULA code, pre-2026-07-31
downsample_ceiling_m fix. The corrected code moved A4's pooled-OOF FPR@90R
0.0529 -> 0.0280 by making the downsample-ceiling augmentation MILDER (mean
draw factor 0.818 -> 0.868 at s=0.33) while moving the centre-AUC confound
distance from the null 0.0117 -> 0.0339 (bar 0.0081, resolvable) -- i.e. the
milder augmentation bought in-distribution FPR@90R at some cost to how
cleanly A4 has scrubbed the hospital-identity shortcut. Whether that same
trade costs HELD-OUT-CENTRE performance is a separate, unanswered question:
nothing before this run tested a corrected-code A4 on an unseen centre. This
is the only run that can answer it.

configs/sweep_a4_checkpointed.yaml VERBATIM (mode=loco instead of its default
cv), both directions x seeds 0-4 = 10 units, save_checkpoint: true (already
set in that config). Comparison baseline for the report stays
runs/a4_checkpointed (0.0280 pooled) per the standing decision -- this run
does not touch reference_a4.

    python scripts/49_a4_loco_chain.py
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

PID_FILE = os.path.join(LOG_DIR, "a4_loco_chain.pid")
SENTINEL_DONE = os.path.join(LOG_DIR, "a4_loco_chain_done")

OUT_DIR = "runs/a4_checkpointed_loco"
A0_LOCO_DIR = "runs/noise_floor_b"
A4_OLD_LOCO_DIR = "runs/sweep_a2_loco"  # OLD-FORMULA -- reused per brief, not re-run
SEEDS = (0, 1, 2, 3, 4)

A4_POOLED_OLD = 0.0529
A4_POOLED_NEW = 0.0280
CENTRE_AUC_OLD = 0.0117   # |0.4883 - 0.5|, old-formula A4, distance from the 0.5 null
CENTRE_AUC_NEW = 0.0339   # |0.4661 - 0.5|, corrected-code A4
CENTRE_AUC_BAR = 0.0081


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[a4-loco-chain {ts}] {msg}", flush=True)


def touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body or time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


def write_pid() -> None:
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


def run_arm() -> int:
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_cv.py"),
          "--config", "configs/sweep_a4_checkpointed.yaml", "--mode", "loco",
          "--centres", "1,2", "--seeds", "0,1,2,3,4",
          "--out-dir", OUT_DIR]
    log(f"RUN: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    return r.returncode


def compute_report() -> dict:
    for p in (REPO_ROOT, SCRIPTS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "_ms", os.path.join(SCRIPTS_DIR, "26_magnitude_sweep.py"))
    MS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MS)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))

    a0 = MS.analyse_loco(A0_LOCO_DIR, manifest, seeds=SEEDS)
    a4_old = MS.analyse_loco(A4_OLD_LOCO_DIR, manifest, seeds=SEEDS)
    a4_new = MS.analyse_loco(OUT_DIR, manifest, seeds=SEEDS)

    per_centre = {}
    for c in (1, 2):
        old, new = a4_old[c], a4_new[c]
        delta_vs_old = new["k5_fpr"] - old["k5_fpr"]
        bar_vs_old = max(new["loo4_spread"]["iqr"], old["loo4_spread"]["iqr"])
        per_centre[c] = {
            "a0_k5_fpr": a0[c]["k5_fpr"],
            "a4_old_k5_fpr": old["k5_fpr"], "a4_old_loo4_iqr": old["loo4_spread"]["iqr"],
            "a4_new_k5_fpr": new["k5_fpr"], "a4_new_loo4_iqr": new["loo4_spread"]["iqr"],
            "delta_new_minus_old": delta_vs_old, "bar_vs_old": bar_vs_old,
            "resolvable_vs_old": abs(delta_vs_old) > bar_vs_old,
            "regressed_vs_old": delta_vs_old > bar_vs_old,
        }

    any_regressed = any(v["regressed_vs_old"] for v in per_centre.values())
    any_improved = any(v["resolvable_vs_old"] and not v["regressed_vs_old"]
                       for v in per_centre.values())

    if any_regressed:
        headline = (
            "YES -- the milder downsample-ceiling augmentation that improved "
            "pooled FPR@90R (0.0529 -> 0.0280) COSTS held-out-centre "
            "performance on at least one direction: corrected-code A4 LOCO "
            "is resolvably WORSE than old-formula A4 LOCO there."
        )
    elif any_improved:
        headline = (
            "NO -- the milder downsample-ceiling augmentation does not cost "
            "held-out-centre performance; corrected-code A4 LOCO is "
            "resolvably no worse (and at least one direction resolvably "
            "better) than old-formula A4 LOCO."
        )
    else:
        headline = (
            "INCONCLUSIVE ON THIS EVIDENCE -- neither direction moves beyond "
            "its own noise bar relative to old-formula A4 LOCO; the milder "
            "augmentation's pooled-OOF gain (0.0529 -> 0.0280) and its "
            "centre-AUC confound cost (0.0117 -> 0.0339, bar 0.0081) do not "
            "resolve into a held-out-centre cost or benefit either way."
        )

    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": "configs/sweep_a4_checkpointed.yaml",
        "out_dir": OUT_DIR,
        "reference_a4_pooled": {"old": A4_POOLED_OLD, "new": A4_POOLED_NEW},
        "centre_auc_confound": {
            "old": CENTRE_AUC_OLD, "new": CENTRE_AUC_NEW, "bar": CENTRE_AUC_BAR,
            "regressed": (CENTRE_AUC_NEW - CENTRE_AUC_OLD) > CENTRE_AUC_BAR,
        },
        "per_centre": per_centre,
        "old_formula_caveat": (
            f"A4-old-formula LOCO figures reused from {A4_OLD_LOCO_DIR} "
            f"(A2's LOCO arm, sampler-degeneracy argument per the brief) -- "
            f"not re-run here. A0 reused from {A0_LOCO_DIR}, context only."
        ),
        "headline": headline,
    }


def render(report: dict) -> None:
    lines = []
    A = lines.append
    A("# A4 LOCO under corrected code -- does the milder downsample-ceiling augmentation cost held-out-centre performance?")
    A("")
    A(f"Generated {report['generated_utc']}. `configs/sweep_a4_checkpointed.yaml` "
     f"verbatim, `mode=loco`, both directions x seeds 0-4 (10 units), "
     f"`save_checkpoint: true`, `out_dir {report['out_dir']}`.")
    A("")
    A("**Context.** The corrected code (post-2026-07-31 `downsample_ceiling_m` "
     "fix) moved A4's pooled-OOF FPR@90R from "
     f"{report['reference_a4_pooled']['old']:.4f} to "
     f"{report['reference_a4_pooled']['new']:.4f} by making the "
     "downsample-ceiling augmentation MILDER (mean draw factor 0.818 -> 0.868 "
     "at s=0.33) -- but the same fix moved the centre-AUC confound distance "
     f"from the 0.5 null {report['centre_auc_confound']['old']:.4f} -> "
     f"{report['centre_auc_confound']['new']:.4f}, against a bar of "
     f"{report['centre_auc_confound']['bar']:.4f} "
     f"({'regressed' if report['centre_auc_confound']['regressed'] else 'not flagged as regressed by itself'}). "
     "This run is the only evidence that can say whether that same milder "
     "augmentation also costs held-out-centre (LOCO) performance.")
    A("")
    A(f"**{report['headline']}**")
    A("")
    A("| direction | A0 | A4 old-formula | A4 corrected-code | delta (new - old) | bar | resolvable? | regressed? |")
    A("|---|---|---|---|---|---|---|---|")
    for c in (1, 2):
        pc = report["per_centre"][c]
        A(f"| holdout_center_{c} | {pc['a0_k5_fpr']:.4f} | "
         f"{pc['a4_old_k5_fpr']:.4f} (IQR {pc['a4_old_loo4_iqr']:.4f}) | "
         f"{pc['a4_new_k5_fpr']:.4f} (IQR {pc['a4_new_loo4_iqr']:.4f}) | "
         f"{pc['delta_new_minus_old']:+.4f} | {pc['bar_vs_old']:.4f} | "
         f"{'**yes**' if pc['resolvable_vs_old'] else 'no'} | "
         f"{'**yes**' if pc['regressed_vs_old'] else 'no'} |")
    A("")
    A(f"_{report['old_formula_caveat']}_")
    A("")

    md_path = os.path.join(REPORT_DIR, "a4_checkpointed_loco.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(REPORT_DIR, "a4_checkpointed_loco.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"written: {md_path}")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    write_pid()
    log("A4 LOCO (corrected-code) chain starting.")
    try:
        if os.path.exists(SENTINEL_DONE):
            log("Already done (sentinel present); skipping straight to report.")
        else:
            rc = run_arm()
            log(f"run_cv.py exited {rc}")
            if rc != 0:
                log("run_arm() did not report a clean 10/10 -- re-run this "
                   "script to resume the remaining units. Not computing a "
                   "report from a partial arm.")
                return 3
            touch(SENTINEL_DONE)

        report = compute_report()
        render(report)
        log(f"HEADLINE: {report['headline']}")
        return 0
    finally:
        cleanup_pid()


if __name__ == "__main__":
    raise SystemExit(main())
