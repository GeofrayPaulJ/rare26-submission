"""Remediation item 5, 2026-08-05. Detached in tmux (session
rare26_pinned_chain), resumable via run_cv.py's own native skip-validated
logic and a PID file (same pattern as scripts/40_overnight_chain.py).

Single linear arm, no branching:
  1. Run configs/a4_pinned085.yaml -> runs/a4_pinned085, repeat 0, 5 folds x
     5 seeds = 25 units, save_checkpoint: true. downsample_ceiling_m pinned
     to EXACTLY 0.85 (verified before launch, see configs/a4_pinned085.yaml's
     own header for the algebra) -- everything else is current code,
     unpinned, identical to configs/sweep_a4_checkpointed.yaml.
  2. Compute pooled-OOF k=5 FPR@90R and its LOO-4 IQR. Report against BOTH
     0.0280 (current code, unpinned, runs/a4_checkpointed) and 0.0529 (old
     code, runs/sweep_a4). State which one it lands near. Also report the
     LOO-4 IQR against 0.0027 (unpinned) and 0.0102 (old) -- the point
     estimate landing near 0.0529 does not by itself confirm this parameter
     also explains the variance collapse; that needs the IQR to move too.
  3. Write reports/a4_pinned085_verdict.md and .json.

    python scripts/44_pinned_arm_chain.py
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

PID_FILE = os.path.join(LOG_DIR, "pinned_chain.pid")
SENTINEL_DONE = os.path.join(LOG_DIR, "pinned_chain_done")

OLD_FPR = 0.0529
UNPINNED_FPR = 0.0280
OLD_IQR = 0.0102
UNPINNED_IQR = 0.0027
# Called "near" if within half the old-vs-unpinned gap of a reference point.
NEAR_TOLERANCE = abs(OLD_FPR - UNPINNED_FPR) / 2.0  # 0.01245


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[pinned-chain {ts}] {msg}", flush=True)


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
          "--config", "configs/a4_pinned085.yaml", "--mode", "cv",
          "--repeats", "0", "--out-dir", "runs/a4_pinned085"]
    log(f"RUN: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    return r.returncode


def compute_verdict() -> dict:
    for p in (REPO_ROOT, SCRIPTS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "_gn", os.path.join(SCRIPTS_DIR, "30_gastronet.py"))
    GN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(GN)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))
    pinned = GN.analyse("runs/a4_pinned085", manifest)
    fpr = pinned["k5"]["fpr_at_90_recall"]
    iqr = pinned["loo4"]["fpr_at_90_recall"]["iqr"]

    dist_old = abs(fpr - OLD_FPR)
    dist_unpinned = abs(fpr - UNPINNED_FPR)
    lands_near_old = dist_old < dist_unpinned

    verdict = {
        "pinned_fpr_at_90_recall": fpr,
        "pinned_loo4_iqr": iqr,
        "old_fpr": OLD_FPR, "unpinned_fpr": UNPINNED_FPR,
        "old_iqr": OLD_IQR, "unpinned_iqr": UNPINNED_IQR,
        "distance_to_old": dist_old, "distance_to_unpinned": dist_unpinned,
        "lands_near_old": lands_near_old,
        "iqr_also_restored": abs(iqr - OLD_IQR) < abs(iqr - UNPINNED_IQR),
        "conclusion": (
            "PARAMETER CONFIRMED AS CAUSE: pinned run lands near old-code "
            "FPR@90R" if lands_near_old else
            "CAUSE IS ELSEWHERE: pinned run still lands near the unpinned "
            "current-code figure despite downsample_ceiling_m being pinned "
            "to the old value -- every pooled-OOF figure computed since "
            "31 July needs re-examination. HALT."
        ),
    }
    return verdict


def render_and_write(verdict: dict) -> None:
    lines = []
    A = lines.append
    A("# Remediation item 5 -- decisive isolation arm verdict")
    A("")
    A(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. "
     f"`runs/a4_pinned085`: current code, `downsample_ceiling_m` pinned to "
     f"exactly 0.85 (verified before launch), everything else identical to "
     f"`configs/sweep_a4_checkpointed.yaml`.")
    A("")
    A("| | FPR@90R | LOO-4 IQR |")
    A("|---|---|---|")
    A(f"| Old code (`runs/sweep_a4`) | {OLD_FPR:.4f} | {OLD_IQR:.4f} |")
    A(f"| Current code, unpinned (`runs/a4_checkpointed`) | {UNPINNED_FPR:.4f} | {UNPINNED_IQR:.4f} |")
    A(f"| **Current code, PINNED to 0.85** | **{verdict['pinned_fpr_at_90_recall']:.4f}** | **{verdict['pinned_loo4_iqr']:.4f}** |")
    A("")
    A(f"Distance to old: {verdict['distance_to_old']:.4f}. "
     f"Distance to unpinned: {verdict['distance_to_unpinned']:.4f}.")
    A("")
    A(f"**{verdict['conclusion']}**")
    A("")
    A(f"IQR also restored toward old (0.0102) rather than staying near "
     f"unpinned (0.0027): **{verdict['iqr_also_restored']}**. If the point "
     f"estimate confirms the parameter but the IQR does not follow, the "
     f"variance-collapse question from item 2 remains open even after this "
     f"result.")
    A("")

    md_path = os.path.join(REPORT_DIR, "a4_pinned085_verdict.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(REPORT_DIR, "a4_pinned085_verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)
    log(f"written: {md_path}")


# ---------------------------------------------------------------------------
# Item 1 (2026-08-06 amendment): G0 LOCO, queued immediately behind item 5.
# configs/g0_rn50_imagenet.yaml VERBATIM, mode=loco, both directions x seeds
# 0-4 = 10 units. RN50 speed (~22-25s/epoch measured elsewhere for RN50
# arms), 10 units x 30 epochs -> ~2h.
# ---------------------------------------------------------------------------
G0_LOCO_SENTINEL_DONE = os.path.join(LOG_DIR, "g0_loco_done")
G0_LOCO_OUT_DIR = "runs/g0_rn50_imagenet_loco"
A0_LOCO_DIR = "runs/noise_floor_b"
A4_LOCO_DIR = "runs/sweep_a2_loco"  # OLD-FORMULA -- see report caveat below
G3_LOCO_DIR = "runs/g3_rn50_gastronet_loco"


def run_g0_loco() -> int:
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_cv.py"),
          "--config", "configs/g0_rn50_imagenet.yaml", "--mode", "loco",
          "--centres", "1,2", "--seeds", "0,1,2,3,4",
          "--out-dir", G0_LOCO_OUT_DIR]
    log(f"RUN: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    return r.returncode


def compute_g0_loco_report() -> dict:
    for p in (REPO_ROOT, SCRIPTS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "_ms", os.path.join(SCRIPTS_DIR, "26_magnitude_sweep.py"))
    MS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MS)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))
    SEEDS = (0, 1, 2, 3, 4)

    a0 = MS.analyse_loco(A0_LOCO_DIR, manifest, seeds=SEEDS)
    a4 = MS.analyse_loco(A4_LOCO_DIR, manifest, seeds=SEEDS)
    g3 = MS.analyse_loco(G3_LOCO_DIR, manifest, seeds=SEEDS)
    g0 = MS.analyse_loco(G0_LOCO_OUT_DIR, manifest, seeds=SEEDS)

    with open(os.path.join(REPO_ROOT, "reports", "gastronet.json")) as fh:
        gj = json.load(fh)
    g3_pooled = gj["results"]["g3_rn50_gastronet"]["pooled"]["k5"]["fpr_at_90_recall"]
    spec2 = importlib.util.spec_from_file_location(
        "_gn", os.path.join(SCRIPTS_DIR, "30_gastronet.py"))
    GN = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(GN)
    g0_pooled_full = GN.analyse("runs/g0_rn50_imagenet", manifest)
    g0_pooled = g0_pooled_full["k5"]["fpr_at_90_recall"]
    pooled_delta = g3_pooled - g0_pooled  # negative: G3 LOWER (better)

    per_centre = {}
    for c in (1, 2):
        d3, d0 = g3[c], g0[c]
        delta = d3["k5_fpr"] - d0["k5_fpr"]
        bar = max(d3["loo4_spread"]["iqr"], d0["loo4_spread"]["iqr"])
        per_centre[c] = {
            "a0_k5_fpr": a0[c]["k5_fpr"], "a4_k5_fpr": a4[c]["k5_fpr"],
            "g3_k5_fpr": d3["k5_fpr"], "g0_k5_fpr": d0["k5_fpr"],
            "g3_minus_g0": delta, "bar": bar, "resolvable": abs(delta) > bar,
        }

    survives = all(abs(v["g3_minus_g0"]) > 1e-6 and
                  abs(v["g3_minus_g0"]) > 0.3 * abs(pooled_delta)
                  for v in per_centre.values())

    return {
        "headline_question": (
            "G3 beats G0 by 0.1420 pooled at fixed capacity -- how much of "
            "that survives to a held-out centre?"
        ),
        "pooled_g3_minus_g0": pooled_delta,
        "per_centre": per_centre,
        "a4_loco_caveat": (
            f"A4 LOCO figures come from {A4_LOCO_DIR}, OLD-FORMULA code "
            f"(pre-2026-07-31 downsample_ceiling_m) -- shown for context "
            f"only, not part of the G3-vs-G0 comparison, which is current-"
            f"code on both sides."
        ),
        "near_zero_on_loco": not survives,
        "conclusion": (
            "LOCO delta is near zero relative to the pooled gap: GastroNet "
            "pretraining bought a large in-distribution gain and little that "
            "is portable to an unseen centre."
            if not survives else
            "LOCO delta remains a substantial fraction of the pooled gap: "
            "GastroNet pretraining's advantage is not purely an "
            "in-distribution artefact."
        ),
    }


def render_g0_loco_report(report: dict) -> None:
    lines = []
    A = lines.append
    A("# G0 LOCO -- does GastroNet pretraining's advantage survive an unseen centre?")
    A("")
    A(f"**{report['headline_question']}**")
    A("")
    A(f"Pooled (current code), from `reports/gastronet.json` / fresh "
     f"`runs/g0_rn50_imagenet` analysis: G3 minus G0 = "
     f"**{report['pooled_g3_minus_g0']:+.4f}**.")
    A("")
    A("| direction | A0 | A4 (old-formula, context only) | G3 | G0 | G3 minus G0 | bar | resolvable? |")
    A("|---|---|---|---|---|---|---|---|")
    for c in (1, 2):
        pc = report["per_centre"][c]
        A(f"| holdout_center_{c} | {pc['a0_k5_fpr']:.4f} | {pc['a4_k5_fpr']:.4f} | "
         f"{pc['g3_k5_fpr']:.4f} | {pc['g0_k5_fpr']:.4f} | "
         f"{pc['g3_minus_g0']:+.4f} | {pc['bar']:.4f} | "
         f"{'**yes**' if pc['resolvable'] else 'no'} |")
    A("")
    A(f"_{report['a4_loco_caveat']}_")
    A("")
    A(f"**{report['conclusion']}**")
    A("")

    md_path = os.path.join(REPORT_DIR, "g0_loco.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(REPORT_DIR, "g0_loco.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"written: {md_path}")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    write_pid()
    log("Pinned-arm chain starting.")
    try:
        if os.path.exists(SENTINEL_DONE):
            log("Already done (sentinel present); skipping straight to verdict.")
        else:
            rc = run_arm()
            log(f"run_cv.py exited {rc}")
            # Resumable natively: re-running this script re-invokes run_cv.py,
            # which skips every already-validated unit. No retry loop needed
            # here beyond that.
            if rc != 0:
                log("run_arm() did not report a clean 25/25 -- re-run this "
                   "script to resume the remaining units. Not computing a "
                   "verdict from a partial arm.")
                return 3
            touch(SENTINEL_DONE)

        verdict = compute_verdict()
        render_and_write(verdict)
        log(f"VERDICT: {verdict['conclusion']}")

        # Item 1 (2026-08-06): G0 LOCO, queued immediately behind item 5.
        # Independent of the pinned-085 verdict -- runs either way.
        if os.path.exists(G0_LOCO_SENTINEL_DONE):
            log("G0 LOCO already done (sentinel present); skipping straight to report.")
        else:
            rc = run_g0_loco()
            log(f"G0 LOCO run_cv.py exited {rc}")
            if rc != 0:
                log("G0 LOCO did not report a clean 10/10 -- re-run this "
                   "script to resume the remaining units.")
                return 3
            touch(G0_LOCO_SENTINEL_DONE)

        g0_loco_report = compute_g0_loco_report()
        render_g0_loco_report(g0_loco_report)
        log(f"G0 LOCO CONCLUSION: {g0_loco_report['conclusion']}")

        return 0
    finally:
        cleanup_pid()


if __name__ == "__main__":
    raise SystemExit(main())
