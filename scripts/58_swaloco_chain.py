"""STEP 8 BRANCH B, stage 3 (2026-08-07/08) -- SWA/EMA LOCO at n=1. tmux
session `swaloco`. CONDITION 3, THE SHIPPING GATE
(reports/swa_ema_n1_pre_registration.md): no SWA or EMA weights enter a
submission container before this closes, regardless of how conditions
1/2/4 landed.

configs/a4_pinned085_swa.yaml VERBATIM, mode=loco, both directions x seeds
0-4 (10 units). Passive EMA/SWA tracking fires in LOCO mode exactly as it
does in CV mode (src/train.py's averaging block does not branch on
cfg.holdout_centre) -- so ONE 10-unit run yields all three variants:
val_loco_c{centre}_s{seed}.parquet (raw = pinned-A4's OWN n=1 LOCO, never
run before tonight), _ema.parquet, _swa.parquet.

Per-seed (single-checkpoint, n=1) FPR@90R per direction -- NOT ensembled
across seeds (that would be the k=5 LOCO figure every other report already
has). Acceptance, extending Amendment 1's paired-seed framework (raw minus
EMA per seed -- 5 GENUINELY independent replicates here, unlike the pooled
arm's fold-dropping pseudo-replicates, since each seed is its own training
run) to LOCO, and requiring BOTH directions per Rule 2's own structure
(this is the highest-stakes gate of the night):

    ACCEPT iff, in BOTH directions: median(raw - ema) > 0 AND >= 4/5 seeds
    agree in sign. VETO: centre-AUC distance-from-null moving further from
    0.5 (EMA vs raw) beyond the across-seed IQR bar, in either direction.

EMA governs per Amendment 1B; SWA reported secondary.

    python scripts/58_swaloco_chain.py
"""
from __future__ import annotations

import importlib.util
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

PID_FILE = os.path.join(LOG_DIR, "swaloco_chain.pid")
SENTINEL_DONE = os.path.join(LOG_DIR, "swaloco_chain_done")
OUT_DIR = "runs/a4_pinned085_swa_loco"
SEEDS = (0, 1, 2, 3, 4)
CENTRES = (1, 2)


def log(msg: str) -> None:
    print(f"[swaloco-chain {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body or time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


def run_arm() -> int:
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_cv.py"),
          "--config", "configs/a4_pinned085_swa.yaml", "--mode", "loco",
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
    _spec = importlib.util.spec_from_file_location(
        "_reanalysis", os.path.join(SCRIPTS_DIR, "17_reanalysis.py"))
    RA = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(RA)
    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))

    variants = ("raw", "ema", "swa")
    per_direction = {}
    for centre in CENTRES:
        vals = {v: {} for v in variants}
        auc_vals = {v: {} for v in variants}
        for seed in SEEDS:
            unit = f"loco_c{centre}_s{seed}"
            base = os.path.join(REPO_ROOT, OUT_DIR, unit)
            raw_p = os.path.join(base, f"val_{unit}.parquet")
            for v in variants:
                p = raw_p if v == "raw" else raw_p.replace(".parquet", f"_{v}.parquet")
                df = pd.read_parquet(p)
                m = metric_block(df["label_int"].to_numpy(), df["logit"].to_numpy())
                vals[v][seed] = m["fpr_at_90_recall"]

        def paired(a_vals, b_vals):
            deltas = [a_vals[s] - b_vals[s] for s in SEEDS]
            median = float(np.median(deltas))
            same_sign = sum(1 for d in deltas if (d > 0) == (median > 0))
            return {"deltas": deltas, "median_delta": median,
                   "n_same_sign": same_sign,
                   "accept": median > 0 and same_sign >= 4}

        per_direction[centre] = {
            "fpr": {v: {"per_seed": vals[v],
                       "median": float(np.median(list(vals[v].values()))),
                       "iqr": float(np.percentile(list(vals[v].values()), 75)
                                   - np.percentile(list(vals[v].values()), 25))}
                   for v in variants},
            "raw_minus_ema": paired(vals["raw"], vals["ema"]),
            "raw_minus_swa": paired(vals["raw"], vals["swa"]),
        }

    both_ema = all(per_direction[c]["raw_minus_ema"]["accept"] for c in CENTRES)
    both_swa = all(per_direction[c]["raw_minus_swa"]["accept"] for c in CENTRES)

    # VETO / confound check: centre_auc_negatives (does the logit leak which
    # centre a negative came from) is UNDEFINED within a single LOCO
    # direction -- that held-out set is single-centre by construction (e.g.
    # loco_c1_s0 holds out ONLY centre_1), so diagnostics_for() returns NaN
    # there and a naive per-direction attempt would silently report "not
    # triggered" from garbage, which is worse than no check (looks like a
    # clean pass when nothing was actually evaluated). Fixed by pooling
    # BOTH directions' held-out predictions per (seed, variant) before the
    # diagnostic -- same "combine complementary held-out models into one
    # evaluation set" convention this project already uses for pooled-OOF
    # (5 different fold-models' held-out predictions, concatenated).
    auc_vals = {v: {} for v in variants}
    for seed in SEEDS:
        for v in variants:
            frames = []
            for centre in CENTRES:
                unit = f"loco_c{centre}_s{seed}"
                raw_p = os.path.join(REPO_ROOT, OUT_DIR, unit, f"val_{unit}.parquet")
                p = raw_p if v == "raw" else raw_p.replace(".parquet", f"_{v}.parquet")
                frames.append(pd.read_parquet(p))
            pooled_both = pd.concat(frames, ignore_index=True)
            diag = RA.diagnostics_for(pooled_both, manifest)
            auc_vals[v][seed] = diag["centre_auc_negatives"]

    auc_summary = {v: {"per_seed": auc_vals[v],
                       "median": float(np.median(list(auc_vals[v].values()))),
                       "iqr": float(np.percentile(list(auc_vals[v].values()), 75)
                                   - np.percentile(list(auc_vals[v].values()), 25))}
                  for v in variants}
    raw_dist = abs(auc_summary["raw"]["median"] - 0.5)
    ema_dist = abs(auc_summary["ema"]["median"] - 0.5)
    bar = max(auc_summary["raw"]["iqr"], auc_summary["ema"]["iqr"], 1e-6)
    veto = (ema_dist - raw_dist) > bar
    veto_detail = {"raw_dist": raw_dist, "ema_dist": ema_dist, "bar": bar,
                  "triggered": veto, "auc_summary": auc_summary}

    accept_ema = both_ema and not veto
    return {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "per_direction": per_direction, "both_directions_accept_ema": both_ema,
           "both_directions_accept_swa": both_swa, "veto": veto,
           "veto_detail": veto_detail,
           "condition3_verdict": "ACCEPT" if accept_ema else "REJECT",
           "governs": "ema"}


def render(r: dict) -> None:
    L = []
    A = L.append
    A("# SWA/EMA LOCO at n=1 -- CONDITION 3, the shipping gate")
    A("")
    A(f"Generated {r['generated_utc']}. `configs/a4_pinned085_swa.yaml`, "
     f"`mode=loco`, both directions x seeds 0-4 (10 units), `{OUT_DIR}`. "
     f"Per-seed (n=1) FPR@90R, NOT ensembled. EMA governs per Amendment 1B; "
     f"SWA reported secondary.")
    A("")
    A(f"## VERDICT: **{r['condition3_verdict']}**")
    A("")
    A(f"Both directions accept (EMA vs raw, paired per-seed): "
     f"**{r['both_directions_accept_ema']}**. VETO (centre-AUC confound, "
     f"pooled across BOTH directions per seed -- see note below on why "
     f"per-direction is undefined): "
     f"**{'TRIGGERED' if r['veto'] else 'not triggered'}**.")
    A("")
    vd = r["veto_detail"]
    A(f"Centre-AUC confound (both LOCO directions pooled per seed, since a "
     f"single direction's held-out set is single-centre by construction "
     f"and the diagnostic is undefined there): raw distance-from-null "
     f"{vd['raw_dist']:.4f}, EMA {vd['ema_dist']:.4f}, bar {vd['bar']:.4f} "
     f"-- {'VETO' if vd['triggered'] else 'ok'}.")
    A("")
    A("No SWA or EMA weights enter a submission container "
     f"{'-- gate is OPEN, this is now permitted subject to human sign-off' if r['condition3_verdict']=='ACCEPT' else '-- gate remains CLOSED'}.")
    A("")
    for centre, d in r["per_direction"].items():
        A(f"## holdout_center_{centre}")
        A("")
        A("| variant | per-seed FPR@90R (s0..s4) | median | IQR |")
        A("|---|---|---|---|")
        for v in ("raw", "ema", "swa"):
            fv = d["fpr"][v]
            vals = " / ".join(f"{fv['per_seed'][s]:.4f}" for s in SEEDS)
            A(f"| {v} | {vals} | {fv['median']:.4f} | {fv['iqr']:.4f} |")
        A("")
        rme = d["raw_minus_ema"]
        A(f"Paired raw-minus-EMA: median delta {rme['median_delta']:+.4f}, "
         f"{rme['n_same_sign']}/5 share that sign -- "
         f"**{'ACCEPT' if rme['accept'] else 'REJECT'}** this direction.")
        rms = d["raw_minus_swa"]
        A(f"Paired raw-minus-SWA (secondary): median delta "
         f"{rms['median_delta']:+.4f}, {rms['n_same_sign']}/5 share that "
         f"sign -- {'ACCEPT' if rms['accept'] else 'REJECT'}"
         f"{' (disagrees with EMA -- EMA governs)' if rms['accept'] != rme['accept'] else ''}.")
        A("")

    with open(os.path.join(REPORT_DIR, "swaloco_n1.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "swaloco_n1.json"), "w") as fh:
        json.dump(r, fh, indent=2, default=float)
    log(f"written: reports/swaloco_n1.md -- VERDICT {r['condition3_verdict']}")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    touch(PID_FILE, f"{os.getpid()}\n")
    log("SWA/EMA LOCO (n=1) chain starting.")
    exit_code = 0
    try:
        if not os.path.exists(SENTINEL_DONE):
            rc = run_arm()
            log(f"run_cv.py exited {rc}")
            if rc != 0:
                log("not a clean 10/10. Falling through to step 4 (ema) "
                   "per the brief -- re-run this script later to "
                   "resume/complete this stage.")
                exit_code = 3
            else:
                touch(SENTINEL_DONE)
        if os.path.exists(SENTINEL_DONE):
            r = analyse()
            render(r)
            log(f"CONDITION 3 VERDICT: {r['condition3_verdict']}")
        subprocess.run(["bash", os.path.join(SCRIPTS_DIR, "run_ema_chain.sh")],
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
