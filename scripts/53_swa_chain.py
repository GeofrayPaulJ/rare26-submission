"""STEP 8 BRANCH B, stage 1 (2026-08-07) -- SWA/EMA arm. tmux session `swa`.

Runs configs/a4_pinned085_swa.yaml (pinned config + passive EMA/SWA tracking,
5 folds x seed 0, save_checkpoint: true), then:

  1. BIT-IDENTITY CROSS-CHECK: the raw parquets must be bit-identical to
     runs/a4_pinned085's r0_f*_s0 parquets -- averaging is passive by
     construction (src/train.py's averaging block), so any mismatch is a
     FINDING about that claim, reported loudly, not worked around.
  2. Pooled-OOF (5 folds, seed 0) FPR@90R for raw vs EMA vs SWA, plus
     per-fold single-checkpoint figures -- the single-checkpoint tail is the
     number that matters: EMA/SWA buy ensemble-like variance reduction inside
     ONE checkpoint at zero inference cost, the only kind of gain the
     10-minute case budget admits.
  3. reports/swa_arm.{md,json}; sentinel logs/swa_chain_done; then chains
     into the pAUC smoke (scripts/run_pauc_chain.sh, tmux `pauc`).

    python scripts/53_swa_chain.py
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

PID_FILE = os.path.join(LOG_DIR, "swa_chain.pid")
SENTINEL_DONE = os.path.join(LOG_DIR, "swa_chain_done")
SENTINEL_EXT_DONE = os.path.join(LOG_DIR, "swa_chain_seeds12_done")
OUT_DIR = "runs/a4_pinned085_swa"
BASELINE_DIR = "runs/a4_pinned085"
FOLDS = (0, 1, 2, 3, 4)


def log(msg: str) -> None:
    print(f"[swa-chain {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def touch(path: str, body: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body or time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


def run_arm(seeds: str = "0") -> int:
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS_DIR, "run_cv.py"),
          "--config", "configs/a4_pinned085_swa.yaml", "--mode", "cv",
          "--repeats", "0", "--folds", "0,1,2,3,4", "--seeds", seeds,
          "--out-dir", OUT_DIR]
    log(f"RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def available_seeds() -> list:
    seeds = []
    for s in (0, 1, 2, 3, 4):
        if all(os.path.exists(os.path.join(REPO_ROOT, OUT_DIR, f"r0_f{f}_s{s}",
                                           f"val_r0_f{f}_s{s}.parquet"))
              for f in FOLDS):
            seeds.append(s)
    return seeds


def _load(name: str, filename: str):
    import importlib.util
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pinned_a4_single_seed_baseline(manifest) -> dict:
    """Zero-GPU: runs/a4_pinned085 already has all 5 seeds (25/25 units).
    reports/swa_ema_n1_pre_registration.md's condition 1 comparator --
    pinned-A4's OWN single_seed distribution (median/IQR of its 5 per-seed
    pooled-OOF figures), the established convention from
    reports/gastronet.json (scripts/30_gastronet.py's analyse())."""
    GN = _load("_gastronet", "30_gastronet.py")
    return GN.analyse(BASELINE_DIR, manifest)["single_seed"]


def loo4_over_folds_paired(frames_by_seed_a: dict, frames_by_seed_b: dict,
                           metric_block) -> dict:
    """Amendment 1, condition 1: for each trained seed, 5 pseudo-replicate
    pooled FPR@90R deltas (a minus b), each dropping a different one of the
    5 folds from that seed's pooled set. Pseudo-replicates share folds
    across the drop and are NOT independent -- no p-value, per the
    amendment. Deltas pool across every trained seed (5 at seed-0-only, 15
    once/if seeds 1-2 are added)."""
    import pandas as pd
    deltas = []
    per_seed_drop = {}
    for seed in sorted(frames_by_seed_a):
        fa, fb = frames_by_seed_a[seed], frames_by_seed_b[seed]
        for dropped in FOLDS:
            pooled_a = pd.concat([fr for f, fr in zip(FOLDS, fa) if f != dropped],
                                 ignore_index=True)
            pooled_b = pd.concat([fr for f, fr in zip(FOLDS, fb) if f != dropped],
                                 ignore_index=True)
            fpr_a = metric_block(pooled_a["label_int"].to_numpy(),
                                 pooled_a["logit"].to_numpy())["fpr_at_90_recall"]
            fpr_b = metric_block(pooled_b["label_int"].to_numpy(),
                                 pooled_b["logit"].to_numpy())["fpr_at_90_recall"]
            delta = fpr_a - fpr_b
            deltas.append(delta)
            per_seed_drop[f"s{seed}_drop_f{dropped}"] = {
                "fpr_a": fpr_a, "fpr_b": fpr_b, "delta": delta}
    median_delta = float(np.median(deltas))
    same_sign = sum(1 for d in deltas if (d > 0) == (median_delta > 0))
    ambiguous = median_delta > 0 and same_sign < 4 and len(frames_by_seed_a) == 1
    accept = median_delta > 0 and same_sign >= 4
    return {"seeds_used": sorted(frames_by_seed_a), "n_replicates": len(deltas),
           "per_seed_drop": per_seed_drop, "deltas": deltas,
           "median_delta": median_delta, "n_same_sign": same_sign,
           "n_total": len(deltas), "accept": accept, "ambiguous": ambiguous}


def analyse() -> dict:
    import pandas as pd
    for p in (REPO_ROOT, SCRIPTS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    from src.evaluate import metric_block
    RA = _load("_reanalysis", "17_reanalysis.py")
    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))

    variants = ("raw", "ema", "swa")
    seeds = available_seeds()
    log(f"analysing seeds {seeds}")
    # frames_all[variant][seed] = [fold0_df, fold1_df, ...] in FOLDS order
    frames_all = {v: {} for v in variants}
    bit_identity = {}
    for seed in seeds:
        for v in variants:
            frames_all[v][seed] = []
        for f in FOLDS:
            unit = f"r0_f{f}_s{seed}"
            base = os.path.join(REPO_ROOT, OUT_DIR, unit)
            raw_p = os.path.join(base, f"val_{unit}.parquet")
            for v in variants:
                p = raw_p if v == "raw" else raw_p.replace(".parquet", f"_{v}.parquet")
                frames_all[v][seed].append(pd.read_parquet(p).assign(_fold=f))
            ref_p = os.path.join(REPO_ROOT, BASELINE_DIR, unit, f"val_{unit}.parquet")
            if os.path.exists(ref_p):
                a = pd.read_parquet(raw_p).sort_values("filepath")["logit"].to_numpy()
                b = pd.read_parquet(ref_p).sort_values("filepath")["logit"].to_numpy()
                bit_identity[unit] = bool(a.shape == b.shape and np.array_equal(a, b))
            else:
                bit_identity[unit] = None

    # seed-0-only pooled/per-fold figures (conditions 2 and 4 stay at seed 0
    # as originally written -- Amendment 1 point D)
    out = {"seeds_used": seeds,
          "bit_identity_vs_pinned085": bit_identity,
          "bit_identity_all": all(v for v in bit_identity.values() if v is not None),
          "pooled": {}, "per_fold": {}}
    for v in variants:
        frames_s0 = frames_all[v][0]
        pooled = pd.concat(frames_s0, ignore_index=True)
        out["pooled"][v] = metric_block(pooled["label_int"].to_numpy(),
                                        pooled["logit"].to_numpy())
        # metric_block() does not include centre_auc_negatives -- that is a
        # separate confound diagnostic (RA.diagnostics_for), needed here
        # because Amendment 1's condition 4 (VETO) reads it.
        diag = RA.diagnostics_for(pooled, manifest)
        out["pooled"][v]["centre_auc_negatives"] = diag["centre_auc_negatives"]
        per_fold = {f: metric_block(fr["label_int"].to_numpy(), fr["logit"].to_numpy())
                   for f, fr in zip(FOLDS, frames_s0)}
        fprs = [per_fold[f]["fpr_at_90_recall"] for f in FOLDS]
        out["per_fold"][v] = {
            "fpr_at_90_recall": {str(f): per_fold[f]["fpr_at_90_recall"] for f in FOLDS},
            "fpr_mean": float(np.mean(fprs)), "fpr_max": float(np.max(fprs)),
            "fpr_sd": float(np.std(fprs, ddof=1)),
        }

    # Amendment 1, condition 1: paired LOO-4-over-folds, EMA primary vs SWA secondary
    out["condition1_ema"] = loo4_over_folds_paired(
        frames_all["raw"], frames_all["ema"], metric_block)
    out["condition1_swa"] = loo4_over_folds_paired(
        frames_all["raw"], frames_all["swa"], metric_block)

    out["pinned_a4_single_seed"] = pinned_a4_single_seed_baseline(manifest)
    return out


def render(r: dict) -> None:
    L = []
    A = L.append
    A("# SWA/EMA arm -- single-checkpoint tail variance at zero inference cost")
    A("")
    A(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. "
     f"`configs/a4_pinned085_swa.yaml` (pinned config + passive EMA "
     f"decay=0.999/step + SWA over the last 25% of epochs), 5 folds x "
     f"seed(s) {r['seeds_used']}, `{OUT_DIR}`. EMA/SWA are PASSIVE: raw "
     f"parquets are asserted bit-identical to `{BASELINE_DIR}` below.")
    A("")
    bi = r["bit_identity_vs_pinned085"]
    A(f"**Bit-identity (raw vs runs/a4_pinned085): "
     f"{'PASS -- all folds identical' if r['bit_identity_all'] else 'FAIL -- see per-fold'}**"
     f" ({', '.join(f'{k}:{v}' for k, v in bi.items())})")
    if not r["bit_identity_all"]:
        A("")
        A("**FINDING: the passivity claim did not hold -- EMA/SWA tracking "
         "perturbed the training trajectory. Do not trust the comparison "
         "below until this is explained.**")
    A("")
    A("## Pooled OOF (5 folds, seed 0)")
    A("")
    A("| variant | FPR@90R | ROC-AUC | pAUC15 (McClish) |")
    A("|---|---|---|---|")
    for v in ("raw", "ema", "swa"):
        m = r["pooled"][v]
        A(f"| {v} | {m['fpr_at_90_recall']:.4f} | "
         f"{m['roc_auc']:.4f} | {m['pauc_15_std']:.4f} |")
    A("")
    A("## Single-checkpoint tail (per-fold FPR@90R) -- the figure that matters")
    A("")
    A("A submission that ships ONE checkpoint cares about the WORST single "
     "checkpoint it might have shipped, not the mean: `max` and `sd` are the "
     "tail-variance columns.")
    A("")
    A("| variant | per-fold FPR@90R (f0..f4) | mean | max | sd |")
    A("|---|---|---|---|---|")
    for v in ("raw", "ema", "swa"):
        pf = r["per_fold"][v]
        vals = " / ".join(f"{pf['fpr_at_90_recall'][str(f)]:.4f}" for f in FOLDS)
        A(f"| {v} | {vals} | {pf['fpr_mean']:.4f} | {pf['fpr_max']:.4f} | "
         f"{pf['fpr_sd']:.4f} |")
    A("")

    pa4 = r["pinned_a4_single_seed"]["fpr_at_90_recall"]
    pa4_auc = r["pinned_a4_single_seed"]["centre_auc_negatives"]
    A("## Pre-registration application (reports/swa_ema_n1_pre_registration.md, Amendment 1)")
    A("")
    A(f"Seeds used: {r['seeds_used']}. EMA is PRIMARY (governs on any "
     f"disagreement with SWA, per amendment point B); SWA reported as "
     f"SECONDARY, not gating; raw final weights are the CONTROL.")
    A("")
    c1 = r["condition1_ema"]
    c1s = r["condition1_swa"]
    A(f"**1. Paired LOO-4-over-folds (Amendment 1A) -- raw minus EMA, "
     f"{c1['n_total']} pseudo-replicate(s) across seed(s) {c1['seeds_used']} "
     f"(NOT independent -- no p-value):** median delta "
     f"{c1['median_delta']:+.4f}, {c1['n_same_sign']}/{c1['n_total']} share "
     f"that sign. **{'ACCEPT' if c1['accept'] else ('AMBIGUOUS' if c1['ambiguous'] else 'REJECT')}** "
     f"(accept iff median > 0 and >= 4/5 agree per seed).")
    A(f"  - SWA (secondary, informational): median delta "
     f"{c1s['median_delta']:+.4f}, {c1s['n_same_sign']}/{c1s['n_total']} "
     f"share that sign -- "
     f"{'ACCEPT' if c1s['accept'] else ('AMBIGUOUS' if c1s['ambiguous'] else 'REJECT')}"
     f"{' (disagrees with EMA -- EMA governs per point B)' if c1s['accept'] != c1['accept'] else ''}.")
    A("")
    bar2 = pa4["iqr"]
    A(f"**2. Per-fold tail must not worsen beyond that bar.** Bar is "
     f"PARTIAL here (pinned-A4's own across-seed IQR only, "
     f"{bar2:.4f} -- the SWA-arm side of the pre-registered bar is "
     f"unavailable per condition 1): ")
    for v in ("raw", "ema", "swa"):
        d = r["per_fold"][v]["fpr_max"] - pa4["median"]
        A(f"  - {v}: tail (max per-fold) {r['per_fold'][v]['fpr_max']:.4f}, "
         f"delta vs pinned-A4 median {d:+.4f}, "
         f"{'beyond partial bar' if abs(d) > bar2 else 'within partial bar'}")
    A("")
    A("**3. LOCO at n=1, both directions, before any SWA weights ship.** "
     "Not available tonight, as pre-registered -- deployment stays "
     "conditional on it.")
    A("")
    A(f"**4. VETO -- centre-AUC distance from 0.5 moving further beyond its "
     f"own bar.** pinned-A4 median centre-AUC(logit->centre, neg) "
     f"{pa4_auc['median']:.4f} (distance from null "
     f"{abs(pa4_auc['median'] - 0.5):.4f}), IQR {pa4_auc['iqr']:.4f} "
     f"(PARTIAL bar, same caveat as condition 2):")
    for v in ("raw", "ema", "swa"):
        av = r["pooled"][v]["centre_auc_negatives"]
        dist = abs(av - 0.5)
        ddist = dist - abs(pa4_auc["median"] - 0.5)
        veto = ddist > pa4_auc["iqr"]
        A(f"  - {v}: centre-AUC {av:.4f} (distance {dist:.4f}, "
         f"delta-in-distance {ddist:+.4f}) -- "
         f"{'VETO TRIGGERED (partial bar)' if veto else 'not triggered'}")
    A("")
    c1_status = "ACCEPT" if c1["accept"] else ("AMBIGUOUS" if c1["ambiguous"] else "REJECT")
    A(f"**Net: condition 1 (EMA, governing) is {c1_status}; condition 3 "
     f"(LOCO at n=1) is not available tonight and remains the shipping "
     f"gate regardless of 1/2/4 -- no SWA/EMA weights enter a submission "
     f"container before it closes.**")
    A("")

    with open(os.path.join(REPORT_DIR, "swa_arm.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(REPORT_DIR, "swa_arm.json"), "w") as fh:
        json.dump(r, fh, indent=2, default=float)
    log("written: reports/swa_arm.md")


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    touch(PID_FILE, f"{os.getpid()}\n")
    log("SWA/EMA chain starting.")
    try:
        if not os.path.exists(SENTINEL_DONE):
            rc = run_arm(seeds="0")
            log(f"run_cv.py (seed 0) exited {rc}")
            if rc != 0:
                log("not a clean 5/5 -- re-run this script to resume.")
                return 3
            touch(SENTINEL_DONE)

        r = analyse()
        log(f"seed-0 condition1(EMA): median={r['condition1_ema']['median_delta']:+.4f} "
            f"same_sign={r['condition1_ema']['n_same_sign']}/{r['condition1_ema']['n_total']} "
            f"ambiguous={r['condition1_ema']['ambiguous']}")

        # Amendment 1A: ambiguous (median>0, <4/5 agree) at seed-0-only
        # triggers the PRE-AUTHORISED bounded follow-up -- seeds 1-2 only,
        # not the full 4-seed set. Not re-triggered once done (sentinel).
        if r["condition1_ema"]["ambiguous"] and not os.path.exists(SENTINEL_EXT_DONE):
            log("condition 1 AMBIGUOUS at seed 0 -- launching pre-authorised "
               "seeds 1-2 follow-up (10 units, ~4.2h, per Amendment 1A)")
            rc = run_arm(seeds="1,2")
            log(f"run_cv.py (seeds 1,2) exited {rc}")
            if rc != 0:
                log("seeds-1-2 follow-up not a clean 10/10 -- re-run this "
                   "script to resume; report reflects seed-0-only until then.")
                render(r)
                return 3
            touch(SENTINEL_EXT_DONE)
            r = analyse()
            log(f"post-followup condition1(EMA): "
                f"median={r['condition1_ema']['median_delta']:+.4f} "
                f"same_sign={r['condition1_ema']['n_same_sign']}/{r['condition1_ema']['n_total']}")

        render(r)
        log(f"pooled raw={r['pooled']['raw']['fpr_at_90_recall']:.4f} "
            f"ema={r['pooled']['ema']['fpr_at_90_recall']:.4f} "
            f"swa={r['pooled']['swa']['fpr_at_90_recall']:.4f}")
        # chain into the pAUC smoke (its launcher self-guards on our sentinel)
        subprocess.run(["bash", os.path.join(SCRIPTS_DIR, "run_pauc_chain.sh")],
                       cwd=REPO_ROOT)
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
