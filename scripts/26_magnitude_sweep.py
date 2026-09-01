"""Magnitude sweep report. NO TRAINING, NO GPU.

PRIMARY decision unit: pooled out-of-fold, repeat 0, five folds, seeds 0-4.
Both LOCO directions are SECONDARY -- a stress test, reported separately and
never averaged, because their two directions differ in training size, positive
count, test size and prevalence all at once.

ACCEPTANCE RULE, fixed before results were seen:

    accept on pooled-OOF ensembled FPR@90R improvement exceeding the
    leave-one-out-4 IQR, PROVIDED neither LOCO direction regresses by more than
    its own LOO-4 IQR.

The pooled gain and both LOCO deltas are printed for EVERY arm, including
rejected ones -- the point of paying for the LOCO runs of a failing arm is to
learn why it failed, not merely that it did.

    python scripts/26_magnitude_sweep.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import MAGNITUDE_IDENTITY, AugConfig, Config  # noqa: E402
from src.evaluate import (  # noqa: E402
    box, fmt_spread, metric_block, per_centre_fpr_at_recall,
    prior_equalised_fpr_at_recall, prior_equalised_fpr_bootstrap, spread,
)
from src.io import write_text_durable  # noqa: E402


def _load(name: str, filename: str):
    path = os.path.join(REPO_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RA = _load("_reanalysis", "17_reanalysis.py")

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
REPEAT = 0
CENTRES = (1, 2)
FUSION = "mean_logit"

# `pooled`/`loco` are run directories; None means "reuse the arm named in
# reuse_from", which is only ever set where reuse has been proven bit-identical.
ARMS: List[Dict[str, Any]] = [
    {"key": "A0", "label": "baseline",
     "config": "configs/sweep_a0.yaml",
     "pooled": "runs/noise_floor_a", "loco": "runs/noise_floor_b",
     "reuse_note": "both reused; bit-identity re-verified before the sweep"},
    {"key": "A1", "label": "centre x class sampler",
     "config": "configs/sweep_a1.yaml",
     "pooled": "runs/sweep_a1", "loco": "runs/noise_floor_b",
     "reuse_note": "LOCO reused from A0 -- the sampler provably degenerates to "
                   "the class-balanced baseline on a single-centre training set"},
    {"key": "A2", "label": "stack at 0.33x magnitude",
     "config": "configs/sweep_a2.yaml",
     "pooled": "runs/sweep_a2", "loco": "runs/sweep_a2_loco",
     "reuse_note": ""},
    {"key": "A3", "label": "stack at 1.0x magnitude",
     "config": "configs/sweep_a3.yaml",
     "pooled": "runs/sweep_a3", "loco": "runs/sweep_a3_loco",
     "reuse_note": ""},
]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def analyse_pooled(run_dir: str, manifest: pd.DataFrame,
                   seeds: Sequence[int] = SEEDS,
                   repeat: int = REPEAT) -> Dict[str, Any]:
    """Single-seed spread AND the 5-seed ensemble with its LOO-4 spread.

    Both, deliberately. The ensemble is the deployment unit and the thing the
    acceptance rule reads; the single-seed spread is the noise estimate that
    says whether the ensemble's movement means anything.

    ``repeat`` selects which of the manifest's independent fold assignments
    (fold_r0..fold_r9) this run_dir's units were trained under -- see
    scripts/29_repeat_gating.py, which is the only caller that passes
    anything other than the default 0.
    """
    wide = {f: RA.build_wide_fold(os.path.join(REPO_ROOT, run_dir), f, manifest,
                                  seeds, repeat=repeat) for f in FOLDS}

    def block(subset):
        pool = RA.ensemble_pool_oof(wide, list(subset), FUSION, manifest, repeat=repeat)
        y, s = pool["label_int"].to_numpy(), pool["logit"].to_numpy()
        m = metric_block(y, s)
        d = RA.diagnostics_for(pool, manifest)
        # PRIMARY DG METRICS -- pre-registered 2026-07-29, see
        # reports/prior_equalised_fpr_pre_registration.md. Computed for every
        # subset (single seed, k=5, each LOO-4), not just the headline point,
        # so the per-arm detail table can show their own spread too.
        pe = prior_equalised_fpr_at_recall(pool)
        pc = per_centre_fpr_at_recall(pool)
        out = {**{k: m[k] for k in ("fpr_at_90_recall", "roc_auc",
                                    "pauc_15_std", "pauc_15_raw")},
               **{k: d[k] for k in ("centre_auc_negatives",
                                    "rho_file_size_positives",
                                    "rho_file_size_negatives")},
               "fpr_prior_equalised": pe["fpr_at_90_recall_prior_equalised"],
               "fpr_asymmetry": pc["asymmetry_max_minus_min"]}
        for c, v in pc["by_centre"].items():
            out[f"fpr_{c}"] = v["fpr"]
        return out

    per_seed = [block([s]) for s in seeds]
    k5 = block(seeds)
    loo = [block([s for s in seeds if s != drop]) for drop in seeds]

    # The 1000-draw subsampling bootstrap is the variance companion to the
    # PRIOR-EQUALISED figure specifically (the reweighting has no resampling
    # distribution of its own to read an IQR off). Run once, on the k=5 pool
    # -- the primary decision figure -- not on all eleven pools: it answers
    # "how much does removing the centre/class entanglement cost in variance",
    # which is a property of the ensemble being deployed, not of any one
    # single-seed or leave-one-out subset.
    k5_pool = RA.ensemble_pool_oof(wide, list(seeds), FUSION, manifest, repeat=repeat)
    k5_bootstrap = prior_equalised_fpr_bootstrap(k5_pool, n_boot=1000, seed=0)

    fields = ("fpr_at_90_recall", "roc_auc", "pauc_15_std", "pauc_15_raw",
              "centre_auc_negatives", "rho_file_size_positives",
              "rho_file_size_negatives", "fpr_prior_equalised",
              "fpr_asymmetry", "fpr_center_1", "fpr_center_2")
    return {
        "single_seed": {f: spread([d[f] for d in per_seed]) for f in fields},
        "single_seed_values": {f: [d[f] for d in per_seed] for f in fields},
        "k5": k5,
        "k5_prior_equalised_bootstrap": k5_bootstrap,
        "loo4": {f: spread([d[f] for d in loo]) for f in fields},
        "loo4_values": {f: [d[f] for d in loo] for f in fields},
    }


def analyse_loco(run_dir: str, manifest: pd.DataFrame,
                 seeds: Sequence[int] = SEEDS) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for centre in CENTRES:
        wide = RA.build_wide_loco(os.path.join(REPO_ROOT, run_dir), centre,
                                  manifest, seeds)

        def fpr(subset):
            y, s, _ = RA.ensemble_loco(wide, list(subset), FUSION)
            return metric_block(y, s)["fpr_at_90_recall"]

        loo = [fpr([s for s in seeds if s != drop]) for drop in seeds]
        out[centre] = {"k5_fpr": fpr(seeds), "loo4_fpr": loo,
                       "loo4_spread": spread(loo)}
    return out


def epoch_seconds(run_dirs: Sequence[str]) -> Dict[str, Any]:
    vals: List[float] = []
    for run_dir in run_dirs:
        base = os.path.join(REPO_ROOT, run_dir)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name, "summary.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as fh:
                    s = json.load(fh)
                if s.get("completed_epochs", 0) >= 30:
                    vals.append(float(s["median_epoch_seconds"]))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
    return spread(vals) if vals else None


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------
def acceptance(arm: Dict[str, Any], control: Dict[str, Any]) -> Dict[str, Any]:
    """The rule, applied. Pooled gate first, LOCO veto second."""
    t, c = arm["pooled"], control["pooled"]
    gain = c["k5"]["fpr_at_90_recall"] - t["k5"]["fpr_at_90_recall"]
    bar = max(c["loo4"]["fpr_at_90_recall"]["iqr"],
              t["loo4"]["fpr_at_90_recall"]["iqr"])
    pooled_pass = bool(gain > bar)

    loco: Dict[int, Dict[str, Any]] = {}
    veto = False
    for centre in CENTRES:
        td, cd = arm["loco"][centre], control["loco"][centre]
        delta = cd["k5_fpr"] - td["k5_fpr"]          # >0 is an improvement
        own_iqr = td["loo4_spread"]["iqr"]
        regressed_too_far = bool(-delta > own_iqr)
        veto = veto or regressed_too_far
        loco[centre] = {
            "control_k5": cd["k5_fpr"], "treatment_k5": td["k5_fpr"],
            "delta": delta, "own_loo4_iqr": own_iqr,
            "regresses_beyond_iqr": regressed_too_far,
        }
    return {
        "pooled_gain": gain, "pooled_iqr_bar": bar,
        "pooled_passes": pooled_pass,
        "loco": loco, "loco_veto": veto,
        "accepted": bool(pooled_pass and not veto),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def magnitude_table() -> List[str]:
    full = dict(photometric=True, optical=True, sensor=True, compression=True)
    a1 = Config(aug=AugConfig(magnitude_scale=1.0, **full)).aug
    a33 = Config(aug=AugConfig(magnitude_scale=0.33, **full)).aug
    rows = ["| parameter | no-op identity | s = 1.00 (A3) | s = 0.33 (A2) |",
            "|---|---|---|---|"]
    for name, ident in MAGNITUDE_IDENTITY.items():
        rows.append(f"| `{name}` | {ident:g} | {getattr(a1, name):g} | "
                    f"{getattr(a33, name):g} |")
    rows.append(f"| `poisson_noise_m` | ∞ (scaled as 1/s²) | "
                f"{a1.poisson_noise_m:g} | {a33.poisson_noise_m:g} |")
    rows.append(f"| `white_balance_tint_frac` | — (shape, not magnitude) | "
                f"{a1.white_balance_tint_frac:g} | "
                f"{a33.white_balance_tint_frac:g} |")
    return rows


def fmt(x, p=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="manifests/rare25_folds_v2.csv")
    ap.add_argument("--out-md", default="reports/magnitude_sweep.md")
    ap.add_argument("--out-json", default="reports/magnitude_sweep.json")
    args = ap.parse_args(argv)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, args.manifest))

    results: Dict[str, Any] = {}
    for arm in ARMS:
        if not os.path.isdir(os.path.join(REPO_ROOT, arm["pooled"])):
            print(f"[skip] {arm['key']}: {arm['pooled']} not present yet")
            continue
        # A mid-sweep retrospective check-in (this script is also run BEFORE
        # the sweep driver's own final call) can catch an arm whose pooled
        # units are complete but whose LOCO stress test has not started yet --
        # RUN_ORDER in scripts/23_sweep.py runs every arm's pooled units before
        # any arm's LOCO units. Skip that arm rather than crash: it is a
        # legitimate, real, transient state, not a data error.
        try:
            results[arm["key"]] = {
                "arm": arm,
                "pooled": analyse_pooled(arm["pooled"], manifest),
                "loco": analyse_loco(arm["loco"], manifest),
                "epoch_seconds": epoch_seconds([arm["pooled"], arm["loco"]]),
            }
        except FileNotFoundError as exc:
            print(f"[skip] {arm['key']}: pooled present, LOCO not yet ({exc})")
            continue

    if "A0" not in results:
        print("A0 is required as the control; nothing to compare against.")
        return 1

    verdicts = {k: acceptance(v, results["A0"])
                for k, v in results.items() if k != "A0"}

    table = render_table(results, verdicts)
    pe_table = render_prior_equalised_table(results)
    print("\n" + table + "\n")
    print("\n" + pe_table + "\n")

    md = render_markdown(results, verdicts, table, pe_table)
    out_md = os.path.join(REPO_ROOT, args.out_md)
    write_text_durable(out_md, md)
    write_text_durable(os.path.join(REPO_ROOT, args.out_json),
                       json.dumps({"results": results, "verdicts": verdicts},
                                  indent=2, default=RA._JD))
    print(f"written: {args.out_md}")
    print(f"written: {args.out_json}")
    return 0


def render_prior_equalised_table(results: Dict[str, Any]) -> str:
    """The primary domain-generalisation metrics, every arm, one table.

    Kept separate from render_table's terse pooled-FPR headline (already at
    twelve columns) rather than crammed into it -- these are wide enough
    (bootstrap IQR, both centres, asymmetry) to need their own space, and
    keeping them visually distinct from the pre-registered ACCEPTANCE gate
    is deliberate: see the pre-registration note for why the two are not
    (yet) the same table.
    """
    head = (f"{'arm':4s} {'component':26s} {'pooled k=5':>11s} "
            f"{'prior-eq k=5':>13s} {'boot med (IQR)':>16s} "
            f"{'c1 FPR':>8s} {'c2 FPR':>8s} {'asymmetry':>10s}")
    lines = [head, "-" * len(head)]
    for arm in ARMS:
        k = arm["key"]
        if k not in results:
            continue
        p = results[k]["pooled"]
        bs = p["k5_prior_equalised_bootstrap"]["spread"]
        lines.append(
            f"{k:4s} {arm['label']:26s} "
            f"{p['k5']['fpr_at_90_recall']:11.4f} "
            f"{p['k5']['fpr_prior_equalised']:13.4f} "
            f"{bs['median']:8.4f} ({bs['iqr']:.4f}) "
            f"{p['k5']['fpr_center_1']:8.4f} {p['k5']['fpr_center_2']:8.4f} "
            f"{p['k5']['fpr_asymmetry']:10.4f}")
    return "\n".join(lines)


def render_table(results: Dict[str, Any], verdicts: Dict[str, Any]) -> str:
    head = (f"{'arm':4s} {'component':26s} {'pooled k=5':>11s} {'LOO4 IQR':>9s} "
            f"{'1-seed med':>11s} {'ROC-AUC':>8s} {'pAUC15':>8s} "
            f"{'ctrAUC':>7s} {'rho+':>7s} {'LOCO c1':>9s} {'LOCO c2':>9s} "
            f"{'s/epoch':>8s} {'verdict':>9s}")
    lines = [head, "-" * len(head)]
    for arm in ARMS:
        k = arm["key"]
        if k not in results:
            continue
        r = results[k]
        p, v = r["pooled"], verdicts.get(k)
        es = r["epoch_seconds"]
        lines.append(
            f"{k:4s} {arm['label']:26s} "
            f"{p['k5']['fpr_at_90_recall']:11.4f} "
            f"{p['loo4']['fpr_at_90_recall']['iqr']:9.4f} "
            f"{p['single_seed']['fpr_at_90_recall']['median']:11.4f} "
            f"{p['k5']['roc_auc']:8.4f} {p['k5']['pauc_15_std']:8.4f} "
            f"{p['k5']['centre_auc_negatives']:7.3f} "
            f"{p['k5']['rho_file_size_positives']:7.3f} "
            f"{r['loco'][1]['k5_fpr']:9.4f} {r['loco'][2]['k5_fpr']:9.4f} "
            f"{(es['median'] if es else float('nan')):8.1f} "
            f"{('ACCEPT' if v['accepted'] else 'reject') if v else 'control':>9s}")
    return "\n".join(lines)


def render_markdown(results, verdicts, table, pe_table) -> str:
    L: List[str] = []
    A = L.append
    A("# RARE26 magnitude sweep — pooled OOF primary, LOCO secondary")
    A("")
    A(f"Generated by `scripts/26_magnitude_sweep.py` on "
      f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. No "
      f"training, no GPU: recomputed from prediction parquets on disk.")
    A("")
    A("**Primary decision unit:** pooled out-of-fold, repeat 0, five folds, "
      "seeds 0-4. **Secondary:** both LOCO directions, reported separately and "
      "never averaged.")
    A("")
    A("## Headline")
    A("")
    A("```")
    A(table)
    A("```")
    A("")
    A("`ctrAUC` is AUC of the logit predicting centre among negatives (0.5 = no "
      "hospital information, baseline 0.300). `rho+` is Spearman rho of logit "
      "against `file_size_bytes` within positives (baseline +0.360).")
    A("")

    A("## 0. Primary domain-generalisation metrics (pre-registered 2026-07-29)")
    A("")
    A("Pooled FPR@90R above is the metric this sweep's ACCEPTANCE RULE was "
      "pre-registered against, and that rule is applied unchanged below. But "
      "pooled FPR@90R has a structural blind spot stated in full in "
      "`reports/prior_equalised_fpr_pre_registration.md`: it is measured on a "
      "pool where centre and class are correlated (center_1 2.7% positive, "
      "center_2 12.0%), so a shortcut of the form \"looks like center_2\" is "
      "PARTLY VALID on this pool and is rewarded by pooled FPR@90R rather than "
      "merely un-penalised. The two metrics below are pre-registered as PRIMARY "
      "from this point forward, specifically because they do not share that "
      "blind spot.")
    A("")
    A("```")
    A(pe_table)
    A("```")
    A("")
    A("- **prior-eq k=5**: FPR@90R with negatives reweighted so "
      "P(centre \\| negative) = P(centre \\| positive) -- hospital identity "
      "carries zero label information under this weighting.")
    A("- **boot med (IQR)**: 1000-draw subsampling bootstrap of the same "
      "quantity -- median and IQR. Wider than the pooled-FPR IQR at the same "
      "k is EXPECTED (effective negatives fall ~2930 → ~1160); that is the "
      "price of removing a shortcut the pooled figure could not see.")
    A("- **c1 FPR / c2 FPR**: FPR@90R within each centre's negatives at ONE "
      "shared threshold set globally from the pooled positives.")
    A("- **asymmetry**: c1 FPR minus c2 FPR (max−min) at that shared "
      "threshold -- a model with zero hospital signal has no mechanism to "
      "produce a nonzero value here.")
    A("")
    A("**This sweep's acceptance verdicts (section 3) still gate on pooled "
      "FPR@90R**, as pre-registered before A1/A2/A3 were run. Changing the gate "
      "metric retroactively, with A2/A3 already ~60% collected under the "
      "original rule, was judged a bigger and more consequential decision than "
      "adding transparent reporting -- so both figures are shown for every arm, "
      "and A4 (and any sweep after this one) should pre-register prior-equalised "
      "FPR@90R as the gate outright rather than inheriting this compromise.")
    A("")

    lines = ["ACCEPTANCE RULE -- fixed before results were seen", "",
             "Accept on pooled-OOF ensembled FPR@90R improvement exceeding the",
             "leave-one-out-4 IQR, PROVIDED neither LOCO direction regresses by",
             "more than its own LOO-4 IQR.", ""]
    for arm in ARMS:
        k = arm["key"]
        if k not in verdicts:
            continue
        v = verdicts[k]
        lines.append(f"{k}  {arm['label']}")
        lines.append(f"   pooled gain {v['pooled_gain']:+.4f} vs IQR bar "
                     f"{v['pooled_iqr_bar']:.4f}   "
                     f"{'PASS' if v['pooled_passes'] else 'fail'}")
        for centre in CENTRES:
            d = v["loco"][centre]
            lines.append(f"   LOCO c{centre} {d['delta']:+.4f} vs own IQR "
                         f"{d['own_loo4_iqr']:.4f}   "
                         f"{'VETO' if d['regresses_beyond_iqr'] else 'ok'}")
        lines.append(f"   VERDICT: {'ACCEPTED' if v['accepted'] else 'REJECTED'}")
        lines.append("")
    A(box(lines))
    A("")

    A("## 1. What the magnitude multiplier means, per parameter")
    A("")
    A("A2 and A3 differ in exactly one config value, `magnitude_scale`. Scaling "
      "is identity-anchored, `value(s) = identity + s·(shipped − identity)`, "
      "where identity is the value at which that transform does nothing. "
      "Probabilities are untouched, so both arms fire the same transforms on "
      "the same images with the same RNG draws and differ only in how hard.")
    A("")
    A("A plain multiply would have been wrong for four parameters, and wrong in "
      "the dangerous direction — it would have made A2 **stronger** than A3. "
      "`downsample_m` is a minimum scale factor whose no-op is 1.0; the two "
      "JPEG quality bounds are no-ops at 100; `poisson_noise_m` is a photon "
      "count whose no-op is infinity and which scales on noise amplitude as "
      "1/s².")
    A("")
    L.extend(magnitude_table())
    A("")
    A("Below roughly s = 0.1 several transforms meet the hardcoded floors of "
      "their internal draw ranges (sigma ≥ 2, defocus radius ≥ 0.5, sharpen "
      "alpha ≥ 0.15) and the scaling stops being linear. Both values used here "
      "are far above that.")
    A("")

    A("## 2. Per-arm detail")
    A("")
    for arm in ARMS:
        k = arm["key"]
        if k not in results:
            continue
        r = results[k]
        p = r["pooled"]
        A(f"### {k} — {arm['label']}")
        A("")
        A(f"`{arm['config']}` · pooled `{arm['pooled']}` · LOCO `{arm['loco']}`"
          + (f" · {arm['reuse_note']}" if arm["reuse_note"] else ""))
        A("")
        A("| metric | single-seed median | single-seed IQR | k=5 ensemble | "
          "k=5 LOO-4 IQR |")
        A("|---|---|---|---|---|")
        for f, label in (("fpr_at_90_recall", "**FPR@90R**"),
                         ("roc_auc", "ROC-AUC"),
                         ("pauc_15_std", "pAUC[0,0.15] (McClish)"),
                         ("pauc_15_raw", "pAUC[0,0.15] (area/0.15)"),
                         ("centre_auc_negatives", "AUC(logit→centre \\| neg)"),
                         ("rho_file_size_positives", "rho(file_size) \\| pos"),
                         ("rho_file_size_negatives", "rho(file_size) \\| neg"),
                         ("fpr_prior_equalised", "**FPR@90R, prior-equalised**"),
                         ("fpr_center_1", "FPR@90R \\| center_1"),
                         ("fpr_center_2", "FPR@90R \\| center_2"),
                         ("fpr_asymmetry", "asymmetry (c1 − c2)")):
            ss, lo = p["single_seed"][f], p["loo4"][f]
            A(f"| {label} | {ss['median']:.4f} | {ss['iqr']:.4f} | "
              f"{p['k5'][f]:.4f} | {lo['iqr']:.4f} |")
        A("")
        bs = p["k5_prior_equalised_bootstrap"]
        A(f"- prior-equalised FPR@90R, 1000-draw subsampling bootstrap: "
          f"{fmt_spread(bs['spread'])}; anchor centre `{bs['anchor_centre']}`, "
          f"subsample targets {bs['target_counts_by_centre']}")
        A(f"- five LOO-4 pooled ensembles (FPR@90R): "
          + ", ".join(f"{v:.4f}" for v in p["loo4_values"]["fpr_at_90_recall"]))
        A(f"- five single seeds (FPR@90R): "
          + ", ".join(f"{v:.4f}" for v in
                      p["single_seed_values"]["fpr_at_90_recall"]))
        for centre in CENTRES:
            d = r["loco"][centre]
            A(f"- LOCO `holdout_center_{centre}` (secondary): k=5 "
              f"{d['k5_fpr']:.4f}, {fmt_spread(d['loo4_spread'])}")
        es = r["epoch_seconds"]
        if es:
            A(f"- epoch seconds: median {es['median']:.1f}, IQR {es['iqr']:.1f}, "
              f"range [{es['min']:.1f}, {es['max']:.1f}]")
        A("")

    A("## 3. Verdicts")
    A("")
    for arm in ARMS:
        k = arm["key"]
        if k not in verdicts:
            continue
        v = verdicts[k]
        A(f"### {k} — {arm['label']}: "
          f"{'**ACCEPTED**' if v['accepted'] else '**REJECTED**'}")
        A("")
        A(f"- Pooled-OOF ensembled FPR@90R "
          f"{results['A0']['pooled']['k5']['fpr_at_90_recall']:.4f} → "
          f"{results[k]['pooled']['k5']['fpr_at_90_recall']:.4f}, a gain of "
          f"{v['pooled_gain']:+.4f} against an IQR bar of "
          f"{v['pooled_iqr_bar']:.4f} — "
          f"{'clears the bar' if v['pooled_passes'] else 'does not clear the bar'}.")
        for centre in CENTRES:
            d = v["loco"][centre]
            A(f"- LOCO `holdout_center_{centre}` {d['control_k5']:.4f} → "
              f"{d['treatment_k5']:.4f} ({d['delta']:+.4f}) against its own "
              f"LOO-4 IQR of {d['own_loo4_iqr']:.4f} — "
              f"{'**VETO**: regresses further than its own noise' if d['regresses_beyond_iqr'] else 'within tolerance'}.")
        A("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
