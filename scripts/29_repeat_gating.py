"""3-repeat gating report. NO TRAINING, NO GPU -- recomputed from parquets.

Reuses tested plumbing rather than re-deriving it: scripts/26_magnitude_sweep.py's
analyse_pooled (now repeat-parametrised) for the per-repeat k5 ensemble and its
within-repeat LOO-4 spread, scripts/17_reanalysis.py's build_wide_fold /
ensemble_pool_oof for the raw k5-ensembled pools this script concatenates
across repeats, and src.evaluate's per_centre_fpr_at_recall / spread for the
four DG-diagnostic metrics themselves.

FOUR METRICS, THREE VIEWS, PER ARM (A0, A4, A2, A1 -- priority order):

    fpr_asymmetry          centre asymmetry (c1 - c2 FPR@90R)
    fpr_center_1           FPR@90R | center_1
    fpr_center_2           FPR@90R | center_2
    centre_auc_negatives   AUC(logit -> centre | neg)

  * PER-REPEAT: each available repeat's own 5-seed k5 ensemble value, with its
    own within-repeat LOO-4 IQR (the existing single-repeat noise estimate).
  * POOLED-ACROSS-REPEATS: every available repeat's k5-ensembled OOF pool
    concatenated (each image contributes one row per repeat, since a repeat is
    a different fold re-split of the same image set, not new images) and the
    four metrics recomputed on that larger pool -- the primary point estimate
    once more than one repeat exists.
  * ACROSS-REPEAT SPREAD (once 2+ repeats exist): median + IQR of the k5
    value across independent repeats, one point per repeat. Independent
    repeats do not share the 4/5-overlap problem the within-repeat LOO-4
    family has, so this is the tighter, more honest noise-floor estimate --
    used alongside LOO-4 wherever both exist, never in place of it.

A0's repeat 0 lives in runs/noise_floor_a (pre-existing noise-floor run), not
runs/sweep_a0 -- repeats 1-2 land in runs/sweep_a0. A1/A2/A4 keep all repeats
in their one existing out_dir. See ARM_SOURCES below.

RESTARTS. Call --restart to append one timestamped line (arm, repeat, which
units were found mid-flight) to logs/repeat_gating_restarts.log before any
of that repeat's GPU time is spent on this driver invocation. The full log is
rendered into the report every time this script regenerates it, so restart
history survives across the many regenerations an unattended run produces.

    python scripts/29_repeat_gating.py                  # regenerate from disk
    python scripts/29_repeat_gating.py --restart a0_repeats 1 r1_f2_s3 r1_f4_s0
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.evaluate import (  # noqa: E402
    fmt_spread, metric_block, per_centre_fpr_at_recall,
    prior_equalised_fpr_at_recall, spread,
)
from src.io import fsync_dir, write_text_durable  # noqa: E402
from run_cv import Unit, expected_split, run_state  # noqa: E402


def _load(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MS = _load("_magnitude_sweep", "26_magnitude_sweep.py")   # analyse_pooled (repeat-aware)
RA = _load("_reanalysis", "17_reanalysis.py")              # build_wide_fold, ensemble_pool_oof, diagnostics_for

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
REPEATS = (0, 1, 2)
FUSION = "mean_logit"

FIELDS = ("fpr_at_90_recall", "fpr_prior_equalised", "fpr_asymmetry",
          "fpr_center_1", "fpr_center_2", "centre_auc_negatives")
LABELS = {
    "fpr_at_90_recall": "FPR@90R (pooled)",
    "fpr_prior_equalised": "FPR@90R, prior-equalised",
    "fpr_asymmetry": "centre asymmetry (c1 - c2 FPR@90R)",
    # No raw "|" inside the label: currently only used as a bold heading here
    # (safe), but the identical string inside a markdown TABLE cell broke
    # reports/gastronet.md and reports/a4.md by splitting the row on the pipe.
    # Kept consistent so this never becomes a landmine if this dict is ever
    # reused in a table.
    "fpr_center_1": "FPR@90R (center_1)",
    "fpr_center_2": "FPR@90R (center_2)",
    "centre_auc_negatives": "AUC(logit -> centre, neg)",
}
# fpr_at_90_recall and fpr_prior_equalised are Rule 2's veto conditions
# (reports/a4_pre_registration.md): the DG gate cannot be formally re-applied
# without them, which is why they are added here rather than left implicit in
# analyse_pooled's k5/loo4 dicts.
RULE2_VETO_FIELDS = ("fpr_at_90_recall", "fpr_prior_equalised")

# priority order from the 48h brief: A0 -> A4 -> A2 -> A1. Repeat 0 for every
# arm already exists on disk; only A0 keeps it in a different directory.
ARM_SOURCES: Dict[str, Dict[str, Any]] = {
    "A0": {"config": "configs/sweep_a0.yaml",
          "dirs": {0: "runs/noise_floor_a", 1: "runs/sweep_a0", 2: "runs/sweep_a0"}},
    "A4": {"config": "configs/sweep_a4.yaml",
          "dirs": {0: "runs/sweep_a4", 1: "runs/sweep_a4", 2: "runs/sweep_a4"}},
    "A2": {"config": "configs/sweep_a2.yaml",
          "dirs": {0: "runs/sweep_a2", 1: "runs/sweep_a2", 2: "runs/sweep_a2"}},
    "A1": {"config": "configs/sweep_a1.yaml",
          "dirs": {0: "runs/sweep_a1", 1: "runs/sweep_a1", 2: "runs/sweep_a1"}},
}
PRIORITY = ("A0", "A4", "A2", "A1")

REPORT_MD = os.path.join(REPO_ROOT, "reports", "repeat_gating.md")
REPORT_JSON = os.path.join(REPO_ROOT, "reports", "repeat_gating.json")
RESTART_LOG = os.path.join(REPO_ROOT, "logs", "repeat_gating_restarts.log")


# ---------------------------------------------------------------------------
# Readiness -- reuse run_state's own definition of "done", not a re-derived one
# ---------------------------------------------------------------------------
def repeat_ready(config_path: str, out_dir: str, repeat: int) -> bool:
    base_cfg = Config.from_yaml(os.path.join(REPO_ROOT, config_path))
    abs_out = os.path.join(REPO_ROOT, out_dir)
    for s in SEEDS:
        for f in FOLDS:
            unit = Unit(seed=s, repeat=repeat, fold=f)
            cfg = unit.apply(dataclasses.replace(base_cfg, out_dir=abs_out))
            run_dir = os.path.join(abs_out, unit.name)
            expected = expected_split(unit, cfg)
            status, _ = run_state(run_dir, unit, cfg, expected)
            if status != "done":
                return False
    return True


# ---------------------------------------------------------------------------
# Pooling across repeats
# ---------------------------------------------------------------------------
def k5_pool_for_repeat(out_dir: str, manifest: pd.DataFrame, repeat: int) -> pd.DataFrame:
    wide = {f: RA.build_wide_fold(os.path.join(REPO_ROOT, out_dir), f, manifest,
                                  SEEDS, repeat=repeat) for f in FOLDS}
    return RA.ensemble_pool_oof(wide, list(SEEDS), FUSION, manifest, repeat=repeat)


def four_metrics(pool: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, float]:
    """Despite the name, six fields now: the original four DG diagnostics plus
    pooled FPR@90R and prior-equalised FPR@90R, Rule 2's veto conditions."""
    y, s = pool["label_int"].to_numpy(), pool["logit"].to_numpy()
    m = metric_block(y, s)
    pe = prior_equalised_fpr_at_recall(pool)
    pc = per_centre_fpr_at_recall(pool)
    d = RA.diagnostics_for(pool, manifest)
    return {
        "fpr_at_90_recall": m["fpr_at_90_recall"],
        "fpr_prior_equalised": pe["fpr_at_90_recall_prior_equalised"],
        "fpr_asymmetry": pc["asymmetry_max_minus_min"],
        "fpr_center_1": pc["by_centre"].get("center_1", {}).get("fpr", float("nan")),
        "fpr_center_2": pc["by_centre"].get("center_2", {}).get("fpr", float("nan")),
        "centre_auc_negatives": d["centre_auc_negatives"],
    }


def analyse_arm(key: str, manifest: pd.DataFrame) -> Dict[str, Any]:
    src = ARM_SOURCES[key]
    available = [r for r in REPEATS if repeat_ready(src["config"], src["dirs"][r], r)]
    per_repeat: Dict[int, Dict[str, Any]] = {}
    pools: Dict[int, pd.DataFrame] = {}
    for r in available:
        ap = MS.analyse_pooled(src["dirs"][r], manifest, seeds=SEEDS, repeat=r)
        per_repeat[r] = {
            "k5": {f: ap["k5"][f] for f in FIELDS},
            "loo4": {f: ap["loo4"][f] for f in FIELDS},
        }
        pools[r] = k5_pool_for_repeat(src["dirs"][r], manifest, r)

    pooled_across: Optional[Dict[str, float]] = None
    across_repeat_spread: Optional[Dict[str, Dict[str, float]]] = None
    if available:
        concat = pd.concat([pools[r] for r in available], ignore_index=True)
        pooled_across = four_metrics(concat, manifest)
    if len(available) >= 2:
        across_repeat_spread = {
            f: spread([per_repeat[r]["k5"][f] for r in available]) for f in FIELDS
        }
    return {
        "available_repeats": available,
        "per_repeat": per_repeat,
        "pooled_across_repeats": pooled_across,
        "across_repeat_spread": across_repeat_spread,
    }


# ---------------------------------------------------------------------------
# Rule 2 (DG gate), re-applied at 3 repeats -- reports/a4_pre_registration.md
# ---------------------------------------------------------------------------
# Rule 2 has FOUR conditions:
#   1. both LOCO directions improve
#   2. at least one LOCO direction exceeds its conservative bar
#   3. centre asymmetry reduces beyond its conservative bar
#   4. veto: neither fpr_at_90_recall nor fpr_prior_equalised regresses beyond
#      its own conservative bar (in-distribution pooled metrics veto only)
#
# Conditions 1-2 are LOCO-based. LOCO was run at repeat 0 only -- there is no
# 3-repeat LOCO evidence, so those two conditions CANNOT be re-tested here.
# Conditions 3-4 are pooled-OOF-based and CAN be re-tested, now that repeats
# 1-2 exist. This function re-tests exactly those two and returns their
# result; it deliberately does not compute or imply an overall ACCEPTED/
# REJECTED verdict, because that requires all four conditions from the same
# evidence base and only half of them have 3-repeat evidence.
def conservative_bar_across_repeat(a_result: Dict[str, Any], b_result: Dict[str, Any],
                                   field: str) -> Optional[float]:
    """max of the two arms' across-repeat spread IQR -- the conservative bar
    at 3 repeats, replacing the within-repeat LOO-4 IQR used at 1 repeat.
    Independent repeats do not share the 4/5-overlap problem LOO-4 has, so
    this is the tighter, more honest bar once it exists (needs 2+ repeats for
    BOTH arms)."""
    sa, sb = a_result["across_repeat_spread"], b_result["across_repeat_spread"]
    if sa is None or sb is None:
        return None
    return max(sa[field]["iqr"], sb[field]["iqr"])


def rule2_at_3_repeats(a0_result: Dict[str, Any],
                       a4_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if a0_result["pooled_across_repeats"] is None or a4_result["pooled_across_repeats"] is None:
        return None  # fewer than 1 complete repeat for one of the arms
    a0p, a4p = a0_result["pooled_across_repeats"], a4_result["pooled_across_repeats"]

    asym_bar = conservative_bar_across_repeat(a0_result, a4_result, "fpr_asymmetry")
    ca, ta = a0p["fpr_asymmetry"], a4p["fpr_asymmetry"]
    a_delta = abs(ca) - abs(ta)  # >0 == A4's asymmetry magnitude is smaller
    asymmetry_reduced = bool(asym_bar is not None and a_delta > asym_bar)

    veto = False
    veto_detail: Dict[str, Any] = {}
    for f in RULE2_VETO_FIELDS:
        bar = conservative_bar_across_repeat(a0_result, a4_result, f)
        c, t = a0p[f], a4p[f]
        regression = t - c  # >0 == A4 got worse than A0
        vetoed = bool(bar is not None and regression > bar)
        veto = veto or vetoed
        veto_detail[f] = {"a0": c, "a4": t, "regression": regression,
                          "conservative_bar": bar, "vetoes": vetoed}

    return {
        "n_repeats_a0": len(a0_result["available_repeats"]),
        "n_repeats_a4": len(a4_result["available_repeats"]),
        "bars_available": asym_bar is not None,
        "asymmetry": {"a0": ca, "a4": ta, "delta_magnitude": a_delta,
                     "conservative_bar": asym_bar,
                     "reduced_beyond_bar": asymmetry_reduced},
        "veto": veto, "veto_detail": veto_detail,
        "conditions_3_4_pass": bool(asymmetry_reduced and not veto),
    }


# ---------------------------------------------------------------------------
# Restarts
# ---------------------------------------------------------------------------
def record_restart(arm_key: str, repeat: int, redone_units: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(RESTART_LOG), exist_ok=True)
    line = (f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  "
            f"arm={arm_key} repeat={repeat}  "
            f"redone=[{', '.join(redone_units)}]")
    existed = os.path.exists(RESTART_LOG)
    with open(RESTART_LOG, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    if not existed:
        fsync_dir(os.path.dirname(RESTART_LOG))


def read_restarts() -> List[str]:
    if not os.path.exists(RESTART_LOG):
        return []
    with open(RESTART_LOG) as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt(x: Optional[float], p: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def render_rule2_section(rule2: Optional[Dict[str, Any]],
                         loco_ref: Optional[Dict[str, Any]]) -> List[str]:
    L: List[str] = []
    A = L.append
    A("## Rule 2 (DG gate) re-applied at 3 repeats: A4 vs A0")
    A("")
    A("Rule 2 has four conditions (see `reports/a4_pre_registration.md`): (1) "
     "both LOCO directions improve, (2) at least one LOCO direction exceeds "
     "its conservative bar, (3) centre asymmetry reduces beyond its "
     "conservative bar, (4) veto -- neither pooled FPR@90R nor prior-"
     "equalised FPR@90R regresses beyond its own conservative bar.")
    A("")
    A("**Conditions 1 and 2 are NOT re-tested here.** LOCO has only ever been "
     "run at repeat 0 -- there is no 3-repeat LOCO evidence for either arm, "
     "so those two conditions cannot be re-evaluated at 3 repeats. Reusing "
     "the repeat-0 LOCO verdict as though it were part of a 3-repeat gate "
     "would silently mix evidence bases; this report does not do that. For "
     "reference only, unchanged from `reports/a4.md`, repeat 0:")
    A("")
    if loco_ref:
        for centre, d in loco_ref.items():
            A(f"- LOCO `holdout_center_{centre}`: control {d['control_k5']:.4f} "
             f"-> treatment {d['treatment_k5']:.4f} ({d['delta']:+.4f}), "
             f"{'exceeds' if d['exceeds_bar'] else 'within'} its repeat-0 "
             f"LOO-4 bar {d['conservative_bar']:.4f}")
    else:
        A("- (reports/a4.json not found -- repeat-0 LOCO reference unavailable)")
    A("")

    if rule2 is None:
        A("**Conditions 3 and 4 not yet re-testable**: fewer than 2 complete "
         "repeats for A0 and/or A4.")
        A("")
        return L

    A(f"**Conditions 3 and 4, re-tested at 3 repeats** (A0: "
     f"{rule2['n_repeats_a0']} repeats, A4: {rule2['n_repeats_a4']} repeats). "
     f"Point estimate is each arm's pooled-across-repeats value; the "
     f"conservative bar is the larger of the two arms' ACROSS-REPEAT spread "
     f"IQR (not the within-repeat LOO-4 IQR).")
    A("")
    if not rule2["bars_available"]:
        A("Across-repeat spread requires 2+ complete repeats for BOTH arms; "
         "not yet available for at least one.")
        A("")
    else:
        asym = rule2["asymmetry"]
        A(f"- **Condition 3 (asymmetry reduced)**: |{asym['a0']:.4f}| -> "
         f"|{asym['a4']:.4f}|, magnitude reduced {asym['delta_magnitude']:+.4f} "
         f"against a bar of {asym['conservative_bar']:.4f} -- "
         f"{'**yes, beyond bar**' if asym['reduced_beyond_bar'] else '**no, not beyond bar**'}.")
        for f, d in rule2["veto_detail"].items():
            A(f"- **Condition 4 veto check [{LABELS[f]}]**: A0 {d['a0']:.4f} "
             f"-> A4 {d['a4']:.4f}, regression {d['regression']:+.4f} against "
             f"a bar of {fmt(d['conservative_bar'])} -- "
             f"{'**VETOES**' if d['vetoes'] else 'ok, no veto'}.")
        A("")
        A(f"**Conditions 3+4 at 3 repeats: "
         f"{'PASS' if rule2['conditions_3_4_pass'] else 'FAIL'}**. This is "
         f"NOT a full Rule 2 verdict -- conditions 1-2 remain repeat-0-only "
         f"and are not part of this result.")
        A("")
    return L


def render_markdown(results: Dict[str, Dict[str, Any]], restarts: List[str],
                    rule2: Optional[Dict[str, Any]] = None,
                    loco_ref: Optional[Dict[str, Any]] = None) -> str:
    L: List[str] = []
    A = L.append
    A("# 3-repeat gating -- A0 / A4 / A2 / A1")
    A("")
    A(f"Generated by `scripts/29_repeat_gating.py` on "
     f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. No "
     f"training, no GPU: recomputed from prediction parquets on disk. "
     f"Regenerated after every repeat boundary, so this file is legible at "
     f"any point in an interrupted run.")
    A("")
    A("Repeat 0 is reused for every arm (already on disk before this run "
     "started); repeats 1 and 2 are the new units this driver produces. "
     "A0's repeat 0 lives in `runs/noise_floor_a`; its repeats 1-2 land in "
     "`runs/sweep_a0`. A1/A2/A4 keep all three repeats in their existing "
     "out_dir.")
    A("")

    L.extend(render_rule2_section(rule2, loco_ref))

    for key in PRIORITY:
        r = results[key]
        avail = r["available_repeats"]
        A(f"## {key}")
        A("")
        if not avail:
            A("No complete repeat yet for this arm.")
            A("")
            continue
        A(f"Repeats complete: {', '.join(str(x) for x in avail)}")
        A("")
        for f in FIELDS:
            A(f"**{LABELS[f]}**")
            A("")
            A("| view | value | within-repeat LOO-4 IQR |")
            A("|---|---|---|")
            for rep in avail:
                pr = r["per_repeat"][rep]
                A(f"| repeat {rep} | {fmt(pr['k5'][f])} | "
                 f"{fmt(pr['loo4'][f]['iqr'])} |")
            pooled = r["pooled_across_repeats"][f] if r["pooled_across_repeats"] else None
            A(f"| **pooled across repeats** (n_repeats={len(avail)}) | "
             f"**{fmt(pooled)}** | -- |")
            A("")
            if r["across_repeat_spread"] is not None:
                s = r["across_repeat_spread"][f]
                A(f"Across-repeat spread (primary noise floor once 2+ repeats "
                 f"exist): {fmt_spread(s)}")
            else:
                A("Across-repeat spread: not available yet (needs 2+ complete "
                 "repeats).")
            A("")

    A("## Restarts")
    A("")
    if not restarts:
        A("None detected.")
    else:
        A("Each line: a driver invocation found unit(s) mid-flight (checkpoint "
         "present, canonical parquet not yet valid) before spending any GPU "
         "time on them -- i.e. a crash or a planned stop/restart, with the "
         "exact units that were redone.")
        A("")
        for ln in restarts:
            A(f"- `{ln}`")
    A("")
    return "\n".join(L)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_a4_loco_reference() -> Optional[Dict[str, Any]]:
    """Repeat-0 LOCO condition 1/2 detail from the already-written
    reports/a4.json, shown for context only -- never re-gated here."""
    path = os.path.join(REPO_ROOT, "reports", "a4.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)["rule2_dg_gate"]["loco"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def update_report(manifest_path: str = "manifests/rare25_folds_v2.csv") -> None:
    manifest = pd.read_csv(os.path.join(REPO_ROOT, manifest_path))
    results = {key: analyse_arm(key, manifest) for key in PRIORITY}
    restarts = read_restarts()
    rule2 = rule2_at_3_repeats(results["A0"], results["A4"])
    loco_ref = load_a4_loco_reference()
    md = render_markdown(results, restarts, rule2, loco_ref)
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    # Durable, and for a specific reason: this file is regenerated on every
    # repeat boundary of an unattended run and is the artefact that is read on
    # return. A half-written report after a host reset would be read as the
    # result, not as a crash -- the failure would be silent and believed.
    write_text_durable(REPORT_MD, md)
    write_text_durable(REPORT_JSON,
                       json.dumps({"results": results, "restarts": restarts,
                                  "rule2_at_3_repeats": rule2},
                                  indent=2, default=_json_default))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restart", nargs="+", metavar=("ARM", "REPEAT"),
                    help="ARM REPEAT [UNIT ...] -- record a detected restart, "
                         "then regenerate the report")
    ap.add_argument("--manifest", default="manifests/rare25_folds_v2.csv")
    args = ap.parse_args(argv)

    if args.restart:
        if len(args.restart) < 2:
            ap.error("--restart needs at least ARM and REPEAT")
        arm_key, repeat_s, *units = args.restart
        record_restart(arm_key, int(repeat_s), units)

    update_report(args.manifest)
    print(f"written: {os.path.relpath(REPORT_MD, REPO_ROOT)}")
    print(f"written: {os.path.relpath(REPORT_JSON, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
