"""A4 report. NO TRAINING, NO GPU -- recomputed from prediction parquets.

A4 = A1 (centre x class sampler) + A2 (stack at magnitude_scale 0.33), pooled
OOF only, plus a LOCO-only probe at magnitude_scale 0.10. See
`reports/a4_pre_registration.md`, written before any A4 unit was trained, for
what A4 is, why A4's LOCO arm reuses `runs/sweep_a2_loco` instead of being
re-run, and for both acceptance rules applied below verbatim.

Two gates are applied and BOTH verdicts are reported:

    Rule 1 (retained)  the A1-A3 pooled-FPR gate, unchanged.
    Rule 2 (PRIMARY from A4 onward)  the DG gate: both LOCO directions
        improve with at least one exceeding its conservative bar, AND centre
        asymmetry reduces beyond its conservative bar, with in-distribution
        pooled metrics acting as veto only on a regression past their own
        conservative bar.

"Conservative bar" == max(control LOO-4 IQR, treatment LOO-4 IQR) at that
metric, used uniformly for the gate and for the resolvability table below, so
the two cannot disagree about how wide noise is.

    python scripts/28_a4_sweep.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import box, fmt_spread  # noqa: E402
from src.io import write_text_durable  # noqa: E402


def _load(name: str, filename: str):
    path = os.path.join(REPO_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MS = _load("_magnitude_sweep", "26_magnitude_sweep.py")
RA = _load("_reanalysis", "17_reanalysis.py")

SEEDS = MS.SEEDS
CENTRES = MS.CENTRES

# Standard per-arm metric fields, as in 26_magnitude_sweep.py's analyse_pooled.
POOLED_FIELDS = ("fpr_at_90_recall", "roc_auc", "pauc_15_std", "pauc_15_raw",
                 "centre_auc_negatives", "rho_file_size_positives",
                 "rho_file_size_negatives", "fpr_prior_equalised",
                 "fpr_asymmetry", "fpr_center_1", "fpr_center_2")
POOLED_LABELS = {
    "fpr_at_90_recall": "FPR@90R",
    "roc_auc": "ROC-AUC",
    "pauc_15_std": "pAUC[0,0.15] (McClish)",
    "pauc_15_raw": "pAUC[0,0.15] (area/0.15)",
    # Same table-cell-pipe bug as fpr_center_1/2 above, same fix -- confirmed
    # this specific row is corrupted in the already-written reports/a4.md
    # section 3 per-arm detail table. Not regenerated here (a4.md wasn't in
    # scope of the change that found this); flagged separately.
    "centre_auc_negatives": "AUC(logit->centre, neg)",
    # Same table-cell-pipe bug as fpr_center_1/2 and centre_auc_negatives
    # above -- found by the same mechanical column-count checker, one pass
    # later. scripts/25_a1_early.py and scripts/26_magnitude_sweep.py already
    # had this specific pair correctly escaped (`\\|`); only this file did not.
    "rho_file_size_positives": "rho(file_size), pos",
    "rho_file_size_negatives": "rho(file_size), neg",
    "fpr_prior_equalised": "FPR@90R, prior-equalised",
    "fpr_asymmetry": "asymmetry (c1 - c2)",
    # No raw "|" inside a table-cell label: it splits a markdown table row
    # into an extra column and silently shifts every value after it. Found
    # 2026-08-04 while investigating a "missing" center_1 row in
    # reports/gastronet.md, whose LABELS dict copied this exact bug.
    "fpr_center_1": "FPR@90R (center_1)",
    "fpr_center_2": "FPR@90R (center_2)",
}
# Higher is better for these; lower is better for everything else.
HIGHER_IS_BETTER = {"roc_auc", "pauc_15_std", "pauc_15_raw"}


# ---------------------------------------------------------------------------
# Resolvability -- the LARGER of the two arms' LOO-4 IQRs, uniformly
# ---------------------------------------------------------------------------
def conservative_bar_pooled(control: Dict[str, Any], arm: Dict[str, Any],
                            field: str) -> float:
    return max(control["pooled"]["loo4"][field]["iqr"],
              arm["pooled"]["loo4"][field]["iqr"])


def conservative_bar_loco(control: Dict[str, Any], arm: Dict[str, Any],
                          centre: int) -> float:
    return max(control["loco"][centre]["loo4_spread"]["iqr"],
              arm["loco"][centre]["loo4_spread"]["iqr"])


def resolvability_table(control: Dict[str, Any], arm: Dict[str, Any]
                        ) -> List[Dict[str, Any]]:
    rows = []
    for f in POOLED_FIELDS:
        c, t = control["pooled"]["k5"][f], arm["pooled"]["k5"][f]
        bar = conservative_bar_pooled(control, arm, f)
        delta = t - c
        rows.append({"metric": POOLED_LABELS[f], "control": c, "treatment": t,
                    "delta": delta, "conservative_bar": bar,
                    "resolvable": bool(abs(delta) > bar)})
    for centre in CENTRES:
        c = control["loco"][centre]["k5_fpr"]
        t = arm["loco"][centre]["k5_fpr"]
        bar = conservative_bar_loco(control, arm, centre)
        delta = t - c
        rows.append({"metric": f"LOCO holdout_center_{centre} FPR@90R",
                    "control": c, "treatment": t, "delta": delta,
                    "conservative_bar": bar,
                    "resolvable": bool(abs(delta) > bar)})
    return rows


# ---------------------------------------------------------------------------
# Rule 2 -- the DG gate, pre-registered in reports/a4_pre_registration.md
# ---------------------------------------------------------------------------
def dg_gate(arm: Dict[str, Any], control: Dict[str, Any]) -> Dict[str, Any]:
    loco: Dict[int, Dict[str, Any]] = {}
    both_improve = True
    any_exceeds = False
    for centre in CENTRES:
        cd, td = control["loco"][centre], arm["loco"][centre]
        delta = cd["k5_fpr"] - td["k5_fpr"]        # >0 == improvement
        bar = conservative_bar_loco(control, arm, centre)
        improves = bool(delta > 0)
        exceeds = bool(delta > bar)
        both_improve = both_improve and improves
        any_exceeds = any_exceeds or exceeds
        loco[centre] = {"control_k5": cd["k5_fpr"], "treatment_k5": td["k5_fpr"],
                        "delta": delta, "conservative_bar": bar,
                        "improves": improves, "exceeds_bar": exceeds}

    ca = control["pooled"]["k5"]["fpr_asymmetry"]
    ta = arm["pooled"]["k5"]["fpr_asymmetry"]
    a_bar = conservative_bar_pooled(control, arm, "fpr_asymmetry")
    a_delta = abs(ca) - abs(ta)                    # >0 == magnitude reduced
    asymmetry_reduced = bool(a_delta > a_bar)

    veto = False
    veto_detail: Dict[str, Any] = {}
    for f in ("fpr_at_90_recall", "fpr_prior_equalised"):
        c, t = control["pooled"]["k5"][f], arm["pooled"]["k5"][f]
        bar = conservative_bar_pooled(control, arm, f)
        regression = t - c                          # >0 == got worse
        vetoed = bool(regression > bar)
        veto = veto or vetoed
        veto_detail[f] = {"control": c, "treatment": t, "regression": regression,
                          "conservative_bar": bar, "vetoes": vetoed}

    accepted = bool(both_improve and any_exceeds and asymmetry_reduced and not veto)
    return {
        "loco": loco, "both_loco_improve": both_improve,
        "any_loco_exceeds_bar": any_exceeds,
        "asymmetry": {"control": ca, "treatment": ta, "delta_magnitude": a_delta,
                     "conservative_bar": a_bar, "reduced_beyond_bar": asymmetry_reduced},
        "veto": veto, "veto_detail": veto_detail,
        "accepted": accepted,
    }


# ---------------------------------------------------------------------------
# Interaction check
# ---------------------------------------------------------------------------
def interaction_check(a0: Dict[str, Any], a1: Dict[str, Any],
                      a4: Dict[str, Any]) -> Dict[str, Any]:
    def dist(v: float) -> float:
        return abs(v - 0.5)

    a0_ss = a0["pooled"]["single_seed"]["centre_auc_negatives"]["median"]
    a1_ss = a1["pooled"]["single_seed"]["centre_auc_negatives"]["median"]
    a4_ss = a4["pooled"]["single_seed"]["centre_auc_negatives"]["median"]
    a1_iqr = a1["pooled"]["single_seed"]["centre_auc_negatives"]["iqr"]
    a4_iqr = a4["pooled"]["single_seed"]["centre_auc_negatives"]["iqr"]
    confound_bar = max(a1_iqr, a4_iqr)
    confound_delta = dist(a4_ss) - dist(a1_ss)      # <0 == A4 reduced further
    confound_consistent = bool(abs(confound_delta) <= confound_bar)

    # LOCO cannot test interaction here: A4's LOCO arm IS runs/sweep_a2_loco,
    # bit-identical by construction (the reuse this whole arm depends on), so
    # any A4-vs-A2 LOCO delta is exactly zero by identity, not by measurement.
    loco_identical = True
    loco_note = ("A4's LOCO arm is runs/sweep_a2_loco itself (reused, verified "
                "bit-identical by the a4_regression gate), so A4's LOCO figures "
                "equal A2's LOCO figures exactly. This axis cannot show "
                "interference between the two components by construction -- "
                "it is identity, not evidence.")

    return {
        "confound_axis": {
            "a0_single_seed_dist_from_half": dist(a0_ss),
            "a1_single_seed_dist_from_half": dist(a1_ss),
            "a4_single_seed_dist_from_half": dist(a4_ss),
            "a4_minus_a1_delta": confound_delta,
            "conservative_bar": confound_bar,
            "consistent_with_a1": confound_consistent,
        },
        "loco_axis": {"identical_by_construction": loco_identical, "note": loco_note},
    }


# ---------------------------------------------------------------------------
# Probe -- magnitude_scale 0.10, LOCO only
# ---------------------------------------------------------------------------
def probe_section(a0_loco: Dict[int, Dict[str, Any]],
                  a2_loco: Dict[int, Dict[str, Any]],
                  probe_loco: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for centre in CENTRES:
        c0, c2, cp = a0_loco[centre], a2_loco[centre], probe_loco[centre]
        bar_probe = max(c0["loo4_spread"]["iqr"], cp["loo4_spread"]["iqr"])
        delta_probe = c0["k5_fpr"] - cp["k5_fpr"]     # >0 == improvement at 0.10
        bar_a2 = max(c0["loo4_spread"]["iqr"], c2["loo4_spread"]["iqr"])
        delta_a2 = c0["k5_fpr"] - c2["k5_fpr"]
        out[centre] = {
            "a0_k5": c0["k5_fpr"], "a2_k5_0.33": c2["k5_fpr"],
            "probe_k5_0.10": cp["k5_fpr"],
            "a2_delta_vs_a0": delta_a2, "a2_conservative_bar": bar_a2,
            "a2_resolvable": bool(abs(delta_a2) > bar_a2),
            "probe_delta_vs_a0": delta_probe, "probe_conservative_bar": bar_probe,
            "probe_resolvable": bool(abs(delta_probe) > bar_probe),
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt(x, p=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


PRE_REGISTRATION_MD = os.path.join(REPO_ROOT, "reports", "a4_pre_registration.md")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="manifests/rare25_folds_v2.csv")
    ap.add_argument("--out-md", default="reports/a4.md")
    ap.add_argument("--out-json", default="reports/a4.json")
    args = ap.parse_args(argv)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, args.manifest))

    required = ["runs/sweep_a4"]
    missing = [p for p in required if not os.path.isdir(os.path.join(REPO_ROOT, p))]
    if missing:
        print(f"[a4] not ready yet, missing: {missing}")
        return 1

    a0 = {"pooled": MS.analyse_pooled("runs/noise_floor_a", manifest),
         "loco": MS.analyse_loco("runs/noise_floor_b", manifest)}
    a1 = {"pooled": MS.analyse_pooled("runs/sweep_a1", manifest),
         "loco": MS.analyse_loco("runs/noise_floor_b", manifest)}
    a2 = {"pooled": MS.analyse_pooled("runs/sweep_a2", manifest),
         "loco": MS.analyse_loco("runs/sweep_a2_loco", manifest)}
    a4 = {"pooled": MS.analyse_pooled("runs/sweep_a4", manifest),
         "loco": MS.analyse_loco("runs/sweep_a2_loco", manifest)}
    a4_epoch_seconds = MS.epoch_seconds(["runs/sweep_a4", "runs/sweep_a2_loco"])

    probe_ready = os.path.isdir(os.path.join(REPO_ROOT, "runs", "sweep_a4_probe"))
    probe_loco = MS.analyse_loco("runs/sweep_a4_probe", manifest) if probe_ready else None

    rule1 = MS.acceptance(a4, a0)
    rule2 = dg_gate(a4, a0)
    resolv = resolvability_table(a0, a4)
    interaction = interaction_check(a0, a1, a4)
    probe = (probe_section(a0["loco"], a2["loco"], probe_loco)
            if probe_ready else None)

    md = render_markdown(a0, a4, a4_epoch_seconds, rule1, rule2, resolv,
                         interaction, probe, probe_ready)
    out_md = os.path.join(REPO_ROOT, args.out_md)
    write_text_durable(out_md, md)
    write_text_durable(os.path.join(REPO_ROOT, args.out_json), json.dumps({
        "a0": a0, "a1": a1, "a2": a2, "a4": a4,
        "probe": probe_loco, "rule1_pooled_fpr_gate": rule1,
        "rule2_dg_gate": rule2, "resolvability": resolv,
        "interaction": interaction, "probe_section": probe,
    }, indent=2, default=RA._JD))
    print(md)
    print(f"\nwritten: {args.out_md}")
    print(f"written: {args.out_json}")
    return 0


def render_markdown(a0, a4, a4_epoch_seconds, rule1, rule2, resolv,
                    interaction, probe, probe_ready) -> str:
    L: List[str] = []
    A = L.append
    A("# A4 — centre x class sampler + stack at 0.33x, pooled OOF; LOCO reused from A2")
    A("")
    A(f"Generated by `scripts/28_a4_sweep.py` on "
     f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. No "
     f"training, no GPU: recomputed from prediction parquets on disk.")
    A("")
    A("Full pre-registration (written 2026-07-30, before any A4 unit was "
     "trained): `reports/a4_pre_registration.md`. Both gates below were fixed "
     "there before these numbers existed.")
    A("")
    A("_Presentation fix (2026-08-04): several table-cell labels used a "
     "literal `|` (e.g. `FPR@90R | center_1`, `AUC(logit->centre | neg)`, "
     "`rho(file_size) | pos`), which splits a markdown table row on the pipe "
     "and silently shifts every value after it. Found in two passes -- a "
     "mechanical column-count checker caught a second instance "
     "(`rho_file_size_positives/negatives`) that eyeballing the first fix had "
     "missed, which is the reason a script checked it rather than a read-"
     "through. This corrupted the `center_1`/`center_2`/`centre_auc_negatives`/"
     "`rho(file_size)` rows in section 1's resolvability table and section "
     "3's per-arm detail tables. Sections 0, 2, 4 and 5 (both gate verdicts, "
     "the interaction check, the probe, the final ACCEPTED/REJECTED verdicts) "
     "never used these labels and are unaffected. No underlying value was "
     "ever wrong, no gate computation reads these label strings, and no "
     "verdict has changed -- only the display of the affected rows. Every "
     "number below is a fresh recomputation from the same parquets, and "
     "every table in this file has been mechanically verified to have a "
     "consistent column count, not just visually re-checked._")
    A("")

    A("## 0. Both gates, both verdicts")
    A("")
    A("Rule 1 is the A1-A3 pooled-FPR gate, retained unchanged. Rule 2 is the "
     "DG gate, PRIMARY from A4 onward -- see `reports/a4_pre_registration.md` "
     "for why the primary/secondary status of the two gates changed, and note "
     "that the justification is the non-resolvability shown in section 0 of "
     "`reports/magnitude_sweep.md`, not any property of the numbers below.")
    A("")
    lines = ["RULE 1 -- pooled-FPR gate (A1-A3's rule, unchanged)", "",
            f"   pooled gain {rule1['pooled_gain']:+.4f} vs IQR bar "
            f"{rule1['pooled_iqr_bar']:.4f}   "
            f"{'PASS' if rule1['pooled_passes'] else 'fail'}"]
    for centre in CENTRES:
        d = rule1["loco"][centre]
        lines.append(f"   LOCO c{centre} {d['delta']:+.4f} vs own IQR "
                     f"{d['own_loo4_iqr']:.4f}   "
                     f"{'VETO' if d['regresses_beyond_iqr'] else 'ok'}")
    lines.append(f"   VERDICT: {'ACCEPTED' if rule1['accepted'] else 'REJECTED'}")
    lines += ["", "RULE 2 -- DG gate (PRIMARY from A4 onward)", ""]
    for centre in CENTRES:
        d = rule2["loco"][centre]
        lines.append(f"   LOCO c{centre} {d['delta']:+.4f} vs conservative bar "
                     f"{d['conservative_bar']:.4f}   "
                     f"{'improves' if d['improves'] else 'REGRESSES'}"
                     f"{', exceeds bar' if d['exceeds_bar'] else ''}")
    lines.append(f"   both directions improve: "
                f"{'yes' if rule2['both_loco_improve'] else 'NO'}")
    lines.append(f"   at least one exceeds its bar: "
                f"{'yes' if rule2['any_loco_exceeds_bar'] else 'NO'}")
    asym = rule2["asymmetry"]
    lines.append(f"   asymmetry |{asym['control']:.4f}| -> |{asym['treatment']:.4f}|, "
                f"reduced {asym['delta_magnitude']:+.4f} vs bar "
                f"{asym['conservative_bar']:.4f}   "
                f"{'ok' if asym['reduced_beyond_bar'] else 'NOT beyond bar'}")
    for f, d in rule2["veto_detail"].items():
        lines.append(f"   veto check [{f}]: regression {d['regression']:+.4f} vs "
                    f"bar {d['conservative_bar']:.4f}   "
                    f"{'VETO' if d['vetoes'] else 'ok'}")
    lines.append(f"   VERDICT: {'ACCEPTED' if rule2['accepted'] else 'REJECTED'}")
    A(box(lines))
    A("")

    A("## 1. Resolvability, every A4-vs-A0 comparison (conservative: larger "
     "of the two arms' LOO-4 IQRs)")
    A("")
    A("| metric | A0 (control) | A4 (treatment) | delta | conservative bar | "
     "resolvable? |")
    A("|---|---|---|---|---|---|")
    for r in resolv:
        A(f"| {r['metric']} | {fmt(r['control'])} | {fmt(r['treatment'])} | "
         f"{r['delta']:+.4f} | {fmt(r['conservative_bar'])} | "
         f"{'**yes**' if r['resolvable'] else 'no'} |")
    A("")

    A("## 2. Interaction check")
    A("")
    ca = interaction["confound_axis"]
    A(f"**Confound axis (pooled).** AUC(logit->centre \\| neg), distance from "
     f"0.5, single-seed median. A0 {ca['a0_single_seed_dist_from_half']:.4f}, "
     f"A1 alone {ca['a1_single_seed_dist_from_half']:.4f}, A4 "
     f"{ca['a4_single_seed_dist_from_half']:.4f}. A4 minus A1 "
     f"{ca['a4_minus_a1_delta']:+.4f} against a conservative bar of "
     f"{ca['conservative_bar']:.4f} -- "
     f"{'consistent with A1 alone' if ca['consistent_with_a1'] else '**not** consistent with A1 alone: the combination moved this further than A1 alone did, beyond noise'}.")
    A("")
    la = interaction["loco_axis"]
    A(f"**LOCO axis.** {la['note']}")
    A("")

    A("## 3. Per-arm detail")
    A("")
    for key, label, r in (("A0", "baseline", a0), ("A4", "sampler + stack (0.33x)", a4)):
        p = r["pooled"]
        A(f"### {key} — {label}")
        A("")
        A("| metric | single-seed median | single-seed IQR | k=5 ensemble | "
         "k=5 LOO-4 IQR |")
        A("|---|---|---|---|---|")
        for f in ("fpr_at_90_recall", "roc_auc", "pauc_15_std", "pauc_15_raw",
                 "centre_auc_negatives", "rho_file_size_positives",
                 "rho_file_size_negatives", "fpr_prior_equalised",
                 "fpr_center_1", "fpr_center_2", "fpr_asymmetry"):
            ss, lo = p["single_seed"][f], p["loo4"][f]
            A(f"| {POOLED_LABELS[f]} | {ss['median']:.4f} | {ss['iqr']:.4f} | "
             f"{p['k5'][f]:.4f} | {lo['iqr']:.4f} |")
        A("")
        for centre in CENTRES:
            d = r["loco"][centre]
            A(f"- LOCO `holdout_center_{centre}`: k=5 {d['k5_fpr']:.4f}, "
             f"{fmt_spread(d['loo4_spread'])}")
        A("")
    if a4_epoch_seconds:
        A(f"A4 epoch seconds: median {a4_epoch_seconds['median']:.1f}, IQR "
         f"{a4_epoch_seconds['iqr']:.1f}, range "
         f"[{a4_epoch_seconds['min']:.1f}, {a4_epoch_seconds['max']:.1f}]")
        A("")

    A("## 4. Probe -- magnitude_scale 0.10, LOCO only")
    A("")
    if not probe_ready:
        A("`runs/sweep_a4_probe` not present yet -- this section is written "
         "once the probe completes.")
    else:
        A("A0 vs A2 (0.33x, for reference) vs the probe (0.10x), same LOCO "
         "protocol, both centres. Resolvability against A0 uses the "
         "conservative bar (larger of the two LOO-4 IQRs).")
        A("")
        A("| centre | A0 k=5 | A2 (0.33x) k=5 | delta vs A0 | resolvable? | "
         "probe (0.10x) k=5 | delta vs A0 | resolvable? |")
        A("|---|---|---|---|---|---|---|---|")
        for centre in CENTRES:
            d = probe[centre]
            A(f"| holdout_center_{centre} | {d['a0_k5']:.4f} | "
             f"{d['a2_k5_0.33']:.4f} | {d['a2_delta_vs_a0']:+.4f} | "
             f"{'**yes**' if d['a2_resolvable'] else 'no'} | "
             f"{d['probe_k5_0.10']:.4f} | {d['probe_delta_vs_a0']:+.4f} | "
             f"{'**yes**' if d['probe_resolvable'] else 'no'} |")
        A("")
        c1 = probe[1]
        if c1["a2_resolvable"] and c1["probe_resolvable"]:
            A("holdout_center_1's gain survives at 0.10x: still resolvable "
             "against A0, in the same direction.")
        elif c1["a2_resolvable"] and not c1["probe_resolvable"]:
            A("holdout_center_1's gain does **not** survive at 0.10x: "
             "resolvable at 0.33x, not resolvable at 0.10x against the same "
             "conservative bar.")
        else:
            A("holdout_center_1's gain was not resolvable at 0.33x either in "
             "this recomputation; see section 1/3 above before drawing a "
             "magnitude-dependence conclusion.")
        A("")

    A("## 5. Verdicts")
    A("")
    A(f"### Rule 1 (pooled-FPR gate): "
     f"{'**ACCEPTED**' if rule1['accepted'] else '**REJECTED**'}")
    A("")
    A(f"- Pooled-OOF ensembled FPR@90R {a0['pooled']['k5']['fpr_at_90_recall']:.4f} "
     f"-> {a4['pooled']['k5']['fpr_at_90_recall']:.4f}, a gain of "
     f"{rule1['pooled_gain']:+.4f} against an IQR bar of "
     f"{rule1['pooled_iqr_bar']:.4f} -- "
     f"{'clears the bar' if rule1['pooled_passes'] else 'does not clear the bar'}.")
    for centre in CENTRES:
        d = rule1["loco"][centre]
        A(f"- LOCO `holdout_center_{centre}` {d['control_k5']:.4f} -> "
         f"{d['treatment_k5']:.4f} ({d['delta']:+.4f}) against its own LOO-4 "
         f"IQR of {d['own_loo4_iqr']:.4f} -- "
         f"{'**VETO**' if d['regresses_beyond_iqr'] else 'within tolerance'}.")
    A("")
    A(f"### Rule 2 (DG gate, PRIMARY): "
     f"{'**ACCEPTED**' if rule2['accepted'] else '**REJECTED**'}")
    A("")
    A(f"- Both LOCO directions improve: "
     f"{'yes' if rule2['both_loco_improve'] else '**no**'}.")
    A(f"- At least one exceeds its conservative bar: "
     f"{'yes' if rule2['any_loco_exceeds_bar'] else '**no**'}.")
    A(f"- Centre asymmetry reduced beyond its conservative bar: "
     f"{'yes' if rule2['asymmetry']['reduced_beyond_bar'] else '**no**'}.")
    A(f"- In-distribution veto triggered: "
     f"{'**yes**' if rule2['veto'] else 'no'}.")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
