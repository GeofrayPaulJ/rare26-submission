"""The n=1 re-frame (2026-08-07/08). NO TRAINING, NO GPU: recomputed from
prediction parquets already on disk for every arm.

WHY. STEP 3's measured T4 throughput admits exactly ONE ensemble member.
Every standing verdict (A1-A4 Rule 1, G1/G3 Rule 2) was computed on k=5
SEED-ENSEMBLED logits -- a quantity the deployed artefact cannot produce.
This script recomputes the SINGLE-CHECKPOINT analogue of every headline
figure and re-applies each arm's own pre-registered gate with n=1
quantities substituted for the k=5 ones, using the SAME structural pattern
those gates already use (point estimate vs. a conservative noise bar) --
only the point estimate and the bar change from "k5 ensemble" /
"leave-one-seed-out-4 IQR" to "median of 5 single-seed pooled figures" /
"across-seed IQR of those same 5 figures". This is a re-interpretation
made explicit here, not an invention of new gate math.

SINGLE-CHECKPOINT POOLED-OOF = MS.analyse_pooled()'s existing "single_seed"
block (median/IQR of 5 per-seed pooled-OOF figures) -- already the
established convention (reports/gastronet.json). Nothing new there.

SINGLE-CHECKPOINT LOCO (n=1) = per-seed metric_block on that seed's own
LOCO held-out predictions, NOT ensembled across seeds -- new here, since
every existing LOCO report only shows k5/loo4-over-seeds.

PER-FOLD TAIL = for each seed, the worst (max) single-FOLD FPR@90R across
its 5 folds (a true single-checkpoint number: one fold's held-out
predictions from one specific trained model, no pooling at all) --
summarised by median and max across the 5 seeds.

    python scripts/56_single_checkpoint_reframe.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.evaluate import metric_block, spread  # noqa: E402


def _load(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MS = _load("_magnitude_sweep", "26_magnitude_sweep.py")
RA = _load("_reanalysis", "17_reanalysis.py")

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
CENTRES = (1, 2)
REPEAT = 0
FIELDS = ("fpr_at_90_recall", "fpr_prior_equalised", "fpr_center_1",
         "fpr_center_2", "fpr_asymmetry", "centre_auc_negatives")

# --- arm registry -----------------------------------------------------------
# (key, pooled_dir or None, loco_dir or None). None means "not on disk" --
# reported as such, never fabricated.
POOLED_ARMS: List[tuple] = [
    ("A0", "runs/noise_floor_a"),
    ("A1", "runs/sweep_a1"),
    ("A2", "runs/sweep_a2"),
    ("A3", "runs/sweep_a3"),
    ("A4-pinned", "runs/a4_pinned085"),
    ("A4-corrected", "runs/a4_checkpointed"),
    ("G0", "runs/g0_rn50_imagenet"),
    ("G1", "runs/g1_rn50_swsl"),
    ("G2", "runs/g2_vitb_dinov2"),
    ("G3", "runs/g3_rn50_gastronet"),
]
LOCO_ARMS: Dict[str, Optional[str]] = {
    "A0": "runs/noise_floor_b",
    "A1": "runs/noise_floor_b",          # reused, sampler-degeneracy argument
    "A2": "runs/sweep_a2_loco",
    "A3": "runs/sweep_a3_loco",
    # runs/a4_pinned085_swa_loco's "raw" variant IS pinned-A4's own n=1 LOCO
    # -- passivity (EMA/SWA tracking never alters the raw trajectory) was
    # bit-identity-verified for the pooled arm (reports/swa_arm.md); same
    # code path, same config, for LOCO. val_loco_c{c}_s{s}.parquet (no
    # suffix) is exactly the "raw" variant this reframe's n1_loco() already
    # knows how to read.
    "A4-pinned": "runs/a4_pinned085_swa_loco",
    "A4-corrected": "runs/a4_checkpointed_loco",
    "G0": "runs/g0_rn50_imagenet_loco",
    "G1": "runs/g1_rn50_swsl_loco",
    "G2": None,                           # never run
    "G3": "runs/g3_rn50_gastronet_loco",
}

# Original k=5 verdicts, sourced from this session's own reading of
# reports/{magnitude_sweep,gastronet,a4_pre_registration}.md and tonight's
# scripts/50 output. Documented per-entry so a stale citation is easy to
# spot and fix, not silently trusted.
ORIGINAL_K5_VERDICTS = {
    "A1": {"rule": "Rule 1 (pooled-FPR vs A0)", "verdict": "REJECTED",
          "source": "reports/magnitude_sweep.md section 0: no arm resolved on pooled-OOF"},
    "A2": {"rule": "Rule 1 (pooled-FPR vs A0)", "verdict": "REJECTED",
          "source": "reports/magnitude_sweep.md section 0 (LOCO gain was the "
                    "finding, not a Rule-1 accept)"},
    "A3": {"rule": "Rule 1 (pooled-FPR vs A0)", "verdict": "REJECTED",
          "source": "reports/magnitude_sweep.md section 0"},
    "A4-pinned": {"rule": "Rule 1 + Rule 2 (vs A0)", "verdict": "N/A -- diagnostic isolation arm, never gated",
                 "source": "reports/a4_pinned085_verdict.md (parameter-isolation, not an acceptance run)"},
    "A4-corrected": {"rule": "Rule 1 + Rule 2 (vs A0)", "verdict": "ACCEPTED (A4 lineage, standing reference)",
                     "source": "reports/a4.md / reports/gastronet.json reference_a4"},
    "G1": {"rule": "Rule 2 (vs A4)", "verdict": "no formal verdict existed before tonight",
          "source": "gastronet.md's G1 LOCO section had deltas/fusion only, never a Rule-2 computation"},
    "G3": {"rule": "Rule 2 (vs A4-corrected)", "verdict": "REJECTED",
          "source": "reports/g3_loco_rule2_corrected.json (computed tonight, scripts/50)"},
}


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))


def per_fold_tail(run_dir: str, manifest: pd.DataFrame,
                  seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """For each seed, max single-fold FPR@90R across its 5 folds (a genuine
    single-checkpoint number). Summarised by median/max across seeds."""
    per_seed_max: Dict[int, float] = {s: -1.0 for s in seeds}
    per_seed_per_fold: Dict[int, Dict[int, float]] = {s: {} for s in seeds}
    for f in tqdm(FOLDS, desc=f"  per-fold tail {os.path.basename(run_dir)}",
                 unit="fold", leave=False, file=sys.stderr):
        base, logit_wide, _ = RA.build_wide_fold(
            os.path.join(REPO_ROOT, run_dir), f, manifest, seeds, repeat=REPEAT)
        y = base["label_int"].to_numpy()
        for s in seeds:
            fpr = metric_block(y, logit_wide[s].to_numpy())["fpr_at_90_recall"]
            per_seed_per_fold[s][f] = fpr
            per_seed_max[s] = max(per_seed_max[s], fpr)
    vals = list(per_seed_max.values())
    return {"per_seed_max": per_seed_max, "per_seed_per_fold": per_seed_per_fold,
           "median": float(np.median(vals)), "max": float(np.max(vals))}


def n1_loco(run_dir: str, manifest: pd.DataFrame,
           seeds: Sequence[int] = SEEDS) -> Dict[int, Dict[str, Any]]:
    """Per-seed (single-checkpoint) LOCO FPR@90R, both directions -- median
    and IQR across seeds, plus the raw 5 values for pairing."""
    out = {}
    for centre in CENTRES:
        base, logit_wide, _ = RA.build_wide_loco(
            os.path.join(REPO_ROOT, run_dir), centre, manifest, seeds)
        y = base["label_int"].to_numpy()
        vals = {s: metric_block(y, logit_wide[s].to_numpy())["fpr_at_90_recall"]
               for s in seeds}
        out[centre] = {"per_seed": vals, **spread(list(vals.values()))}
    return out


def apply_rule1(arm_n1: dict, control_n1: dict,
                arm_loco_n1: Optional[Dict[int, Any]],
                control_loco_n1: Optional[Dict[int, Any]]) -> dict:
    """Rule 1 (reports/a4_pre_registration.md), n=1 quantities substituted:
    point estimate = single_seed median (was: k5), bar = across-seed IQR
    (was: leave-one-seed-out-4 IQR). Same structural comparison, smaller n."""
    av = arm_n1["fpr_at_90_recall"]["median"]
    cv = control_n1["fpr_at_90_recall"]["median"]
    bar = max(arm_n1["fpr_at_90_recall"]["iqr"], control_n1["fpr_at_90_recall"]["iqr"])
    delta = cv - av  # positive = arm better (lower FPR)
    improves = delta > bar

    loco_ok = None
    loco_detail = {}
    if arm_loco_n1 is not None and control_loco_n1 is not None:
        loco_ok = True
        for c in CENTRES:
            a = arm_loco_n1[c]["median"]
            k = control_loco_n1[c]["median"]
            b = max(arm_loco_n1[c]["iqr"], control_loco_n1[c]["iqr"])
            regressed = (a - k) > b
            loco_detail[c] = {"arm": a, "control": k, "bar": b, "regressed": regressed}
            if regressed:
                loco_ok = False

    accept = improves and (loco_ok is not False)
    return {"arm_median": av, "control_median": cv, "delta": delta, "bar": bar,
           "improves": improves, "loco_ok": loco_ok, "loco_detail": loco_detail,
           "accept": accept,
           "note": ("LOCO not available for one side -- pooled-only verdict"
                    if loco_ok is None else "")}


def apply_rule2(arm_n1: dict, control_n1: dict,
                arm_loco_n1: Dict[int, Any], control_loco_n1: Dict[int, Any]) -> dict:
    """Rule 2 (DG gate), n=1 quantities substituted throughout."""
    per_centre = {}
    both_improve = True
    any_beyond_bar = False
    for c in CENTRES:
        a = arm_loco_n1[c]["median"]
        k = control_loco_n1[c]["median"]
        b = max(arm_loco_n1[c]["iqr"], control_loco_n1[c]["iqr"])
        delta = k - a  # positive = arm better
        beyond = delta > b
        per_centre[c] = {"arm": a, "control": k, "delta": delta, "bar": b,
                         "beyond_bar": beyond}
        if delta <= 0:
            both_improve = False
        if beyond:
            any_beyond_bar = True

    a_asym = arm_n1["fpr_asymmetry"]["median"]
    k_asym = control_n1["fpr_asymmetry"]["median"]
    asym_bar = max(arm_n1["fpr_asymmetry"]["iqr"], control_n1["fpr_asymmetry"]["iqr"])
    asym_reduced = (abs(k_asym) - abs(a_asym)) > asym_bar

    veto = False
    veto_detail = []
    for field in ("fpr_at_90_recall", "fpr_prior_equalised"):
        av = arm_n1[field]["median"]
        kv = control_n1[field]["median"]
        b = max(arm_n1[field]["iqr"], control_n1[field]["iqr"])
        regressed = (av - kv) > b
        veto_detail.append({"field": field, "arm": av, "control": kv, "bar": b,
                            "regressed": regressed})
        if regressed:
            veto = True

    accept = both_improve and any_beyond_bar and asym_reduced and not veto
    return {"per_centre": per_centre, "both_improve": both_improve,
           "any_beyond_bar": any_beyond_bar,
           "asymmetry": {"arm": a_asym, "control": k_asym, "bar": asym_bar,
                        "reduced": asym_reduced},
           "veto": {"triggered": veto, "detail": veto_detail}, "accept": accept}


def main() -> int:
    t0 = time.time()
    manifest = load_manifest()

    pooled: Dict[str, Any] = {}
    for key, run_dir in tqdm(POOLED_ARMS, desc="pooled arms", unit="arm",
                             file=sys.stderr):
        if not os.path.isdir(os.path.join(REPO_ROOT, run_dir)):
            pooled[key] = None
            continue
        a = MS.analyse_pooled(run_dir, manifest)
        tail = per_fold_tail(run_dir, manifest)
        pooled[key] = {"run_dir": run_dir, "single_seed": a["single_seed"],
                       "single_seed_values": a["single_seed_values"],
                       "k5": a["k5"], "per_fold_tail": tail}

    loco: Dict[str, Any] = {}
    for key, run_dir in tqdm(LOCO_ARMS.items(), desc="LOCO arms", unit="arm",
                             file=sys.stderr):
        if run_dir is None or not os.path.isdir(os.path.join(REPO_ROOT, run_dir)):
            loco[key] = None
            continue
        n1 = n1_loco(run_dir, manifest)
        k5 = MS.analyse_loco(run_dir, manifest)
        loco[key] = {"run_dir": run_dir, "n1": n1,
                    "k5": {c: k5[c]["k5_fpr"] for c in CENTRES}}

    # --- gate re-application ---
    gates: Dict[str, Any] = {}
    a0 = pooled.get("A0")
    a0_loco = loco.get("A0")
    for key in ("A1", "A2", "A3", "A4-pinned", "A4-corrected"):
        arm = pooled.get(key)
        if arm is None or a0 is None:
            gates[key] = {"rule1": None, "note": "arm or A0 not on disk"}
            continue
        arm_loco = loco.get(key)
        gates[key] = {"rule1": apply_rule1(arm["single_seed"], a0["single_seed"],
                                           arm_loco["n1"] if arm_loco else None,
                                           a0_loco["n1"] if a0_loco else None)}

    a4c = pooled.get("A4-corrected")
    a4c_loco = loco.get("A4-corrected")
    for key in ("G1", "G3"):
        arm = pooled.get(key)
        arm_loco = loco.get(key)
        if arm is None or a4c is None or arm_loco is None or a4c_loco is None:
            gates[key] = {"rule2": None,
                         "note": "arm, A4-corrected, or one side's LOCO not on disk"}
            continue
        gates[key] = {"rule2": apply_rule2(arm["single_seed"], a4c["single_seed"],
                                           arm_loco["n1"], a4c_loco["n1"])}

    # --- ensembling gain (k5 minus n=1 median), pooled FPR@90R ---
    gains = {}
    for key, arm in pooled.items():
        if arm is None:
            continue
        k5v = arm["k5"]["fpr_at_90_recall"]
        n1v = arm["single_seed"]["fpr_at_90_recall"]["median"]
        gains[key] = {"k5": k5v, "n1_median": n1v, "gain": n1v - k5v}
    gain_vals = [g["gain"] for g in gains.values()]
    gain_uniform = (max(gain_vals) - min(gain_vals) < 2 * float(np.median(gain_vals) or 1e-6)
                    if gain_vals else None)

    payload = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "pooled": pooled, "loco": loco, "gates": gates, "gains": gains,
              "gain_uniform_heuristic": gain_uniform,
              "wall_seconds": time.time() - t0}
    with open(os.path.join(REPORT_DIR, "single_checkpoint_reframe.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    render(payload)
    print(f"[reframe] done in {payload['wall_seconds']:.0f}s")
    return 0


def fmt(x, p=4):
    return "n/a" if x is None else f"{x:.{p}f}"


def render(payload: dict) -> None:
    L = []
    A = L.append
    A("# Single-checkpoint (n=1) re-frame")
    A("")
    A(f"Generated {payload['generated_utc']} by "
     f"`scripts/56_single_checkpoint_reframe.py`, from prediction parquets "
     f"already on disk -- no training, no GPU. Every verdict on record was "
     f"computed on k=5 SEED-ENSEMBLED logits; the deployed artefact (STEP "
     f"3's measured T4 throughput) is n=1. This is the single-checkpoint "
     f"analogue of every headline figure, and each arm's own pre-registered "
     f"gate re-applied with n=1 quantities substituted for k=5 ones -- "
     f"point estimate = median of 5 single-seed pooled figures (was: the "
     f"k=5 ensemble), bar = across-seed IQR of those same 5 figures (was: "
     f"leave-one-seed-out-4 IQR). Same structural comparison, smaller n; "
     f"this substitution is the re-interpretation this report makes "
     f"explicit, not new gate math.")
    A("")

    A("## Pooled-OOF: single-checkpoint (n=1) vs k=5, per arm")
    A("")
    A("| arm | n=1 median | n=1 IQR | k=5 | ensembling gain (n1-k5) | "
     "per-fold tail median | per-fold tail max |")
    A("|---|---|---|---|---|---|---|")
    for key, _ in POOLED_ARMS:
        arm = payload["pooled"].get(key)
        if arm is None:
            A(f"| {key} | -- | -- | -- | -- | -- | -- | *(not on disk)* |")
            continue
        ss = arm["single_seed"]["fpr_at_90_recall"]
        tail = arm["per_fold_tail"]
        A(f"| {key} | {fmt(ss['median'])} | {fmt(ss['iqr'])} | "
         f"{fmt(arm['k5']['fpr_at_90_recall'])} | "
         f"{payload['gains'][key]['gain']:+.4f} | {fmt(tail['median'])} | "
         f"{fmt(tail['max'])} |")
    A("")
    gu = payload["gain_uniform_heuristic"]
    A(f"**Ensembling gain uniformity (heuristic: range < 2x median gain): "
     f"{'roughly uniform' if gu else 'NOT uniform -- see per-arm table above'}.**")
    A("")

    A("## LOCO: single-checkpoint (n=1) vs k=5, both directions")
    A("")
    A("| arm | direction | n=1 median | n=1 IQR | k=5 |")
    A("|---|---|---|---|---|")
    for key in LOCO_ARMS:
        larm = payload["loco"].get(key)
        if larm is None:
            A(f"| {key} | -- | -- | -- | -- | *(not on disk)* |")
            continue
        for c in CENTRES:
            n1 = larm["n1"][c]
            A(f"| {key} | holdout_center_{c} | {fmt(n1['median'])} | "
             f"{fmt(n1['iqr'])} | {fmt(larm['k5'][c])} |")
    A("")

    A("## Verdict-change table")
    A("")
    A("| arm | k=5 verdict | n=1 verdict | changed? |")
    A("|---|---|---|---|")
    for key, meta in ORIGINAL_K5_VERDICTS.items():
        g = payload["gates"].get(key, {})
        if "rule1" in g and g["rule1"] is not None:
            n1v = "ACCEPTED" if g["rule1"]["accept"] else "REJECTED"
        elif "rule2" in g and g["rule2"] is not None:
            n1v = "ACCEPTED" if g["rule2"]["accept"] else "REJECTED"
        else:
            n1v = f"N/A ({g.get('note', 'not computed')})"
        old = meta["verdict"]
        changed = ("yes" if (("ACCEPT" in old) != ("ACCEPT" in n1v)) else
                  ("n/a" if "N/A" in old or "N/A" in n1v else "no"))
        A(f"| {key} | {old} ({meta['rule']}) | {n1v} | {changed} |")
    A("")
    A("_Sources for each k=5 verdict are in this script's "
     "`ORIGINAL_K5_VERDICTS` dict, cited per entry._")
    A("")

    A("## Gate detail")
    A("")
    for key, g in payload["gates"].items():
        A(f"### {key}")
        A("")
        if g.get("rule1") is not None:
            r1 = g["rule1"]
            A(f"Rule 1 (n=1): arm median {fmt(r1['arm_median'])} vs control "
             f"median {fmt(r1['control_median'])}, delta {r1['delta']:+.4f}, "
             f"bar {fmt(r1['bar'])} -- improves: **{r1['improves']}**. "
             f"LOCO ok: **{r1['loco_ok']}**{(' -- ' + r1['note']) if r1['note'] else ''}. "
             f"**{'ACCEPT' if r1['accept'] else 'REJECT'}**")
        if g.get("rule2") is not None:
            r2 = g["rule2"]
            A(f"Rule 2 (n=1): both directions improve: **{r2['both_improve']}**, "
             f"at least one beyond bar: **{r2['any_beyond_bar']}**, asymmetry "
             f"reduced: **{r2['asymmetry']['reduced']}** "
             f"(arm {fmt(r2['asymmetry']['arm'])} vs control "
             f"{fmt(r2['asymmetry']['control'])}, bar "
             f"{fmt(r2['asymmetry']['bar'])}), veto: "
             f"**{'TRIGGERED' if r2['veto']['triggered'] else 'not triggered'}**. "
             f"**{'ACCEPT' if r2['accept'] else 'REJECT'}**")
        if g.get("rule1") is None and g.get("rule2") is None:
            A(f"_{g.get('note', 'not computed')}_")
        A("")

    with open(os.path.join(REPORT_DIR, "single_checkpoint_reframe.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[reframe] written: {os.path.join(REPORT_DIR, 'single_checkpoint_reframe.md')}")


if __name__ == "__main__":
    raise SystemExit(main())
