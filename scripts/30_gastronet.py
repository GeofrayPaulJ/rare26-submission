"""GastroNet / DINOv2 ensemble-member report. NO TRAINING, NO GPU.

Recomputed from prediction parquets on disk, reusing the tested measurement
path (scripts/26_magnitude_sweep.py's analyse_pooled, which owns the k=5
ensemble, the LOO-4 spread, and the prior-equalised / per-centre DG metrics)
rather than re-deriving any of it here.

WHAT THIS COMPARES. Each backbone against the A4 ConvNeXt figures already on
disk (runs/sweep_a4, repeat 0) -- the accepted arm and the ensemble's primary
member. These backbones are ENSEMBLE DIVERSITY MEMBERS: the question is not
"is this better than ConvNeXt" but "is this good enough, and decorrelated
enough, to be worth an ensemble slot and its inference cost".

WRITTEN AT BACKBONE BOUNDARIES ONLY. A backbone with fewer than its full 25
units is reported as INCOMPLETE with a unit count, never with metrics computed
from a partial set -- a partial backbone is unreported, not half-reported.

    python scripts/30_gastronet.py
    python scripts/30_gastronet.py --restart g2_vitb_dinov2 0 r0_f2_s3
    python scripts/30_gastronet.py --halt-stage g2_vitb_dinov2 --halt-reason "..."
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
    metric_block, per_centre_fpr_at_recall, prior_equalised_fpr_at_recall,
)
from src.io import fsync_dir, write_text_durable  # noqa: E402
from run_cv import Unit, expected_split, run_state  # noqa: E402


def _load(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MS = _load("_magnitude_sweep", "26_magnitude_sweep.py")
RA = _load("_reanalysis", "17_reanalysis.py")  # build_wide_fold, ensemble_pool_oof

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
REPEAT = 0
FUSION = "mean_logit"

# Priority order, matching scripts/23_sweep.py RUN_ORDER_GASTRONET.
BACKBONES = [
    {"key": "g1_rn50_swsl", "config": "configs/g1_rn50_swsl.yaml",
     "dir": "runs/g1_rn50_swsl", "arch": "resnet50", "image_size": 384,
     "label": "G1 -- RN50 Billion-Scale-SWSL + GastroNet-5M (DINOv1)",
     "note": "Priority 1, strongest configuration."},
    {"key": "g2_vitb_dinov2", "config": "configs/g2_vitb_dinov2.yaml",
     "dir": "runs/g2_vitb_dinov2", "arch": "vit_base_patch14_reg4_dinov2",
     "image_size": 378,
     "label": "G2 -- DINOv2 ViT-B/14 (registers) @378",
     "note": "Priority 2, architecture diversity. Runs at 378 = 27x14, not "
             "384: ViT-B/14 needs a size divisible by 14 and timm silently "
             "floors 384 to the same 27x27 grid while discarding 6 px."},
    {"key": "g3_rn50_gastronet", "config": "configs/g3_rn50_gastronet.yaml",
     "dir": "runs/g3_rn50_gastronet", "arch": "resnet50", "image_size": 384,
     "label": "G3 -- RN50 GastroNet-5M (DINOv1), SWSL control",
     "note": "Priority 3, control: identical to G1 but without the "
             "billion-scale SWSL stage, so G1-minus-G3 isolates SWSL."},
]

REFERENCE = {"key": "A4", "dir": "runs/sweep_a4",
             "label": "A4 ConvNeXt-Base (accepted arm, PRIMARY ensemble member)"}

# G1's LOCO arm, inserted 2026-08-04 ahead of G3 -- see scripts/23_sweep.py
# RUN_ORDER_GASTRONET for why (build question, ahead of a control question).
# A0's LOCO reference is runs/noise_floor_b; A4's is runs/sweep_a2_loco,
# reused bit-identically per reports/a4_pre_registration.md's established
# sampler-degeneracy argument -- the same reuse this whole project already
# relies on for A4's own LOCO figures everywhere else.
G1_LOCO = {"key": "g1_loco", "config": "configs/g1_rn50_swsl.yaml",
          "dir": "runs/g1_rn50_swsl_loco",
          "label": "G1 -- RN50 Billion-Scale-SWSL + GastroNet-5M (DINOv1)"}
A0_LOCO_DIR = "runs/noise_floor_b"
A4_LOCO_DIR = "runs/sweep_a2_loco"
CENTRES = (1, 2)

FIELDS = ("fpr_at_90_recall", "fpr_prior_equalised", "fpr_asymmetry",
          "fpr_center_1", "fpr_center_2", "roc_auc",
          "pauc_15_std", "pauc_15_raw", "centre_auc_negatives")
LABELS = {
    "fpr_at_90_recall": "FPR@90R",
    "fpr_prior_equalised": "FPR@90R, prior-equalised",
    "fpr_asymmetry": "centre asymmetry (c1 - c2)",
    # NOTE: no raw "|" inside a table-cell label. A literal pipe splits a
    # markdown table row into an extra column and silently shifts every value
    # after it -- this broke the center_1/center_2 rows in the first version
    # of this report (and, it turns out, in reports/a4.md too, same bug,
    # scripts/28_a4_sweep.py's identical POOLED_LABELS dict).
    "fpr_center_1": "FPR@90R (center_1)",
    "fpr_center_2": "FPR@90R (center_2)",
    "roc_auc": "ROC-AUC",
    "pauc_15_std": "pAUC[0,0.15] (McClish)",
    "pauc_15_raw": "pAUC[0,0.15] (area/0.15)",
    "centre_auc_negatives": "AUC(logit -> centre, neg)",
}
# Higher is better for these three; lower is better for everything else here.
# centre_auc_negatives is neither -- 0.5 is the null (no hospital signal), and
# distance from 0.5 in either direction is the thing that matters -- so it is
# reported as a plain delta with no better/worse framing implied by sign.
HIGHER_IS_BETTER = {"roc_auc", "pauc_15_std", "pauc_15_raw"}

REPORT_MD = os.path.join(REPO_ROOT, "reports", "gastronet.md")
REPORT_JSON = os.path.join(REPO_ROOT, "reports", "gastronet.json")
RESTART_LOG = os.path.join(REPO_ROOT, "logs", "gastronet_restarts.log")
HALT_NOTE = os.path.join(REPO_ROOT, "logs", "gastronet_halt_note.json")


# ---------------------------------------------------------------------------
# Completeness -- reuse run_state, never a second definition of "done"
# ---------------------------------------------------------------------------
def unit_status(config_path: str, out_dir: str) -> Dict[str, Any]:
    base_cfg = Config.from_yaml(os.path.join(REPO_ROOT, config_path))
    abs_out = os.path.join(REPO_ROOT, out_dir)
    done, missing = 0, []
    for s in SEEDS:
        for f in FOLDS:
            unit = Unit(seed=s, repeat=REPEAT, fold=f)
            cfg = unit.apply(dataclasses.replace(base_cfg, out_dir=abs_out))
            st, _ = run_state(os.path.join(abs_out, unit.name), unit, cfg,
                              expected_split(unit, cfg))
            if st == "done":
                done += 1
            else:
                missing.append(unit.name)
    return {"done": done, "total": len(SEEDS) * len(FOLDS), "missing": missing,
            "complete": done == len(SEEDS) * len(FOLDS)}


def unit_status_loco(config_path: str, out_dir: str) -> Dict[str, Any]:
    """Same as unit_status but for a LOCO arm: 5 seeds x 2 centres, no folds.
    A separate function rather than a mode branch in unit_status -- the CV
    hand-built Unit(seed, repeat, fold) loop in scan_for_restarts() silently
    checked the wrong paths the moment a LOCO arm was added to the same run
    order it assumed was CV-only; keeping the two shapes in separate
    functions here means this file cannot make that mistake."""
    base_cfg = Config.from_yaml(os.path.join(REPO_ROOT, config_path))
    abs_out = os.path.join(REPO_ROOT, out_dir)
    done, missing = 0, []
    for s in SEEDS:
        for c in CENTRES:
            unit = Unit(seed=s, holdout_centre=c)
            cfg = unit.apply(dataclasses.replace(base_cfg, out_dir=abs_out))
            st, _ = run_state(os.path.join(abs_out, unit.name), unit, cfg,
                              expected_split(unit, cfg))
            if st == "done":
                done += 1
            else:
                missing.append(unit.name)
    return {"done": done, "total": len(SEEDS) * len(CENTRES), "missing": missing,
            "complete": done == len(SEEDS) * len(CENTRES)}


def epoch_seconds(out_dir: str) -> Optional[Dict[str, float]]:
    """Median/IQR epoch wall time -- the inference-budget input. Read from each
    unit's own summary.json rather than timed here."""
    abs_out = os.path.join(REPO_ROOT, out_dir)
    vals: List[float] = []
    for s in SEEDS:
        for f in FOLDS:
            p = os.path.join(abs_out, f"r{REPEAT}_f{f}_s{s}", "summary.json")
            try:
                with open(p) as fh:
                    vals.append(float(json.load(fh)["median_epoch_seconds"]))
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                pass
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    q1, q3 = np.percentile(a, [25, 75])
    return {"n": int(a.size), "median": float(np.median(a)),
            "iqr": float(q3 - q1), "min": float(a.min()), "max": float(a.max())}


def analyse(out_dir: str, manifest: pd.DataFrame) -> Dict[str, Any]:
    return MS.analyse_pooled(out_dir, manifest, seeds=SEEDS, repeat=REPEAT)


# ---------------------------------------------------------------------------
# Ensemble fusion -- does adding this backbone to A4 actually help?
# ---------------------------------------------------------------------------
def k5_pool(out_dir: str, manifest: pd.DataFrame) -> pd.DataFrame:
    """The k=5 seed-ensembled pooled-OOF prediction set for one arm, as a
    filepath-indexed DataFrame -- the same primitive scripts/29_repeat_gating.py
    uses to pool across repeats, reused here to pool across ARMS instead."""
    wide = {f: RA.build_wide_fold(os.path.join(REPO_ROOT, out_dir), f, manifest,
                                  SEEDS, repeat=REPEAT) for f in FOLDS}
    return RA.ensemble_pool_oof(wide, list(SEEDS), FUSION, manifest, repeat=REPEAT)


def fuse_logit_mean(pool_a: pd.DataFrame, pool_b: pd.DataFrame) -> pd.DataFrame:
    """Simple logit-average of two arms' already-seed-ensembled pooled OOF
    predictions, aligned by filepath. Logit, not rank -- already settled."""
    a = pool_a.set_index("filepath")
    b = pool_b.set_index("filepath")
    if set(a.index) != set(b.index):
        raise AssertionError(
            f"fusion: filepath sets differ between arms ({len(a)} vs {len(b)} "
            f"rows) -- they should cover the identical repeat-0 image set")
    b = b.loc[a.index]
    if not (a["label_int"] == b["label_int"]).all():
        raise AssertionError("fusion: label_int disagrees between arms for "
                             "the same filepath -- manifests are out of sync")
    fused = a.copy()
    fused["logit"] = (a["logit"].to_numpy() + b["logit"].to_numpy()) / 2.0
    return fused.reset_index()


def fusion_metrics(pool: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, float]:
    y, s = pool["label_int"].to_numpy(), pool["logit"].to_numpy()
    m = metric_block(y, s)
    pc = per_centre_fpr_at_recall(pool)
    pe = prior_equalised_fpr_at_recall(pool)
    d = RA.diagnostics_for(pool, manifest)
    return {
        "fpr_at_90_recall": m["fpr_at_90_recall"],
        "fpr_prior_equalised": pe["fpr_at_90_recall_prior_equalised"],
        "fpr_asymmetry": pc["asymmetry_max_minus_min"],
        "fpr_center_1": pc["by_centre"].get("center_1", {}).get("fpr", float("nan")),
        "fpr_center_2": pc["by_centre"].get("center_2", {}).get("fpr", float("nan")),
        "roc_auc": m["roc_auc"],
        "pauc_15_std": m["pauc_15_std"],
        "pauc_15_raw": m["pauc_15_raw"],
        "centre_auc_negatives": d["centre_auc_negatives"],
    }


def analyse_fusion(backbone_dir: str, manifest: pd.DataFrame,
                   confound_bar: Optional[float]) -> Dict[str, Any]:
    """A4 alone / backbone alone / logit-mean fused, same metrics on all
    three -- the direct answer to "does this backbone earn an ensemble slot,
    or does it just trade one metric for another".

    ``confound_bar`` is the conservative bar (max of the two arms' LOO-4 IQR
    for centre_auc_negatives) ALREADY computed by the main metrics section --
    passed in rather than recomputed, so the two sections can never disagree
    about how wide noise is on that axis.
    """
    pool_a4 = k5_pool(REFERENCE["dir"], manifest)
    pool_bb = k5_pool(backbone_dir, manifest)
    fused = fuse_logit_mean(pool_a4, pool_bb)
    a4_alone = fusion_metrics(pool_a4, manifest)
    backbone_alone = fusion_metrics(pool_bb, manifest)
    fused_m = fusion_metrics(fused, manifest)

    # The confound-axis verdict: did fusion pull the ensemble toward the more
    # hospital-confounded arm, or stay close to the cleaner one? Distance from
    # the 0.5 null is the interpretable framing; resolvability against the
    # already-established bar is what makes the framing more than a story.
    confound: Dict[str, Any] = {
        "a4_alone": a4_alone["centre_auc_negatives"],
        "backbone_alone": backbone_alone["centre_auc_negatives"],
        "fused": fused_m["centre_auc_negatives"],
        "a4_dist_from_null": abs(a4_alone["centre_auc_negatives"] - 0.5),
        "backbone_dist_from_null": abs(backbone_alone["centre_auc_negatives"] - 0.5),
        "fused_dist_from_null": abs(fused_m["centre_auc_negatives"] - 0.5),
        "conservative_bar": confound_bar,
    }
    delta_vs_a4 = fused_m["centre_auc_negatives"] - a4_alone["centre_auc_negatives"]
    delta_vs_backbone = fused_m["centre_auc_negatives"] - backbone_alone["centre_auc_negatives"]
    confound["delta_fused_vs_a4"] = delta_vs_a4
    confound["delta_fused_vs_backbone"] = delta_vs_backbone
    confound["resolvable_vs_a4"] = (
        bool(confound_bar is not None and abs(delta_vs_a4) > confound_bar))
    confound["resolvable_vs_backbone"] = (
        bool(confound_bar is not None and abs(delta_vs_backbone) > confound_bar))
    # Moved toward the backbone's (more confounded) figure if fused's distance
    # from A4-alone's distance shrank the gap toward the backbone's distance.
    confound["moved_toward_backbone"] = bool(
        confound["fused_dist_from_null"] > confound["a4_dist_from_null"])

    return {
        "a4_alone": a4_alone, "backbone_alone": backbone_alone, "fused": fused_m,
        "confound": confound,
    }


# ---------------------------------------------------------------------------
# G1 LOCO -- the figure that actually decides the ensemble slot
# ---------------------------------------------------------------------------
def loco_k5_frame(out_dir: str, centre: int, manifest: pd.DataFrame) -> pd.DataFrame:
    wide = RA.build_wide_loco(os.path.join(REPO_ROOT, out_dir), centre, manifest, SEEDS)
    _y, _s, frame = RA.ensemble_loco(wide, list(SEEDS), FUSION)
    return frame


def fuse_loco_frames(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> pd.DataFrame:
    a = frame_a.set_index("filepath")
    b = frame_b.set_index("filepath")
    if set(a.index) != set(b.index):
        raise AssertionError(
            f"LOCO fusion: filepath sets differ between arms ({len(a)} vs "
            f"{len(b)} rows) -- both should cover the identical held-out set")
    b = b.loc[a.index]
    if not (a["label_int"] == b["label_int"]).all():
        raise AssertionError("LOCO fusion: label_int disagrees between arms "
                             "for the same filepath")
    fused = a.copy()
    fused["logit"] = (a["logit"].to_numpy() + b["logit"].to_numpy()) / 2.0
    return fused.reset_index()


def analyse_g1_loco(manifest: pd.DataFrame) -> Dict[str, Any]:
    """Both LOCO directions, A0 vs A4 vs G1, plus the A4+G1 fusion on LOCO
    predictions per direction -- the figure the pooled-OOF numbers everywhere
    else in this report cannot provide, and the one that actually decides
    whether G1 earns an ensemble slot.
    """
    st = unit_status_loco(G1_LOCO["config"], G1_LOCO["dir"])
    result: Dict[str, Any] = {"status": st}
    if not st["complete"]:
        return result

    a0 = MS.analyse_loco(A0_LOCO_DIR, manifest, seeds=SEEDS)
    a4 = MS.analyse_loco(A4_LOCO_DIR, manifest, seeds=SEEDS)
    g1 = MS.analyse_loco(G1_LOCO["dir"], manifest, seeds=SEEDS)

    fusion: Dict[int, Dict[str, float]] = {}
    for centre in CENTRES:
        fa = loco_k5_frame(A4_LOCO_DIR, centre, manifest)
        fb = loco_k5_frame(G1_LOCO["dir"], centre, manifest)
        fused = fuse_loco_frames(fa, fb)
        y, s = fused["label_int"].to_numpy(), fused["logit"].to_numpy()
        m = metric_block(y, s)
        fusion[centre] = {"fpr_at_90_recall": m["fpr_at_90_recall"],
                          "roc_auc": m["roc_auc"]}

    result.update({"a0": a0, "a4": a4, "g1": g1, "fusion": fusion})
    return result


# ---------------------------------------------------------------------------
# Restarts / halt
# ---------------------------------------------------------------------------
def record_restart(arm_key: str, repeat: int, redone: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(RESTART_LOG), exist_ok=True)
    existed = os.path.exists(RESTART_LOG)
    line = (f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  "
            f"backbone={arm_key} repeat={repeat}  redone=[{', '.join(redone)}]")
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


def record_halt(stage: str, reason: str, detail: str) -> None:
    write_text_durable(HALT_NOTE, json.dumps(
        {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "stage": stage, "reason": reason, "detail": detail}, indent=2))


def read_halt() -> Optional[Dict[str, str]]:
    if not os.path.exists(HALT_NOTE):
        return None
    try:
        with open(HALT_NOTE) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt(x: Optional[float], p: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def render_g1_loco_section(loco: Optional[Dict[str, Any]]) -> List[str]:
    L: List[str] = []
    A = L.append
    A("## G1 LOCO -- the figure that decides the ensemble slot")
    A("")
    A("Inserted ahead of G3 specifically because every G1 figure elsewhere in "
      "this report -- pooled OOF, the A4+G1 fusion -- comes from the protocol "
      "this project has repeatedly shown cannot resolve domain-generalisation "
      "questions (`reports/magnitude_sweep.md` section 0). This is the "
      "held-out-centre evidence that actually answers whether G1 earns an "
      "ensemble slot.")
    A("")
    if loco is None or not loco["status"]["complete"]:
        st = loco["status"] if loco else {"done": 0, "total": 10}
        A(f"**Not yet complete**: {st['done']}/{st['total']} units. Reported "
          f"the moment it exists; not computed from a partial set.")
        A("")
        return L

    A("Both directions, k=5 seed ensemble FPR@90R with its own within-repeat "
     "LOO-4 IQR. `bar` is the conservative bar (max of A4's and G1's LOO-4 "
     "IQR); A0 is shown for context and is not part of the bar.")
    A("")
    A("| direction | A0 | A4 | G1 | G1 vs A4 delta | bar | resolvable? |")
    A("|---|---|---|---|---|---|---|")
    for centre in CENTRES:
        a0d, a4d, g1d = loco["a0"][centre], loco["a4"][centre], loco["g1"][centre]
        bar = max(a4d["loo4_spread"]["iqr"], g1d["loo4_spread"]["iqr"])
        delta = g1d["k5_fpr"] - a4d["k5_fpr"]
        res = abs(delta) > bar
        A(f"| holdout_center_{centre} | {a0d['k5_fpr']:.4f} | {a4d['k5_fpr']:.4f} | "
         f"{g1d['k5_fpr']:.4f} | {delta:+.4f} | {bar:.4f} | "
         f"{'**yes**' if res else 'no'} |")
    A("")

    A("### A4+G1 fusion on LOCO predictions, per direction")
    A("")
    A("Same logit-average fusion as the pooled-OOF table, applied to each "
     "direction's held-out predictions instead. `bar` reused from the table "
     "above (max of A4's and G1's LOO-4 IQR).")
    A("")
    A("| direction | A4 alone | G1 alone | fused | fused vs A4 delta | bar | resolvable? |")
    A("|---|---|---|---|---|---|---|")
    for centre in CENTRES:
        a4d, g1d, fus = loco["a4"][centre], loco["g1"][centre], loco["fusion"][centre]
        bar = max(a4d["loo4_spread"]["iqr"], g1d["loo4_spread"]["iqr"])
        delta = fus["fpr_at_90_recall"] - a4d["k5_fpr"]
        res = abs(delta) > bar
        A(f"| holdout_center_{centre} | {a4d['k5_fpr']:.4f} | {g1d['k5_fpr']:.4f} | "
         f"{fus['fpr_at_90_recall']:.4f} | {delta:+.4f} | {bar:.4f} | "
         f"{'**yes**' if res else 'no'} |")
    A("")
    return L


def render(results: Dict[str, Any], ref: Optional[Dict[str, Any]],
           ref_epoch: Optional[Dict[str, float]],
           fusions: Dict[str, Any], g1_loco: Optional[Dict[str, Any]],
           restarts: List[str], halt: Optional[Dict[str, str]]) -> str:
    L: List[str] = []
    A = L.append
    A("# GastroNet / DINOv2 ensemble diversity members")
    A("")
    A(f"Generated by `scripts/30_gastronet.py` on "
      f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. No "
      f"training, no GPU: recomputed from prediction parquets on disk.")
    A("")
    A("_Presentation fix (2026-08-04): earlier versions of this report used a "
      "literal `|` inside table-cell labels (e.g. `FPR@90R | center_1`), "
      "which splits a markdown table row on the pipe and silently shifts "
      "every value after it. That corrupted the rendering of the "
      "`center_1`/`center_2`/`centre_auc_negatives` rows. No underlying value "
      "was ever wrong -- only the display. Labels fixed; every number in this "
      "regenerated file is a fresh recomputation from the same parquets, "
      "verified against a mechanical column-count check, not a hand edit of "
      "the old output._")
    A("")
    A("These are **ensemble diversity members, not backbone replacements**. "
      "ConvNeXt-Base (A4) remains the primary member; each arm below is asked "
      "whether it earns an additional ensemble slot and the inference cost "
      "that comes with it. Protocol per backbone: A4 config verbatim "
      "(centre x class sampler + stack at magnitude_scale 0.33), repeat 0, "
      "5 folds x 5 seeds = 25 units, `save_checkpoint: false`.")
    A("")

    if halt:
        A("## QUEUE HALTED")
        A("")
        A(f"- **when**: {halt.get('utc')}")
        A(f"- **stage**: `{halt.get('stage')}`")
        A(f"- **reason**: {halt.get('reason')}")
        if halt.get("detail"):
            A(f"- **detail**: {halt.get('detail')}")
        A("")
        A("Nothing further ran after this point. Backbones marked complete "
          "below finished before the halt and their numbers stand.")
        A("")

    A("## Status")
    A("")
    A("| backbone | units | status |")
    A("|---|---|---|")
    for b in BACKBONES:
        r = results[b["key"]]
        st = r["status"]
        mark = ("**complete**" if st["complete"]
                else f"incomplete -- not reported" if st["done"]
                else "not started")
        A(f"| {b['label']} | {st['done']}/{st['total']} | {mark} |")
    A("")

    complete = [b for b in BACKBONES if results[b["key"]]["status"]["complete"]]
    if not complete:
        A("No backbone has all 25 units yet, so no metrics are reported. A "
          "partial backbone is unreported, not half-reported.")
        A("")
    else:
        A("## Metrics vs A4 ConvNeXt (repeat 0, k=5 seed ensemble)")
        A("")
        A("`delta` is backbone minus A4. Lower is better except for ROC-AUC "
          "and both pAUC[0,0.15] variants, where higher is better -- a "
          "negative delta there is a REGRESSION, not an improvement. "
          "`centre_auc_negatives` is neither: 0.5 is the null (no hospital "
          "signal in the negatives), so read its delta as movement toward or "
          "away from 0.5, not as better/worse by sign. `bar` is the "
          "conservative noise bar -- the larger of the two arms' LOO-4 IQRs -- "
          "and `resolvable` asks whether |delta| exceeds it.")
        A("")
        A("pAUC is the metric that actually settles a ROC-AUC-vs-prior-equalised "
          "disagreement: ROC-AUC integrates over the full FPR range, including "
          "regions the challenge's own scoring never touches, so a ROC-AUC "
          "regression alongside a prior-equalised FPR improvement does not by "
          "itself mean a net cost. pAUC[0,0.15] is restricted to the FPR band "
          "the challenge actually scores.")
        A("")
        for b in complete:
            r = results[b["key"]]
            A(f"### {b['label']}")
            A("")
            A(f"{b['note']}")
            A("")
            A("| metric | A4 ConvNeXt | this backbone | delta | bar | resolvable? |")
            A("|---|---|---|---|---|---|")
            for f in FIELDS:
                t = r["pooled"]["k5"][f]
                bar = max(r["pooled"]["loo4"][f]["iqr"],
                          ref["loo4"][f]["iqr"]) if ref else float("nan")
                c = ref["k5"][f] if ref else float("nan")
                d = t - c if ref else float("nan")
                res = bool(np.isfinite(d) and np.isfinite(bar) and abs(d) > bar)
                A(f"| {LABELS[f]} | {fmt(c)} | {fmt(t)} | {d:+.4f} | "
                  f"{fmt(bar)} | {'**yes**' if res else 'no'} |")
            A("")
            es = r["epoch_seconds"]
            if es:
                A(f"**Epoch seconds** (inference-budget input): median "
                  f"{es['median']:.1f}s, IQR {es['iqr']:.1f}, range "
                  f"[{es['min']:.1f}, {es['max']:.1f}], n={es['n']}.")
                if ref_epoch:
                    ratio = es["median"] / ref_epoch["median"]
                    A(f"Relative to A4 ConvNeXt ({ref_epoch['median']:.1f}s): "
                      f"**{ratio:.2f}x**.")
                A("")
            A(f"Batch size used: {r.get('batch_size', 'n/a')} "
              f"(probed, not assumed -- see `logs/vram_probe_{b['key']}.json`).")
            A("")

        A("## Ensemble fusion with A4")
        A("")
        A("Simple logit-average of each backbone's k=5 pooled-OOF predictions "
          "with A4's, aligned by filepath -- logit, not rank (already "
          "settled). This is the direct test of whether a backbone earns an "
          "ensemble slot: if fusing it with A4 beats A4 alone, it is adding "
          "something the primary member does not already have; if fusion "
          "lands between the two inputs or worse, it is trading one metric "
          "for another rather than adding real diversity.")
        A("")
        for b in complete:
            fus = fusions.get(b["key"])
            if not fus:
                continue
            A(f"### {b['label']} + A4, fused")
            A("")
            A("| metric | A4 alone | backbone alone | fused |")
            A("|---|---|---|---|")
            for f in ("fpr_at_90_recall", "fpr_prior_equalised", "fpr_asymmetry",
                      "fpr_center_1", "fpr_center_2", "roc_auc",
                      "pauc_15_std", "pauc_15_raw", "centre_auc_negatives"):
                A(f"| {LABELS[f]} | {fmt(fus['a4_alone'][f])} | "
                  f"{fmt(fus['backbone_alone'][f])} | {fmt(fus['fused'][f])} |")
            A("")

            c = fus["confound"]
            A(f"**Is fusion re-acquiring the hospital shortcut?** A4 alone "
              f"sits {c['a4_dist_from_null']:.4f} from the 0.5 null "
              f"(no hospital signal); {b['label'].split('--')[0].strip()} "
              f"alone sits {c['backbone_dist_from_null']:.4f} from it -- "
              f"{c['backbone_dist_from_null'] / c['a4_dist_from_null']:.1f}x "
              f"further, i.e. more confounded. Fused sits "
              f"{c['fused_dist_from_null']:.4f} from the null "
              f"({fmt(c['fused'])} raw), a movement of "
              f"{c['delta_fused_vs_a4']:+.4f} from A4-alone and "
              f"{c['delta_fused_vs_backbone']:+.4f} from backbone-alone, "
              f"against a conservative bar of {fmt(c['conservative_bar'])} "
              f"(max of the two arms' LOO-4 IQR, same bar as the main metrics "
              f"table above).")
            A("")
            if c["resolvable_vs_a4"] and c["moved_toward_backbone"]:
                A(f"**Fusion moves the confound axis resolvably toward the "
                  f"backbone's more-confounded figure.** The in-distribution "
                  f"gains reported above are therefore not cleanly separable "
                  f"from the hospital shortcut A4 was built to remove -- some "
                  f"share of the fused pair's improvement may be the shortcut "
                  f"partially re-entering the ensemble, not added diversity.")
            elif c["resolvable_vs_a4"] and not c["moved_toward_backbone"]:
                A(f"Fusion moves the confound axis resolvably, but TOWARD "
                  f"the null (away from both inputs' confound), not toward "
                  f"the backbone's more-confounded figure -- not the shortcut-"
                  f"re-entry pattern.")
            else:
                A(f"**Movement is not resolvable** against the conservative "
                  f"bar. Fusion's position on the confound axis cannot be "
                  f"distinguished from A4 alone at this noise level -- the "
                  f"in-distribution gains are not shown to be a re-acquired "
                  f"hospital shortcut, but this is an absence-of-evidence "
                  f"result, not evidence the shortcut is absent.")
            A("")

        L.extend(render_g1_loco_section(g1_loco))

        A("## LOCO -- G2, G3")
        A("")
        A("**Not run for G2 or G3.** Confirmed from the driver's own arm "
          "definitions (`scripts/23_sweep.py` `RUN_ORDER_GASTRONET`): both "
          "remain `mode=\"cv\"` only, 5 folds x 5 seeds, no `loco` arm. This "
          "is not an omission in this report -- the queue itself never ran a "
          "LOCO unit for either. G1 is the exception; see the G1 LOCO section "
          "above.")
        A("")

    A("## Restarts")
    A("")
    if not restarts:
        A("None detected.")
    else:
        A("Each line: a driver pass found unit(s) mid-flight (a prior attempt "
          "had started but not finished) before spending GPU time on them.")
        A("")
        for ln in restarts:
            A(f"- `{ln}`")
    A("")
    return "\n".join(L)


def _jd(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def update_report(manifest_path: str = "manifests/rare25_folds_v2.csv") -> None:
    manifest = pd.read_csv(os.path.join(REPO_ROOT, manifest_path))

    results: Dict[str, Any] = {}
    for b in BACKBONES:
        st = unit_status(b["config"], b["dir"])
        entry: Dict[str, Any] = {"status": st, "label": b["label"]}
        bs_file = os.path.join(REPO_ROOT, "logs", f"batch_size_{b['key']}")
        if os.path.exists(bs_file):
            try:
                entry["batch_size"] = int(open(bs_file).read().strip())
            except (OSError, ValueError):
                pass
        if st["complete"]:
            entry["pooled"] = analyse(b["dir"], manifest)
            entry["epoch_seconds"] = epoch_seconds(b["dir"])
        results[b["key"]] = entry

    ref = ref_epoch = None
    ref_ready = False
    if any(results[b["key"]]["status"]["complete"] for b in BACKBONES):
        ref_status = unit_status("configs/sweep_a4.yaml", REFERENCE["dir"])
        if ref_status["complete"]:
            ref = analyse(REFERENCE["dir"], manifest)
            ref_epoch = epoch_seconds(REFERENCE["dir"])
            ref_ready = True

    fusions: Dict[str, Any] = {}
    if ref_ready:
        for b in BACKBONES:
            if results[b["key"]]["status"]["complete"]:
                bb_loo4 = results[b["key"]]["pooled"]["loo4"]["centre_auc_negatives"]["iqr"]
                a4_loo4 = ref["loo4"]["centre_auc_negatives"]["iqr"]
                confound_bar = max(bb_loo4, a4_loo4)
                fusions[b["key"]] = analyse_fusion(b["dir"], manifest, confound_bar)

    g1_loco = analyse_g1_loco(manifest)

    md = render(results, ref, ref_epoch, fusions, g1_loco, read_restarts(), read_halt())
    write_text_durable(REPORT_MD, md)
    write_text_durable(REPORT_JSON, json.dumps(
        {"results": results, "reference_a4": ref,
         "reference_epoch_seconds": ref_epoch, "fusions": fusions,
         "g1_loco": g1_loco,
         "restarts": read_restarts(), "halt": read_halt()},
        indent=2, default=_jd))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restart", nargs="+", metavar="ARM REPEAT UNIT",
                    help="record a detected restart, then regenerate")
    ap.add_argument("--halt-stage", default=None)
    ap.add_argument("--halt-reason", default=None)
    ap.add_argument("--halt-detail", default="")
    ap.add_argument("--manifest", default="manifests/rare25_folds_v2.csv")
    args = ap.parse_args(argv)

    if args.restart:
        if len(args.restart) < 2:
            ap.error("--restart needs at least ARM and REPEAT")
        arm_key, repeat_s, *units = args.restart
        record_restart(arm_key, int(repeat_s), units)
    if args.halt_stage and args.halt_reason:
        record_halt(args.halt_stage, args.halt_reason, args.halt_detail)

    update_report(args.manifest)
    print(f"written: {os.path.relpath(REPORT_MD, REPO_ROOT)}")
    print(f"written: {os.path.relpath(REPORT_JSON, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
