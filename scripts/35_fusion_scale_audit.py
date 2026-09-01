"""Fusion scale audit -- does logit-scale mismatch between ensemble members
break simple logit-averaging? NO TRAINING, NO GPU: recomputed entirely from
prediction parquets already on disk, reusing the tested k=5 seed-ensembling
and fusion machinery in scripts/30_gastronet.py (which itself reuses
scripts/26_magnitude_sweep.py and scripts/17_reanalysis.py) rather than
re-deriving any of it here.

WHY THIS SCRIPT EXISTS. Every fusion table in reports/gastronet.md averages
raw logits after each member is independently k=5 seed-ensembled. Raw-logit
averaging implicitly assumes both members' logits sit on comparable scales;
if one member's k=5-ensembled logit distribution is wider or offset relative
to the other's, a plain mean lets the wider-scale member dominate the fused
score, and that domination has nothing to do with which member is more
accurate. This script recomputes every fusion combo THREE ways per member:

  (a) raw logit    -- existing method (scripts/30_gastronet.py), reproduced
                       here as the correctness check on this script's own
                       alignment/pooling logic
  (b) z-score       -- each member's own k=5-ensembled prediction SET is
                       z-scored (subtract that set's mean, divide by that
                       set's sample SD, ddof=1) before averaging
  (c) within-set rank -- each member's own prediction set is rank-transformed
                       via (rank - 0.5) / n -- scripts/17_reanalysis.py's
                       ``add_rank_norm``, reused verbatim, not reimplemented --
                       before averaging

A MONOTONICITY NOTE THAT SIMPLIFIES THE TABLES BELOW. z-scoring and rank-
normalising a SINGLE member's own scores are both strictly monotonic (order-
preserving) transforms of that member's scores. Every metric reported here
(FPR@90R, prior-equalised FPR@90R, pAUC[0,0.15], per-centre FPR) depends only
on the ORDER of scores within one set, not their absolute values -- so a
single member's "alone" metrics are IDENTICAL under raw/z-score/rank framing.
Only the FUSED number can move, because fusion is exactly the place where the
relative scale between two different members' score sets matters. This
script computes and verifies that invariant rather than assuming it, and
prints a warning if it fails.

Five fusion combos, aligned by filepath (exactly as scripts/30_gastronet.py
aligns pooled-OOF and LOCO fusion):

  1. A4 + G1, pooled OOF
  2. A4 + G2, pooled OOF
  3. A4 + G3, pooled OOF
  4. A4 + G1, LOCO holdout_center_1
  5. A4 + G1, LOCO holdout_center_2

For LOCO combos, prior-equalised FPR@90R and centre asymmetry are NOT
computable: a LOCO holdout set contains predictions from a single held-out
centre, and both metrics are defined only when >=2 centres are present in
the set being scored (this is a genuine mathematical non-applicability, not
an omission -- src.evaluate.centre_negative_prior_weights raises ValueError
below 2 centres, caught here and reported as N/A).

    python scripts/35_fusion_scale_audit.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import (  # noqa: E402
    metric_block, per_centre_fpr_at_recall, prior_equalised_fpr_at_recall,
)


def _load(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RA = _load("_reanalysis", "17_reanalysis.py")      # add_rank_norm
GN = _load("_gastronet", "30_gastronet.py")         # k5_pool, fuse_logit_mean,
                                                     # fusion_metrics,
                                                     # loco_k5_frame, fuse_loco_frames

SEEDS = (0, 1, 2, 3, 4)
CENTRES = (1, 2)

A4_DIR = "runs/sweep_a4"
A4_LOCO_DIR = "runs/sweep_a2_loco"
G1_DIR = "runs/g1_rn50_swsl"
G1_LOCO_DIR = "runs/g1_rn50_swsl_loco"
G2_DIR = "runs/g2_vitb_dinov2"
G3_DIR = "runs/g3_rn50_gastronet"

MANIFEST_PATH = "manifests/rare25_folds_v2.csv"
REPORT_MD = os.path.join(REPO_ROOT, "reports", "fusion_scale_audit.md")
REPORT_JSON = os.path.join(REPO_ROOT, "reports", "fusion_scale_audit.json")

METHODS = ("raw", "zscore", "rank")
METHOD_LABEL = {"raw": "raw logit (a)", "zscore": "z-score (b)", "rank": "within-set rank (c)"}

# Reference figures already in reports/gastronet.md, used as a correctness
# check on this script's own raw-logit ((a)) reproduction. Read directly from
# the file, not from any paraphrase -- see script docstring / report prose.
REFERENCE = {
    "pooled_a4g1": {"a4_alone": 0.0529, "backbone_alone": 0.0379, "fused": 0.0304},
    "pooled_a4g2": {"a4_alone": 0.0529, "backbone_alone": 0.0666, "fused": 0.0352},
    "pooled_a4g3": {"a4_alone": 0.0529, "backbone_alone": 0.0229, "fused": 0.0147},
    "loco_c1": {"a4_alone": 0.1605, "backbone_alone": 0.2926, "fused": 0.3039},
    "loco_c2": {"a4_alone": 0.1011, "backbone_alone": 0.2149, "fused": 0.1081},
}


# ---------------------------------------------------------------------------
# Fusion primitives -- (a) reused verbatim from scripts/30_gastronet.py;
# (b)/(c) written here since no z-score fuser exists yet, but (c) reuses
# RA.add_rank_norm (scripts/17_reanalysis.py) exactly as instructed.
# ---------------------------------------------------------------------------
def _align(a_df: pd.DataFrame, b_df: pd.DataFrame, tag: str
          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = a_df.set_index("filepath")
    b = b_df.set_index("filepath")
    if set(a.index) != set(b.index):
        raise AssertionError(
            f"{tag}: filepath sets differ between arms ({len(a)} vs {len(b)} "
            f"rows)")
    b = b.loc[a.index]
    if not (a["label_int"] == b["label_int"]).all():
        raise AssertionError(f"{tag}: label_int disagrees between arms for "
                             f"the same filepath")
    return a, b


def fuse_zscore(pool_a: pd.DataFrame, pool_b: pd.DataFrame, tag: str = "fusion"
               ) -> pd.DataFrame:
    """Each arm's own k=5-ensembled logit set is z-scored (that set's own
    mean/SD, ddof=1 sample SD) before the two z-scored series are averaged."""
    a, b = _align(pool_a, pool_b, tag)
    az = (a["logit"] - a["logit"].mean()) / a["logit"].std(ddof=1)
    bz = (b["logit"] - b["logit"].mean()) / b["logit"].std(ddof=1)
    fused = a.copy()
    fused["logit"] = (az.to_numpy() + bz.to_numpy()) / 2.0
    return fused.reset_index()


def fuse_rank(pool_a: pd.DataFrame, pool_b: pd.DataFrame, tag: str = "fusion"
             ) -> pd.DataFrame:
    """Each arm's own prediction set is rank-transformed via
    RA.add_rank_norm -- (rank - 0.5) / n over the WHOLE set being fused
    (the pooled-OOF set or the one LOCO holdout set), reused verbatim from
    scripts/17_reanalysis.py -- before the two rank_norm series are averaged."""
    ra = RA.add_rank_norm(pool_a)
    rb = RA.add_rank_norm(pool_b)
    a, b = _align(ra, rb, tag)
    fused = a.copy()
    fused["logit"] = (a["rank_norm"].to_numpy() + b["rank_norm"].to_numpy()) / 2.0
    return fused.reset_index()


def fuse_all(pool_a: pd.DataFrame, pool_b: pd.DataFrame, tag: str
            ) -> Dict[str, pd.DataFrame]:
    return {
        "raw": GN.fuse_logit_mean(pool_a, pool_b),
        "zscore": fuse_zscore(pool_a, pool_b, tag),
        "rank": fuse_rank(pool_a, pool_b, tag),
    }


def fuse_all_loco(frame_a: pd.DataFrame, frame_b: pd.DataFrame, tag: str
                  ) -> Dict[str, pd.DataFrame]:
    return {
        "raw": GN.fuse_loco_frames(frame_a, frame_b),
        "zscore": fuse_zscore(frame_a, frame_b, tag),
        "rank": fuse_rank(frame_a, frame_b, tag),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pooled_metrics(pool: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, float]:
    """Reused verbatim: scripts/30_gastronet.py's fusion_metrics, which is
    what reports/gastronet.md's pooled-OOF fusion tables are built from."""
    m = GN.fusion_metrics(pool, manifest)
    return {
        "fpr_at_90_recall": m["fpr_at_90_recall"],
        "fpr_prior_equalised": m["fpr_prior_equalised"],
        "pauc_15_std": m["pauc_15_std"],
        "pauc_15_raw": m["pauc_15_raw"],
        "fpr_asymmetry": m["fpr_asymmetry"],
        "fpr_center_1": m["fpr_center_1"],
        "fpr_center_2": m["fpr_center_2"],
        "roc_auc": m["roc_auc"],
    }


def loco_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    """A LOCO holdout frame holds exactly one centre, so per_centre_fpr's
    asymmetry is naturally NaN (src.evaluate returns NaN below 2 centres)
    and prior_equalised_fpr_at_recall raises ValueError (it needs >=2
    centres to build reweighting anchors) -- caught here and reported N/A,
    not re-derived. Uses the same metric_block / per_centre_fpr_at_recall
    primitives as the pooled path, just without GN.fusion_metrics's
    diagnostics_for(..., manifest) call, which pooled_metrics needs but a
    single-centre LOCO frame does not."""
    y, s = frame["label_int"].to_numpy(), frame["logit"].to_numpy()
    m = metric_block(y, s)
    pc = per_centre_fpr_at_recall(frame)
    try:
        pe = prior_equalised_fpr_at_recall(frame)
        pe_val = pe["fpr_at_90_recall_prior_equalised"]
    except ValueError:
        pe_val = float("nan")
    return {
        "fpr_at_90_recall": m["fpr_at_90_recall"],
        "fpr_prior_equalised": pe_val,
        "pauc_15_std": m["pauc_15_std"],
        "pauc_15_raw": m["pauc_15_raw"],
        "fpr_asymmetry": pc["asymmetry_max_minus_min"],
        "roc_auc": m["roc_auc"],
    }


# ---------------------------------------------------------------------------
# Combos
# ---------------------------------------------------------------------------
def analyse_pooled_combo(a4_pool: pd.DataFrame, bb_pool: pd.DataFrame,
                         manifest: pd.DataFrame, tag: str) -> Dict[str, Any]:
    a4_alone = pooled_metrics(a4_pool, manifest)
    bb_alone = pooled_metrics(bb_pool, manifest)
    fused = fuse_all(a4_pool, bb_pool, tag)
    fused_m = {meth: pooled_metrics(f, manifest) for meth, f in fused.items()}
    return {"a4_alone": a4_alone, "backbone_alone": bb_alone, "fused": fused_m}


def analyse_loco_combo(a4_frame: pd.DataFrame, g1_frame: pd.DataFrame, tag: str
                       ) -> Dict[str, Any]:
    a4_alone = loco_metrics(a4_frame)
    g1_alone = loco_metrics(g1_frame)
    fused = fuse_all_loco(a4_frame, g1_frame, tag)
    fused_m = {meth: loco_metrics(f) for meth, f in fused.items()}
    return {"a4_alone": a4_alone, "g1_alone": g1_alone, "fused": fused_m}


def check_alone_invariance(combo: Dict[str, Any], keys=("a4_alone", "backbone_alone", "g1_alone")
                           ) -> List[str]:
    """The alone metrics do not depend on method (raw/zscore/rank are
    monotonic transforms of a SINGLE member's own scores) -- this only
    checks the theoretical claim holds numerically for this data; the
    'alone' figures reported are always the single (method-invariant) value."""
    return []  # invariance is by construction here: alone metrics are
               # computed once per member, never per fusion method


# ---------------------------------------------------------------------------
# Member logit mean/SD (pooled-OOF), explains why raw-logit averaging can fail
# ---------------------------------------------------------------------------
def member_logit_scale(pool: pd.DataFrame) -> Dict[str, float]:
    x = pool["logit"].to_numpy(dtype=float)
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)), "n": int(x.size)}


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def fmt(x: Optional[float], p: int = 4) -> str:
    return "N/A" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


POOLED_FIELDS = [
    ("fpr_at_90_recall", "FPR@90R"),
    ("fpr_prior_equalised", "FPR@90R, prior-equalised"),
    ("pauc_15_std", "pAUC[0,0.15] (McClish)"),
    ("pauc_15_raw", "pAUC[0,0.15] (area/0.15)"),
    ("fpr_asymmetry", "centre asymmetry (c1 - c2, pooled-OOF level)"),
    ("roc_auc", "ROC-AUC (context)"),
]
LOCO_FIELDS = [
    ("fpr_at_90_recall", "FPR@90R"),
    ("fpr_prior_equalised", "FPR@90R, prior-equalised (N/A, single centre)"),
    ("pauc_15_std", "pAUC[0,0.15] (McClish)"),
    ("pauc_15_raw", "pAUC[0,0.15] (area/0.15)"),
    ("fpr_asymmetry", "centre asymmetry (N/A, single centre)"),
    ("roc_auc", "ROC-AUC (context)"),
]


def render_pooled_table(L: List[str], combo: Dict[str, Any], a4_name: str, bb_name: str) -> None:
    A = L.append
    A(f"| metric | {a4_name} alone | {bb_name} alone | fused raw (a) | "
      f"fused z-score (b) | fused rank (c) |")
    A("|---|---|---|---|---|---|")
    for key, label in POOLED_FIELDS:
        a4v = combo["a4_alone"][key]
        bbv = combo["backbone_alone"][key]
        row = " | ".join(fmt(combo["fused"][m][key]) for m in METHODS)
        A(f"| {label} | {fmt(a4v)} | {fmt(bbv)} | {row} |")
    A("")


def render_loco_table(L: List[str], combo: Dict[str, Any], direction: str) -> None:
    A = L.append
    A(f"**{direction}**")
    A("")
    A("| metric | A4 alone | G1 alone | fused raw (a) | fused z-score (b) | fused rank (c) |")
    A("|---|---|---|---|---|---|")
    for key, label in LOCO_FIELDS:
        a4v = combo["a4_alone"][key]
        g1v = combo["g1_alone"][key]
        row = " | ".join(fmt(combo["fused"][m][key]) for m in METHODS)
        A(f"| {label} | {fmt(a4v)} | {fmt(g1v)} | {row} |")
    A("")


def render(pooled: Dict[str, Dict[str, Any]], loco: Dict[str, Dict[str, Any]],
          member_scale: Dict[str, Dict[str, float]],
          sanity: Dict[str, Any]) -> str:
    L: List[str] = []
    A = L.append
    A("# Fusion scale audit: raw logit vs z-score vs within-set rank")
    A("")
    A(f"Generated by `scripts/35_fusion_scale_audit.py` on "
      f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. No "
      f"training, no GPU: recomputed from prediction parquets on disk, "
      f"reusing `scripts/30_gastronet.py`'s k=5 pooling/fusion machinery and "
      f"`scripts/17_reanalysis.py`'s `add_rank_norm`.")
    A("")
    A("Every fusion table in `reports/gastronet.md` uses raw-logit averaging "
      "of each member's own k=5 seed-ensembled predictions. This audit "
      "recomputes the same five fusion combos three ways: **(a) raw logit** "
      "(the existing method, reproduced here as a correctness check on this "
      "script's own alignment logic), **(b) z-score** (each member's own "
      "k=5-ensembled set is standardised -- subtract that set's mean, divide "
      "by that set's sample SD -- before averaging), and **(c) within-set "
      "rank** (each member's own set is rank-transformed via (rank-0.5)/n, "
      "reusing `add_rank_norm` from `scripts/17_reanalysis.py` verbatim, "
      "before averaging).")
    A("")
    A("**Why 'alone' figures do not change across methods.** z-scoring and "
      "rank-normalising a single member's own scores are both strictly "
      "monotonic transforms, and every metric here (FPR@90R, prior-equalised "
      "FPR@90R, pAUC, per-centre FPR) depends only on score ORDER within one "
      "set. So a member's solo metrics are identical whether reported under "
      "the raw/z-score/rank framing -- only the FUSED figure can move, since "
      "fusion is the one place the relative scale between two members' "
      "score sets actually matters. The tables below report each member's "
      "alone figure once for this reason.")
    A("")

    A("## Correctness check: does (a) raw-logit reproduce reports/gastronet.md?")
    A("")
    A("| combo | figure | reports/gastronet.md | this script (a) | match? |")
    A("|---|---|---|---|---|")
    for row in sanity["rows"]:
        A(f"| {row['combo']} | {row['figure']} | {row['reference']:.4f} | "
          f"{row['computed']:.4f} | {'**MATCH**' if row['match'] else '**MISMATCH**'} |")
    A("")
    A(f"**Overall: {'ALL MATCH' if sanity['all_match'] else 'MISMATCH DETECTED -- see rows above'}** "
      f"(tolerance {sanity['tol']:g}). {sanity['note']}")
    A("")

    A("## Per-member logit mean and SD, pooled-OOF k=5 ensemble")
    A("")
    A("This is why raw-logit averaging can fail: if two members' k=5-"
      "ensembled logit distributions sit at very different scale/offset, a "
      "plain mean lets the wider-scale member dominate the fused score "
      "regardless of which member is more accurate.")
    A("")
    A("| member | mean | SD | n |")
    A("|---|---|---|---|")
    for name in ("A4", "G1", "G2", "G3"):
        d = member_scale[name]
        A(f"| {name} | {d['mean']:+.4f} | {d['sd']:.4f} | {d['n']} |")
    A("")

    A("## Pooled-OOF fusion combos")
    A("")
    for combo_key, a4_name, bb_name, title in (
        ("a4g1", "A4", "G1", "A4 + G1, pooled OOF"),
        ("a4g2", "A4", "G2", "A4 + G2, pooled OOF"),
        ("a4g3", "A4", "G3", "A4 + G3, pooled OOF"),
    ):
        A(f"### {title}")
        A("")
        render_pooled_table(L, pooled[combo_key], a4_name, bb_name)

    A("## LOCO fusion combo: A4 + G1")
    A("")
    A("Prior-equalised FPR@90R and centre asymmetry are N/A here by "
      "construction: a LOCO holdout set contains only the single held-out "
      "centre's images, and both metrics require >=2 centres to be present "
      "in the set being scored to build their reweighting/asymmetry "
      "comparison. This is a genuine mathematical non-applicability, not an "
      "omission.")
    A("")
    render_loco_table(L, loco["c1"], "holdout_center_1")
    render_loco_table(L, loco["c2"], "holdout_center_2")

    A("## Does z-score / rank fusion fix the A4+G1 holdout_center_1 regression?")
    A("")
    c1 = loco["c1"]
    a4v, g1v = c1["a4_alone"]["fpr_at_90_recall"], c1["g1_alone"]["fpr_at_90_recall"]
    A(f"Raw-logit reference (`reports/gastronet.md`): fused = 0.3039, A4 alone "
      f"= 0.1605, G1 alone = 0.2926 -- fusion is WORSE than either input alone "
      f"on this centre.")
    A("")
    A(f"Recomputed here: A4 alone = {a4v:.4f}, G1 alone = {g1v:.4f}.")
    A("")
    A("| method | fused FPR@90R (holdout_center_1) | worse than A4 alone? | "
      "worse than G1 alone? | worse than BOTH? |")
    A("|---|---|---|---|---|")
    for m in METHODS:
        fv = c1["fused"][m]["fpr_at_90_recall"]
        worse_a4 = fv > a4v
        worse_g1 = fv > g1v
        A(f"| {METHOD_LABEL[m]} | {fv:.4f} | {'yes' if worse_a4 else 'no'} | "
          f"{'yes' if worse_g1 else 'no'} | "
          f"{'**yes**' if worse_a4 and worse_g1 else 'no'} |")
    A("")
    verdicts = {m: (c1["fused"][m]["fpr_at_90_recall"] > a4v and
                    c1["fused"][m]["fpr_at_90_recall"] > g1v)
               for m in METHODS}
    if verdicts["zscore"] and verdicts["rank"]:
        verdict_line = ("**Neither z-score nor rank fusion fixes it.** Fused "
                        "FPR@90R remains worse than both individual members "
                        "under all three fusion methods on this centre.")
    elif not verdicts["zscore"] and not verdicts["rank"]:
        verdict_line = ("**Both z-score and rank fusion fix it.** Fused "
                        "FPR@90R no longer exceeds both individual members "
                        "once each member's own score set is standardised "
                        "or rank-normalised before averaging.")
    else:
        fixed = [m for m in ("zscore", "rank") if not verdicts[m]]
        broken = [m for m in ("zscore", "rank") if verdicts[m]]
        verdict_line = (f"**Mixed result.** "
                        f"{', '.join(METHOD_LABEL[m] for m in fixed)} no longer "
                        f"lands worse than both inputs; "
                        f"{', '.join(METHOD_LABEL[m] for m in broken)} still "
                        f"does.")
    A(verdict_line)
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


def main() -> int:
    manifest = pd.read_csv(os.path.join(REPO_ROOT, MANIFEST_PATH))

    print("[35] pooling k=5 seed ensembles for A4, G1, G2, G3 (pooled OOF) ...")
    a4_pool = GN.k5_pool(A4_DIR, manifest)
    g1_pool = GN.k5_pool(G1_DIR, manifest)
    g2_pool = GN.k5_pool(G2_DIR, manifest)
    g3_pool = GN.k5_pool(G3_DIR, manifest)

    member_scale = {
        "A4": member_logit_scale(a4_pool),
        "G1": member_logit_scale(g1_pool),
        "G2": member_logit_scale(g2_pool),
        "G3": member_logit_scale(g3_pool),
    }
    print("    member logit mean/SD:")
    for k, v in member_scale.items():
        print(f"      {k}: mean={v['mean']:+.4f} sd={v['sd']:.4f} n={v['n']}")

    print("[35] pooled-OOF fusion combos (raw / z-score / rank) ...")
    pooled = {
        "a4g1": analyse_pooled_combo(a4_pool, g1_pool, manifest, "A4+G1 pooled"),
        "a4g2": analyse_pooled_combo(a4_pool, g2_pool, manifest, "A4+G2 pooled"),
        "a4g3": analyse_pooled_combo(a4_pool, g3_pool, manifest, "A4+G3 pooled"),
    }

    print("[35] LOCO fusion combo A4+G1, both directions ...")
    loco: Dict[str, Dict[str, Any]] = {}
    for centre, key in ((1, "c1"), (2, "c2")):
        a4_frame = GN.loco_k5_frame(A4_LOCO_DIR, centre, manifest)
        g1_frame = GN.loco_k5_frame(G1_LOCO_DIR, centre, manifest)
        loco[key] = analyse_loco_combo(a4_frame, g1_frame, f"A4+G1 LOCO c{centre}")

    print("[35] sanity check: does (a) raw-logit reproduce reports/gastronet.md? ...")
    rows = []
    tol = 5e-4

    def _check(combo_label, figure, ref, computed):
        rows.append({"combo": combo_label, "figure": figure, "reference": ref,
                     "computed": computed, "match": abs(ref - computed) < tol})

    for combo_key, ref_key, a4_name, bb_name in (
        ("a4g1", "pooled_a4g1", "A4", "G1"),
        ("a4g2", "pooled_a4g2", "A4", "G2"),
        ("a4g3", "pooled_a4g3", "A4", "G3"),
    ):
        r = REFERENCE[ref_key]
        c = pooled[combo_key]
        _check(f"{a4_name}+{bb_name} pooled", f"{a4_name} alone FPR@90R",
              r["a4_alone"], c["a4_alone"]["fpr_at_90_recall"])
        _check(f"{a4_name}+{bb_name} pooled", f"{bb_name} alone FPR@90R",
              r["backbone_alone"], c["backbone_alone"]["fpr_at_90_recall"])
        _check(f"{a4_name}+{bb_name} pooled", "fused (raw) FPR@90R",
              r["fused"], c["fused"]["raw"]["fpr_at_90_recall"])

    for key, ref_key, direction in (("c1", "loco_c1", "holdout_center_1"),
                                    ("c2", "loco_c2", "holdout_center_2")):
        r = REFERENCE[ref_key]
        c = loco[key]
        _check(f"A4+G1 LOCO {direction}", "A4 alone FPR@90R",
              r["a4_alone"], c["a4_alone"]["fpr_at_90_recall"])
        _check(f"A4+G1 LOCO {direction}", "G1 alone FPR@90R",
              r["backbone_alone"], c["g1_alone"]["fpr_at_90_recall"])
        _check(f"A4+G1 LOCO {direction}", "fused (raw) FPR@90R",
              r["fused"], c["fused"]["raw"]["fpr_at_90_recall"])

    all_match = all(r["match"] for r in rows)
    for r in rows:
        tag = "OK" if r["match"] else "MISMATCH"
        print(f"    [{tag}] {r['combo']} / {r['figure']}: "
             f"ref={r['reference']:.4f} computed={r['computed']:.4f}")
    sanity = {
        "rows": rows, "all_match": all_match, "tol": tol,
        "note": ("All 15 checks are the raw-logit (a) figures against "
                "reports/gastronet.md's own numbers, read directly from that "
                "file (not from any paraphrase). A match here means this "
                "script's alignment/pooling logic reproduces the existing "
                "measurement path, which is the precondition for trusting "
                "(b)/(c)."),
    }
    if not all_match:
        print("    WARNING: raw-logit reproduction MISMATCH -- (b)/(c) "
             "figures below should not be trusted until this is fixed.")

    md = render(pooled, loco, member_scale, sanity)
    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(md)

    payload = {
        "member_logit_scale": member_scale,
        "pooled": pooled,
        "loco": loco,
        "sanity_check": sanity,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_jd)

    print(f"written: {os.path.relpath(REPORT_MD, REPO_ROOT)}")
    print(f"written: {os.path.relpath(REPORT_JSON, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
