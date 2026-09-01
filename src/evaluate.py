"""Three reporting levels over the prediction dumps, and the noise floor.

THE DISTINCTION BETWEEN THE LEVELS IS THE POINT OF THIS MODULE.

LEVEL 1 -- per fold. Logged, never selected on. A validation fold holds ~31
    positives and 586 negatives, so 90% recall is the operating point set by
    three images and FPR moves in steps of 1/586 = 0.0017. Every Level-1 number
    is printed with that granularity next to it so nobody reads a 0.01 movement
    as signal.

LEVEL 2 -- pooled out-of-fold, per (repeat, seed). THE PRIMARY UNIT OF
    MEASUREMENT. All five folds' validation logits concatenated into one
    3088-row set (158 positives, 2930 negatives), each image appearing exactly
    once. The threshold and FPR are computed ON THAT POOL.

    Averaging the five per-fold FPRs is a DIFFERENT STATISTIC and it is wrong.
    Not merely noisier -- different. Each fold's FPR is measured at that fold's
    own threshold, so the average is "mean FPR across five different operating
    points", which corresponds to no single decision rule. The pooled FPR is the
    false-positive rate of one threshold applied to every image once, which is
    what deploying a model actually does. pooled_vs_averaged() computes both and
    reports the gap, because the two are easy to confuse and only one is
    meaningful. tests/test_evaluate.py pins a case where they differ.

LEVEL 3 -- across repeats and seeds. Median and IQR of every Level-2 figure.
    No single number is reported anywhere without its spread.

METRIC CONVENTIONS, stated because they are ambiguous otherwise:

  * pAUC over FPR [0, 0.15] is reported in BOTH conventions on every line.
    ``pauc_15_raw`` divides the restricted area by 0.15, so a random ranker
    scores 0.075 and a perfect one 1.0. ``pauc_15_std`` is the McClish
    standardisation, rescaled so random = 0.5 and perfect = 1.0. Both describe
    the same area. 0.075 and 0.5 both mean "no signal", which is exactly why the
    convention has to travel with the number.

  * FPR@90R is PRIMARY. It is prevalence-invariant -- FP/(FP+TN) depends only on
    the negatives' score distribution and where the recall threshold falls -- so
    it is comparable between a 5.1%-positive local pool and a ~1%-positive
    leaderboard, which PPV is not.

  * PPV_leaderboard is a PROJECTION, not a measurement. It maps a local FPR onto
    the competition's test shape (232 positives, 23,176 negatives) analytically.
    It measures nothing about the competition data; it restates local FPR in
    units that are easier to argue about.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.io import read_predictions  # noqa: E402
from src.metrics import (  # noqa: E402
    PAUC_MAX_FPR, TARGET_RECALL, fpr_at_recall, partial_auc, ppv_at_recall,
    roc_auc,
)

# Competition test-set shape, used only by the PPV projection.
LEADERBOARD_POS = 232
LEADERBOARD_NEG = 23176

# |rho| at or above this is called out as a possible hospital-identity shortcut.
SHORTCUT_RHO = 0.30

# Manifest columns the diagnostics correlate the logit against. Every one of
# these is a DIAGNOSTIC column per manifests/DATA_README.md -- none is a model
# input -- so a strong correlation means the model reconstructed it from pixels.
CONFOUND_COLS = ("rg_ratio", "file_size_bytes", "fov_radius", "has_redaction")
DIAGNOSTIC_JOIN_COLS = CONFOUND_COLS + ("centre", "visibility", "class_label")

VISIBILITY_ORDER = ("obvious", "moderate", "would_have_missed")


# ---------------------------------------------------------------------------
# The official scorer, loaded verbatim rather than reimplemented
# ---------------------------------------------------------------------------
def load_official_score():
    """Import ``official_score`` straight out of scripts/08_score.py.

    Loaded by path because the filename starts with a digit and cannot be a
    normal import. The point is that the bootstrap figure in this report comes
    from the organisers' own code, byte for byte, and not from a local
    re-derivation that might drift from it.
    """
    path = os.path.join(REPO_ROOT, "scripts", "08_score.py")
    spec = importlib.util.spec_from_file_location("_official_score", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.official_score


# ---------------------------------------------------------------------------
# Loading and pooling
# ---------------------------------------------------------------------------
def load_unit(out_dir: str, name: str) -> pd.DataFrame:
    """One unit's canonical (last-epoch) prediction dump."""
    path = os.path.join(out_dir, name, f"val_{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return read_predictions(path)


def pool_oof(
    out_dir: str, repeat: int, seed: int, folds: Sequence[int] = (0, 1, 2, 3, 4),
    manifest: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Concatenate one (repeat, seed)'s five folds into the out-of-fold set.

    The coverage assertions are not decoration. A pooled statistic silently
    stops meaning what it claims the moment an image appears twice or goes
    missing, and both are one-line mistakes to make. So this verifies that every
    usable manifest row appears EXACTLY once before returning anything.
    """
    parts = [load_unit(out_dir, f"r{repeat}_f{f}_s{seed}") for f in folds]
    pool = pd.concat(parts, ignore_index=True)

    dupes = pool["filepath"].duplicated()
    if dupes.any():
        raise AssertionError(
            f"pooled OOF r{repeat} s{seed}: {int(dupes.sum())} filepaths appear "
            f"more than once -- folds overlap, the pool is not out-of-fold"
        )
    if manifest is not None:
        usable = set(manifest.loc[manifest[f"fold_r{repeat}"] != -1, "filepath"])
        got = set(pool["filepath"])
        if got != usable:
            raise AssertionError(
                f"pooled OOF r{repeat} s{seed}: covers {len(got)} rows but the "
                f"manifest has {len(usable)} usable; missing "
                f"{len(usable - got)}, unexpected {len(got - usable)}"
            )
    seeds = pool["seed"].unique()
    if len(seeds) != 1 or int(seeds[0]) != seed:
        raise AssertionError(f"pooled OOF mixes seeds: {seeds.tolist()}")
    return pool


# ---------------------------------------------------------------------------
# The metric block, computed identically at every level
# ---------------------------------------------------------------------------
def ppv_leaderboard(fpr: float) -> float:
    """Project a local FPR onto the competition's test shape.

    PROJECTION, NOT MEASUREMENT. Assumes the model's FPR on 23,176 unseen
    competition negatives equals its FPR here, and that recall lands at exactly
    0.90 there. Neither is measurable locally. What this genuinely provides is a
    monotone restatement of FPR in leaderboard units -- it adds no information
    to FPR and cannot disagree with it about which of two models is better.
    """
    tp = TARGET_RECALL * LEADERBOARD_POS
    fp = fpr * LEADERBOARD_NEG
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def operating_point_detail(y: np.ndarray, s: np.ndarray) -> Dict[str, Any]:
    """How coarse is FPR@90R on THIS set?

    0.9 * n_pos is essentially never an integer, so the reported FPR is a linear
    interpolation between two achievable operating points that differ by one
    true positive. Reporting that bracket alongside the number is what stops a
    reader treating a movement smaller than one image as a result.
    """
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"n_pos": n_pos, "n_neg": n_neg}
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, s)
    tp_needed = TARGET_RECALL * n_pos
    lo_tp, hi_tp = int(np.floor(tp_needed)), int(np.ceil(tp_needed))
    out: Dict[str, Any] = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "fpr_step_one_fp": 1.0 / n_neg,
        "recall_step_one_pos": 1.0 / n_pos,
        "tp_needed": tp_needed,
        "tp_bracket": [lo_tp, hi_tp],
    }
    for tag, k in (("lo", lo_tp), ("hi", hi_tp)):
        i = int(np.searchsorted(tpr, k / n_pos, side="left"))
        i = min(i, len(fpr) - 1)
        out[f"fpr_at_tp_{tag}"] = float(fpr[i])
        out[f"fp_at_tp_{tag}"] = int(round(fpr[i] * n_neg))
    out["fpr_bracket_width"] = abs(out["fpr_at_tp_hi"] - out["fpr_at_tp_lo"])
    return out


def metric_block(
    y: np.ndarray, s: np.ndarray, bootstrap: bool = False,
    n_boot: int = 1000, boot_seed: int = 0,
) -> Dict[str, Any]:
    """Every headline metric for one set of labels and logits.

    ``bootstrap`` runs the organisers' scorer (1000 prevalence-matched draws);
    it is off by default because it costs seconds, not milliseconds, and Level 1
    does not need it.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    pauc = partial_auc(y, s, PAUC_MAX_FPR)
    fpr90 = fpr_at_recall(y, s, TARGET_RECALL)
    out: Dict[str, Any] = {
        "n": int(y.size),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "prevalence": float((y == 1).mean()),
        "roc_auc": roc_auc(y, s),
        "pauc_15_raw": pauc["raw"],
        "pauc_15_std": pauc["std"],
        "fpr_at_90_recall": fpr90,
        "ppv_at_90_recall_local": ppv_at_recall(y, s, TARGET_RECALL),
        "ppv_leaderboard_projected": ppv_leaderboard(fpr90),
        "operating_point": operating_point_detail(y, s),
    }
    if bootstrap:
        official_score = load_official_score()
        r = official_score(y, s, n_iterations=n_boot, seed=boot_seed)
        out["bootstrap_official"] = {
            "median_ppv": float(r["Score"]),
            "ci_lo": float(r["PPV@90RECALL CI Lower"]),
            "ci_hi": float(r["PPV@90RECALL CI Upper"]),
            "iqr": float(r["PPV@90RECALL CI Upper"] - r["PPV@90RECALL CI Lower"]),
            "auroc_median": float(r["AUROC"]),
            "auprc_median": float(r["AUPRC"]),
            "ppv_full": float(r["PPV@90RECALL Full"]),
            "n_iterations": n_boot,
        }
    return out


# ---------------------------------------------------------------------------
# LEVEL 1 -- per fold
# ---------------------------------------------------------------------------
def level1(out_dir: str, repeat: int, seed: int,
           folds: Sequence[int] = (0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Per-fold metrics. LOG ONLY -- never select a model or a change on these."""
    rows = []
    for f in folds:
        df = load_unit(out_dir, f"r{repeat}_f{f}_s{seed}")
        m = metric_block(df["label_int"].to_numpy(), df["logit"].to_numpy())
        rows.append({
            "repeat": repeat, "seed": seed, "fold": f,
            "n": m["n"], "n_pos": m["n_pos"],
            "roc_auc": m["roc_auc"], "pauc_15_std": m["pauc_15_std"],
            "fpr_at_90_recall": m["fpr_at_90_recall"],
            "ppv_at_90_recall_local": m["ppv_at_90_recall_local"],
            "fpr_step_one_fp": m["operating_point"]["fpr_step_one_fp"],
            "tp_bracket_lo": m["operating_point"]["tp_bracket"][0],
            "tp_bracket_hi": m["operating_point"]["tp_bracket"][1],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LEVEL 2 -- pooled out-of-fold
# ---------------------------------------------------------------------------
def level2(pool: pd.DataFrame, bootstrap: bool = True,
           n_boot: int = 1000) -> Dict[str, Any]:
    """The primary measurement: one threshold over all 3088 images."""
    return metric_block(
        pool["label_int"].to_numpy(), pool["logit"].to_numpy(),
        bootstrap=bootstrap, n_boot=n_boot,
        boot_seed=int(pool["seed"].iloc[0]),
    )


def pooled_vs_averaged(pool: pd.DataFrame, per_fold: pd.DataFrame) -> Dict[str, Any]:
    """Quantify the difference between the right statistic and the wrong one.

    Reported so the choice to pool is defensible with a number rather than an
    assertion. ``mean_of_fold_fpr`` is the quantity this codebase must never
    report as its FPR; it appears here only as the thing being ruled out.
    """
    y, s = pool["label_int"].to_numpy(), pool["logit"].to_numpy()
    pooled = fpr_at_recall(y, s, TARGET_RECALL)
    mean_fold = float(per_fold["fpr_at_90_recall"].mean())
    return {
        "pooled_fpr": pooled,
        "mean_of_fold_fpr": mean_fold,
        "absolute_difference": abs(pooled - mean_fold),
        "fold_fprs": per_fold["fpr_at_90_recall"].tolist(),
        "note": ("mean_of_fold_fpr measures five different thresholds and "
                 "corresponds to no single decision rule; it is shown only to "
                 "quantify what pooling avoids"),
    }


def fold_contribution(pool: pd.DataFrame) -> Dict[str, Any]:
    """Is the pooled threshold dominated by one fold's logit scale?

    POOLING VALIDITY CHECK. The five folds are five separately trained models,
    and concatenating their raw logits assumes those logits are on a comparable
    scale. If fold 3's outputs run systematically high, fold 3 supplies most of
    the false positives at the pooled threshold and the pooled FPR is partly a
    statement about calibration drift rather than about discrimination.

    Under comparable scales each fold contributes about 1/5 of the false
    positives, in proportion to its share of the negatives. A large deviation
    does not invalidate the pooled number -- pooled FPR is still the honest
    false-positive rate of one threshold -- but it does mean seed-to-seed
    movement partly reflects recalibration, so it is reported, not corrected.
    """
    y, s = pool["label_int"].to_numpy(), pool["logit"].to_numpy()
    thr = threshold_at_recall(y, s, TARGET_RECALL)
    neg = pool[pool["label_int"] == 0]
    fp = neg[neg["logit"] >= thr]
    share = (fp.groupby("fold").size() / max(1, len(fp))).to_dict()
    expected = (neg.groupby("fold").size() / max(1, len(neg))).to_dict()
    per_fold_logits = pool.groupby("fold")["logit"].agg(["mean", "std"]).to_dict("index")
    dev = {int(k): float(share.get(k, 0.0) - expected.get(k, 0.0)) for k in expected}
    return {
        "threshold": float(thr),
        "n_false_positives": int(len(fp)),
        "fp_share_by_fold": {int(k): float(v) for k, v in share.items()},
        "expected_share_by_fold": {int(k): float(v) for k, v in expected.items()},
        "share_deviation_by_fold": dev,
        "max_abs_deviation": float(max(abs(v) for v in dev.values())) if dev else 0.0,
        "logit_mean_sd_by_fold": {
            int(k): {"mean": float(v["mean"]), "sd": float(v["std"])}
            for k, v in per_fold_logits.items()
        },
    }


def threshold_at_recall(y: np.ndarray, s: np.ndarray,
                        recall: float = TARGET_RECALL) -> float:
    """Lowest score still counted positive at >= `recall`.

    Uses the score of the k-th highest positive where k = ceil(recall * n_pos),
    i.e. the smallest achievable operating point that MEETS the recall target
    rather than interpolating past it. Interpolation is right for reporting a
    continuous FPR; a threshold has to be an actual attainable score.
    """
    pos = np.sort(s[y == 1])[::-1]
    k = int(np.ceil(recall * pos.size))
    return float(pos[min(k, pos.size) - 1])


# ---------------------------------------------------------------------------
# PRIOR-EQUALISED FPR@90R and PER-CENTRE FPR@90R
# ---------------------------------------------------------------------------
# Pre-registered 2026-07-29 (see reports/prior_equalised_fpr_pre_registration.md
# for the timestamped statement this implements). Pooled FPR@90R is the primary
# metric everywhere else in this file, and it has a structural blind spot: it
# is computed on the pool's OWN centre-conditioned class balance. On the
# repeat-0 pooled-OOF validation set, center_1 supplies 61/2279 = 2.7% positives
# and center_2 supplies 97/809 = 12.0% -- so "this looks like center_2" is a
# VALID predictor of the label on THIS pool. A model that partly reads hospital
# identity is REWARDED by pooled FPR@90R, not merely un-penalised, because the
# shortcut and the pathology signal are entangled in the pool being scored.
# Twelve unseen evaluation centres will not share this particular entanglement,
# so a shortcut that is free here is not free there.
#
# These two metrics measure what pooled FPR@90R structurally cannot:
#   * prior-equalised FPR@90R strips the entanglement by reweighting negatives
#     so hospital identity carries zero label information, then asks what FPR
#     survives;
#   * per-centre FPR@90R asks the different question of whether ONE deployed
#     threshold treats the two hospitals' negatives asymmetrically, which a
#     model with no hospital signal has no mechanism to do.
def centre_negative_prior_weights(pool: pd.DataFrame) -> Dict[str, Any]:
    """Per-centre weight for NEGATIVE images making
    P(centre | weighted negative) == P(centre | positive) exactly.

    ANCHOR SELECTION. A weight vector achieving that equality is defined only
    up to an overall scale -- the equalised PROPORTION is unchanged by
    multiplying every weight by the same constant -- so one centre's weight
    has to be pinned to fix the scale. The centre with the SMALLEST
    negative:positive ratio is used as that anchor (weight 1.0, negatives used
    exactly as they are); every other centre's negatives are scaled DOWN to
    match that same ratio. Every returned weight therefore lies in (0, 1]: no
    centre's negatives are ever upweighted or invented, only the excess in a
    more negative-heavy centre is discounted. That is what makes the companion
    bootstrap a SUBSAMPLING procedure rather than a resampling one -- the
    anchor centre's negatives are always used in full, every draw.

    On the repeat-0 pooled-OOF validation set (identical across every arm of
    this sweep -- same manifest, same splits, every seed): 61 center_1 / 97
    center_2 positives, 2218 center_1 / 712 center_2 negatives. center_2 has
    the smaller neg:pos ratio (7.34 vs 36.36) and is the anchor; center_1's
    negatives are weighted 0.2019. tests/test_prior_equalised.py pins these
    exact numbers.
    """
    npos = pool.loc[pool["label_int"] == 1, "centre"].value_counts()
    nneg = pool.loc[pool["label_int"] == 0, "centre"].value_counts()
    centres = sorted(set(npos.index) & set(nneg.index))
    if len(centres) < 2:
        raise ValueError(
            f"prior-equalisation needs >= 2 centres with both classes "
            f"present, found {centres}"
        )
    ratio = {c: float(nneg[c]) / float(npos[c]) for c in centres}
    anchor = min(ratio, key=ratio.get)
    target_ratio = ratio[anchor]
    weights = {c: target_ratio * float(npos[c]) / float(nneg[c])
              for c in centres}
    return {
        "weights": weights, "anchor": anchor,
        "target_neg_pos_ratio": target_ratio,
        "n_pos_by_centre": {c: int(npos[c]) for c in centres},
        "n_neg_by_centre": {c: int(nneg[c]) for c in centres},
    }


def prior_equalised_fpr_at_recall(
    pool: pd.DataFrame, recall: float = TARGET_RECALL
) -> Dict[str, Any]:
    """FPR@90R with negatives reweighted so hospital identity carries zero
    label information: P(centre | weighted negative) == P(centre | positive).

    The THRESHOLD is computed exactly as everywhere else in this file --
    ``threshold_at_recall`` on the UNWEIGHTED positives, an attainable score.
    Reweighting touches only which negatives count, and how much, once that
    threshold is fixed: recall is a property of the positives alone, and
    reweighting them too would conflate two different corrections in one
    number.

    Effective sample size is reported two ways. ``effective_n`` is the plain
    sum of weights under THIS weighting's anchor convention, and is anchor-
    DEPENDENT -- an artefact of which centre got pinned to 1.0, not on its own
    a statistical quantity. ``kish_ess`` -- (sum w)^2 / sum(w^2) -- is the
    standard effective-sample-size correction for unequal weights and is
    anchor-INVARIANT (scaling every weight by a constant leaves it unchanged).
    Quote kish_ess for "how much data is this really"; effective_n only to
    reproduce the specific weights used here.
    """
    y = pool["label_int"].to_numpy()
    s = pool["logit"].to_numpy()
    thr = threshold_at_recall(y, s, recall)

    wdata = centre_negative_prior_weights(pool)
    neg = pool[pool["label_int"] == 0].copy()
    neg["weight"] = neg["centre"].map(wdata["weights"])
    if neg["weight"].isna().any():
        missing = sorted(neg.loc[neg["weight"].isna(), "centre"].unique())
        raise ValueError(f"no prior weight for centre(s) {missing}")

    above = (neg["logit"] >= thr).to_numpy()
    w = neg["weight"].to_numpy()
    fpr = float(w[above].sum() / w.sum())

    return {
        "threshold": float(thr),
        "fpr_at_90_recall_prior_equalised": fpr,
        "weights": wdata["weights"],
        "anchor_centre": wdata["anchor"],
        "effective_n": float(w.sum()),
        "kish_ess": float(w.sum() ** 2 / np.sum(w ** 2)),
        "n_neg_raw": int(len(neg)),
    }


def prior_equalised_fpr_bootstrap(
    pool: pd.DataFrame, recall: float = TARGET_RECALL, n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Variance of the prior-equalised figure via repeated SUBSAMPLING.

    Reweighting and subsampling answer the same question (what would this look
    like with hospital-balanced negatives) but only subsampling has a
    resampling distribution to read variance off directly: draw each
    down-weighted centre to its weight-implied integer count WITHOUT
    replacement, keep the anchor centre's negatives in full every time (its
    weight is exactly 1.0), recombine with every positive (unchanged across
    draws), and compute the plain UNWEIGHTED FPR@90R at the SAME threshold
    used throughout -- recall depends only on the positives, which never
    change. ``n_boot`` independent subsamples give the median and IQR.

    This is a real cost, stated here so it cannot be missed: the subsampled
    set holds roughly 40% of the pooled negatives (~1160 of 2930 on this
    validation set), so this IQR is WIDER than the pooled-FPR IQR at the same
    k. That widening is the honest price of removing a shortcut the pooled
    figure could not see.
    """
    rng = np.random.default_rng(seed)
    y = pool["label_int"].to_numpy()
    s = pool["logit"].to_numpy()
    thr = threshold_at_recall(y, s, recall)

    wdata = centre_negative_prior_weights(pool)
    anchor = wdata["anchor"]
    neg = pool[pool["label_int"] == 0]
    by_centre = {c: neg.loc[neg["centre"] == c, "logit"].to_numpy()
                for c in wdata["weights"]}
    target_counts: Dict[str, int] = {}
    for c, scores in by_centre.items():
        if c == anchor:
            target_counts[c] = len(scores)
        else:
            k = int(round(wdata["weights"][c] * len(scores)))
            if k > len(scores):
                raise AssertionError(
                    f"centre {c}: subsample target {k} exceeds available "
                    f"{len(scores)} -- a non-anchor weight exceeded 1.0, "
                    f"which centre_negative_prior_weights should never return"
                )
            target_counts[c] = k

    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        above = 0
        total = 0
        for c, scores in by_centre.items():
            k = target_counts[c]
            if c == anchor:
                above += int((scores >= thr).sum())
            else:
                idx = rng.choice(len(scores), size=k, replace=False)
                above += int((scores[idx] >= thr).sum())
            total += k
        draws[i] = above / total

    return {
        "threshold": float(thr),
        "target_counts_by_centre": target_counts,
        "anchor_centre": anchor,
        "n_boot": n_boot,
        "seed": seed,
        "draws": draws.tolist(),
        "spread": spread(draws.tolist()),
    }


def per_centre_fpr_at_recall(
    pool: pd.DataFrame, recall: float = TARGET_RECALL
) -> Dict[str, Any]:
    """FPR@90R within each centre, at ONE threshold set globally from the
    pooled positives.

    Deliberately not each centre's own threshold: the question is "how does
    the SAME deployed decision rule behave in each hospital", not "what
    threshold would each hospital want in isolation". Asymmetry between
    centres at that one shared threshold is a direct fingerprint of the model
    reading hospital identity -- a model with zero hospital signal has no
    mechanism to prefer one centre's negatives over another's at a fixed cut.
    """
    y = pool["label_int"].to_numpy()
    s = pool["logit"].to_numpy()
    thr = threshold_at_recall(y, s, recall)

    neg = pool[pool["label_int"] == 0]
    by_centre: Dict[str, Any] = {}
    for c, grp in neg.groupby("centre"):
        n = len(grp)
        above = int((grp["logit"].to_numpy() >= thr).sum())
        by_centre[c] = {"n_neg": int(n), "n_above": above,
                        "fpr": float(above / n) if n else float("nan")}

    fprs = [v["fpr"] for v in by_centre.values() if np.isfinite(v["fpr"])]
    n_total = sum(v["n_neg"] for v in by_centre.values())
    above_total = sum(v["n_above"] for v in by_centre.values())
    return {
        "threshold": float(thr),
        "by_centre": by_centre,
        "asymmetry_max_minus_min": (float(max(fprs) - min(fprs))
                                    if len(fprs) >= 2 else float("nan")),
        # sanity check: recombining the per-centre counts must reproduce the
        # plain unweighted pooled FPR at this same threshold, or the split
        # above has silently dropped or double-counted rows
        "pooled_check": (float(above_total / n_total) if n_total
                         else float("nan")),
    }


# ---------------------------------------------------------------------------
# LEVEL 3 -- across repeats and seeds
# ---------------------------------------------------------------------------
def spread(values: Sequence[float]) -> Dict[str, float]:
    """Median, IQR and range. Never a bare mean: these are 5-point samples where
    one bad run moves a mean and does not move a median."""
    a = np.asarray([v for v in values], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: float("nan") for k in
                ("n", "median", "iqr", "q1", "q3", "min", "max", "range", "sd")}
    q1, q3 = np.percentile(a, [25, 75])
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "iqr": float(q3 - q1),
        "q1": float(q1), "q3": float(q3),
        "min": float(a.min()), "max": float(a.max()),
        "range": float(a.max() - a.min()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


LEVEL3_FIELDS = (
    "roc_auc", "pauc_15_raw", "pauc_15_std", "fpr_at_90_recall",
    "ppv_at_90_recall_local", "ppv_leaderboard_projected",
)


def level3(level2_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Median and IQR of every Level-2 figure across repeats and seeds."""
    out: Dict[str, Any] = {}
    for f in LEVEL3_FIELDS:
        out[f] = spread([r[f] for r in level2_results])
    boots = [r["bootstrap_official"]["median_ppv"] for r in level2_results
             if "bootstrap_official" in r]
    if boots:
        out["bootstrap_median_ppv"] = spread(boots)
    return out


# ---------------------------------------------------------------------------
# DOMAIN GAP -- LOCO, reported per direction and never averaged
# ---------------------------------------------------------------------------
def loco_results(out_dir: str, centres: Sequence[int], seeds: Sequence[int],
                 manifest: pd.DataFrame, keep_filter: bool = True,
                 bootstrap: bool = False) -> pd.DataFrame:
    """Per-direction, per-seed LOCO metrics.

    ``keep_filter`` drops rows flagged keep_for_training=False. src/folds.py
    applies that filter to the train side of a holdout split but not to the test
    side, so the held-out center_2 set otherwise carries 7 rows that are exact
    SHA-256 duplicates of other rows in the same set -- 7 images counted twice.
    Filtering them here makes the LOCO test sets consistent with the pooled OOF
    set, which excludes them. Both numbers are reported so the choice is visible.
    """
    keep = manifest.set_index("filepath")["keep_for_training"]
    rows = []
    for centre in centres:
        for seed in seeds:
            name = f"loco_c{centre}_s{seed}"
            try:
                df = load_unit(out_dir, name)
            except FileNotFoundError:
                continue
            n_raw = len(df)
            if keep_filter:
                df = df[df["filepath"].map(keep).fillna(True).astype(bool)]
            m = metric_block(df["label_int"].to_numpy(), df["logit"].to_numpy(),
                             bootstrap=bootstrap, boot_seed=seed)
            rows.append({
                "holdout_centre": centre, "seed": seed, "unit": name,
                "n_raw": n_raw, "n": m["n"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                "roc_auc": m["roc_auc"], "pauc_15_std": m["pauc_15_std"],
                "pauc_15_raw": m["pauc_15_raw"],
                "fpr_at_90_recall": m["fpr_at_90_recall"],
                "ppv_at_90_recall_local": m["ppv_at_90_recall_local"],
                "ppv_leaderboard_projected": m["ppv_leaderboard_projected"],
            })
    return pd.DataFrame(rows)


def domain_gap(loco: pd.DataFrame, pooled_fpr: Dict[str, float],
               manifest: pd.DataFrame) -> Dict[str, Any]:
    """LOCO FPR@90R per direction, and its gap against pooled OOF.

    THE TWO DIRECTIONS ARE NOT INTERCHANGEABLE AND ARE NEVER AVERAGED. Holding
    out center_1 leaves 809 training images with 97 positives; holding out
    center_2 leaves 2279 with 61. Different training-set sizes, different
    positive counts, different held-out set sizes -- two separate experiments
    that happen to share a script. A single "LOCO number" would be the mean of
    two things that measure different quantities.
    """
    out: Dict[str, Any] = {
        "pooled_oof_fpr_median": pooled_fpr["median"],
        "pooled_oof_fpr_iqr": pooled_fpr["iqr"],
        "directions": {},
        "never_average_note": (
            "the two directions differ in training-set size (809 vs 2279), "
            "positive count (97 vs 61) and held-out size (2279 vs 816); they are "
            "not two measurements of one quantity and must not be averaged"),
    }
    for centre in sorted(loco["holdout_centre"].unique()):
        sub = loco[loco["holdout_centre"] == centre]
        train_col = f"holdout_center_{centre}"
        tr = manifest[(manifest[train_col] == "train") & (manifest["fold_r0"] != -1)]
        sp = spread(sub["fpr_at_90_recall"].tolist())
        out["directions"][f"holdout_center_{centre}"] = {
            "held_out_centre": f"center_{centre}",
            "n_train": int(len(tr)),
            "n_train_pos": int((tr["class_label"] == "neoplasia").sum()),
            "n_test": int(sub["n"].iloc[0]) if len(sub) else 0,
            "n_test_pos": int(sub["n_pos"].iloc[0]) if len(sub) else 0,
            "fpr_at_90_recall": sp,
            "roc_auc": spread(sub["roc_auc"].tolist()),
            "per_seed_fpr": dict(zip(sub["seed"].astype(int),
                                     sub["fpr_at_90_recall"])),
            "domain_gap_vs_pooled_oof": sp["median"] - pooled_fpr["median"],
            "domain_gap_note": (
                "domain_gap = median LOCO FPR@90R - median pooled-OOF FPR@90R; "
                "positive means the held-out centre is harder than in-distribution "
                "cross-validation"),
        }
    return out


# ---------------------------------------------------------------------------
# DIAGNOSTICS on the pooled OOF set
# ---------------------------------------------------------------------------
def attach_manifest(pool: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Join the diagnostic manifest columns onto a prediction set."""
    cols = [c for c in DIAGNOSTIC_JOIN_COLS if c in manifest.columns]
    m = manifest.set_index("filepath")[cols]
    joined = pool.join(m, on="filepath", rsuffix="_manifest")
    joined["rank"] = joined["logit"].rank(ascending=False, method="min").astype(int)
    joined["pct_rank"] = joined["logit"].rank(pct=True, ascending=False)
    return joined


def diag_positive_ranks(joined: pd.DataFrame, n_lowest: int = 16) -> Dict[str, Any]:
    """DIAGNOSTIC 1 -- where every positive sits in the pooled ranking.

    ``n_lowest`` defaults to 16 because that is what sets the operating point:
    158 positives at 90% recall means missing 15.8, so the 16 lowest-ranked
    positives are exactly the ones given up at the threshold, and the boundary
    between the 16th and 17th is where FPR@90R is decided.
    """
    pos = joined[joined["label_int"] == 1].sort_values("logit")
    lowest = pos.head(n_lowest)
    cols = ["filepath", "centre", "visibility", "logit", "rank", "pct_rank", "fold"]
    return {
        "n_positives": int(len(pos)),
        "n_lowest_reported": int(n_lowest),
        "why_16": ("0.9 * 158 = 142.2 positives retained, so 16 are given up at "
                   "90% recall; these are those 16"),
        "rank_summary": {
            "median_rank": float(pos["rank"].median()),
            "best_rank": int(pos["rank"].min()),
            "worst_rank": int(pos["rank"].max()),
            "n_in_top_158": int((pos["rank"] <= 158).sum()),
            "n_below_median_of_all": int((pos["pct_rank"] > 0.5).sum()),
        },
        "lowest": lowest[[c for c in cols if c in lowest.columns]].to_dict("records"),
    }


def diag_visibility_crosstab(joined: pd.DataFrame, n_lowest: int = 16) -> Dict[str, Any]:
    """DIAGNOSTIC 2 -- are the threshold-setting positives the invisible ones?

    Two definitions of "lowest decile", because they answer different questions
    and are easy to conflate:

      * ``bottom_16_positives``  -- the lowest decile OF THE POSITIVES (10% of
        158 = 15.8, so 16 images). These are exactly the images given up at the
        90%-recall operating point, which is why the two diagnostics use the
        same cut.
      * ``global_bottom_decile`` -- positives whose score falls in the bottom 10%
        of ALL 3088 images, i.e. positives the model ranks below most negatives.
        This set is often EMPTY, and that is a finding rather than a gap: it
        means even the worst-ranked positives still outrank most negatives.

    Enrichment is tested against the base rate with a hypergeometric tail
    probability rather than chi-square: at n=16 the chi-square approximation is
    not trustworthy and would manufacture significance.
    """
    pos = joined[joined["label_int"] == 1]
    base = pos["visibility"].value_counts()
    n_pos = len(pos)

    def _tab(sub: pd.DataFrame, label: str) -> Dict[str, Any]:
        if len(sub) == 0:
            return {"label": label, "n": 0, "by_visibility": {}, "empty": True}
        obs = sub["visibility"].value_counts()
        rows = {}
        for cat in VISIBILITY_ORDER:
            k = int(obs.get(cat, 0))
            K = int(base.get(cat, 0))
            n = len(sub)
            expected = n * K / n_pos if n_pos else float("nan")
            # P(X >= k) for X ~ Hypergeometric(n_pos, K, n)
            p_enrich = (float(stats.hypergeom.sf(k - 1, n_pos, K, n))
                        if K > 0 and n > 0 else float("nan"))
            rows[cat] = {
                "observed": k,
                "expected": float(expected),
                "base_rate_n": K,
                "enrichment": float(k / expected) if expected else float("nan"),
                "p_enrichment_hypergeom": p_enrich,
            }
        return {"label": label, "n": len(sub), "by_visibility": rows}

    bottom16 = pos.nsmallest(n_lowest, "logit")
    global_decile = pos[pos["pct_rank"] > 0.90]
    return {
        "positives_total": n_pos,
        "visibility_base_rates": {c: int(base.get(c, 0)) for c in VISIBILITY_ORDER},
        "bottom_16_positives": _tab(
            bottom16,
            f"lowest decile of positives -- the {n_lowest} worst-ranked of "
            f"{n_pos}, i.e. exactly those given up at 90% recall "
            f"(the threshold setters)"),
        "global_bottom_decile": _tab(global_decile, "positives in the bottom 10% "
                                                    "of all 3088 images"),
        "test_note": ("hypergeometric tail P(X >= observed); chi-square is not "
                      "used because expected counts below 5 make it unreliable"),
    }


def diag_hard_negatives(joined: pd.DataFrame, n: int = 50) -> Dict[str, Any]:
    """DIAGNOSTIC 3 -- the highest-scoring negatives, which are what set FPR.

    Base rates travel with the list: '31 of the top 50 are center_2' means
    nothing until you know center_2 is 24% of negatives.
    """
    neg = joined[joined["label_int"] == 0]
    top = neg.nlargest(n, "logit")
    cols = ["filepath", "centre", "has_redaction", "logit", "rank", "fold"]
    out: Dict[str, Any] = {
        "n_reported": int(len(top)),
        "n_negatives": int(len(neg)),
        "rows": top[[c for c in cols if c in top.columns]].to_dict("records"),
        "enrichment": {},
    }
    for col in ("centre", "has_redaction"):
        if col not in neg.columns:
            continue
        base = neg[col].value_counts(normalize=True)
        got = top[col].value_counts()
        out["enrichment"][col] = {
            str(k): {
                "count_in_top": int(got.get(k, 0)),
                "share_in_top": float(got.get(k, 0) / len(top)),
                "base_rate": float(base.get(k, 0.0)),
                "ratio": float((got.get(k, 0) / len(top)) / base[k])
                if base.get(k, 0) else float("nan"),
            }
            for k in base.index
        }
    return out


def diag_confound_correlations(joined: pd.DataFrame) -> Dict[str, Any]:
    """DIAGNOSTIC 4 -- Spearman between logit and each hospital-identity proxy.

    Reported three ways, because the unconditional correlation is the one that
    is easiest to misread. Class label is associated with both the logit (by
    construction, if the model works) and with several of these columns, so an
    ALL-ROWS correlation is confounded by class and can be large even for a
    model reading nothing but pathology. The within-class correlations are the
    ones that answer the actual question -- among images of the SAME class, does
    the score still track hospital identity?

    has_redaction is boolean, so its "Spearman" is a rank-biserial correlation.
    Same statistic, and monotone association is still what it measures, but the
    magnitude is bounded by the class balance and is not comparable to a
    continuous column's rho.
    """
    out: Dict[str, Any] = {"threshold": SHORTCUT_RHO, "columns": {}}
    subsets = {
        "all_rows": joined,
        "negatives_only": joined[joined["label_int"] == 0],
        "positives_only": joined[joined["label_int"] == 1],
    }
    for col in CONFOUND_COLS:
        if col not in joined.columns:
            out["columns"][col] = {"error": "column absent from manifest"}
            continue
        entry: Dict[str, Any] = {"binary": bool(joined[col].dropna().isin(
            [0, 1, True, False]).all())}
        for tag, sub in subsets.items():
            x = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            yv = sub["logit"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(yv)
            if ok.sum() < 3 or np.unique(x[ok]).size < 2:
                entry[tag] = {"rho": float("nan"), "p": float("nan"),
                              "n": int(ok.sum())}
                continue
            rho, p = stats.spearmanr(x[ok], yv[ok])
            entry[tag] = {"rho": float(rho), "p": float(p), "n": int(ok.sum()),
                          "flagged": bool(abs(rho) >= SHORTCUT_RHO)}
        out["columns"][col] = entry
    # a separate, blunter question: can the logit alone separate the two centres?
    if "centre" in joined.columns:
        neg = joined[joined["label_int"] == 0]
        c1 = neg.loc[neg["centre"] == "center_1", "logit"].to_numpy()
        c2 = neg.loc[neg["centre"] == "center_2", "logit"].to_numpy()
        if c1.size and c2.size:
            u = stats.mannwhitneyu(c1, c2, alternative="two-sided")
            auc = float(u.statistic / (c1.size * c2.size))
            out["centre_separability_among_negatives"] = {
                "auc_logit_predicts_centre": auc,
                "interpretation": ("0.5 = the logit carries no hospital "
                                   "information among negatives; far from 0.5 "
                                   "means it does"),
                "n_center_1": int(c1.size), "n_center_2": int(c2.size),
                "p": float(u.pvalue),
            }
    return out


def aggregate_diagnostics(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Level-3 treatment of the diagnostics: how stable are they across seeds?

    A bottom-16 list from one seed is an anecdote. The same image appearing in
    the bottom 16 of all five seeds is a property of the image.
    """
    counter: Dict[str, Dict[str, Any]] = {}
    for d in per_seed:
        for row in d["positive_ranks"]["lowest"]:
            e = counter.setdefault(row["filepath"], {
                "filepath": row["filepath"], "centre": row.get("centre"),
                "visibility": row.get("visibility"), "n_seeds_in_bottom_16": 0,
                "ranks": [],
            })
            e["n_seeds_in_bottom_16"] += 1
            e["ranks"].append(int(row["rank"]))
    persistent = sorted(counter.values(),
                        key=lambda e: (-e["n_seeds_in_bottom_16"],
                                       float(np.mean(e["ranks"]))))
    for e in persistent:
        e["median_rank"] = float(np.median(e["ranks"]))

    rho_agg: Dict[str, Any] = {}
    for col in CONFOUND_COLS:
        for subset in ("all_rows", "negatives_only", "positives_only"):
            vals = [d["confounds"]["columns"][col][subset]["rho"]
                    for d in per_seed
                    if col in d["confounds"]["columns"]
                    and subset in d["confounds"]["columns"][col]]
            if vals:
                rho_agg.setdefault(col, {})[subset] = spread(vals)

    centre_auc = [d["confounds"]["centre_separability_among_negatives"]
                  ["auc_logit_predicts_centre"]
                  for d in per_seed
                  if "centre_separability_among_negatives" in d["confounds"]]
    return {
        "bottom_16_persistence": persistent,
        "n_distinct_images_ever_in_bottom_16": len(counter),
        "n_in_all_seeds": sum(1 for e in persistent
                              if e["n_seeds_in_bottom_16"] == len(per_seed)),
        "confound_rho_across_seeds": rho_agg,
        "centre_separability_across_seeds": spread(centre_auc) if centre_auc else None,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def box(lines: Sequence[str], width: int = 78) -> str:
    top = "+" + "-" * (width - 2) + "+"
    body = [f"| {ln:<{width - 4}} |" for ln in lines]
    return "\n".join([top, *body, top])


def fmt_spread(s: Dict[str, float], p: int = 4) -> str:
    return (f"median {s['median']:.{p}f}  IQR {s['iqr']:.{p}f} "
            f"[{s['q1']:.{p}f}, {s['q3']:.{p}f}]  range {s['range']:.{p}f} "
            f"[{s['min']:.{p}f}, {s['max']:.{p}f}]  n={s['n']}")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Level 1/2/3 evaluation, domain gap and diagnostics."
    )
    ap.add_argument("--job-a", default="runs/noise_floor_a",
                    help="CV run directory (Level 1/2/3 + diagnostics)")
    ap.add_argument("--job-b", default="runs/noise_floor_b",
                    help="LOCO run directory (domain gap)")
    ap.add_argument("--manifest", default="manifests/rare25_folds_v2.csv")
    ap.add_argument("--repeats", default="0")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--centres", default="1,2")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="skip the official scorer (fast preview)")
    ap.add_argument("--out", default="reports/noise_floor",
                    help="output stem; writes .md, .json and diagnostic CSVs")
    args = ap.parse_args(argv)

    ints = lambda s: [int(x) for x in str(s).replace(" ", "").split(",") if x]
    repeats, folds = ints(args.repeats), ints(args.folds)
    seeds, centres = ints(args.seeds), ints(args.centres)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, args.manifest)
                           if not os.path.isabs(args.manifest) else args.manifest)
    job_a = args.job_a if os.path.isabs(args.job_a) else os.path.join(REPO_ROOT, args.job_a)
    job_b = args.job_b if os.path.isabs(args.job_b) else os.path.join(REPO_ROOT, args.job_b)

    report: Dict[str, Any] = {"job_a": job_a, "job_b": job_b,
                              "repeats": repeats, "folds": folds, "seeds": seeds}
    out_lines: List[str] = []
    W = out_lines.append

    W("# RARE26 noise floor\n")
    W("Generated by `src/evaluate.py`. Every figure carries its spread; no "
      "single number in this report is reportable on its own.\n")

    # ---------------- Level 1 / 2 ----------------
    l1_all, l2_all, diag_per_seed = [], [], []
    for repeat in repeats:
        for seed in seeds:
            try:
                pool = pool_oof(job_a, repeat, seed, folds, manifest)
            except FileNotFoundError as exc:
                W(f"- missing unit for r{repeat} s{seed}: `{exc}` -- skipped\n")
                continue
            per_fold = level1(job_a, repeat, seed, folds)
            l1_all.append(per_fold)
            m2 = level2(pool, bootstrap=not args.no_bootstrap, n_boot=args.n_boot)
            m2.update(repeat=repeat, seed=seed)
            m2["pooled_vs_averaged"] = pooled_vs_averaged(pool, per_fold)
            m2["fold_contribution"] = fold_contribution(pool)
            l2_all.append(m2)

            joined = attach_manifest(pool, manifest)
            diag_per_seed.append({
                "repeat": repeat, "seed": seed,
                "positive_ranks": diag_positive_ranks(joined),
                "visibility": diag_visibility_crosstab(joined),
                "hard_negatives": diag_hard_negatives(joined),
                "confounds": diag_confound_correlations(joined),
            })

    if not l2_all:
        W("\n**No complete (repeat, seed) pools found. Nothing to report.**\n")
        print("\n".join(out_lines))
        return 1

    l1 = pd.concat(l1_all, ignore_index=True)
    report["level1"] = l1.to_dict("records")
    report["level2"] = l2_all
    report["level3"] = level3(l2_all)

    # ---------------- the headline ----------------
    fprs = [r["fpr_at_90_recall"] for r in l2_all]
    sp = spread(fprs)
    mde = sp["range"]
    W("\n## 1. Minimum detectable effect (Job A)\n")
    W("```")
    W(box([
        "MINIMUM DETECTABLE EFFECT -- pooled-OOF FPR@90R, 5 seeds",
        "",
        *[f"  seed {r['seed']}   FPR@90R = {r['fpr_at_90_recall']:.4f}"
          f"   ({round(r['fpr_at_90_recall'] * r['n_neg'])} FP of {r['n_neg']})"
          for r in sorted(l2_all, key=lambda r: r["seed"])],
        "",
        f"  median  {sp['median']:.4f}",
        f"  IQR     {sp['iqr']:.4f}   [{sp['q1']:.4f}, {sp['q3']:.4f}]",
        f"  range   {sp['range']:.4f}   [{sp['min']:.4f}, {sp['max']:.4f}]",
        f"  SD      {sp['sd']:.4f}",
        "",
        f"  MINIMUM DETECTABLE EFFECT = {mde:.4f} FPR ({mde * 100:.2f} pp)",
        f"  = {mde * l2_all[0]['n_neg']:.1f} false positives out of "
        f"{l2_all[0]['n_neg']}",
        "",
        "  Any change in pooled-OOF FPR@90R smaller than this is NOT an",
        "  improvement. It is the same model with a different seed.",
    ]))
    W("```\n")
    report["minimum_detectable_effect"] = {
        "metric": "pooled_oof_fpr_at_90_recall",
        "definition": "max - min across seeds (matches noise_floor() in scripts/08_score.py)",
        "value": mde, "spread": sp,
        "per_seed": {int(r["seed"]): r["fpr_at_90_recall"] for r in l2_all},
    }

    # ---------------- Level 3 table ----------------
    W("\n## 2. Level 3 -- across seeds (median / IQR of every Level-2 figure)\n")
    W("| metric | median | IQR | range | min | max |")
    W("|---|---|---|---|---|---|")
    for f in LEVEL3_FIELDS:
        s = report["level3"][f]
        W(f"| {f} | {s['median']:.4f} | {s['iqr']:.4f} | {s['range']:.4f} | "
          f"{s['min']:.4f} | {s['max']:.4f} |")
    if "bootstrap_median_ppv" in report["level3"]:
        s = report["level3"]["bootstrap_median_ppv"]
        W(f"| bootstrap_median_ppv (official scorer) | {s['median']:.4f} | "
          f"{s['iqr']:.4f} | {s['range']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")
    W("")
    W("pAUC convention: `pauc_15_raw` is area/0.15 (random = 0.075, perfect = 1.0); "
      "`pauc_15_std` is McClish-standardised (random = 0.5, perfect = 1.0). Both "
      "describe the same area over FPR in [0, 0.15].\n")
    W("`ppv_leaderboard_projected` is a PROJECTION of local FPR onto the "
      f"competition's test shape ({LEADERBOARD_POS} positives, {LEADERBOARD_NEG} "
      "negatives), not a measurement of anything on the competition data.\n")

    # ---------------- pooling vs averaging ----------------
    W("\n## 3. Why pool: per-fold spread vs pooled spread\n")
    fold_seed_spread = []
    for f in sorted(l1["fold"].unique()):
        vals = l1.loc[l1["fold"] == f, "fpr_at_90_recall"].tolist()
        s = spread(vals)
        fold_seed_spread.append({"fold": int(f), **s})
    within_seed_across_fold = []
    for (r, sd), g in l1.groupby(["repeat", "seed"]):
        within_seed_across_fold.append(
            {"repeat": int(r), "seed": int(sd), **spread(g["fpr_at_90_recall"].tolist())})

    mean_fold_sd = float(np.mean([d["sd"] for d in fold_seed_spread]))
    mean_fold_range = float(np.mean([d["range"] for d in fold_seed_spread]))
    W("Seed-to-seed noise, measured the same way at both levels:\n")
    W("| level | unit of measurement | SD across seeds | range across seeds |")
    W("|---|---|---|---|")
    for d in fold_seed_spread:
        W(f"| 1 | fold {d['fold']} alone (~31 pos, 586 neg) | {d['sd']:.4f} | "
          f"{d['range']:.4f} |")
    W(f"| 1 | **mean over the 5 folds** | **{mean_fold_sd:.4f}** | "
      f"**{mean_fold_range:.4f}** |")
    W(f"| 2 | **pooled OOF (158 pos, 2930 neg)** | **{sp['sd']:.4f}** | "
      f"**{sp['range']:.4f}** |")
    ratio_sd = mean_fold_sd / sp["sd"] if sp["sd"] else float("nan")
    ratio_rg = mean_fold_range / sp["range"] if sp["range"] else float("nan")
    W("")
    W(f"**Pooling shrinks the seed-noise SD by {ratio_sd:.1f}x and the range by "
      f"{ratio_rg:.1f}x.** A per-fold experiment would need an effect "
      f"{ratio_rg:.1f} times larger before it cleared its own noise floor.\n")
    W("Fold-to-fold spread within a single seed -- variation pooling removes "
      "entirely, because the pooled estimate has no fold dimension:\n")
    W("| repeat | seed | fold FPR@90R median | IQR | range |")
    W("|---|---|---|---|---|")
    for d in within_seed_across_fold:
        W(f"| {d['repeat']} | {d['seed']} | {d['median']:.4f} | {d['iqr']:.4f} | "
          f"{d['range']:.4f} |")
    W("")
    W("Operating-point granularity, which is the mechanism behind the difference:\n")
    op = l2_all[0]["operating_point"]
    W(f"- pooled: 158 positives, so 90% recall retains {op['tp_needed']:.1f}; one "
      f"false positive moves FPR by {op['fpr_step_one_fp']:.5f}")
    f0 = l1.iloc[0]
    W(f"- per fold: ~{int(f0['n_pos'])} positives, so the threshold is set by "
      f"~3 images; one false positive moves FPR by {f0['fpr_step_one_fp']:.5f} "
      f"({f0['fpr_step_one_fp'] / op['fpr_step_one_fp']:.0f}x coarser)\n")

    pva = [r["pooled_vs_averaged"] for r in l2_all]
    W("Pooled FPR vs the average of the five fold FPRs -- the statistic this "
      "codebase must not report:\n")
    W("| seed | pooled FPR@90R | mean of 5 fold FPRs | difference |")
    W("|---|---|---|---|")
    for r, d in zip(sorted(l2_all, key=lambda r: r["seed"]), pva):
        W(f"| {r['seed']} | {d['pooled_fpr']:.4f} | {d['mean_of_fold_fpr']:.4f} | "
          f"{d['absolute_difference']:.4f} |")
    W("")
    report["pooling_case"] = {
        "per_fold_across_seed": fold_seed_spread,
        "within_seed_across_fold": within_seed_across_fold,
        "pooled_across_seed": sp,
        "sd_shrink_factor": ratio_sd,
        "range_shrink_factor": ratio_rg,
    }

    # pooling validity
    fc = [r["fold_contribution"] for r in l2_all]
    maxdev = spread([d["max_abs_deviation"] for d in fc])
    W("**Pooling validity check.** The five folds are five separately trained "
      "models, so pooling raw logits assumes comparable logit scales. If one "
      "fold's outputs ran high it would supply most of the false positives at "
      "the pooled threshold. Each fold holds ~20% of the negatives, so its "
      "expected share of false positives is ~20%:\n")
    W(f"- max |share - expected| across folds: {fmt_spread(maxdev, 3)}")
    W(f"- per-fold false-positive share, seed {l2_all[0]['seed']}: "
      + ", ".join(f"f{k}={v:.2f}" for k, v in
                  sorted(fc[0]["fp_share_by_fold"].items())))
    W("")
    report["pooling_validity"] = fc

    # ---------------- bootstrap ----------------
    if not args.no_bootstrap:
        boots = [r["bootstrap_official"] for r in l2_all]
        bs = spread([b["median_ppv"] for b in boots])
        W("\n## 4. Bootstrap PPV (official scorer, for variance only)\n")
        W(f"`official_score` from `scripts/08_score.py`, applied verbatim to each "
          f"pooled OOF set ({args.n_boot} draws):\n")
        W(f"- median PPV@90R across seeds: {fmt_spread(bs)}")
        W("")
        n_neg = l2_all[0]["n_neg"]
        boot_prev = (n_neg // 100) / (n_neg + n_neg // 100)
        W("```")
        W(box([
            "THIS BOOTSTRAP FIGURE IS NOT COMPARABLE TO THE LEADERBOARD.",
            "FPR@90R IS THE COMPARABLE QUANTITY. USE IT INSTEAD.",
            "",
            f"Local pooled OOF prevalence: {l2_all[0]['prevalence'] * 100:.1f}% "
            f"({l2_all[0]['n_pos']} pos / {l2_all[0]['n']}).",
            f"Leaderboard prevalence:      "
            f"{LEADERBOARD_POS / (LEADERBOARD_POS + LEADERBOARD_NEG) * 100:.1f}% "
            f"({LEADERBOARD_POS} / {LEADERBOARD_POS + LEADERBOARD_NEG}).",
            "",
            "NOTE ON THE REASON: the official scorer already resamples positives",
            f"down to a 1:100 ratio, so each draw sits at {boot_prev * 100:.1f}% "
            "prevalence,",
            "which does match the leaderboard. Prevalence is therefore NOT what",
            "makes this incomparable. The actual reasons are:",
            "",
            f"  1. All {n_neg} negatives appear in EVERY draw, so the interval",
            f"     carries no negative-sampling variance at all. The leaderboard",
            f"     has {LEADERBOARD_NEG} different negatives.",
            f"  2. Positives are resampled with replacement from only "
            f"{l2_all[0]['n_pos']} unique",
            f"     images down to {n_neg // 100}; the draws are not independent and",
            "     repeated images create score ties.",
            f"  3. {n_neg // 100} positives means 90% recall is set by ~3 images,",
            "     the same fragility Level 1 exists to avoid.",
            "  4. These are the training centres. The test set is other data;",
            "     see the LOCO domain gap below.",
        ]))
        W("```\n")
        report["bootstrap"] = {
            "per_seed": boots, "spread": bs,
            "draw_prevalence": boot_prev,
            "comparability": "NOT comparable to leaderboard; use FPR@90R",
        }

    # ---------------- domain gap ----------------
    W("\n## 5. Domain gap (Job B -- LOCO)\n")
    loco = loco_results(job_b, centres, seeds, manifest, keep_filter=True)
    loco_raw = loco_results(job_b, centres, seeds, manifest, keep_filter=False)
    if loco.empty:
        W("_No LOCO runs found yet._\n")
    else:
        dg = domain_gap(loco, sp, manifest)
        report["domain_gap"] = dg
        report["loco_per_seed"] = loco.to_dict("records")
        W("**The two directions are reported separately and are never averaged.**")
        W(dg["never_average_note"] + ".\n")
        W("| direction | held out | train n (pos) | test n (pos) | FPR@90R median | "
          "IQR | range | domain gap vs pooled OOF |")
        W("|---|---|---|---|---|---|---|---|")
        for k, d in dg["directions"].items():
            s = d["fpr_at_90_recall"]
            W(f"| `{k}` | {d['held_out_centre']} | {d['n_train']} "
              f"({d['n_train_pos']}) | {d['n_test']} ({d['n_test_pos']}) | "
              f"{s['median']:.4f} | {s['iqr']:.4f} | {s['range']:.4f} | "
              f"{d['domain_gap_vs_pooled_oof']:+.4f} |")
        W("")
        W(f"Pooled-OOF reference: FPR@90R median {sp['median']:.4f}, "
          f"IQR {sp['iqr']:.4f}.\n")
        W("Per seed:\n")
        W("| direction | " + " | ".join(f"seed {s_}" for s_ in seeds) +
          " | median | IQR |")
        W("|---|" + "---|" * (len(seeds) + 2))
        for centre in centres:
            sub = loco[loco["holdout_centre"] == centre].set_index("seed")
            cells = [f"{sub.loc[s_, 'fpr_at_90_recall']:.4f}"
                     if s_ in sub.index else "--" for s_ in seeds]
            ss = spread(sub["fpr_at_90_recall"].tolist())
            W(f"| holdout_center_{centre} | " + " | ".join(cells) +
              f" | {ss['median']:.4f} | {ss['iqr']:.4f} |")
        W("")
        W("**LOCO noise floor** (this is the screening instrument's own MDE):\n")
        for centre in centres:
            sub = loco[loco["holdout_centre"] == centre]
            ss = spread(sub["fpr_at_90_recall"].tolist())
            W(f"- `holdout_center_{centre}`: MDE = {ss['range']:.4f} FPR "
              f"({ss['range'] * 100:.2f} pp), IQR {ss['iqr']:.4f}")
        W("")
        if not loco_raw.empty:
            W("Held-out center_2 contains 7 rows flagged `keep_for_training=False` "
              "(exact SHA-256 duplicates of other rows in the same set); "
              "`src/folds.py` filters those from the train side of a holdout split "
              "but not the test side. Numbers above exclude them, matching the "
              "pooled OOF set. Unfiltered, for comparison:\n")
            W("| direction | seed | FPR@90R filtered | FPR@90R unfiltered | n |")
            W("|---|---|---|---|---|")
            for centre in centres:
                a = loco[loco["holdout_centre"] == centre].set_index("seed")
                b = loco_raw[loco_raw["holdout_centre"] == centre].set_index("seed")
                for s_ in seeds:
                    if s_ in a.index and s_ in b.index:
                        W(f"| holdout_center_{centre} | {s_} | "
                          f"{a.loc[s_, 'fpr_at_90_recall']:.4f} | "
                          f"{b.loc[s_, 'fpr_at_90_recall']:.4f} | "
                          f"{int(a.loc[s_, 'n'])} vs {int(b.loc[s_, 'n'])} |")
            W("")

    # ---------------- diagnostics ----------------
    W("\n## 6. Diagnostics on the pooled OOF set\n")
    agg = aggregate_diagnostics(diag_per_seed)
    report["diagnostics_per_seed"] = diag_per_seed
    report["diagnostics_aggregated"] = agg
    d0 = diag_per_seed[0]

    W("### 6.1 Rank of every positive; the 16 that set the threshold\n")
    rs = d0["positive_ranks"]["rank_summary"]
    W(f"Seed {d0['seed']}: median positive rank {rs['median_rank']:.0f} of 3088; "
      f"best {rs['best_rank']}, worst {rs['worst_rank']}; "
      f"{rs['n_in_top_158']}/158 positives rank inside the top 158; "
      f"{rs['n_below_median_of_all']} rank below the median image.\n")
    W(f"{d0['positive_ranks']['why_16']}.\n")
    W("Persistence across seeds -- how often each image lands in the bottom 16 "
      "(an image that does so every time is a property of the image, not of a "
      "seed):\n")
    W("| filepath | centre | visibility | seeds in bottom 16 | median rank |")
    W("|---|---|---|---|---|")
    for e in agg["bottom_16_persistence"][:25]:
        W(f"| `{e['filepath']}` | {e['centre']} | {e['visibility']} | "
          f"{e['n_seeds_in_bottom_16']}/{len(diag_per_seed)} | "
          f"{e['median_rank']:.0f} |")
    W("")
    W(f"{agg['n_distinct_images_ever_in_bottom_16']} distinct images appear in "
      f"some seed's bottom 16; {agg['n_in_all_seeds']} appear in every seed's.\n")

    W("### 6.2 Threshold-setting positives vs visibility\n")
    vis = d0["visibility"]
    W(f"Base rates over all {vis['positives_total']} positives: "
      + ", ".join(f"{k} {v}" for k, v in vis["visibility_base_rates"].items()) + ".\n")
    for key in ("bottom_16_positives", "global_bottom_decile"):
        t = vis[key]
        W(f"**{t['label']}** (n={t['n']}):\n")
        if t.get("empty"):
            W("_Empty, and that is the finding: no positive falls in the bottom "
              "10% of all 3088 images, so even the worst-ranked positives still "
              "outrank most negatives. The operating point is set by the "
              "positives in the table above, not by positives buried among the "
              "negatives._\n")
            continue
        W("| visibility | observed | expected | enrichment | P(X>=obs) |")
        W("|---|---|---|---|---|")
        for cat in VISIBILITY_ORDER:
            r = t["by_visibility"][cat]
            W(f"| {cat} | {r['observed']} | {r['expected']:.1f} | "
              f"{r['enrichment']:.2f}x | {r['p_enrichment_hypergeom']:.3f} |")
        W("")
    W(vis["test_note"] + ".\n")

    W("### 6.3 The 50 highest-scoring negatives\n")
    hn = d0["hard_negatives"]
    W(f"Seed {d0['seed']}, {hn['n_reported']} of {hn['n_negatives']} negatives. "
      "Enrichment against base rate:\n")
    W("| attribute | value | count in top 50 | share | base rate | ratio |")
    W("|---|---|---|---|---|---|")
    for col, entries in hn["enrichment"].items():
        for val, r in entries.items():
            W(f"| {col} | {val} | {r['count_in_top']} | {r['share_in_top']:.2f} | "
              f"{r['base_rate']:.2f} | {r['ratio']:.2f}x |")
    W("")
    W("Top 20 by logit:\n")
    W("| # | filepath | centre | has_redaction | logit |")
    W("|---|---|---|---|---|")
    for i, r in enumerate(hn["rows"][:20], 1):
        W(f"| {i} | `{r['filepath']}` | {r['centre']} | {r.get('has_redaction')} | "
          f"{r['logit']:.3f} |")
    W("")

    W("### 6.4 Spearman correlation between logit and hospital-identity proxies\n")
    W(f"Flagged at |rho| >= {SHORTCUT_RHO}. All four columns are DIAGNOSTIC per "
      "`manifests/DATA_README.md` -- none is a model input -- so a strong "
      "correlation means the model reconstructed it from pixels.\n")
    W("The all-rows column is confounded by class: class label drives the logit "
      "(by design) and is itself associated with these columns, so a large "
      "all-rows rho is expected even for a model reading only pathology. "
      "**The within-class columns are the ones that answer the question.**\n")
    W("| column | all rows (median rho) | negatives only | positives only | flagged |")
    W("|---|---|---|---|---|")
    for col in CONFOUND_COLS:
        e = agg["confound_rho_across_seeds"].get(col)
        if not e:
            W(f"| {col} | -- | -- | -- | -- |")
            continue
        flag = any(abs(e[s_]["median"]) >= SHORTCUT_RHO
                   for s_ in ("all_rows", "negatives_only", "positives_only")
                   if s_ in e)
        W(f"| {col} | {e['all_rows']['median']:+.3f} "
          f"(IQR {e['all_rows']['iqr']:.3f}) | "
          f"{e['negatives_only']['median']:+.3f} "
          f"(IQR {e['negatives_only']['iqr']:.3f}) | "
          f"{e['positives_only']['median']:+.3f} "
          f"(IQR {e['positives_only']['iqr']:.3f}) | "
          f"{'**YES**' if flag else 'no'} |")
    W("")
    W("`has_redaction` is boolean, so its coefficient is a rank-biserial "
      "correlation; it measures the same monotone association but its magnitude "
      "is bounded by the class balance and is not comparable to a continuous "
      "column's rho.\n")
    if agg.get("centre_separability_across_seeds"):
        cs = agg["centre_separability_across_seeds"]
        W(f"**Can the logit alone identify the hospital?** Among negatives only, "
          f"AUC of logit predicting centre: {fmt_spread(cs, 3)}. 0.5 means no "
          f"hospital information; distance from 0.5 in either direction means "
          f"the score carries hospital identity.\n")

    # ---------------- cost ----------------
    W("\n## 7. Wall clock and projected cost\n")
    cost = wallclock_summary(job_a, job_b)
    report["wallclock"] = cost
    if cost["units"]:
        W("| job | units | total wall | median per unit | mean per unit |")
        W("|---|---|---|---|---|")
        for job, d in cost["by_job"].items():
            W(f"| {job} | {d['n']} | {d['total_hours']:.2f} h | "
              f"{d['median_minutes']:.1f} min | {d['mean_minutes']:.1f} min |")
        W("")
        per_cv = cost["by_job"].get("job_a", {}).get("median_minutes")
        if per_cv:
            W(f"**Projected cost of one 10-repeat experiment** (10 repeats x 5 "
              f"folds x 1 seed = 50 units at {per_cv:.1f} min): "
              f"**{50 * per_cv / 60:.1f} h**.")
            W(f"- with 5 seeds (250 units): {250 * per_cv / 60:.1f} h "
              f"({250 * per_cv / 60 / 24:.1f} days)")
            W(f"- one repeat x 5 folds x 5 seeds (this Job A, 25 units): "
              f"{25 * per_cv / 60:.1f} h")
        W("")

    text = "\n".join(out_lines)
    out_stem = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    with open(out_stem + ".md", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(out_stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    l1.to_csv(out_stem + "_level1.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items()
                   if not isinstance(v, (dict, list))} for r in l2_all]).to_csv(
        out_stem + "_level2.csv", index=False)
    if not loco.empty:
        loco.to_csv(out_stem + "_loco.csv", index=False)

    print(text)
    print(f"\nwritten: {out_stem}.md / .json / _level1.csv / _level2.csv"
          + (" / _loco.csv" if not loco.empty else ""))
    return 0


def wallclock_summary(job_a: str, job_b: str) -> Dict[str, Any]:
    """Measured per-unit wall clock, read from each runner's index."""
    out: Dict[str, Any] = {"by_job": {}, "units": {}}
    for tag, d in (("job_a", job_a), ("job_b", job_b)):
        p = os.path.join(d, "run_index.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as fh:
                idx = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        secs = [u["seconds"] for u in idx.get("units", {}).values()
                if u.get("status") == "ok" and "seconds" in u]
        out["units"][tag] = idx.get("units", {})
        if secs:
            a = np.asarray(secs, dtype=float)
            out["by_job"][tag] = {
                "n": int(a.size),
                "total_hours": float(a.sum() / 3600),
                "median_minutes": float(np.median(a) / 60),
                "mean_minutes": float(a.mean() / 60),
            }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
