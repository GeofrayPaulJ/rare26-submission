"""Validation metrics computed on raw logits.

Four numbers per epoch, and they are not equally trustworthy:

  * ``roc_auc``   -- ranks every validation image against every other. All 617
                    rows contribute. This is the stable one, and the only one
                    fit to select a checkpoint on.
  * ``pauc_15``   -- ROC-AUC restricted to FPR in [0, 0.15], the low-false-alarm
                    region a screening tool actually operates in. Uses fewer
                    effective comparisons than full AUC, so it is noisier.
  * ``ppv_at_90_recall`` -- the competition metric's shape, but at fold scale.
                    A validation fold holds ~31 positives; 90% recall means
                    missing ~3 of them, so the operating threshold is pinned by
                    a handful of images and the resulting PPV swings wildly
                    epoch to epoch. LOG IT, PLOT IT, NEVER SELECT ON IT.
  * ``fpr_at_90_recall`` -- the false positive rate at the same operating
                    point. Unlike PPV, FPR does not depend on the class
                    prevalence of whatever set it is measured on, so a fold's
                    5.1% positive rate and the leaderboard's ~1% are directly
                    comparable in FPR even though their PPVs are not. This is
                    the quantity all planning should be done in.

Everything takes raw logits. Sigmoid is monotonic, so it changes none of these
metrics -- and not applying it avoids the saturation-to-ties problem that
src/io.py exists to prevent.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

PAUC_MAX_FPR = 0.15
TARGET_RECALL = 0.90


def _degenerate(y_true: np.ndarray) -> bool:
    """True when one class is missing and every ranking metric is undefined."""
    return len(np.unique(y_true)) < 2


def roc_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    if _degenerate(y_true):
        logger.warning("roc_auc undefined: only one class present")
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_score)))


def partial_auc(y_true, y_score, max_fpr: float = PAUC_MAX_FPR) -> Dict[str, float]:
    """Partial AUC over FPR in [0, max_fpr], in both reporting conventions.

    Returns ``raw`` (the restricted area divided by ``max_fpr``, so a random
    ranker scores max_fpr/2 = 0.075 and a perfect one 1.0) and ``std`` (the
    McClish standardisation, which rescales the same area so random = 0.5 and
    perfect = 1.0, directly comparable to full ROC-AUC).

    Both come from one curve; they are the same quantity on two scales. The
    convention has to be stated whenever the number is, because 0.075 and 0.5
    both mean "no signal" and they are easy to confuse.
    """
    y_true = np.asarray(y_true)
    if _degenerate(y_true):
        return {"raw": float("nan"), "std": float("nan")}

    fpr, tpr, _ = roc_curve(y_true, np.asarray(y_score))

    # Truncate the ROC curve at max_fpr, interpolating the exact crossing point
    # so the area does not depend on where the empirical curve happens to have
    # a vertex. Same construction sklearn uses internally for max_fpr.
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    if stop < len(fpr):
        tpr_at = np.interp(max_fpr, fpr[stop - 1:stop + 1], tpr[stop - 1:stop + 1])
        fpr_c = np.append(fpr[:stop], max_fpr)
        tpr_c = np.append(tpr[:stop], tpr_at)
    else:
        fpr_c, tpr_c = fpr, tpr

    area = float(auc(fpr_c, tpr_c))
    min_area = 0.5 * max_fpr ** 2   # random ranker
    max_area = max_fpr              # perfect ranker
    return {
        "raw": area / max_fpr,
        "std": 0.5 * (1.0 + (area - min_area) / (max_area - min_area)),
    }


def ppv_at_recall(y_true, y_score, recall: float = TARGET_RECALL) -> float:
    """Precision at the operating point that achieves `recall`.

    Deliberately the same interpolation the organisers' scorer uses for its
    "PPV@90RECALL Full" field (scripts/08_score.py), so this number and the
    official one mean the same thing on the same inputs. tests/test_metrics.py
    pins that equivalence.
    """
    y_true = np.asarray(y_true)
    if _degenerate(y_true):
        return float("nan")
    prec, rec, _ = precision_recall_curve(y_true, np.asarray(y_score))
    return float(np.interp(recall, rec[::-1], prec[::-1]))


def fpr_at_recall(y_true, y_score, recall: float = TARGET_RECALL) -> float:
    """False positive rate at the operating point that achieves `recall`.

    Prevalence-invariant, unlike PPV: FPR = FP / (FP + TN) depends only on the
    score distribution of the negatives and where the recall threshold falls,
    not on how many positives are in the set. That makes it the number that
    can actually be compared across a fold (5.1% positive) and the
    leaderboard (~1% positive), which PPV@90R cannot be.

    ``tpr`` from ``roc_curve`` is already non-decreasing along the array (as
    the threshold sweeps from high to low), so it can be interpolated over
    directly -- unlike ``ppv_at_recall``, which has to reverse
    ``precision_recall_curve``'s arrays first.
    """
    y_true = np.asarray(y_true)
    if _degenerate(y_true):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, np.asarray(y_score))
    return float(np.interp(recall, tpr, fpr))


def evaluate(y_true, y_score) -> Dict[str, float]:
    """All validation metrics for one epoch's logits."""
    pauc = partial_auc(y_true, y_score)
    return {
        "roc_auc": roc_auc(y_true, y_score),
        "pauc_15_raw": pauc["raw"],
        "pauc_15_std": pauc["std"],
        "ppv_at_90_recall": ppv_at_recall(y_true, y_score),
        "fpr_at_90_recall": fpr_at_recall(y_true, y_score),
    }
