"""
08_score.py -- The official scoring function, copied exactly.

WHY: This is how the challenge scores you. It is NOT accuracy and NOT AUC.
Do not approximate it, do not substitute something easier, do not re-derive it.
The bootstrap logic below is taken from the organisers' own
evaluation_Grand-Challenge.py so your local numbers mean the same thing theirs do.

ALSO INCLUDED: noise_floor(). Run this once before you tune anything. It tells
you how much your score moves between identical runs with different random
seeds. Any "improvement" smaller than that number is not an improvement -- it
is noise, and chasing it will waste weeks.

USAGE (as a library):
    from score import official_score, noise_floor
    result = official_score(y_true, y_pred)
    print(result["Score"])
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def official_score(y_true, y_pred, n_iterations=1000, imbalance_ratio=100, seed=None):
    """Median PPV@90%Recall across 1000 prevalence-matched bootstrap draws.

    All negatives are kept in every draw. Positives are resampled WITH
    replacement down to a 1:100 ratio, mimicking real clinical prevalence.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)

    ndbe_idx = np.where(y_true == 0)[0]
    neo_idx = np.where(y_true == 1)[0]

    auc_full = roc_auc_score(y_true, y_pred)
    auprc_full = average_precision_score(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    ppv90_full = np.interp(0.9, rec[::-1], prec[::-1])

    n_sample = max(1, int(len(ndbe_idx) / imbalance_ratio))
    boot = []
    for _ in range(n_iterations):
        sampled = np.concatenate([
            ndbe_idx,
            rng.choice(neo_idx, size=n_sample, replace=True),
        ])
        yt, yp = y_true[sampled], y_pred[sampled]
        prec, rec, _ = precision_recall_curve(yt, yp)
        boot.append((
            roc_auc_score(yt, yp),
            average_precision_score(yt, yp),
            np.interp(0.9, rec[::-1], prec[::-1]),
        ))

    boot = np.array(boot)
    return {
        "Score": np.median(boot[:, 2]),
        "PPV@90RECALL": np.median(boot[:, 2]),
        "PPV@90RECALL CI Lower": np.percentile(boot[:, 2], 2.5),
        "PPV@90RECALL CI Upper": np.percentile(boot[:, 2], 97.5),
        "AUROC": np.median(boot[:, 0]),
        "AUPRC": np.median(boot[:, 1]),
        "AUROC Full": auc_full,
        "AUPRC Full": auprc_full,
        "PPV@90RECALL Full": ppv90_full,
    }


def noise_floor(scores):
    """Given scores from N identical runs at different seeds, report the spread.

    Anything smaller than this is not a real improvement. Write the number down
    and hold to it.
    """
    scores = np.asarray(scores)
    lo, hi = np.percentile(scores, [25, 75])
    return {
        "n_runs": len(scores),
        "median": float(np.median(scores)),
        "iqr": float(hi - lo),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "range": float(scores.max() - scores.min()),
        "minimum_detectable_effect": float(scores.max() - scores.min()),
    }


if __name__ == "__main__":
    # Smoke test on synthetic data.
    rng = np.random.default_rng(0)
    n_neg, n_pos = 2937, 158
    y = np.r_[np.zeros(n_neg), np.ones(n_pos)]
    p = np.r_[rng.beta(2, 8, n_neg), rng.beta(6, 4, n_pos)]
    r = official_score(y, p, n_iterations=200, seed=0)
    print("Smoke test (synthetic scores, not real):")
    for k, v in r.items():
        print(f"  {k:26s} {v:.4f}")
