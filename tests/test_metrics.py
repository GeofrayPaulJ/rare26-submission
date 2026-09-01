"""Test 9: the validation metrics mean what they claim to mean.

The one that matters is PPV@90R agreeing with the organisers' scorer. If our
number and theirs are computed differently, every local decision is made against
a metric the leaderboard does not use.
"""
import importlib.util
import os

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.metrics import evaluate, partial_auc, ppv_at_recall, roc_auc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_official_scorer():
    """Import scripts/08_score.py, whose filename is not a valid identifier."""
    path = os.path.join(REPO_ROOT, "scripts", "08_score.py")
    spec = importlib.util.spec_from_file_location("official_score_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(seed=0, n_neg=586, n_pos=31, sep=1.2):
    """Logit-scale scores with the fold's real class ratio (617 rows, 31 pos)."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n_neg, dtype=np.int64), np.ones(n_pos, dtype=np.int64)]
    s = np.r_[rng.normal(0.0, 1.0, n_neg), rng.normal(sep, 1.0, n_pos)]
    return y, s


def test_ppv_at_90_recall_matches_official_scorer():
    official = _load_official_scorer()
    for seed in range(5):
        y, s = _synthetic(seed=seed)
        # "PPV@90RECALL Full" is the un-bootstrapped point estimate, which is
        # what a per-epoch validation number is
        expected = official.official_score(y, s, n_iterations=1, seed=0)["PPV@90RECALL Full"]
        assert ppv_at_recall(y, s) == pytest.approx(expected, abs=1e-12)


def test_partial_auc_standardised_matches_sklearn():
    for seed in range(5):
        y, s = _synthetic(seed=seed)
        assert partial_auc(y, s, 0.15)["std"] == pytest.approx(
            roc_auc_score(y, s, max_fpr=0.15), abs=1e-9
        )


def test_partial_auc_conventions_at_the_extremes():
    """raw: random = max_fpr/2 = 0.075, perfect = 1.0. std: 0.5 and 1.0."""
    y = np.r_[np.zeros(500, dtype=np.int64), np.ones(500, dtype=np.int64)]

    perfect = np.r_[np.zeros(500), np.ones(500)]
    p = partial_auc(y, perfect, 0.15)
    assert p["raw"] == pytest.approx(1.0, abs=1e-6)
    assert p["std"] == pytest.approx(1.0, abs=1e-6)

    # a genuinely uninformative ranker, averaged over draws to beat sampling noise
    raws, stds = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        r = partial_auc(y, rng.normal(size=1000), 0.15)
        raws.append(r["raw"])
        stds.append(r["std"])
    assert np.mean(raws) == pytest.approx(0.075, abs=0.02)
    assert np.mean(stds) == pytest.approx(0.5, abs=0.05)


def test_metrics_are_invariant_to_monotone_rescaling():
    """All three are ranking metrics, so applying a sigmoid must not move them.
    This is why src/io.py can store raw logits and lose nothing."""
    y, s = _synthetic(seed=3)
    a = evaluate(y, s)
    b = evaluate(y, 1.0 / (1.0 + np.exp(-s)))
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_single_class_is_nan_not_a_crash():
    """A degenerate fold must not take a 30-epoch run down at epoch 17."""
    y = np.zeros(50, dtype=np.int64)
    s = np.linspace(0, 1, 50)
    assert np.isnan(roc_auc(y, s))
    assert np.isnan(ppv_at_recall(y, s))
    assert np.isnan(partial_auc(y, s)["std"])


def test_evaluate_reports_all_five_fields():
    y, s = _synthetic(seed=1)
    m = evaluate(y, s)
    assert set(m) == {
        "roc_auc", "pauc_15_raw", "pauc_15_std",
        "ppv_at_90_recall", "fpr_at_90_recall",
    }
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["fpr_at_90_recall"] <= 1.0


def test_fpr_at_90_recall_is_sane():
    """FPR at 90% recall must fall as the classes separate further, and must
    sit strictly between 0 and 1 for an imperfect-but-informative ranker."""
    from src.metrics import fpr_at_recall

    y_easy, s_easy = _synthetic(seed=0, sep=4.0)
    y_hard, s_hard = _synthetic(seed=0, sep=0.5)
    fpr_easy = fpr_at_recall(y_easy, s_easy)
    fpr_hard = fpr_at_recall(y_hard, s_hard)
    assert 0.0 <= fpr_easy <= fpr_hard <= 1.0

    # a perfect ranker reaches 90% recall with zero false positives
    y = np.r_[np.zeros(500), np.ones(500)]
    perfect = np.r_[np.zeros(500), np.ones(500)]
    assert fpr_at_recall(y, perfect) == pytest.approx(0.0, abs=1e-9)
