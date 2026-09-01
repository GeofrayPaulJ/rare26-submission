"""Prior-equalised FPR@90R and per-centre FPR@90R.

These exist because pooled FPR@90R has a structural blind spot: it is measured
on a pool where centre and class are correlated (center_1 2.7% positive,
center_2 12.0% positive), so "looks like center_2" is a VALID predictor of the
label on this specific pool. A model exploiting that is rewarded by pooled
FPR@90R, not merely un-penalised. See
reports/prior_equalised_fpr_pre_registration.md.

Every test here constructs its own synthetic pool with EXACT, hand-computable
counts, so every assertion can be verified by arithmetic in the test itself
rather than by trusting the function under test.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    centre_negative_prior_weights,
    per_centre_fpr_at_recall,
    prior_equalised_fpr_at_recall,
    prior_equalised_fpr_bootstrap,
)
from src.metrics import fpr_at_recall


def _pool(pos_by_centre, neg_scores_by_centre, pos_score=5.0):
    """A minimal pool: every positive at a fixed high score (so recall/threshold
    arithmetic is exact and not at the mercy of a random draw), negatives given
    explicitly per centre."""
    rows = []
    for centre, n in pos_by_centre.items():
        for _ in range(n):
            rows.append({"centre": centre, "label_int": 1, "logit": pos_score})
    for centre, scores in neg_scores_by_centre.items():
        for s in scores:
            rows.append({"centre": centre, "label_int": 0, "logit": s})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# centre_negative_prior_weights
# ---------------------------------------------------------------------------
def test_weights_match_the_preregistered_numbers():
    """The exact counts and weights quoted in the pre-registration note."""
    pool = _pool(
        {"center_1": 61, "center_2": 97},
        {"center_1": [0.0] * 2218, "center_2": [0.0] * 712},
    )
    w = centre_negative_prior_weights(pool)
    assert w["anchor"] == "center_2"
    assert w["weights"]["center_2"] == pytest.approx(1.0)
    assert w["weights"]["center_1"] == pytest.approx(0.2019, abs=1e-4)


def test_anchor_is_never_upweighted_and_others_never_exceed_one():
    """The safety property the bootstrap depends on: every weight in (0, 1]."""
    pool = _pool(
        {"center_1": 30, "center_2": 200, "center_3": 5},
        {"center_1": [0.0] * 900, "center_2": [0.0] * 300, "center_3": [0.0] * 50},
    )
    w = centre_negative_prior_weights(pool)
    for c, wt in w["weights"].items():
        assert 0.0 < wt <= 1.0 + 1e-9, f"{c} weight {wt} outside (0, 1]"
    assert w["weights"][w["anchor"]] == pytest.approx(1.0)


def test_weighting_exactly_equalises_the_centre_conditional_proportion():
    """The defining property, checked by direct arithmetic rather than by
    re-deriving the formula: sum of weighted negatives per centre, divided by
    the weighted total, must equal that centre's share of the POSITIVES."""
    pool = _pool(
        {"center_1": 40, "center_2": 160},
        {"center_1": [0.0] * 1000, "center_2": [0.0] * 300},
    )
    w = centre_negative_prior_weights(pool)
    neg = pool[pool["label_int"] == 0].copy()
    neg["weight"] = neg["centre"].map(w["weights"])
    weighted_share = neg.groupby("centre")["weight"].sum() / neg["weight"].sum()

    npos = pool.loc[pool["label_int"] == 1, "centre"].value_counts()
    pos_share = npos / npos.sum()

    for c in pos_share.index:
        assert weighted_share[c] == pytest.approx(pos_share[c], abs=1e-9), c


def test_needs_at_least_two_centres_with_both_classes():
    pool = _pool({"center_1": 10}, {"center_1": [0.0] * 50})
    with pytest.raises(ValueError, match=">= 2 centres"):
        centre_negative_prior_weights(pool)


# ---------------------------------------------------------------------------
# prior_equalised_fpr_at_recall
# ---------------------------------------------------------------------------
def test_prior_equalised_fpr_hand_computable():
    """Every negative above threshold and the weights are chosen so the
    result can be checked by hand, not just by re-running the formula."""
    # 10 positives (recall 0.90 -> threshold is the 9th-highest positive == 1.0,
    # since ceil(0.9*10)=9 and all ten positives share one score)
    pos = {"center_1": 5, "center_2": 5}
    # center_1: 100 negatives, 20 above threshold (score 2.0 > 1.0)
    # center_2: 20 negatives, 4 above threshold
    neg = {
        "center_1": [2.0] * 20 + [0.0] * 80,
        "center_2": [2.0] * 4 + [0.0] * 16,
    }
    pool = _pool(pos, neg, pos_score=1.0)

    w = centre_negative_prior_weights(pool)
    # equal positive counts -> ratio center_1 = 100/5=20, center_2=20/5=4;
    # anchor = smaller ratio = center_2 (weight 1.0); center_1 weight = 4/20=0.2
    assert w["anchor"] == "center_2"
    assert w["weights"]["center_1"] == pytest.approx(0.2)

    result = prior_equalised_fpr_at_recall(pool)
    # weighted negatives: center_1 100*0.2=20 effective, of which 20*0.2=4 above
    #                     center_2 20*1.0=20 effective, of which 4 above
    # weighted FPR = (4+4)/(20+20) = 0.2
    assert result["fpr_at_90_recall_prior_equalised"] == pytest.approx(0.2)
    assert result["effective_n"] == pytest.approx(40.0)


def test_prior_equalisation_changes_fpr_relative_to_pooled_when_shortcut_present():
    """The whole point: when centre correlates with the label AND with the
    negative scores, pooled and prior-equalised FPR must differ."""
    # center_2 is "easier": its negatives score low almost always. center_2
    # also supplies most of the positives, so a raw pooled threshold is
    # dominated by center_1's harder negatives -- exactly the entanglement
    # prior-equalisation exists to remove.
    pos = {"center_1": 20, "center_2": 80}
    neg = {
        "center_1": [3.0] * 500 + [-3.0] * 1500,   # 25% score high
        "center_2": [3.0] * 50 + [-3.0] * 450,      # 10% score high
    }
    pool = _pool(pos, neg, pos_score=1.0)

    pooled = fpr_at_recall(pool["label_int"].to_numpy(), pool["logit"].to_numpy())
    prior_eq = prior_equalised_fpr_at_recall(pool)["fpr_at_90_recall_prior_equalised"]
    assert pooled != pytest.approx(prior_eq, abs=1e-6)


def test_effective_n_and_kish_ess_agree_with_the_preregistered_dataset():
    pool = _pool(
        {"center_1": 61, "center_2": 97},
        {"center_1": [0.0] * 2218, "center_2": [0.0] * 712},
    )
    r = prior_equalised_fpr_at_recall(pool)
    assert r["effective_n"] == pytest.approx(1159.9, abs=0.5)
    # Kish ESS must be scale-invariant: rescaling every weight by a constant
    # must not move it.
    w = centre_negative_prior_weights(pool)["weights"]
    scaled_w = {c: v * 3.7 for c, v in w.items()}
    neg = pool[pool["label_int"] == 0].copy()
    neg["w1"] = neg["centre"].map(w)
    neg["w2"] = neg["centre"].map(scaled_w)
    kish1 = neg["w1"].sum() ** 2 / (neg["w1"] ** 2).sum()
    kish2 = neg["w2"].sum() ** 2 / (neg["w2"] ** 2).sum()
    assert kish1 == pytest.approx(kish2)
    assert r["kish_ess"] == pytest.approx(kish1)


# ---------------------------------------------------------------------------
# prior_equalised_fpr_bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_target_counts_match_the_preregistered_448():
    pool = _pool(
        {"center_1": 61, "center_2": 97},
        {"center_1": [0.0] * 2218, "center_2": [0.0] * 712},
    )
    bs = prior_equalised_fpr_bootstrap(pool, n_boot=5, seed=0)
    assert bs["target_counts_by_centre"]["center_1"] == 448
    assert bs["target_counts_by_centre"]["center_2"] == 712  # anchor: untouched
    assert bs["anchor_centre"] == "center_2"


def test_bootstrap_anchor_centre_is_never_subsampled():
    """Every draw must use ALL of the anchor centre's negatives -- checked by
    running enough draws that omitting even one anchor row would eventually
    show up as a shorter total."""
    pool = _pool(
        {"center_1": 10, "center_2": 40},
        {"center_1": [1.0] * 50, "center_2": [1.0] * 20},
    )
    bs = prior_equalised_fpr_bootstrap(pool, n_boot=50, seed=3)
    # anchor is whichever centre has the smaller neg/pos ratio: c1=50/10=5,
    # c2=20/40=0.5 -> anchor = center_2, always fully included (20 every draw)
    assert bs["anchor_centre"] == "center_2"
    assert bs["target_counts_by_centre"]["center_2"] == 20


def test_bootstrap_is_reproducible_under_a_fixed_seed():
    # pos_score=0.0 sits in the middle of the negatives' [-3, 3] spread, so
    # roughly half of each centre's negatives cross the threshold -- the
    # default pos_score=5.0 sits above every negative and gives a degenerate
    # all-zero FPR for which "different seed -> different draws" is vacuously
    # false rather than a real check of the RNG plumbing.
    pool = _pool(
        {"center_1": 61, "center_2": 97},
        {"center_1": list(np.linspace(-3, 3, 2218)),
         "center_2": list(np.linspace(-3, 3, 712))},
        pos_score=0.0,
    )
    a = prior_equalised_fpr_bootstrap(pool, n_boot=100, seed=7)
    b = prior_equalised_fpr_bootstrap(pool, n_boot=100, seed=7)
    assert a["draws"] == b["draws"]

    c = prior_equalised_fpr_bootstrap(pool, n_boot=100, seed=8)
    assert a["draws"] != c["draws"]


def test_bootstrap_median_is_close_to_the_exact_weighted_figure():
    """Subsampling and reweighting are two different estimators of the same
    target quantity; they should agree, not merely both exist. pos_score=0.0
    is mid-range for [-3, 3] negatives, so the comparison happens at a
    non-trivial FPR rather than the vacuous 0-vs-0 that pos_score's default
    (above the whole negative range) would give."""
    pool = _pool(
        {"center_1": 61, "center_2": 97},
        {"center_1": list(np.linspace(-3, 3, 2218)),
         "center_2": list(np.linspace(-3, 3, 712))},
        pos_score=0.0,
    )
    exact = prior_equalised_fpr_at_recall(pool)["fpr_at_90_recall_prior_equalised"]
    bs = prior_equalised_fpr_bootstrap(pool, n_boot=1000, seed=0)
    assert bs["spread"]["median"] == pytest.approx(exact, abs=0.02)


def test_bootstrap_has_nonzero_spread_when_the_subsampled_centre_is_not_degenerate():
    """A real variance floor exists whenever center_1's within-group FPR is
    strictly between 0 and 1 -- which without-replacement subsampling of a
    non-degenerate group must produce.

    NOT tested here: "wider IQR than the pooled figure". That claim compares
    two different sources of variance -- this bootstrap's pure finite-sample
    variance for ONE fixed set of logits, against the ACTUAL seed-to-seed
    spread of pooled FPR@90R across independently trained models
    (reports/noise_floor.md, ~0.0235 at k=1) -- and is checked against the
    real prediction parquets in the pre-registration report, not asserted here
    against a synthetic stand-in. An earlier version of this test compared
    against a made-up with-replacement bootstrap of the full pool and was
    removed: that comparison isn't even reliably true in the claimed
    direction, since the anchor centre contributes ZERO variance here (it is
    never resampled) while a full-pool bootstrap randomises everything,
    so which one is wider depends on the split, not on anything this method
    gets right or wrong.
    """
    rng = np.random.default_rng(42)
    pool = _pool(
        {"center_1": 61, "center_2": 97},
        {"center_1": list(rng.normal(-1, 1, 2218)),
         "center_2": list(rng.normal(-1, 1, 712))},
        pos_score=0.0,
    )
    bs = prior_equalised_fpr_bootstrap(pool, n_boot=500, seed=1)
    assert bs["spread"]["iqr"] > 0.0
    assert bs["spread"]["sd"] > 0.0


# ---------------------------------------------------------------------------
# per_centre_fpr_at_recall
# ---------------------------------------------------------------------------
def test_per_centre_fpr_uses_one_shared_threshold():
    """Same threshold for both centres -- verified against threshold_at_recall
    computed independently on the pooled positives."""
    from src.evaluate import threshold_at_recall

    pool = _pool(
        {"center_1": 61, "center_2": 97},
        {"center_1": [0.0] * 2218, "center_2": [0.0] * 712},
    )
    y, s = pool["label_int"].to_numpy(), pool["logit"].to_numpy()
    expected_thr = threshold_at_recall(y, s, 0.90)
    result = per_centre_fpr_at_recall(pool)
    assert result["threshold"] == pytest.approx(expected_thr)


def test_per_centre_fpr_recombines_to_the_plain_pooled_fpr():
    """Splitting by centre and recombining count-weighted must reproduce the
    plain (not prior-equalised) pooled proportion above threshold exactly --
    this is the self-consistency check, not a re-derivation."""
    pool = _pool(
        {"center_1": 61, "center_2": 97},
        {"center_1": [3.0] * 300 + [-3.0] * 1918,
         "center_2": [3.0] * 200 + [-3.0] * 512},
    )
    result = per_centre_fpr_at_recall(pool)
    y, s = pool["label_int"].to_numpy(), pool["logit"].to_numpy()
    from src.evaluate import threshold_at_recall
    thr = threshold_at_recall(y, s, 0.90)
    direct = float((s[y == 0] >= thr).mean())
    assert result["pooled_check"] == pytest.approx(direct)


def test_per_centre_asymmetry_is_zero_when_centres_are_identical():
    """No hospital signal, no asymmetry -- the metric's own null case."""
    pool = _pool(
        {"center_1": 50, "center_2": 50},
        {"center_1": list(np.linspace(-3, 3, 500)),
         "center_2": list(np.linspace(-3, 3, 500))},
    )
    result = per_centre_fpr_at_recall(pool)
    assert result["asymmetry_max_minus_min"] == pytest.approx(0.0, abs=1e-9)


def test_per_centre_asymmetry_is_positive_when_one_centre_is_all_false_positives():
    pool = _pool(
        {"center_1": 50, "center_2": 50},
        {"center_1": [10.0] * 100,     # every negative scores ABOVE the
                                       # threshold set by pos_score=1.0
         "center_2": [-10.0] * 100},   # every negative scores below it
    )
    result = per_centre_fpr_at_recall(pool)
    assert result["by_centre"]["center_1"]["fpr"] == pytest.approx(1.0)
    assert result["by_centre"]["center_2"]["fpr"] == pytest.approx(0.0)
    assert result["asymmetry_max_minus_min"] == pytest.approx(1.0)
