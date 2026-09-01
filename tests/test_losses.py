"""Tests for src/losses.py's partial-AUC surrogate.

IMPLEMENTATION-ONLY REVIEW GATE. Nothing in src/losses.py is wired into
training; these tests exist so the surrogate can be reviewed and trusted
BEFORE that decision, not after.

Four things are pinned:

  1. Agreement with src.metrics.partial_auc (the project's one brute-force
     reference) to 1e-6, on synthetic data constructed so the two
     constructions are mathematically exactly equal, not merely close (see
     src/losses.py's module docstring, point 2, and test_exact_equivalence_*
     below for the condition this rests on: max_fpr * n_neg an exact
     integer, no tie straddling the boundary negative).
  2. Gradient finiteness on ordinary random batches.
  3. Behaviour at ties -- both the hard step-function indicator (exactly
     0.5) and the smooth surrogate under heavy ties (finite, no NaN).
  4. Behaviour when a batch has zero positives (or zero negatives) --
     exactly 0.0, connected to autograd, never NaN.
"""
import numpy as np
import pytest
import torch

from src.losses import (
    DEFAULT_MAX_FPR,
    partial_auc_loss,
    partial_auc_surrogate,
)
from src.metrics import partial_auc


def _synthetic_split(seed: int, n_neg: int, n_pos: int, sep: float = 1.5):
    """Continuous scores (normal draws), split by class. n_neg chosen by the
    caller so max_fpr * n_neg is an exact integer for the equivalence tests;
    continuous draws make an exact tie at the top-m boundary a
    probability-zero event, so this also satisfies the "no boundary tie"
    condition in practice."""
    rng = np.random.default_rng(seed)
    neg = rng.normal(0.0, 1.0, n_neg)
    pos = rng.normal(sep, 1.0, n_pos)
    y = np.r_[np.zeros(n_neg, dtype=np.int64), np.ones(n_pos, dtype=np.int64)]
    s = np.r_[neg, pos]
    return y, s, neg, pos


# ---------------------------------------------------------------------------
# 1. Agreement with the brute-force reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_exact_equivalence_hard_mode_matches_brute_force(seed):
    """n_neg=200, max_fpr=0.15 -> m=30 exactly (no rounding). Continuous
    scores make a tie exactly at the m/m+1 boundary a probability-zero
    event. Under those two conditions, the hard-indicator surrogate is
    EXACTLY the trapezoidal partial_auc's "raw" figure -- not an
    approximation -- so 1e-6 is a loose bound, not a target being pushed
    against; this should agree far tighter than that in practice."""
    y, s, neg, pos = _synthetic_split(seed, n_neg=200, n_pos=37)
    ref = partial_auc(y, s, max_fpr=DEFAULT_MAX_FPR)["raw"]

    surrogate = partial_auc_surrogate(
        torch.tensor(pos, dtype=torch.float64),
        torch.tensor(neg, dtype=torch.float64),
        max_fpr=DEFAULT_MAX_FPR, hard=True,
    ).item()

    assert abs(surrogate - ref) < 1e-6, (
        f"seed={seed}: surrogate={surrogate!r} ref={ref!r} "
        f"diff={abs(surrogate - ref):.3e}")


def test_exact_equivalence_holds_across_several_fpr_and_sizes():
    """Same identity, different (n_neg, max_fpr) pairs that keep
    max_fpr * n_neg an exact integer, to rule out the seed=range(10) case
    above being a coincidence of one particular m."""
    cases = [(0.15, 200), (0.15, 400), (0.10, 300), (0.20, 250), (0.05, 400)]
    for max_fpr, n_neg in cases:
        assert (max_fpr * n_neg) == int(max_fpr * n_neg), \
            f"test bug: {max_fpr} * {n_neg} is not an integer"
        y, s, neg, pos = _synthetic_split(seed=123, n_neg=n_neg, n_pos=41)
        ref = partial_auc(y, s, max_fpr=max_fpr)["raw"]
        surrogate = partial_auc_surrogate(
            torch.tensor(pos, dtype=torch.float64),
            torch.tensor(neg, dtype=torch.float64),
            max_fpr=max_fpr, hard=True,
        ).item()
        assert abs(surrogate - ref) < 1e-6, (
            f"max_fpr={max_fpr} n_neg={n_neg}: surrogate={surrogate!r} "
            f"ref={ref!r}")


def test_perfect_ranker_hits_the_documented_ceiling():
    """A ranker that places every positive above every negative should hit
    partial_auc's documented perfect-ranker value: raw = 1.0 exactly (see
    src/metrics.py's partial_auc docstring)."""
    neg = torch.arange(0.0, 200.0)          # 0..199
    pos = torch.arange(1000.0, 1041.0)      # strictly above every negative
    surrogate = partial_auc_surrogate(pos, neg, max_fpr=0.15, hard=True).item()
    assert surrogate == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Gradient finiteness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_pos,n_neg", [(31, 586), (1, 50), (100, 100), (5, 5)])
def test_gradient_finiteness_on_random_batches(n_pos, n_neg):
    torch.manual_seed(0)
    logits_pos = torch.randn(n_pos, requires_grad=True)
    logits_neg = torch.randn(n_neg, requires_grad=True)

    loss = partial_auc_loss(logits_pos, logits_neg)
    assert torch.isfinite(loss)

    loss.backward()
    assert logits_pos.grad is not None and logits_neg.grad is not None
    assert torch.isfinite(logits_pos.grad).all()
    assert torch.isfinite(logits_neg.grad).all()


def test_gradient_finiteness_with_extreme_logit_magnitudes():
    """Logit magnitudes this project actually sees (see src/model.py's fp32
    classifier-head docstring: magnitude ~8.5 at the operating point, and
    training logits can range further before convergence) should not blow
    the sigmoid's gradient up to inf or down to a silent zero everywhere."""
    logits_pos = torch.tensor([50.0, -50.0, 8.5, 0.0], requires_grad=True)
    logits_neg = torch.tensor([-50.0, 50.0, -8.5, 0.0, 100.0], requires_grad=True)
    loss = partial_auc_loss(logits_pos, logits_neg, temperature=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits_pos.grad).all()
    assert torch.isfinite(logits_neg.grad).all()


# ---------------------------------------------------------------------------
# 3. Behaviour at ties
# ---------------------------------------------------------------------------
def test_hard_indicator_gives_exact_half_credit_at_a_tie():
    pos = torch.tensor([3.0, 3.0, 3.0])
    neg = torch.tensor([3.0, 3.0])  # every pos/neg pair is an exact tie
    surrogate = partial_auc_surrogate(pos, neg, max_fpr=1.0, hard=True)
    # max_fpr=1.0 with n_neg=2 -> m=2, every pair compared, every diff==0
    assert surrogate.item() == pytest.approx(0.5, abs=0.0)


def test_soft_surrogate_is_finite_and_gradient_safe_under_heavy_ties():
    """Many exactly-equal scores, including ties that cross the positive/
    negative boundary -- the case the hard top-m SELECTION step handles
    ambiguously (see src/losses.py module docstring) but the smooth
    indicator must still never produce NaN/Inf, in value or gradient."""
    torch.manual_seed(1)
    logits_pos = torch.full((20,), 2.0, requires_grad=True)
    logits_neg = torch.full((30,), 2.0, requires_grad=True)  # ties pos==neg
    loss = partial_auc_loss(logits_pos, logits_neg)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits_pos.grad).all()
    assert torch.isfinite(logits_neg.grad).all()

    # sigmoid(0) == 0.5 exactly -> the all-tied surrogate should be ~0.5
    surrogate = partial_auc_surrogate(
        torch.full((20,), 2.0), torch.full((30,), 2.0), max_fpr=1.0)
    assert surrogate.item() == pytest.approx(0.5, abs=1e-6)


def test_soft_surrogate_converges_toward_hard_as_temperature_shrinks():
    """Sanity check on the temperature parameter's documented role: a much
    smaller temperature should move the smooth surrogate closer to the exact
    hard-indicator value than a larger one does, on a batch with a real
    (non-tied) separation."""
    torch.manual_seed(2)
    pos = torch.randn(20) + 2.0
    neg = torch.randn(40)
    hard = partial_auc_surrogate(pos, neg, max_fpr=0.5, hard=True).item()
    soft_loose = partial_auc_surrogate(pos, neg, max_fpr=0.5, temperature=2.0).item()
    soft_tight = partial_auc_surrogate(pos, neg, max_fpr=0.5, temperature=0.05).item()
    assert abs(soft_tight - hard) < abs(soft_loose - hard)


# ---------------------------------------------------------------------------
# 4. Zero positives (or zero negatives) in a batch
# ---------------------------------------------------------------------------
def test_zero_positives_returns_exact_zero_not_nan():
    logits_pos = torch.zeros(0, requires_grad=True)
    logits_neg = torch.randn(10, requires_grad=True)
    loss = partial_auc_loss(logits_pos, logits_neg)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()  # must not raise
    assert torch.isfinite(logits_neg.grad).all()
    assert torch.count_nonzero(logits_neg.grad).item() == 0


def test_zero_negatives_returns_exact_zero_not_nan():
    logits_pos = torch.randn(10, requires_grad=True)
    logits_neg = torch.zeros(0, requires_grad=True)
    loss = partial_auc_loss(logits_pos, logits_neg)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits_pos.grad).all()
    assert torch.count_nonzero(logits_pos.grad).item() == 0


def test_both_empty_returns_exact_zero_not_nan():
    logits_pos = torch.zeros(0, requires_grad=True)
    logits_neg = torch.zeros(0, requires_grad=True)
    loss = partial_auc_loss(logits_pos, logits_neg)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()  # must not raise even with two empty leaves


def test_surrogate_itself_also_degenerate_safe():
    """partial_auc_surrogate (not just the loss wrapper) is degenerate-safe
    too, since it is documented and usable standalone."""
    out = partial_auc_surrogate(torch.zeros(0), torch.randn(5))
    assert torch.isfinite(out) and out.item() == 0.0


# ---------------------------------------------------------------------------
# 5. Monotonicity -- the test a sign error cannot pass
# ---------------------------------------------------------------------------
def test_loss_decreases_when_a_positive_score_rises():
    """Raise one positive's score (the lowest one, so it is inside the
    bottom-quartile restriction) and the loss must strictly decrease. The
    equality/finiteness tests above would all pass a global sign flip; this
    cannot."""
    torch.manual_seed(3)
    neg = torch.randn(40)
    pos = torch.randn(12)                       # overlapping -> active hinges
    base = partial_auc_loss(pos, neg).item()
    lifted = pos.clone()
    lifted[lifted.argmin()] += 2.0
    assert partial_auc_loss(lifted, neg).item() < base


def test_loss_increases_when_a_selected_negative_rises():
    """The mirror direction: push the top negative higher and the loss must
    not decrease (strictly increases while its hinges are active)."""
    torch.manual_seed(4)
    neg = torch.randn(40)
    pos = torch.randn(12)
    base = partial_auc_loss(pos, neg).item()
    pushed = neg.clone()
    pushed[pushed.argmax()] += 2.0
    assert partial_auc_loss(pos, pushed).item() >= base


# ---------------------------------------------------------------------------
# 6. Gradient CORRECTNESS (autograd vs finite differences), not just finiteness
# ---------------------------------------------------------------------------
def test_gradcheck_fp64_small_batch():
    """torch.autograd.gradcheck on an fp64 batch. Values are drawn
    continuously so no pair sits exactly on the hinge kink or the topk
    boundary, where the loss is (measure-zero) non-differentiable."""
    torch.manual_seed(5)
    pos = (torch.randn(6, dtype=torch.float64) * 1.3).requires_grad_(True)
    neg = (torch.randn(15, dtype=torch.float64) * 1.3).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda p, n: partial_auc_loss(p, n, max_fpr=0.4, pos_fraction=0.5),
        (pos, neg), eps=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# 7. Two-sided restriction and ceil() selection
# ---------------------------------------------------------------------------
def test_two_sided_restriction_gives_top_positives_zero_gradient():
    """With pos_fraction=0.25, only the bottom-quartile positives are in the
    pair grid -- a well-separated top positive must receive exactly zero
    gradient from the loss."""
    torch.manual_seed(6)
    pos = torch.tensor([-1.0, -0.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                       requires_grad=True)
    neg = torch.randn(20)
    loss = partial_auc_loss(pos, neg, pos_fraction=0.25)
    loss.backward()
    # ceil(0.25 * 8) = 2 -> only the two lowest (-1.0, -0.5) selected
    assert torch.count_nonzero(pos.grad[2:]).item() == 0
    assert torch.count_nonzero(pos.grad[:2]).item() > 0


def test_selection_uses_ceil_with_floor_of_one():
    from src.losses import select_restricted
    # 7 negatives at max_fpr=0.15: round() would select 1 (1.05 -> 1), but
    # ceil() selects 2. 3 positives at pos_fraction=0.25: ceil(0.75) = 1.
    pos = torch.arange(3.0)
    neg = torch.arange(7.0)
    sel_pos, sel_neg = select_restricted(pos, neg, max_fpr=0.15, pos_fraction=0.25)
    assert sel_neg.numel() == 2
    assert sel_pos.numel() == 1
    # tiny batch: never zero selected
    sel_pos, sel_neg = select_restricted(torch.zeros(1), torch.zeros(1),
                                         max_fpr=0.05, pos_fraction=0.05)
    assert sel_neg.numel() == 1 and sel_pos.numel() == 1


# ---------------------------------------------------------------------------
# 8. Squared hinge: gradient grows with margin violation (anti-saturation)
# ---------------------------------------------------------------------------
def test_hinge_gradient_grows_with_margin_violation():
    """The sigmoid this replaced SATURATES on badly-misranked pairs; the
    squared hinge's gradient must GROW with the violation instead."""
    def grad_at(sep: float) -> float:
        pos = torch.tensor([-sep], requires_grad=True)   # misranked by `sep`
        neg = torch.tensor([0.0])
        loss = partial_auc_loss(pos, neg, max_fpr=1.0, pos_fraction=1.0)
        loss.backward()
        return abs(float(pos.grad[0]))

    g_small, g_large = grad_at(1.0), grad_at(10.0)
    assert g_large > g_small * 4  # linear growth: (10+1)/(1+1) = 5.5x


# ---------------------------------------------------------------------------
# 9. fp32 under autocast / low-precision inputs
# ---------------------------------------------------------------------------
def test_low_precision_inputs_are_promoted_to_fp32():
    pos16 = torch.randn(8, dtype=torch.bfloat16)
    neg16 = torch.randn(20, dtype=torch.bfloat16)
    loss = partial_auc_loss(pos16, neg16)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    surr = partial_auc_surrogate(pos16, neg16)
    assert surr.dtype == torch.float32


def test_fp64_inputs_stay_fp64_for_gradcheck():
    pos64 = torch.randn(8, dtype=torch.float64)
    neg64 = torch.randn(20, dtype=torch.float64)
    assert partial_auc_loss(pos64, neg64).dtype == torch.float64


# ---------------------------------------------------------------------------
# 10. Warmup / anneal schedule
# ---------------------------------------------------------------------------
def test_beta_schedule_warmup_anneal_and_hold():
    from src.losses import pauc_beta_schedule
    target = 0.15
    # epochs 0-4: BCE only
    for e in range(5):
        assert pauc_beta_schedule(e, target) is None
    # epoch 5: anneal starts at 1.0 exactly
    assert pauc_beta_schedule(5, target) == pytest.approx(1.0)
    # epoch 9: anneal ends at the target exactly
    assert pauc_beta_schedule(9, target) == pytest.approx(target)
    # strictly decreasing across the anneal window
    betas = [pauc_beta_schedule(e, target) for e in range(5, 10)]
    assert all(a > b for a, b in zip(betas, betas[1:]))
    # held at target thereafter
    assert pauc_beta_schedule(29, target) == pytest.approx(target)
    # degenerate anneal window collapses straight to target
    assert pauc_beta_schedule(5, target, anneal_epochs=1) == pytest.approx(target)
