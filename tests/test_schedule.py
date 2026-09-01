"""Test 12: the LR schedule is the shape the config asks for.

Warmup exists because AdamW's second-moment estimate is garbage for the first
few dozen steps, and a pretrained backbone hit with a full-size step before it
settles loses features it will not recover. 3% of 30 epochs is about one epoch,
which is the intent -- so the schedule is checked against total steps, not
against a hardcoded step count that would silently drift if the batch size
changed.
"""
import numpy as np
import pytest
import torch

from src.train import make_scheduler

BASE_LR = 1e-4


def _curve(total_steps, warmup_frac, base_lr=BASE_LR):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=base_lr)
    sched = make_scheduler(opt, total_steps, warmup_frac)
    lrs = []
    for _ in range(total_steps):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return np.array(lrs)


def test_warmup_is_linear_and_spans_three_percent_of_steps():
    total = 1000
    lrs = _curve(total, 0.03)
    warmup = int(round(total * 0.03))  # 30 steps

    assert lrs[0] < BASE_LR, "step 0 must not start at the full learning rate"
    assert lrs[warmup - 1] == pytest.approx(BASE_LR, rel=1e-12), (
        "warmup must reach exactly the base lr on its last step"
    )
    # evenly spaced during warmup
    steps = np.diff(lrs[:warmup])
    assert np.allclose(steps, steps[0])
    assert steps[0] > 0


def test_peak_is_reached_once_and_then_decays_monotonically():
    total = 1000
    lrs = _curve(total, 0.03)
    peak = int(np.argmax(lrs))
    assert lrs[peak] <= BASE_LR + 1e-12, "schedule must never exceed the base lr"
    after = lrs[peak:]
    assert np.all(np.diff(after) <= 1e-15), "cosine leg must not go back up"


def test_cosine_ends_near_zero():
    lrs = _curve(1000, 0.03)
    assert lrs[-1] < BASE_LR * 0.01


def test_cosine_leg_is_invariant_to_step_count():
    """Halving the step count (say, by doubling the batch) must not change the
    curve the model rides -- only how finely it is sampled.

    Compared on the cosine leg only. Warmup cannot match pointwise: 3% of 1000
    steps is 30 linear increments and 3% of 500 is 15, so the two rise in
    different-sized jumps to the same peak. That granularity is the intended
    behaviour, and test_warmup_length_tracks_the_configured_fraction pins it.
    """
    total_a, total_b, frac = 1000, 500, 0.03
    a = _curve(total_a, frac)[int(round(total_a * frac)):]
    b = _curve(total_b, frac)[int(round(total_b * frac)):]

    at = np.linspace(0.0, 1.0, 50)
    a_at = a[(at * (len(a) - 1)).astype(int)]
    b_at = b[(at * (len(b) - 1)).astype(int)]
    assert np.allclose(a_at, b_at, atol=BASE_LR * 0.01)


def test_warmup_length_tracks_the_configured_fraction():
    """Warmup is a fraction of training, not a fixed number of steps, so it
    still covers ~one epoch when the batch size changes."""
    for total in (200, 500, 1000, 4000):
        lrs = _curve(total, 0.03)
        assert int(np.argmax(lrs)) == int(round(total * 0.03)) - 1


def test_zero_warmup_starts_at_full_lr():
    lrs = _curve(100, 0.0)
    assert lrs[0] == BASE_LR
    assert np.all(np.diff(lrs) <= 1e-15)
