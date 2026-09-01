"""The acceptance rule of the domain-generalisation screen.

This is the piece of the deliverable that turns numbers into a decision, so it
is tested against constructed cases rather than only against whatever the sweep
happened to produce. The rule, fixed before any result was seen:

    a component is accepted only if it improves BOTH LOCO directions, and the
    improvement in at least one direction exceeds that direction's ensembled
    leave-one-out IQR.

The cases below are the ones that decide something: a clean pass, a
single-direction gain (which the brief calls an accident), a two-direction gain
that is too small to clear the noise, and an exact null.
"""
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_screen():
    path = os.path.join(REPO_ROOT, "scripts", "22_dg_screen.py")
    spec = importlib.util.spec_from_file_location("_dg_screen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCREEN = _load_screen()


def arm(fpr_c1, iqr_c1, fpr_c2, iqr_c2, med_c1=None, med_c2=None):
    """Minimal stand-in for one configuration's per-direction results.

    The LOO-4 medians default to the k=5 points, which makes the robustness
    check agree by construction; the tests that care about disagreement set
    them explicitly.
    """
    return {
        1: {"k5_fpr": fpr_c1,
            "loo4_spread": {"iqr": iqr_c1,
                            "median": fpr_c1 if med_c1 is None else med_c1}},
        2: {"k5_fpr": fpr_c2,
            "loo4_spread": {"iqr": iqr_c2,
                            "median": fpr_c2 if med_c2 is None else med_c2}},
    }


CONTROL = arm(0.2840, 0.0126, 0.1053, 0.0506)


@pytest.mark.parametrize(
    "name,treatment,accepted,both,exceeds",
    [
        # improves both; direction 1's gain (0.08) clears its bar (0.0126)
        ("clean pass", arm(0.2040, 0.0126, 0.1000, 0.0506), True, True, True),
        # large gain in direction 1, WORSE in direction 2 -- the accident case
        ("single direction", arm(0.1840, 0.0126, 0.1600, 0.0506), False, False, True),
        # improves both, but neither gain clears its IQR
        ("too small", arm(0.2830, 0.0126, 0.1050, 0.0506), False, True, False),
        # exactly nothing
        ("null", arm(0.2840, 0.0126, 0.1053, 0.0506), False, False, False),
        # worse everywhere
        ("regression", arm(0.3200, 0.0126, 0.1400, 0.0506), False, False, False),
    ],
)
def test_rule(name, treatment, accepted, both, exceeds):
    got = SCREEN.apply_acceptance(treatment, CONTROL, name)
    assert got["accepted"] is accepted, name
    assert got["improves_both_directions"] is both, name
    assert got["exceeds_iqr_somewhere"] is exceeds, name


def test_a_single_direction_gain_is_never_accepted():
    """The rule's whole point. However large the gain in one direction, a
    worsening in the other must not be accepted -- twelve unseen centres are
    not obliged to resemble whichever direction happened to improve."""
    huge = arm(0.0100, 0.0126, 0.9000, 0.0506)
    got = SCREEN.apply_acceptance(huge, CONTROL, "enormous but one-sided")
    assert got["exceeds_iqr_somewhere"] is True
    assert got["improves_both_directions"] is False
    assert got["accepted"] is False


def test_iqr_bar_is_the_larger_of_the_two_arms():
    """The quantity under test is a difference between two noisy ensembles, so
    the yardstick must not be whichever arm happens to be quieter."""
    treatment = arm(0.2700, 0.0400, 0.1000, 0.0100)
    got = SCREEN.apply_acceptance(treatment, CONTROL, "x")
    assert got["directions"][1]["iqr_bar"] == pytest.approx(0.0400)  # treatment
    assert got["directions"][2]["iqr_bar"] == pytest.approx(0.0506)  # control


def test_loo4_median_robustness_flags_a_k5_artefact():
    """The k=5 comparison and the LOO-4 median comparison can disagree, and
    when they do the verdict rests on one lucky ensemble. The report says so
    rather than quietly reporting the k=5 answer."""
    # k=5 says the treatment improved direction 1; the LOO-4 medians say the
    # opposite, i.e. the k=5 treatment point was the fortunate one
    treatment = arm(0.2000, 0.0126, 0.1000, 0.0506, med_c1=0.3000)
    got = SCREEN.apply_acceptance(treatment, CONTROL, "artefact")
    d = got["directions"][1]
    assert d["improvement"] > 0
    assert d["improvement_loo4_median"] < 0
    assert d["sign_agrees_with_loo4_median"] is False

    # and the agreeing case is reported as agreeing
    honest = arm(0.2000, 0.0126, 0.1000, 0.0506, med_c1=0.2100)
    assert SCREEN.apply_acceptance(honest, CONTROL, "honest")[
        "directions"][1]["sign_agrees_with_loo4_median"] is True


def test_outcome_labels_distinguish_no_change_from_worse():
    """A zero delta is not a worsening; reporting it as one would misdescribe
    the most likely result of any screen."""
    d = SCREEN.apply_acceptance(CONTROL, CONTROL, "x")["directions"]
    assert SCREEN.outcome_label(d[1]) == "no change"

    worse = SCREEN.apply_acceptance(arm(0.30, 0.0126, 0.12, 0.0506),
                                    CONTROL, "x")["directions"]
    assert SCREEN.outcome_label(worse[1]) == "WORSE"

    small = SCREEN.apply_acceptance(arm(0.2830, 0.0126, 0.1050, 0.0506),
                                    CONTROL, "x")["directions"]
    assert SCREEN.outcome_label(small[1]) == "improved, under IQR"

    big = SCREEN.apply_acceptance(arm(0.2040, 0.0126, 0.1000, 0.0506),
                                  CONTROL, "x")["directions"]
    assert SCREEN.outcome_label(big[1]) == "PASS"
