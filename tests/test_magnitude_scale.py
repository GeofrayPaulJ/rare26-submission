"""The identity-anchored magnitude multiplier.

This single number defines the difference between arms A2 and A3, so the ways
it can be silently wrong all get a test:

  * s=1.0 must reproduce the shipped magnitudes EXACTLY, because A3 is defined
    as "the stack as accepted in the sample grid" and a config that merely
    approximates it is a different arm;
  * applying it twice (dump to YAML, read back) must not compound;
  * probabilities must never move, or A2 and A3 would differ in which
    transforms fire as well as how hard, and the comparison would be confounded;
  * every magnitude must move MONOTONICALLY toward doing less as s falls --
    the four inverted parameters (downsample, JPEG quality, photon count) are
    the entire reason this is interpolation rather than multiplication.
"""
import dataclasses

import numpy as np
import pytest

from src.augment import apply_uint8, load_redaction_dist
from src.config import MAGNITUDE_IDENTITY, AugConfig, Config, with_magnitude_scale
from tests.conftest import REDACTION, make_config

FULL = dict(photometric=True, optical=True, sensor=True, compression=True)


def resolved(s):
    """The effective AugConfig at scale `s`, as a run would see it."""
    return Config(**{**make_config().to_dict(),
                     "aug": AugConfig(magnitude_scale=s, **FULL)}).aug


def test_unit_scale_reproduces_the_shipped_config_exactly():
    got = resolved(1.0)
    want = AugConfig(**FULL)
    assert dataclasses.replace(got, magnitude_scale_applied=False) == want


def test_scaling_is_idempotent_across_a_yaml_round_trip(tmp_path):
    cfg = Config(**{**make_config().to_dict(),
                    "aug": AugConfig(magnitude_scale=0.33, **FULL)})
    path = str(tmp_path / "c.yaml")
    cfg.to_yaml(path)
    assert Config.from_yaml(path).aug == cfg.aug

    # and applying the helper again by hand changes nothing
    assert with_magnitude_scale(cfg.aug) == cfg.aug


def test_probabilities_are_untouched():
    a, b = resolved(1.0), resolved(0.33)
    for name in AugConfig._P_FIELDS:
        assert getattr(a, name) == getattr(b, name), name


@pytest.mark.parametrize("s", [0.1, 0.33, 0.5, 0.75])
def test_every_magnitude_moves_toward_its_no_op(s):
    """The property a naive multiply would violate for four parameters."""
    shipped, scaled = resolved(1.0), resolved(s)
    for name, identity in MAGNITUDE_IDENTITY.items():
        v_ship = getattr(shipped, name)
        v_scale = getattr(scaled, name)
        # scaled must sit between the identity and the shipped value ...
        lo, hi = sorted((identity, float(v_ship)))
        assert lo - 1e-9 <= v_scale <= hi + 1e-9, (
            f"{name}: {v_scale} is outside [{lo}, {hi}]")
        # ... and strictly nearer the identity than the shipped value is
        assert abs(v_scale - identity) <= abs(v_ship - identity) + 1e-9, name


def test_the_four_inverted_parameters_get_gentler_not_harsher():
    """Named explicitly, because a plain multiply makes each of these WORSE.

    downsample_m is a minimum scale factor, the JPEG bounds are qualities, and
    poisson_noise_m is a photon count -- for all four, smaller means MORE
    aggressive, so 0.33 x value would strengthen the transform.
    """
    shipped, scaled = resolved(1.0), resolved(0.33)

    assert scaled.downsample_m > shipped.downsample_m      # less resolution loss
    assert scaled.jpeg_q_min > shipped.jpeg_q_min          # higher quality
    assert scaled.jpeg_q_max >= shipped.jpeg_q_max
    assert scaled.poisson_noise_m > shipped.poisson_noise_m  # more photons, less noise

    # a naive multiply would have gone the other way on all four
    for name in ("downsample_m", "jpeg_q_min", "jpeg_q_max", "poisson_noise_m"):
        naive = 0.33 * float(getattr(shipped, name))
        assert float(getattr(scaled, name)) != pytest.approx(naive), name


def test_photon_count_scales_on_noise_amplitude():
    """Shot-noise sigma goes as 1/sqrt(photon count), so a third of the
    amplitude is nine times the photons, not a third of them."""
    shipped, scaled = resolved(1.0), resolved(0.33)
    assert scaled.poisson_noise_m == pytest.approx(
        shipped.poisson_noise_m / (0.33 ** 2), rel=1e-9)


def test_tint_fraction_is_a_shape_not_a_magnitude():
    assert resolved(0.33).white_balance_tint_frac == \
        resolved(1.0).white_balance_tint_frac


def test_scaled_stack_is_visibly_gentler_but_still_active():
    """End to end: at a third of the magnitude the same transforms still fire
    on the same draws, and the result sits closer to the source image."""
    rng_img = np.random.default_rng(3)
    img = rng_img.integers(0, 256, size=(192, 192, 3), dtype=np.uint8)
    dist = load_redaction_dist(REDACTION)

    def mean_abs_delta(aug):
        deltas = []
        for seed in range(12):
            out = apply_uint8(img, aug, np.random.default_rng(seed), True, dist)
            deltas.append(float(np.abs(out.astype(np.float64)
                                       - img.astype(np.float64)).mean()))
        return float(np.mean(deltas))

    strong = mean_abs_delta(dataclasses.replace(resolved(1.0),
                                                random_black_boxes=False))
    gentle = mean_abs_delta(dataclasses.replace(resolved(0.33),
                                                random_black_boxes=False))
    assert gentle < strong, f"s=0.33 ({gentle:.2f}) not gentler than s=1.0 ({strong:.2f})"
    assert gentle > 0.5, "s=0.33 did essentially nothing; it should still augment"


def test_out_of_range_scale_is_rejected():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="magnitude_scale"):
            AugConfig(magnitude_scale=bad, **FULL).validate()
