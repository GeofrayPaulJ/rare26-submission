"""Component A: the centre x class balanced sampler.

Three claims are pinned here:

  1. On a training set holding both centres, one drawn epoch really does split
     ~25/25/25/25 across the four (centre, class) strata -- measured from real
     draws, not inferred from the weight vector.
  2. Measuring that does not perturb the stream training subsequently consumes.
  3. On a SINGLE-centre training set the new sampler is exactly, not
     approximately, the old class-balanced one. Both leave-one-centre-out
     splits are single-centre by construction, so this is the property that
     decides whether a LOCO arm using it is a new experiment or a duplicate of
     one already run.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    BarrettDataset,
    make_balanced_centre_class_sampler,
    make_sampler,
    make_weighted_sampler,
    stratum_draw_counts,
)
from src.folds import get_holdout_split
from tests.conftest import MANIFEST, make_config


def _dataset(filepaths, manifest_df, **cfg_overrides):
    """Dataset over `filepaths` with no image decoding: the sampler only ever
    reads labels and centres out of the manifest, so paying for a RAM cache of
    2279 images to test a weight vector would be pure waste."""
    cfg = make_config(**cfg_overrides)
    return BarrettDataset(
        filepaths, manifest_df, cfg, train=True, cache=None, build_cache=False,
    )


@pytest.fixture(scope="module")
def manifest():
    return pd.read_csv(MANIFEST)


@pytest.fixture(scope="module")
def two_centre_filepaths(manifest):
    """A training set containing both centres -- i.e. what a pooled CV split
    looks like, which is the only regime in which this sampler does anything."""
    usable = manifest[manifest["fold_r0"] != -1]
    assert usable["centre"].nunique() == 2
    return usable["filepath"].tolist()


def test_four_strata_are_drawn_equally(manifest, two_centre_filepaths):
    ds = _dataset(two_centre_filepaths, manifest)
    gen = torch.Generator()
    gen.manual_seed(0)
    sampler = make_balanced_centre_class_sampler(ds, gen)

    counts = stratum_draw_counts(sampler, ds, gen)
    total = sum(counts.values())

    assert len(counts) == 4, f"expected four strata, drew {list(counts)}"
    assert total == len(ds)
    shares = {k: v / total for k, v in counts.items()}
    # One epoch of len(ds) multinomial draws over four equal strata: the SD of
    # each share is sqrt(0.25*0.75/n) ~= 0.0087 at n=3088, so 0.03 is >3 SD and
    # will not flake, while still failing loudly on a genuinely wrong weighting.
    for name, share in shares.items():
        assert abs(share - 0.25) < 0.03, f"stratum {name} drew {share:.3f}, not ~0.25"


def test_baseline_weighted_sampler_leaves_centres_imbalanced(
    manifest, two_centre_filepaths
):
    """The problem being fixed, stated as a test.

    Class-balanced sampling equalises the classes and nothing else, so the
    hospital prior survives inside each class: center_2 supplies ~12.5% of the
    negatives but well over half the positives. That association is what makes
    hospital identity a predictive training feature.
    """
    ds = _dataset(two_centre_filepaths, manifest)
    gen = torch.Generator()
    gen.manual_seed(0)
    counts = stratum_draw_counts(make_weighted_sampler(ds, gen), ds, gen)
    total = sum(counts.values())

    pos = {k: v for k, v in counts.items() if k.endswith("neoplasia")}
    c2_share_of_pos = (counts["center_2|neoplasia"] / sum(pos.values()))
    assert c2_share_of_pos > 0.5, (
        f"center_2 should dominate the sampled positives under class-only "
        f"balancing; got {c2_share_of_pos:.3f}"
    )
    assert abs(sum(pos.values()) / total - 0.5) < 0.03, "classes should be ~balanced"


def test_measuring_draws_does_not_disturb_the_generator(manifest, two_centre_filepaths):
    """stratum_draw_counts must be an observation, not an intervention."""
    ds = _dataset(two_centre_filepaths, manifest)

    gen_a = torch.Generator()
    gen_a.manual_seed(3)
    sampler_a = make_balanced_centre_class_sampler(ds, gen_a)
    stratum_draw_counts(sampler_a, ds, gen_a)   # the observation
    after_measurement = list(sampler_a)

    gen_b = torch.Generator()
    gen_b.manual_seed(3)
    sampler_b = make_balanced_centre_class_sampler(ds, gen_b)
    without_measurement = list(sampler_b)

    assert after_measurement == without_measurement


@pytest.mark.parametrize("centre", [1, 2])
def test_single_centre_training_set_makes_the_samplers_identical(manifest, centre):
    """THE LOCO DEGENERACY. Both holdout splits train on one centre, so
    1/count(centre, class) and 1/count(class) are the same number for every
    item. Identical weights and an identically seeded generator give an
    identical index sequence, so the two configurations are the same run."""
    train_fps, _ = get_holdout_split(centre, MANIFEST)
    ds = _dataset(train_fps, manifest)
    assert ds.centres().astype(str).tolist().count(
        str(ds.centres()[0])
    ) == len(ds), "this split's training side should hold exactly one centre"

    gen_w = torch.Generator()
    gen_w.manual_seed(11)
    gen_b = torch.Generator()
    gen_b.manual_seed(11)

    weighted = make_weighted_sampler(ds, gen_w)
    balanced = make_balanced_centre_class_sampler(ds, gen_b)

    assert np.array_equal(
        np.asarray(weighted.weights, dtype=np.float64),
        np.asarray(balanced.weights, dtype=np.float64),
    ), "weight vectors differ on a single-centre training set"
    assert list(weighted) == list(balanced), (
        "the two samplers drew different epochs from identically seeded "
        "generators -- the LOCO degeneracy argument would not hold"
    )


def test_config_selects_the_sampler(manifest, two_centre_filepaths):
    """'weighted' and 'none' must still work unchanged."""
    ds = _dataset(two_centre_filepaths, manifest)

    def build(name):
        gen = torch.Generator()
        gen.manual_seed(5)
        return make_sampler(ds, make_config(sampler=name), gen)

    assert isinstance(build("balanced_centre_class"),
                      torch.utils.data.WeightedRandomSampler)
    assert isinstance(build("weighted"), torch.utils.data.WeightedRandomSampler)
    assert isinstance(build("none"), torch.utils.data.RandomSampler)
    assert isinstance(build("random"), torch.utils.data.RandomSampler)
    assert isinstance(build("sequential"), torch.utils.data.SequentialSampler)

    # and 'weighted' still means what it meant before this component existed
    gen_a = torch.Generator()
    gen_a.manual_seed(5)
    gen_b = torch.Generator()
    gen_b.manual_seed(5)
    assert list(make_sampler(ds, make_config(sampler="weighted"), gen_a)) == list(
        make_weighted_sampler(ds, gen_b)
    )
