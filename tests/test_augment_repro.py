"""Component B: the augmentation stack keeps the (seed, epoch, index) contract.

The stack is worthless if it is not reproducible, because every claim this
codebase makes about resume fidelity and about paired comparisons rests on a
fixed (seed, epoch, index) producing a fixed tensor. Three separate things can
break that, and each gets its own test:

  * a transform reaching for a global RNG (numpy's, random's, torch's) instead
    of the generator it was handed -- invisible single-process, catastrophic
    across workers;
  * `persistent_workers` handing a stale epoch to a worker that was alive
    before set_epoch was called;
  * the new stages consuming draws when they are switched OFF, which would
    silently change what the BASELINE arm of a factorial screen does and make
    it non-comparable to runs made before this file existed.
"""
import numpy as np
import pytest
import torch

from src.augment import (
    apply_uint8,
    horizontal_flip,
    load_redaction_dist,
    random_black_boxes,
)
from src.config import AugConfig
from src.data import BarrettDataset, make_loader
from tests.conftest import REDACTION, make_config

# every stage on, at the shipped magnitudes
FULL_STACK = dict(photometric=True, optical=True, sensor=True, compression=True)


def _image(seed=12345, h=192, w=192):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _collect(dataset, cfg, epochs, num_workers):
    """{(epoch, filepath): tensor} over `epochs`, read through a DataLoader with
    `num_workers` worker processes. Sequential sampler so the comparison is
    about pixels, not ordering."""
    cfg = make_config(**{**cfg, "num_workers": num_workers})
    loader = make_loader(
        dataset, cfg, shuffle=False,
        sampler=torch.utils.data.SequentialSampler(dataset),
    )
    out = {}
    for epoch in epochs:
        dataset.set_epoch(epoch)
        for x, _y, fps in loader:
            for j, fp in enumerate(fps):
                out[(epoch, fp)] = x[j].numpy().copy()
    return out


@pytest.fixture(scope="module")
def full_stack_dataset(manifest_df, sample_filepaths):
    cfg = make_config(num_workers=0, aug=AugConfig(**FULL_STACK))
    return BarrettDataset(
        sample_filepaths, manifest_df, cfg, train=True, build_cache=True,
    )


@pytest.fixture(scope="module")
def real_frame(manifest_df, sample_filepaths):
    """One real endoscopy frame, cropped and resized exactly as training does.

    Synthetic noise will not do for the "does it still resemble the source"
    check below: white noise has no spatial structure, so a blur or a JPEG pass
    decorrelates it completely and the test would report destruction for any
    magnitude at all, including zero.
    """
    cfg = make_config(num_workers=0, image_size=384, cache_size=384)
    ds = BarrettDataset(
        sample_filepaths[:1], manifest_df, cfg, train=True, build_cache=True,
    )
    return ds.cache[0].numpy().copy()


def test_identical_across_worker_counts_and_epochs(full_stack_dataset, manifest_df,
                                                   sample_filepaths):
    """Same (seed, epoch, index) -> same bytes, single-process or not.

    Also runs epoch 0 AGAIN after epoch 1 inside the same persistent workers:
    if set_epoch did not reach a worker that was already alive, the second
    visit to epoch 0 would return epoch 1's augmentation and this would fail.
    """
    base = dict(num_workers=0, aug=AugConfig(**FULL_STACK))

    single = _collect(full_stack_dataset, base, [0, 1], num_workers=0)

    workers_cfg = make_config(num_workers=2, aug=AugConfig(**FULL_STACK))
    ds_w = BarrettDataset(
        sample_filepaths, manifest_df, workers_cfg, train=True, build_cache=True,
    )
    assert ds_w.cfg.num_workers == 2
    multi = _collect(ds_w, base, [0, 1, 0], num_workers=2)

    for key, tensor in single.items():
        assert np.array_equal(tensor, multi[key]), (
            f"worker count changed the augmentation for {key}"
        )

    # epoch 0 revisited after epoch 1, in workers that outlived both
    for (epoch, fp), tensor in single.items():
        if epoch == 0:
            assert np.array_equal(tensor, multi[(0, fp)])

    # and the epochs must genuinely differ, or the test above is vacuous
    differing = sum(
        1 for fp in {f for _, f in single}
        if not np.array_equal(single[(0, fp)], single[(1, fp)])
    )
    assert differing > len(sample_filepaths) * 0.5, (
        "epochs 0 and 1 produced near-identical augmentation; the epoch "
        "component of the RNG seed is not doing anything"
    )


def test_no_global_rng_dependence():
    """Perturbing every global RNG between two calls must change nothing."""
    img, dist = _image(), load_redaction_dist(REDACTION)
    aug = AugConfig(**FULL_STACK)

    first = apply_uint8(img, aug, np.random.default_rng(7), True, dist)

    np.random.seed(999)
    np.random.random(1000)
    torch.manual_seed(999)
    torch.rand(1000)
    import random as _random
    _random.seed(999)
    _random.random()

    second = apply_uint8(img, aug, np.random.default_rng(7), True, dist)
    assert np.array_equal(first, second), "output moved with a global RNG"


def test_disabled_stages_consume_no_randomness():
    """With the new gates OFF the pipeline must be the old pipeline, byte for
    byte AND draw for draw. The state comparison is the strict half: equal
    output could be luck, an equal generator state cannot be."""
    img, dist = _image(), load_redaction_dist(REDACTION)
    aug = AugConfig()  # every new gate defaults OFF

    rng_pipeline = np.random.default_rng(11)
    got = apply_uint8(img, aug, rng_pipeline, True, dist)

    rng_manual = np.random.default_rng(11)
    want = random_black_boxes(
        horizontal_flip(img, rng_manual, aug.hflip_p),
        rng_manual, dist, aug.black_boxes_p,
    )

    assert np.array_equal(got, want)
    assert rng_pipeline.bit_generator.state == rng_manual.bit_generator.state, (
        "the gated-off stack consumed draws; every run made before it existed "
        "is no longer reproducible"
    )


def test_stack_changes_the_image_but_still_resembles_it(real_frame):
    """A machine-checkable floor under 'does it still look like endoscopy'.

    The eye check is reports/aug_samples.png and it is the real one; this only
    catches the failure mode where a magnitude typo turns the output into noise
    or a flat field, which would otherwise be discovered halfway through a
    13-hour sweep.
    """
    img, dist = real_frame, load_redaction_dist(REDACTION)
    # black boxes off: a large black rectangle is a legitimate part of the
    # baseline but would dominate the correlation being measured here
    aug = AugConfig(random_black_boxes=False, **FULL_STACK)

    corrs, changed = [], 0
    for s in range(25):
        out = apply_uint8(img, aug, np.random.default_rng(s), True, dist)
        assert out.shape == img.shape and out.dtype == np.uint8
        if not np.array_equal(out, img):
            changed += 1
        a = img.astype(np.float64).ravel()
        b = out.astype(np.float64).ravel()
        if b.std() > 1e-6:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))

    assert changed >= 24, f"only {changed}/25 draws altered the image"
    assert len(corrs) == 25, "a draw produced a constant image"
    median_r = float(np.median(corrs))
    assert median_r > 0.5, (
        f"median correlation with the source image is {median_r:.3f}; the "
        f"magnitudes are destroying the image, not perturbing it"
    )


def test_probabilities_of_zero_are_a_no_op():
    img, dist = _image(), load_redaction_dist(REDACTION)
    off = {f: 0.0 for f in AugConfig._P_FIELDS}
    aug = AugConfig(**{**off, **FULL_STACK})
    out = apply_uint8(img, aug, np.random.default_rng(4), True, dist)
    assert np.array_equal(out, img)


def test_tracing_is_side_effect_free():
    """The trace list exists for reports/aug_samples.png. It is threaded through
    production code, so it has to be provably inert: same pixels, same generator
    state, whether or not anyone is watching."""
    img, dist = _image(), load_redaction_dist(REDACTION)
    aug = AugConfig(**FULL_STACK)

    rng_a = np.random.default_rng(21)
    untraced = apply_uint8(img, aug, rng_a, True, dist)

    rng_b = np.random.default_rng(21)
    trace = []
    traced = apply_uint8(img, aug, rng_b, True, dist, trace=trace)

    assert np.array_equal(untraced, traced), "tracing changed the output"
    assert rng_a.bit_generator.state == rng_b.bit_generator.state, \
        "tracing consumed randomness"
    assert trace, "nothing was recorded despite the whole stack being enabled"
    # names must be usable as labels, and jpeg carries its quality
    assert all(isinstance(n, str) and n for n in trace)


def test_eval_path_is_untouched():
    img, dist = _image(), load_redaction_dist(REDACTION)
    aug = AugConfig(**FULL_STACK)
    rng = np.random.default_rng(4)
    out = apply_uint8(img, aug, rng, False, dist)
    assert np.array_equal(out, img)
    assert rng.bit_generator.state == np.random.default_rng(4).bit_generator.state
