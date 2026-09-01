"""Test 6: synthetic black boxes are reproducible under a fixed seed."""
import numpy as np

from src.augment import load_redaction_dist, random_black_boxes
from tests.conftest import REDACTION


def _fixed_image():
    # deterministic non-trivial image so "all black" cannot pass by accident
    rng = np.random.default_rng(12345)
    return rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)


def test_black_boxes_reproducible():
    dist = load_redaction_dist(REDACTION)
    img = _fixed_image()

    a = random_black_boxes(img, np.random.default_rng(42), dist, p=1.0)
    b = random_black_boxes(img, np.random.default_rng(42), dist, p=1.0)
    assert np.array_equal(a, b), "same seed produced different boxes"

    # a different seed should (essentially always) differ, and boxes actually drew
    c = random_black_boxes(img, np.random.default_rng(43), dist, p=1.0)
    assert not np.array_equal(a, c), "different seed produced identical boxes"
    assert (a == 0).all(axis=2).sum() > 0, "no black pixels were drawn"

    # p=0 is a no-op
    d = random_black_boxes(img, np.random.default_rng(42), dist, p=0.0)
    assert np.array_equal(d, img)


def test_dist_read_from_file():
    """The distribution must come from the file, not invented constants."""
    dist = load_redaction_dist(REDACTION)
    assert set(dist.counts.tolist()) <= {1, 2, 3}
    assert abs(dist.count_probs.sum() - 1.0) < 1e-9
    assert len(dist.w_frac) == 765  # 765 redacted training images
    assert (dist.w_frac > 0).all() and (dist.h_frac > 0).all()
