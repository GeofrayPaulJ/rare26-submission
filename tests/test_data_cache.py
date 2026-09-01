"""Tests 1-3: cache fidelity and DataLoader determinism."""
import numpy as np
import torch

from src.data import BarrettDataset, make_loader
from tests.conftest import make_config


def test_cache_matches_on_the_fly(manifest_df, sample_filepaths):
    """Test 1: cached reads are byte-identical to fresh decodes for 50 images."""
    cfg = make_config()
    fps = sample_filepaths[:50]

    cached = BarrettDataset(fps, manifest_df, cfg, train=False, build_cache=True)
    fresh = BarrettDataset(fps, manifest_df, cfg, train=False, cache=None)

    for i in range(len(fps)):
        tc, lc, fc = cached[i]
        tf, lf, ff = fresh[i]
        assert fc == ff
        assert lc == lf
        assert torch.equal(tc, tf), f"cache != fresh at item {i} ({fc})"


def test_same_seed_identical_batches(manifest_df, sample_filepaths):
    """Test 2: same seed -> bitwise-identical batches across two passes."""
    cfg = make_config(seed=7, batch_size=8, num_workers=0)
    ds = BarrettDataset(
        sample_filepaths, manifest_df, cfg, train=True, build_cache=True
    )

    def collect():
        loader = make_loader(ds, cfg, shuffle=True)
        return [(t.clone(), list(fp)) for t, _, fp in loader]

    a, b = collect(), collect()
    assert len(a) == len(b)
    for (ta, fa), (tb, fb) in zip(a, b):
        assert fa == fb, "batch order differs under same seed"
        assert torch.equal(ta, tb), "batch tensors differ under same seed"


def test_different_seed_different_order(manifest_df, sample_filepaths):
    """Test 3: different seed -> different batch order."""
    ds_kwargs = dict(train=True, build_cache=True)
    cfg0 = make_config(seed=1, batch_size=8, num_workers=0)
    cfg1 = make_config(seed=2, batch_size=8, num_workers=0)
    # one shared cache is fine; ordering comes from the loader generator/seed
    ds0 = BarrettDataset(sample_filepaths, manifest_df, cfg0, **ds_kwargs)
    ds1 = BarrettDataset(sample_filepaths, manifest_df, cfg1, cache=ds0.cache, train=True)

    order0 = [fp for _, _, fp in make_loader(ds0, cfg0, shuffle=True)]
    order1 = [fp for _, _, fp in make_loader(ds1, cfg1, shuffle=True)]
    flat0 = [f for batch in order0 for f in batch]
    flat1 = [f for batch in order1 for f in batch]
    assert set(flat0) == set(flat1)
    assert flat0 != flat1, "different seeds produced identical order"
