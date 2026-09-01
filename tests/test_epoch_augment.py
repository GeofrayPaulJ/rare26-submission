"""Fix 1: augmentation must depend on the epoch, not just (seed, index).

Before this fix, BarrettDataset._rng derived its seed from (seed, index) only,
so every image received the identical flip/black-box decision every epoch --
the training set was effectively frozen after epoch 0. These tests pin the
three properties the fix promises:

  * a fixed (seed, epoch, index) always reproduces the same augmentation
  * two different epochs diverge for a healthy fraction of images
  * that divergence is visible to already-alive persistent_workers=True
    worker processes, not just to a freshly-constructed dataset -- because
    set_epoch must update OS-level shared memory, not a plain attribute a
    forked/spawned worker would never see change.
"""
import torch

from src.data import BarrettDataset, make_loader
from tests.conftest import MANIFEST, make_config
import pandas as pd

MIN_DIFFERING_FRACTION = 0.40


def _full_manifest_df():
    return pd.read_csv(MANIFEST)


def _sample_100(manifest_df):
    usable = manifest_df[manifest_df["fold_r0"] != -1]
    pos = usable[usable["class_label"] == "neoplasia"].head(25)
    neg = usable[usable["class_label"] == "non-dysplastic"].head(75)
    return pd.concat([pos, neg])["filepath"].tolist()


def test_augmentation_differs_across_epochs_but_not_within_one():
    manifest_df = _full_manifest_df()
    fps = _sample_100(manifest_df)
    assert len(fps) == 100

    cfg = make_config(seed=5, num_workers=0)
    ds = BarrettDataset(fps, manifest_df, cfg, train=True, build_cache=True)

    ds.set_epoch(0)
    epoch0 = [ds[i][0].clone() for i in range(len(fps))]
    ds.set_epoch(0)
    epoch0_again = [ds[i][0].clone() for i in range(len(fps))]
    ds.set_epoch(1)
    epoch1 = [ds[i][0].clone() for i in range(len(fps))]

    # re-running the same epoch must be byte-identical
    for a, b in zip(epoch0, epoch0_again):
        assert torch.equal(a, b), "same epoch re-run produced different augmentation"

    # a different epoch must differ for a healthy fraction of images
    n_diff = sum(
        0 if torch.equal(a, b) else 1 for a, b in zip(epoch0, epoch1)
    )
    frac = n_diff / len(fps)
    assert frac >= MIN_DIFFERING_FRACTION, (
        f"only {frac:.0%} of images changed between epoch 0 and epoch 1, "
        f"expected >= {MIN_DIFFERING_FRACTION:.0%}"
    )


def test_persistent_workers_observe_epoch_change(manifest_df, sample_filepaths):
    """The trap: workers alive since before set_epoch() was called must still
    see the new epoch, because it lives in shared memory rather than a plain
    attribute copied into the worker at fork/spawn time."""
    fps = sample_filepaths[:16]
    cfg = make_config(seed=3, batch_size=4, num_workers=2)
    ds = BarrettDataset(fps, manifest_df, cfg, train=True, build_cache=True)

    loader = make_loader(ds, cfg, shuffle=False)
    assert loader.persistent_workers is True, "test requires the persistent-worker path"

    def collect():
        out = {}
        for t, _y, fp in loader:
            for i, f in enumerate(fp):
                out[f] = t[i].clone()
        return out

    ds.set_epoch(0)
    epoch0 = collect()
    ds.set_epoch(1)
    epoch1 = collect()
    ds.set_epoch(0)
    epoch0_again = collect()

    n_diff = sum(0 if torch.equal(epoch0[f], epoch1[f]) else 1 for f in fps)
    assert n_diff > 0, (
        "already-alive persistent workers did not observe the epoch change -- "
        "set_epoch is not reaching them"
    )
    for f in fps:
        assert torch.equal(epoch0[f], epoch0_again[f]), (
            "re-running the same epoch through persistent workers must be "
            "byte-identical"
        )
