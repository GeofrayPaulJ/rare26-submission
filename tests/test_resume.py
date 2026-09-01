"""Test 11: a resumed run retraces the run it replaces, exactly.

The point of --resume is not "training continues from roughly there". It is that
a run killed by a dropped tmux session produces the same model as one that was
never killed. Anything weaker means a crash silently changes the experiment, and
you find out by wondering why fold 3 disagrees with folds 0-2.

So: train 3 epochs straight through, train 2 epochs and resume for the third,
and require the loss curves to agree to 4+ decimal places -- plus the epoch-3
validation logits to be bit-identical, which is the strongest form of the claim.

Runs at toy scale (resnet18, 96px, 96 images) so it costs seconds. The mechanism
under test -- per-epoch seeding, optimiser/scheduler/scaler/RNG restore -- is
the same one the 30-epoch ConvNeXt run uses.
"""
import dataclasses
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.io import read_predictions
from src.train import run
from tests.conftest import MANIFEST, make_config

EPOCHS = 3
PLACES = 4


@pytest.fixture(scope="module")
def subset_manifest(tmp_path_factory):
    """A small manifest keeping the real column set and the real fold_r0 values,
    with both classes present on each side of the split."""
    tmp = tmp_path_factory.mktemp("resume")
    df = pd.read_csv(MANIFEST)
    usable = df[df["fold_r0"] != -1]
    val, train = usable[usable["fold_r0"] == 0], usable[usable["fold_r0"] != 0]
    sub = pd.concat([
        val[val["class_label"] == "neoplasia"].head(8),
        val[val["class_label"] == "non-dysplastic"].head(24),
        train[train["class_label"] == "neoplasia"].head(16),
        train[train["class_label"] == "non-dysplastic"].head(48),
    ])
    path = tmp / "subset_folds.csv"
    sub.to_csv(path, index=False)
    return str(path)


def _config(subset_manifest, out_dir, num_workers=0):
    return make_config(
        arch="resnet18", pretrained=False,
        epochs=EPOCHS, image_size=96, cache_size=128, batch_size=8,
        num_workers=num_workers, sampler="weighted", precision="bf16",
        seed=0, repeat=0, fold=0,
        manifest=subset_manifest, out_dir=str(out_dir),
    )


@pytest.fixture(scope="module", params=[0, 2], ids=["workers0", "workers2"])
def runs(request, subset_manifest, tmp_path_factory):
    """Both worker counts, because they take different code paths.

    With num_workers>0 the sampler is drawn in the main process but consumed by
    persistent worker processes, and each worker reseeds numpy/random from the
    loader generator. The real 30-epoch run uses 8 workers, so a resume that
    only holds for the single-process path would be worthless.
    """
    out = tmp_path_factory.mktemp(f"out_w{request.param}")
    cfg = _config(subset_manifest, out, num_workers=request.param)

    straight = run(cfg, progress=False, tag="straight")

    # stop after 2 completed epochs -- the exact on-disk state a kill leaves,
    # since the checkpoint is written at the end of every epoch
    partial = run(cfg, max_epochs=2, progress=False, tag="killed")
    resumed = run(cfg, resume="auto", progress=False, tag="killed")
    return straight, partial, resumed


def test_loss_curves_match_to_four_places(runs):
    straight, _partial, resumed = runs
    a, b = straight["history"], resumed["history"]
    assert len(a) == len(b) == EPOCHS

    for ea, eb in zip(a, b):
        assert ea["epoch"] == eb["epoch"]
        for field in ("train_loss", "val_loss"):
            assert round(ea[field], PLACES) == round(eb[field], PLACES), (
                f"epoch {ea['epoch']} {field}: uninterrupted {ea[field]!r} != "
                f"resumed {eb[field]!r}"
            )


def test_metrics_match_after_resume(runs):
    straight, _partial, resumed = runs
    for ea, eb in zip(straight["history"], resumed["history"]):
        for field in ("roc_auc", "pauc_15_std", "ppv_at_90_recall"):
            assert round(ea[field], PLACES) == round(eb[field], PLACES), (
                f"epoch {ea['epoch']} {field} diverged"
            )


def test_final_logits_are_bit_identical(runs):
    """The strongest statement: same weights, not merely a similar loss."""
    straight, _partial, resumed = runs
    a = read_predictions(straight["history"][-1]["pred_path"])
    b = read_predictions(resumed["history"][-1]["pred_path"])
    assert a["filepath"].tolist() == b["filepath"].tolist()
    assert (a["logit"].to_numpy() == b["logit"].to_numpy()).all()


def test_resume_actually_skipped_the_first_epochs(runs):
    """Guard against the test passing because --resume quietly retrained from
    scratch, which would also produce matching curves."""
    _straight, partial, resumed = runs
    assert partial["completed_epochs"] == 2
    log = os.path.join(resumed["run_dir"], "train_log.jsonl")
    events = [json.loads(line) for line in open(log)]
    resume_events = [e for e in events if e.get("event") == "resume"]
    assert len(resume_events) == 1
    assert resume_events[0]["completed_epochs"] == 2
    # exactly one epoch record was written after the resume marker
    after = [e for e in events[events.index(resume_events[0]):] if e.get("event") == "epoch"]
    assert [e["epoch"] for e in after] == [3]


def test_truncated_run_emits_no_canonical_prediction_dump(runs):
    """A run that stopped early has no last epoch to speak of, so it must not
    leave a canonical dump for a downstream step to pick up."""
    _straight, partial, resumed = runs
    canonical = os.path.join(partial["run_dir"], "val_r0_f0_s0.parquet")
    assert partial["completed_epochs"] < EPOCHS
    assert resumed["completed_epochs"] == EPOCHS
    assert os.path.exists(canonical), "the completed resume should have created it"


def test_default_config_leaves_no_checkpoint_after_success(runs):
    """save_checkpoint defaults to False: a screening run's checkpoint
    directory is empty once it completes -- predictions parquets only, no
    optimiser state or model weights left behind anywhere."""
    straight, _partial, _resumed = runs
    assert straight["config"]["save_checkpoint"] is False
    ckpt_dir = os.path.join(straight["run_dir"], "checkpoints")
    assert os.listdir(ckpt_dir) == []
    assert straight["canonical_checkpoint"] is None
    # and it is by ROC-AUC, never by PPV@90R
    assert straight["diagnostic_best_roc_auc"]["roc_auc"] == max(
        h["roc_auc"] for h in straight["history"]
    )


def test_ensemble_member_checkpoint_is_weights_only_fp32(subset_manifest, tmp_path_factory):
    """save_checkpoint=True (a run explicitly designated a final ensemble
    member) keeps exactly one file: the LAST epoch's model weights in fp32,
    no optimiser/scheduler/scaler/RNG/history -- that's the whole point of the
    policy, since optimiser state is what made the old checkpoints ~3x the
    size of the weights alone."""
    out = tmp_path_factory.mktemp("ensemble")
    cfg = dataclasses.replace(_config(subset_manifest, out), save_checkpoint=True)
    summary = run(cfg, progress=False, tag="ensemble")

    ckpt_dir = os.path.join(summary["run_dir"], "checkpoints")
    assert os.listdir(ckpt_dir) == ["weights_fp32.pt"]

    ck = torch.load(os.path.join(ckpt_dir, "weights_fp32.pt"), map_location="cpu",
                    weights_only=False)
    for key in ("optimizer", "scheduler", "scaler", "rng", "train_generator",
                "loader_generator", "history"):
        assert key not in ck, f"{key} must not be in the weights-only checkpoint"
    assert ck["canonical"] is True
    assert ck["selection"] == "last"
    assert ck["epoch"] == EPOCHS
    assert all(t.dtype == torch.float32 for t in ck["model"].values())
    assert summary["canonical_checkpoint"] == os.path.join(ckpt_dir, "weights_fp32.pt")


def test_resume_refuses_a_config_that_changes_the_trajectory(subset_manifest,
                                                            tmp_path_factory):
    """Resuming into a changed lr is a different experiment wearing the old
    run's directory. It must fail rather than produce a plausible curve."""
    out = tmp_path_factory.mktemp("mismatch")
    cfg = _config(subset_manifest, out)
    run(cfg, max_epochs=1, progress=False, tag="m")
    with pytest.raises(ValueError, match="trajectory"):
        run(dataclasses.replace(cfg, lr=0.123), resume="auto", progress=False, tag="m")


def test_every_epoch_dumped_its_validation_logits(runs):
    straight, _partial, _resumed = runs
    for h in straight["history"]:
        assert os.path.exists(h["pred_path"])
        df = read_predictions(h["pred_path"])
        assert len(df) == straight["header"]["n_val"]
        assert df["logit"].dtype == np.float64
