"""Statistical correctness of src/evaluate.py.

These tests exist because every error this module can make is SILENT. A pooled
FPR computed the wrong way still returns a plausible number in the right range;
a pAUC in the wrong convention still moves in the right direction; a PPV
projection with a transposed constant still ranks models identically. None of
them raise. Each test below pins one claim the report makes, against a case
where the wrong implementation gives a demonstrably different answer.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluate import (  # noqa: E402
    LEADERBOARD_NEG, LEADERBOARD_POS, load_official_score, metric_block,
    operating_point_detail, pool_oof, ppv_leaderboard, spread,
    threshold_at_recall,
)
from src.io import SCHEMA, build_frame, write_predictions  # noqa: E402
from src.metrics import TARGET_RECALL, fpr_at_recall, partial_auc  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def synth(n_pos, n_neg, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)]
    s = np.r_[rng.normal(sep, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)]
    return y, s


def write_fold(dirpath, repeat, fold, seed, filepaths, labels, logits):
    name = f"r{repeat}_f{fold}_s{seed}"
    run_dir = os.path.join(dirpath, name)
    os.makedirs(run_dir, exist_ok=True)
    df = build_frame(
        filepath=filepaths,
        centre=["center_1"] * len(filepaths),
        class_label=["neoplasia" if v else "non-dysplastic" for v in labels],
        label_int=labels,
        visibility=[None] * len(filepaths),
        group_id_v2=["g"] * len(filepaths),
        repeat=repeat, fold=fold, seed=seed, logit=logits,
    )
    write_predictions(df, os.path.join(run_dir, f"val_{name}.parquet"))
    return run_dir


# ---------------------------------------------------------------------------
# THE central claim: pooling is not averaging
# ---------------------------------------------------------------------------
def test_pooled_fpr_differs_from_mean_of_fold_fprs(tmp_path):
    """Pooled FPR@90R and the mean of per-fold FPR@90R are different statistics.

    Constructed so the difference is unmistakable: two folds with identical
    within-fold separation but logits on shifted scales. Each fold alone
    separates its own classes perfectly, so each fold's FPR@90R is ~0. Pooled,
    fold B's negatives outscore fold A's positives, so one global threshold
    admits a large number of false positives. Averaging the fold numbers reports
    ~0 and hides that completely.
    """
    n = 200
    fps_a = [f"a/{i}.png" for i in range(n)]
    fps_b = [f"b/{i}.png" for i in range(n)]
    lab = np.r_[np.ones(20, dtype=np.int64), np.zeros(n - 20, dtype=np.int64)]
    # fold A on a low scale, fold B on a high scale; both perfectly separated
    log_a = np.r_[np.full(20, 1.0), np.full(n - 20, 0.0)]
    log_b = np.r_[np.full(20, 11.0), np.full(n - 20, 10.0)]

    write_fold(tmp_path, 0, 0, 0, fps_a, lab, log_a)
    write_fold(tmp_path, 0, 1, 0, fps_b, lab, log_b)

    fpr_a = fpr_at_recall(lab, log_a)
    fpr_b = fpr_at_recall(lab, log_b)
    mean_of_folds = (fpr_a + fpr_b) / 2

    pool = pool_oof(str(tmp_path), repeat=0, seed=0, folds=(0, 1))
    pooled = fpr_at_recall(pool["label_int"].to_numpy(), pool["logit"].to_numpy())

    assert mean_of_folds == pytest.approx(0.0, abs=1e-9)
    # every fold-B negative outranks every fold-A positive, so a single
    # threshold reaching 90% recall must admit them
    assert pooled > 0.4, f"pooled FPR {pooled} should be large, not {mean_of_folds}"
    assert abs(pooled - mean_of_folds) > 0.4


def test_pool_oof_rejects_duplicate_filepaths(tmp_path):
    """An image appearing twice silently double-weights it. Must raise."""
    n = 50
    fps = [f"x/{i}.png" for i in range(n)]
    lab = np.r_[np.ones(5, dtype=np.int64), np.zeros(n - 5, dtype=np.int64)]
    log = np.linspace(0, 1, n)
    write_fold(tmp_path, 0, 0, 0, fps, lab, log)
    write_fold(tmp_path, 0, 1, 0, fps, lab, log)  # same filepaths again
    with pytest.raises(AssertionError, match="more than once"):
        pool_oof(str(tmp_path), repeat=0, seed=0, folds=(0, 1))


def test_pool_oof_rejects_incomplete_coverage(tmp_path):
    """A pool missing manifest rows is not out-of-fold. Must raise."""
    n = 40
    fps = [f"x/{i}.png" for i in range(n)]
    lab = np.r_[np.ones(4, dtype=np.int64), np.zeros(n - 4, dtype=np.int64)]
    write_fold(tmp_path, 0, 0, 0, fps, lab, np.linspace(0, 1, n))
    manifest = pd.DataFrame({
        "filepath": fps + ["x/missing.png"],
        "fold_r0": [0] * n + [1],
    })
    with pytest.raises(AssertionError, match="missing"):
        pool_oof(str(tmp_path), repeat=0, seed=0, folds=(0,), manifest=manifest)


def test_pool_oof_rejects_mixed_seeds(tmp_path):
    n = 40
    fps_a = [f"a/{i}.png" for i in range(n)]
    fps_b = [f"b/{i}.png" for i in range(n)]
    lab = np.r_[np.ones(4, dtype=np.int64), np.zeros(n - 4, dtype=np.int64)]
    write_fold(tmp_path, 0, 0, 0, fps_a, lab, np.linspace(0, 1, n))
    # fold 1's dump carries seed 1 but is loaded as part of seed 0's pool
    run = write_fold(tmp_path, 0, 1, 1, fps_b, lab, np.linspace(0, 1, n))
    os.replace(os.path.join(run, "val_r0_f1_s1.parquet"),
               os.path.join(tmp_path, "r0_f1_s0", "val_r0_f1_s0.parquet")
               if os.path.isdir(os.path.join(tmp_path, "r0_f1_s0"))
               else os.path.join(run, "val_r0_f1_s0.parquet"))
    os.rename(run, os.path.join(tmp_path, "r0_f1_s0"))
    with pytest.raises(AssertionError, match="mixes seeds"):
        pool_oof(str(tmp_path), repeat=0, seed=0, folds=(0, 1))


# ---------------------------------------------------------------------------
# FPR is prevalence-invariant; PPV is not. This is why FPR is primary.
# ---------------------------------------------------------------------------
def test_fpr_invariant_to_prevalence_but_ppv_is_not():
    """Duplicating every positive doubles prevalence, leaves the ranking of
    positives relative to negatives untouched, and so must leave FPR@90R
    unchanged while PPV@90R rises. If FPR moved here, it would not be
    comparable between a 5.1% local pool and a 1% leaderboard."""
    y, s = synth(158, 2930, sep=2.0, seed=1)
    y2 = np.r_[y, np.ones(158, dtype=np.int64)]
    s2 = np.r_[s, s[y == 1]]  # exact copies of the positives

    m1 = metric_block(y, s)
    m2 = metric_block(y2, s2)

    assert m2["prevalence"] > m1["prevalence"] * 1.8
    assert m2["fpr_at_90_recall"] == pytest.approx(m1["fpr_at_90_recall"], abs=1e-3)
    assert m2["ppv_at_90_recall_local"] > m1["ppv_at_90_recall_local"] + 0.05


# ---------------------------------------------------------------------------
# PPV projection
# ---------------------------------------------------------------------------
def test_ppv_leaderboard_matches_the_stated_formula():
    for fpr in (0.0, 0.01, 0.05, 0.1234, 0.5, 1.0):
        expected = (0.9 * LEADERBOARD_POS) / (0.9 * LEADERBOARD_POS + fpr * LEADERBOARD_NEG)
        assert ppv_leaderboard(fpr) == pytest.approx(expected)
    assert LEADERBOARD_POS == 232 and LEADERBOARD_NEG == 23176


def test_ppv_leaderboard_is_strictly_decreasing_in_fpr():
    """A projection that is not monotone in FPR would let two experiments
    disagree about which is better depending on which number you read."""
    fprs = np.linspace(0.0, 0.5, 200)
    ppvs = np.array([ppv_leaderboard(f) for f in fprs])
    assert ppv_leaderboard(0.0) == pytest.approx(1.0)
    assert np.all(np.diff(ppvs) < 0)


# ---------------------------------------------------------------------------
# pAUC conventions -- 0.075 and 0.5 both mean "no signal"
# ---------------------------------------------------------------------------
def test_pauc_conventions_random_and_perfect():
    rng = np.random.default_rng(0)
    n = 20000
    y = np.r_[np.ones(n // 2, dtype=np.int64), np.zeros(n // 2, dtype=np.int64)]

    rand = rng.normal(0, 1, n)  # no signal at all
    p = partial_auc(y, rand, 0.15)
    assert p["raw"] == pytest.approx(0.075, abs=0.02), "raw random must be max_fpr/2"
    assert p["std"] == pytest.approx(0.5, abs=0.05), "std random must be 0.5"

    perfect = np.r_[np.ones(n // 2), np.zeros(n // 2)] + rng.normal(0, 1e-6, n)
    p = partial_auc(y, perfect, 0.15)
    assert p["raw"] == pytest.approx(1.0, abs=1e-3)
    assert p["std"] == pytest.approx(1.0, abs=1e-3)


def test_metric_block_reports_both_pauc_conventions():
    y, s = synth(158, 2930, seed=2)
    m = metric_block(y, s)
    assert "pauc_15_raw" in m and "pauc_15_std" in m
    assert m["pauc_15_raw"] != pytest.approx(m["pauc_15_std"], abs=1e-6)


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------
def test_threshold_at_recall_actually_achieves_the_target():
    for seed in range(5):
        y, s = synth(158, 2930, seed=seed)
        thr = threshold_at_recall(y, s, TARGET_RECALL)
        recall = float((s[y == 1] >= thr).sum() / (y == 1).sum())
        assert recall >= TARGET_RECALL, f"threshold gives recall {recall}"
        # and it is the SMALLEST such attainable score: raising it past the next
        # positive would drop below target
        higher = s[y == 1][s[y == 1] > thr]
        if higher.size:
            nxt = higher.min()
            assert float((s[y == 1] >= nxt).sum() / (y == 1).sum()) < TARGET_RECALL


def test_operating_point_brackets_the_interpolated_fpr():
    """The reported FPR@90R is an interpolation between two achievable points
    one true positive apart. It must lie inside that bracket, and the bracket
    width is the resolution of the primary metric."""
    for seed in range(5):
        y, s = synth(158, 2930, seed=seed)
        op = operating_point_detail(y, s)
        f = fpr_at_recall(y, s)
        lo, hi = sorted((op["fpr_at_tp_lo"], op["fpr_at_tp_hi"]))
        assert lo - 1e-9 <= f <= hi + 1e-9, f"{f} outside [{lo}, {hi}]"
        assert op["tp_bracket"] == [142, 143]
        assert op["fpr_step_one_fp"] == pytest.approx(1 / 2930)


def test_pool_scale_resolution_is_five_times_finer_than_fold_scale():
    """The quantitative reason Level 2 exists rather than Level 1."""
    y_pool, s_pool = synth(158, 2930, seed=3)
    y_fold, s_fold = synth(31, 586, seed=3)
    op_pool = operating_point_detail(y_pool, s_pool)
    op_fold = operating_point_detail(y_fold, s_fold)
    assert op_fold["fpr_step_one_fp"] / op_pool["fpr_step_one_fp"] == pytest.approx(5.0)
    assert op_fold["recall_step_one_pos"] / op_pool["recall_step_one_pos"] > 5.0


# ---------------------------------------------------------------------------
# The official scorer is used verbatim, and its draws are at 1% prevalence
# ---------------------------------------------------------------------------
def test_official_score_is_loaded_from_08_score_not_reimplemented():
    fn = load_official_score()
    assert fn.__name__ == "official_score"
    assert fn.__module__ == "_official_score"
    src = os.path.join(REPO_ROOT, "scripts", "08_score.py")
    with open(src) as fh:
        assert "def official_score(" in fh.read()


def test_bootstrap_draws_are_prevalence_matched_to_one_percent():
    """Pins the correction made in the report: the official scorer resamples
    positives down to 1:100, so its draws ARE at leaderboard prevalence.
    Prevalence is therefore not the reason the bootstrap figure is
    incomparable -- the frozen negative set and the tiny positive pool are."""
    n_neg = 2930
    n_sample = max(1, int(n_neg / 100))
    draw_prevalence = n_sample / (n_neg + n_sample)
    leaderboard_prevalence = LEADERBOARD_POS / (LEADERBOARD_POS + LEADERBOARD_NEG)
    assert n_sample == 29
    assert draw_prevalence == pytest.approx(leaderboard_prevalence, abs=0.002)


def test_official_score_runs_on_a_pooled_sized_set():
    y, s = synth(158, 2930, seed=4)
    r = load_official_score()(y, s, n_iterations=50, seed=0)
    assert 0.0 <= r["Score"] <= 1.0
    assert r["PPV@90RECALL CI Lower"] <= r["Score"] <= r["PPV@90RECALL CI Upper"]


# ---------------------------------------------------------------------------
# spread()
# ---------------------------------------------------------------------------
def test_spread_reports_median_iqr_and_range():
    s = spread([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["median"] == 3.0
    assert s["q1"] == 2.0 and s["q3"] == 4.0
    assert s["iqr"] == 2.0
    assert s["range"] == 4.0 and s["min"] == 1.0 and s["max"] == 5.0
    assert s["n"] == 5


def test_spread_ignores_nan_and_survives_empty():
    s = spread([1.0, float("nan"), 3.0])
    assert s["n"] == 2 and s["median"] == 2.0
    assert np.isnan(spread([])["median"])


def test_spread_median_resists_one_bad_run():
    """Why the report leads with medians: one failed run moves a mean and does
    not move a median."""
    good = [0.10, 0.11, 0.12, 0.13, 0.14]
    with_outlier = good[:-1] + [0.90]
    assert spread(with_outlier)["median"] == pytest.approx(
        spread(good)["median"], abs=0.011)
    assert abs(np.mean(with_outlier) - np.mean(good)) > 0.15


# ---------------------------------------------------------------------------
# schema round-trip through the pooling path
# ---------------------------------------------------------------------------
def test_pool_preserves_schema_and_float64_logits(tmp_path):
    n = 60
    lab = np.r_[np.ones(6, dtype=np.int64), np.zeros(n - 6, dtype=np.int64)]
    for f in range(2):
        write_fold(tmp_path, 0, f, 0, [f"f{f}/{i}.png" for i in range(n)],
                   lab, np.linspace(-2, 2, n))
    pool = pool_oof(str(tmp_path), repeat=0, seed=0, folds=(0, 1))
    assert list(pool.columns) == SCHEMA
    assert pool["logit"].dtype == np.float64
    assert len(pool) == 2 * n
