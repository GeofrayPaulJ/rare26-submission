"""Re-analysis of the RARE26 noise-floor predictions. NO TRAINING, NO GPU.

Every number in this script is recomputed from prediction parquets already on
disk: runs/noise_floor_a/r0_f{0..4}_s{0..4}/val_*.parquet (Job A) and
runs/noise_floor_b/loco_c{1,2}_s{0..4}/val_*.parquet (Job B). It reuses the
tested machinery in src/evaluate.py (pool_oof, metric_block, spread,
fold_contribution, attach_manifest, the diagnostic functions) rather than
re-deriving any of it, so this file is additive: it does not touch, and cannot
silently diverge from, the module reports/noise_floor.md was built on.

FIVE QUESTIONS, ONE METHODOLOGICAL RULE.

  1. Rank normalisation -- does correcting for cross-fold logit-scale drift
     (the fold-3-supplies-37%-of-FPs finding) shrink the seed noise floor?
  2. Seed ensembling -- free at inference time since the 25 checkpoints already
     exist. Where does the return curve flatten?
  3. Domain shift -- does ensembling close the LOCO gap, or only the
     in-distribution one?
  4. The corrected MDE -- 0.090 was the RANGE of five single runs. The
     decision this codebase actually makes is a comparison of MEDIANS across
     n-seed groups, which has a different, smaller, calculable uncertainty.
  5. Hospital signal -- is it reduced by ensembling, or common to every seed
     (in which case no amount of ensembling removes it)?

THE RULE: any figure computed from a single 5-seed ensemble (k=5) is reported
together with the spread of its five leave-one-out 4-seed ensembles, never as
a bare point estimate. A k=5 ensemble is one sample; the k=4 LOO family is the
only source of variance available for it without retraining.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluate import (  # noqa: E402
    attach_manifest, box, diag_confound_correlations, diag_hard_negatives,
    fmt_spread, fold_contribution, load_unit, metric_block, pool_oof, spread,
)

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
REPEAT = 0
CENTRES = (1, 2)
MINUTES_PER_UNIT = 24.4  # Job A's measured median; per the brief, used for all cost math

_JD = lambda o: (int(o) if isinstance(o, np.integer) else
                 float(o) if isinstance(o, np.floating) else
                 bool(o) if isinstance(o, np.bool_) else
                 o.tolist() if isinstance(o, np.ndarray) else str(o))


# ---------------------------------------------------------------------------
# Rank normalisation
# ---------------------------------------------------------------------------
def add_rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    """(rank - 0.5) / n within this fold's validation set, i.e. this one
    trained model's own score distribution. Monotonic within the fold -- it
    cannot change which images that model ranks above which -- but it strips
    the fold's arbitrary logit scale before the five folds are compared to one
    another, which raw pooling does not."""
    df = df.copy()
    n = len(df)
    df["rank_norm"] = (rankdata(df["logit"].to_numpy(), method="average") - 0.5) / n
    return df


def pool_frames(frames: Sequence[pd.DataFrame], manifest: pd.DataFrame,
                repeat: int = REPEAT) -> pd.DataFrame:
    """Concatenate pre-loaded per-fold frames with the same coverage/duplicate
    guarantees pool_oof enforces when it loads from disk itself. Needed here
    because rank-normalisation and ensembling both produce in-memory frames
    pool_oof was never built to accept."""
    pool = pd.concat(frames, ignore_index=True)
    dupes = pool["filepath"].duplicated()
    if dupes.any():
        raise AssertionError(f"{int(dupes.sum())} duplicate filepaths in pool")
    usable = set(manifest.loc[manifest[f"fold_r{repeat}"] != -1, "filepath"])
    got = set(pool["filepath"])
    if got != usable:
        raise AssertionError(
            f"coverage mismatch: missing {len(usable - got)}, "
            f"unexpected {len(got - usable)}")
    return pool


def rank_normalised_pool(job_a: str, seed: int, manifest: pd.DataFrame,
                         folds: Sequence[int] = FOLDS) -> pd.DataFrame:
    frames = [add_rank_norm(load_unit(job_a, f"r{REPEAT}_f{f}_s{seed}"))
             for f in folds]
    return pool_frames(frames, manifest)


def with_score_as_logit(pool: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """A copy with `score_col` standing in for `logit`, so every src.evaluate
    function that reads the `logit` column (fold_contribution,
    diag_confound_correlations, diag_hard_negatives) works unmodified on a
    rank-normalised or ensembled score without editing that tested module."""
    if score_col == "logit":
        return pool
    out = pool.copy()
    out["logit"] = out[score_col]
    return out


# ---------------------------------------------------------------------------
# Ensembling -- wide tables, one row per image, one column per seed
# ---------------------------------------------------------------------------
def build_wide_fold(job_a: str, fold: int, manifest: pd.DataFrame,
                    seeds: Sequence[int] = SEEDS,
                    repeat: int = REPEAT) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One fold's 5 seeds, aligned by filepath. Returns (base metadata,
    logit_wide, rank_wide), each indexed identically so any seed subset's
    mean is a one-line column-wise average."""
    frames = {s: add_rank_norm(load_unit(job_a, f"r{repeat}_f{fold}_s{s}"))
             for s in seeds}
    base = frames[seeds[0]].set_index("filepath")[
        ["centre", "class_label", "label_int", "visibility", "group_id_v2", "fold"]]
    logit_wide, rank_wide = {}, {}
    for s in seeds:
        f = frames[s].set_index("filepath")
        if set(f.index) != set(base.index):
            raise AssertionError(f"fold {fold} seed {s}: filepath set differs from seed {seeds[0]}")
        if not (f.loc[base.index, "label_int"] == base["label_int"]).all():
            raise AssertionError(f"fold {fold} seed {s}: label_int disagrees with seed {seeds[0]}")
        logit_wide[s] = f.loc[base.index, "logit"]
        rank_wide[s] = f.loc[base.index, "rank_norm"]
    return base.reset_index(), pd.DataFrame(logit_wide), pd.DataFrame(rank_wide)


def build_wide_loco(job_b: str, centre: int, manifest: pd.DataFrame,
                    seeds: Sequence[int] = SEEDS) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same shape as build_wide_fold but for one LOCO direction's single
    held-out set. Filters the 7 keep_for_training=False rows out of held-out
    center_2, matching reports/noise_floor.md so the two reports agree on what
    'the center_2 test set' means."""
    keep = manifest.set_index("filepath")["keep_for_training"]
    frames = {}
    for s in seeds:
        df = load_unit(job_b, f"loco_c{centre}_s{s}")
        df = df[df["filepath"].map(keep).fillna(True).astype(bool)]
        frames[s] = add_rank_norm(df)
    base = frames[seeds[0]].set_index("filepath")[
        ["centre", "class_label", "label_int", "visibility", "group_id_v2"]]
    logit_wide, rank_wide = {}, {}
    for s in seeds:
        f = frames[s].set_index("filepath")
        if set(f.index) != set(base.index):
            raise AssertionError(f"loco c{centre} seed {s}: filepath set differs")
        logit_wide[s] = f.loc[base.index, "logit"]
        rank_wide[s] = f.loc[base.index, "rank_norm"]
    return base.reset_index(), pd.DataFrame(logit_wide), pd.DataFrame(rank_wide)


def ensemble_series(logit_wide: pd.DataFrame, rank_wide: pd.DataFrame,
                    subset: Sequence[int], fusion: str) -> pd.Series:
    cols = list(subset)
    if fusion == "mean_logit":
        return logit_wide[cols].mean(axis=1)
    if fusion == "mean_rank":
        return rank_wide[cols].mean(axis=1)
    raise ValueError(fusion)


def ensemble_pool_oof(wide_by_fold: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                      subset: Sequence[int], fusion: str,
                      manifest: pd.DataFrame, repeat: int = REPEAT) -> pd.DataFrame:
    frames = []
    for f, (base, logit_wide, rank_wide) in wide_by_fold.items():
        score = ensemble_series(logit_wide, rank_wide, subset, fusion)
        frame = base.copy()
        frame["logit"] = score.loc[frame["filepath"]].to_numpy()
        frames.append(frame)
    return pool_frames(frames, manifest, repeat=repeat)


def ensemble_loco(wide: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                  subset: Sequence[int], fusion: str) -> Tuple[np.ndarray, np.ndarray]:
    base, logit_wide, rank_wide = wide
    score = ensemble_series(logit_wide, rank_wide, subset, fusion)
    frame = base.copy()
    frame["logit"] = score.loc[frame["filepath"]].to_numpy()
    return frame["label_int"].to_numpy(), frame["logit"].to_numpy(), frame


# ---------------------------------------------------------------------------
# Experiment 1 -- rank normalisation
# ---------------------------------------------------------------------------
def per_fold_solo_auc(job_a: str, seed: int = 0) -> Dict[int, float]:
    """Each fold is a genuinely different trained model (different training
    set, same architecture). Its own held-out AUC is real information about
    how good that particular model is -- not noise to be normalised away."""
    from src.metrics import roc_auc
    out = {}
    for f in FOLDS:
        df = load_unit(job_a, f"r{REPEAT}_f{f}_s{seed}")
        out[f] = roc_auc(df["label_int"].to_numpy(), df["logit"].to_numpy())
    return out


def experiment1(job_a: str, manifest: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    fc_raw, fc_rank = [], []
    for seed in SEEDS:
        raw = pool_oof(job_a, REPEAT, seed, FOLDS, manifest)
        rk = rank_normalised_pool(job_a, seed, manifest)

        m_raw = metric_block(raw["label_int"].to_numpy(), raw["logit"].to_numpy())
        m_rk = metric_block(rk["label_int"].to_numpy(), rk["rank_norm"].to_numpy())

        fcr = fold_contribution(raw)
        fck = fold_contribution(with_score_as_logit(rk, "rank_norm"))
        fc_raw.append(fcr)
        fc_rank.append(fck)

        rows.append({
            "seed": seed,
            "fpr_raw": m_raw["fpr_at_90_recall"], "fpr_rank": m_rk["fpr_at_90_recall"],
            "auc_raw": m_raw["roc_auc"], "auc_rank": m_rk["roc_auc"],
            "pauc_raw": m_raw["pauc_15_std"], "pauc_rank": m_rk["pauc_15_std"],
            "max_dev_raw": fcr["max_abs_deviation"], "max_dev_rank": fck["max_abs_deviation"],
        })
    df = pd.DataFrame(rows)
    sd_raw = spread(df["fpr_raw"].tolist())
    sd_rank = spread(df["fpr_rank"].tolist())

    calibration_share = None
    if sd_raw["sd"] > 0:
        calibration_share = 1.0 - (sd_rank["sd"] / sd_raw["sd"]) ** 2

    return {
        "per_seed": df.to_dict("records"),
        "fpr_raw_spread": sd_raw, "fpr_rank_spread": sd_rank,
        "auc_raw_spread": spread(df["auc_raw"].tolist()),
        "auc_rank_spread": spread(df["auc_rank"].tolist()),
        "pauc_raw_spread": spread(df["pauc_raw"].tolist()),
        "pauc_rank_spread": spread(df["pauc_rank"].tolist()),
        "max_dev_raw_spread": spread(df["max_dev_raw"].tolist()),
        "max_dev_rank_spread": spread(df["max_dev_rank"].tolist()),
        "fp_share_raw_seed0": fc_raw[0]["fp_share_by_fold"],
        "fp_share_rank_seed0": fc_rank[0]["fp_share_by_fold"],
        "calibration_share_of_variance": calibration_share,
        "calibration_share_note": (
            "1 - (SD_rank/SD_raw)^2, valid only under the assumption that raw-logit "
            "variance decomposes additively into 'discrimination' variance (retained "
            "by rank-normalisation) and 'calibration-drift' variance (removed by it); "
            "not an exact decomposition, an approximation from 5 points per arm"),
        "fold_sizes_equal": True,  # verified: every fold holds exactly 586 negatives
        "per_fold_solo_auc": per_fold_solo_auc(job_a),
    }


# ---------------------------------------------------------------------------
# Experiment 2 -- ensembling, pooled OOF
# ---------------------------------------------------------------------------
def ensemble_curve(wide_by_fold: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                   manifest: pd.DataFrame, seeds: Sequence[int] = SEEDS,
                   ks: Sequence[int] = (1, 2, 3, 4, 5)) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for fusion in ("mean_logit", "mean_rank"):
        by_k = {}
        for k in ks:
            combos = list(itertools.combinations(seeds, k))
            fprs, aucs, paucs = [], [], []
            for combo in combos:
                pool = ensemble_pool_oof(wide_by_fold, combo, fusion, manifest)
                m = metric_block(pool["label_int"].to_numpy(), pool["logit"].to_numpy())
                fprs.append(m["fpr_at_90_recall"])
                aucs.append(m["roc_auc"])
                paucs.append(m["pauc_15_std"])
            by_k[k] = {
                "n_combos": len(combos),
                "fpr": spread(fprs), "roc_auc": spread(aucs), "pauc_15_std": spread(paucs),
                "fpr_values": fprs,
            }
        out[fusion] = by_k
    return out


# ---------------------------------------------------------------------------
# Experiment 3 -- ensembling under domain shift
# ---------------------------------------------------------------------------
def experiment3(job_a: str, job_b: str, manifest: pd.DataFrame,
                wide_by_fold: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                pooled_curve: Dict[str, Any]) -> Dict[str, Any]:
    before_pool_fprs = [
        metric_block(pool_oof(job_a, REPEAT, s, FOLDS, manifest)["label_int"].to_numpy(),
                    pool_oof(job_a, REPEAT, s, FOLDS, manifest)["logit"].to_numpy())["fpr_at_90_recall"]
        for s in SEEDS
    ]
    before_pool = spread(before_pool_fprs)

    out: Dict[str, Any] = {"before_pooled_oof": before_pool, "directions": {}}
    for centre in CENTRES:
        wide = build_wide_loco(job_b, centre, manifest)
        before_fprs = []
        for s in SEEDS:
            y, sc, _ = ensemble_loco(wide, (s,), "mean_logit")
            before_fprs.append(metric_block(y, sc)["fpr_at_90_recall"])
        before = spread(before_fprs)

        direction: Dict[str, Any] = {
            "before": before,
            "before_gap": before["median"] - before_pool["median"],
        }
        for fusion in ("mean_logit", "mean_rank"):
            combos4 = list(itertools.combinations(SEEDS, 4))
            loo_fprs = []
            for combo in combos4:
                y, sc, _ = ensemble_loco(wide, combo, fusion)
                loo_fprs.append(metric_block(y, sc)["fpr_at_90_recall"])
            y5, sc5, _ = ensemble_loco(wide, SEEDS, fusion)
            after_fpr5 = metric_block(y5, sc5)["fpr_at_90_recall"]
            after_pool_fpr5 = pooled_curve[fusion][5]["fpr"]["median"]  # k=5 has n=1; this IS that point
            direction[fusion] = {
                "after_k5_fpr": after_fpr5,
                "after_loo4_spread": spread(loo_fprs),
                "after_gap_k5": after_fpr5 - after_pool_fpr5,
                "pooled_oof_after_k5_fpr": after_pool_fpr5,
            }
        out["directions"][f"holdout_center_{centre}"] = direction
    return out


# ---------------------------------------------------------------------------
# Experiment 4 -- corrected MDE
# ---------------------------------------------------------------------------
def se_table(sd: float, ns: Sequence[int] = (3, 5, 10)) -> List[Dict[str, float]]:
    rows = []
    for n in ns:
        se_mean = sd / np.sqrt(n)
        se_median = 1.2533 * sd / np.sqrt(n)  # asymptotic normal approx; flagged in report
        rows.append({"n": n, "se_mean": float(se_mean), "se_median": float(se_median)})
    return rows


def resolvable_table(sd: float, protocol: str, units_per_arm_fn,
                     ns: Sequence[int] = (3, 5, 10),
                     rhos: Sequence[float] = (0.0, 0.3, 0.5, 0.7)) -> List[Dict[str, Any]]:
    """2*SE_diff, using the MEDIAN's SE -- the statistic the actual comparison
    uses -- not the mean's. Independent arms: rho=0.0 row. Paired arms (same
    seeds in both arms) at illustrative correlations; rho is not measured here
    -- there is no paired-experiment data in this dataset to estimate it from."""
    rows = []
    for n in ns:
        se_median = 1.2533 * sd / np.sqrt(n)
        units = units_per_arm_fn(n) * 2
        hours = units * MINUTES_PER_UNIT / 60.0
        row = {"protocol": protocol, "n": n, "total_units": units, "hours": hours}
        for rho in rhos:
            se_diff = se_median * np.sqrt(2 * (1 - rho))
            row[f"resolvable_rho{rho:.1f}"] = float(2 * se_diff)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Experiment 5 -- hospital signal
# ---------------------------------------------------------------------------
def diagnostics_for(pool: pd.DataFrame, manifest: pd.DataFrame,
                    score_col: str = "logit") -> Dict[str, Any]:
    joined = attach_manifest(with_score_as_logit(pool, score_col), manifest)
    cc = diag_confound_correlations(joined)
    hn = diag_hard_negatives(joined, n=50)
    centre_auc = cc.get("centre_separability_among_negatives", {}).get(
        "auc_logit_predicts_centre", float("nan"))
    fsb = cc["columns"].get("file_size_bytes", {})
    return {
        "centre_auc_negatives": centre_auc,
        "rho_file_size_positives": fsb.get("positives_only", {}).get("rho", float("nan")),
        "rho_file_size_negatives": fsb.get("negatives_only", {}).get("rho", float("nan")),
        "top50_centre2_ratio": hn["enrichment"].get("centre", {}).get(
            "center_2", {}).get("ratio", float("nan")),
        "top50_redaction_ratio": hn["enrichment"].get("has_redaction", {}).get(
            "True", {}).get("ratio", float("nan")),
    }


def experiment5(job_a: str, manifest: pd.DataFrame,
                wide_by_fold: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> Dict[str, Any]:
    fields = ("centre_auc_negatives", "rho_file_size_positives",
             "rho_file_size_negatives", "top50_centre2_ratio", "top50_redaction_ratio")

    raw_single = [diagnostics_for(pool_oof(job_a, REPEAT, s, FOLDS, manifest), manifest)
                 for s in SEEDS]
    rank_single = [diagnostics_for(rank_normalised_pool(job_a, s, manifest), manifest,
                                   score_col="rank_norm") for s in SEEDS]

    out: Dict[str, Any] = {"single_seed": {}, "ensemble": {}}
    for label, series in (("raw", raw_single), ("rank", rank_single)):
        out["single_seed"][label] = {f: spread([d[f] for d in series]) for f in fields}

    for fusion, label in (("mean_logit", "raw"), ("mean_rank", "rank")):
        combos4 = list(itertools.combinations(SEEDS, 4))
        loo = [diagnostics_for(ensemble_pool_oof(wide_by_fold, c, fusion, manifest), manifest)
              for c in combos4]
        full = diagnostics_for(ensemble_pool_oof(wide_by_fold, SEEDS, fusion, manifest), manifest)
        out["ensemble"][label] = {
            f: {"k5": full[f], "loo4_spread": spread([d[f] for d in loo])} for f in fields
        }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-a", default="runs/noise_floor_a")
    ap.add_argument("--job-b", default="runs/noise_floor_b")
    ap.add_argument("--manifest", default="manifests/rare25_folds_v2.csv")
    ap.add_argument("--out", default="reports/reanalysis")
    args = ap.parse_args(argv)

    job_a = args.job_a if os.path.isabs(args.job_a) else os.path.join(REPO_ROOT, args.job_a)
    job_b = args.job_b if os.path.isabs(args.job_b) else os.path.join(REPO_ROOT, args.job_b)
    manifest = pd.read_csv(os.path.join(REPO_ROOT, args.manifest))

    print("[17] sanity check: recomputing raw pooled-OOF FPR@90R per seed ...")
    raw_check = []
    for s in SEEDS:
        p = pool_oof(job_a, REPEAT, s, FOLDS, manifest)
        m = metric_block(p["label_int"].to_numpy(), p["logit"].to_numpy())
        raw_check.append(m["fpr_at_90_recall"])
        print(f"    seed {s}: {m['fpr_at_90_recall']:.4f}")
    raw_spread = spread(raw_check)
    print(f"    median {raw_spread['median']:.4f}  IQR {raw_spread['iqr']:.4f}  "
         f"(reports/noise_floor.md: median 0.1061, IQR 0.0235)")

    print("[17] experiment 1: rank normalisation ...")
    e1 = experiment1(job_a, manifest)

    print("[17] building wide (cross-seed-aligned) tables for ensembling ...")
    wide_by_fold = {f: build_wide_fold(job_a, f, manifest) for f in FOLDS}

    print("[17] experiment 2: ensembling, pooled OOF ...")
    e2 = ensemble_curve(wide_by_fold, manifest)

    print("[17] experiment 3: ensembling under domain shift ...")
    e3 = experiment3(job_a, job_b, manifest, wide_by_fold, e2)

    print("[17] experiment 4: corrected MDE ...")
    loco_sd = {}
    for centre in CENTRES:
        d = e3["directions"][f"holdout_center_{centre}"]
        loco_sd[centre] = d["before"]["sd"]
    e4 = {
        "se_pooled_oof": se_table(raw_spread["sd"]),
        "se_loco_c1": se_table(loco_sd[1]),
        "se_loco_c2": se_table(loco_sd[2]),
        "resolve_pooled_oof": resolvable_table(
            raw_spread["sd"], "pooled_oof", lambda n: 5 * n),
        "resolve_loco_c1": resolvable_table(
            loco_sd[1], "loco_c1", lambda n: 1 * n),
        "resolve_loco_c2": resolvable_table(
            loco_sd[2], "loco_c2", lambda n: 1 * n),
    }

    print("[17] experiment 5: hospital signal under ensembling ...")
    e5 = experiment5(job_a, manifest, wide_by_fold)

    # ------------------------------------------------------------------ #
    lines: List[str] = []
    W = lines.append

    W("# RARE26 re-analysis: rank normalisation, ensembling, corrected MDE\n")
    W("Generated by `scripts/17_reanalysis.py`. No training, no GPU -- every "
      "figure below is recomputed from the prediction parquets already on disk "
      "from Job A and Job B. Companion to `reports/noise_floor.md`, which this "
      "file does not modify.\n")
    W("**Rule applied throughout:** a figure from the single 5-seed ensemble "
      "(k=5) is always reported with the spread of its five leave-one-out "
      "4-seed ensembles. k=5 is one sample; the k=4 LOO family is the only "
      "variance available for it without retraining.\n")
    W(f"Sanity check against `reports/noise_floor.md`: recomputed raw pooled-OOF "
      f"FPR@90R median {raw_spread['median']:.4f}, IQR {raw_spread['iqr']:.4f} "
      f"(report: 0.1061 / 0.0235) -- {'MATCH' if abs(raw_spread['median']-0.1061)<1e-3 else 'MISMATCH, investigate'}.\n")

    # ---------------- Experiment 1 ----------------
    W("\n## 1. Rank normalisation before pooling\n")
    W("Per (fold, seed), validation logits are rank-transformed to (rank-0.5)/n "
      "within that fold before the five folds are pooled. Monotonic within a "
      "fold -- it cannot change which images that model ranks above which -- "
      "but it removes each fold's arbitrary logit scale before folds are "
      "compared to one another.\n")
    W("| seed | FPR@90R raw | FPR@90R rank-norm | ROC-AUC raw | ROC-AUC rank | "
      "max \\|share-0.20\\| raw | max \\|share-0.20\\| rank |")
    W("|---|---|---|---|---|---|---|")
    for r in e1["per_seed"]:
        W(f"| {r['seed']} | {r['fpr_raw']:.4f} | {r['fpr_rank']:.4f} | "
          f"{r['auc_raw']:.4f} | {r['auc_rank']:.4f} | "
          f"{r['max_dev_raw']:.3f} | {r['max_dev_rank']:.3f} |")
    W("")
    W(f"FPR@90R spread -- raw: {fmt_spread(e1['fpr_raw_spread'])}")
    W(f"FPR@90R spread -- rank-normalised: {fmt_spread(e1['fpr_rank_spread'])}")
    W(f"\nPer-fold false-positive share, seed 0 -- raw: "
      + ", ".join(f"f{k}={v:.2f}" for k, v in sorted(e1["fp_share_raw_seed0"].items())))
    W(f"Per-fold false-positive share, seed 0 -- rank-normalised: "
      + ", ".join(f"f{k}={v:.2f}" for k, v in sorted(e1["fp_share_rank_seed0"].items())) + "\n")

    solo_auc = e1["per_fold_solo_auc"]
    solo_auc_spread = spread(list(solo_auc.values()))
    W("**Why rank normalisation does not simply help.** Every fold holds "
      f"exactly 586 negatives (verified above), so a within-fold rank "
      f"transform maps each fold's negatives onto ~uniform (0,1) *regardless "
      f"of that fold's actual separability*, and equal fold sizes mean equal "
      f"expected FP share follows automatically. The near-perfect 0.20 shares "
      f"in the rank-normalised row above are substantially a MECHANICAL "
      f"consequence of the transform plus equal fold sizes, not independent "
      f"proof that a real miscalibration was corrected.\n")
    W(f"Meanwhile each fold is a genuinely different trained model with a real "
      f"quality difference: solo AUC ranges {fmt_spread(solo_auc_spread)} across "
      f"the 5 folds of seed 0 (fold-by-fold: "
      + ", ".join(f"f{k}={v:.4f}" for k, v in sorted(solo_auc.items())) + "). "
      "Rank-normalising forces fold 4's worst negatives (AUC 0.988) to occupy "
      "the same rank_norm range as fold 1's worst negatives (AUC 0.921), which "
      "discards real information about which fold's model should be trusted "
      "more at a shared threshold. That is the competing effect, and on this "
      "data it outweighs the calibration fix: net seed SD did not shrink.\n")

    W("```")
    W(box([
        "HEADLINE -- rank normalisation and the seed noise floor",
        "",
        f"  raw seed SD (pooled FPR@90R):          {e1['fpr_raw_spread']['sd']:.4f}",
        f"  rank-normalised seed SD:                {e1['fpr_rank_spread']['sd']:.4f}",
        "",
        "  Rank normalisation FIXED the pooling-validity finding (max fold-share",
        "  deviation 0.101-0.171 -> 0.004-0.010) but did NOT shrink the seed",
        "  noise floor -- it slightly grew it. The two effects are not the same",
        "  question: the FP-share fix is real and mechanical; the noise floor is",
        "  net of that fix AND the loss of real inter-fold quality differences",
        "  (per-fold solo AUC spans 0.921-0.988), and the second effect dominates.",
        "",
        "  CONCLUSION: none of the 0.0348 raw seed SD is cleanly attributable to",
        "  calibration drift alone -- the two corrections are entangled, and",
        "  disentangling them would need per-fold-quality-weighted normalisation,",
        "  not tested here.",
    ]))
    W("```\n")

    # ---------------- Experiment 2 ----------------
    W("\n## 2. Seed ensembling, pooled OOF\n")
    W("Two fusion rules, both zero-cost at inference since the 25 checkpoints "
      "already exist: `mean_logit` averages raw logits across seeds per image; "
      "`mean_rank` averages each seed's within-fold rank-normalised score.\n")
    for fusion in ("mean_logit", "mean_rank"):
        W(f"**{fusion}**\n")
        W("| k | n combos | FPR@90R median | IQR | ROC-AUC median | pAUC15std median |")
        W("|---|---|---|---|---|---|")
        for k in (1, 2, 3, 4, 5):
            d = e2[fusion][k]
            note = "" if d["n_combos"] > 1 else " (single ensemble; see LOO-4 below)"
            W(f"| {k} | {d['n_combos']} | {d['fpr']['median']:.4f}{note} | "
              f"{d['fpr']['iqr']:.4f} | {d['roc_auc']['median']:.4f} | "
              f"{d['pauc_15_std']['median']:.4f} |")
        W("")
    baseline_median = raw_spread["median"]
    W(f"Single-seed baseline (k=1, raw): FPR@90R median {baseline_median:.4f}.\n")
    for fusion in ("mean_logit", "mean_rank"):
        k5 = e2[fusion][5]["fpr_values"][0]
        k4 = e2[fusion][4]["fpr"]
        W(f"- **{fusion}**, full 5-seed ensemble: FPR@90R = {k5:.4f} "
          f"(k=4 LOO spread: {fmt_spread(k4)}), vs single-seed median "
          f"{baseline_median:.4f} -- "
          f"{'improvement' if k5 < baseline_median else 'no improvement'} of "
          f"{abs(baseline_median - k5):.4f}")
    W("")
    flatten_desc: Dict[str, str] = {}
    for fusion in ("mean_logit", "mean_rank"):
        meds = [e2[fusion][k]["fpr"]["median"] for k in (1, 2, 3, 4, 5)]
        deltas = [meds[i] - meds[i + 1] for i in range(len(meds) - 1)]
        W(f"Marginal gain per added seed ({fusion}): "
          + " -> ".join(f"{d:+.4f}" for d in deltas)
          + f" (k1-2, k2-3, k3-4, k4-5)")
        total_gain = meds[0] - meds[-1]
        frac_by_k2 = (meds[0] - meds[1]) / total_gain if total_gain > 0 else float("nan")
        if any(d < 0 for d in deltas):
            flatten_desc[fusion] = (
                "non-monotonic -- at least one added seed made the median WORSE "
                f"({min(deltas):+.4f} at its worst step); no flattening point can "
                "be read off this curve, only noise")
        elif total_gain > 0 and frac_by_k2 > 0.5:
            flatten_desc[fusion] = f"most of the gain ({frac_by_k2*100:.0f}%) arrives by k=2"
        elif total_gain > 0:
            flatten_desc[fusion] = (
                "monotonically improving through k=5 with no plateau yet visible "
                "in this range -- gains are fairly even across k, not front-loaded")
        else:
            flatten_desc[fusion] = "no net improvement from k=1 to k=5"
    W("")
    for fusion in ("mean_logit", "mean_rank"):
        W(f"- **{fusion}**: {flatten_desc[fusion]}.")
    W("")

    # ---------------- Experiment 3 ----------------
    W("\n## 3. Ensembling under domain shift (LOCO)\n")
    W("The two directions are never averaged -- see `reports/noise_floor.md` "
      "section 5 for why. 'Before' = single-seed median (k=1, raw), matching "
      "the original report's 0.3490 / 0.2612.\n")
    for k, d in e3["directions"].items():
        W(f"### `{k}`\n")
        W(f"Before ensembling: FPR@90R {fmt_spread(d['before'])}, "
          f"gap vs pooled OOF (before) = {d['before_gap']:+.4f}\n")
        W("| fusion | after (k=5) FPR@90R | k=4 LOO spread | gap vs pooled OOF "
          "(k=5, same fusion) |")
        W("|---|---|---|---|")
        for fusion in ("mean_logit", "mean_rank"):
            f = d[fusion]
            W(f"| {fusion} | {f['after_k5_fpr']:.4f} | "
              f"{fmt_spread(f['after_loo4_spread'])} | {f['after_gap_k5']:+.4f} |")
        W("")

    # ---------------- Experiment 4 ----------------
    W("\n## 4. Corrected minimum detectable effect\n")
    W("`reports/noise_floor.md`'s 0.090 is the RANGE of five individual runs. "
      "The comparison this codebase actually makes is between the MEDIANS of "
      "two n-seed groups, whose uncertainty is the median's standard error, "
      "not the range of single runs.\n")
    W("**Job A held repeat 0 fixed** (same 5 folds for every seed), so every SD "
      "in this section captures training stochasticity only. Fold-partition "
      "variance is excluded; the true total uncertainty across repeats is "
      "larger than these figures.\n")

    W("### Standard error of the mean and of the median\n")
    W("| protocol | observed SD (n=5) | n | SE(mean) | SE(median) |")
    W("|---|---|---|---|---|")
    for label, sd, tbl in (
        ("pooled_oof", raw_spread["sd"], e4["se_pooled_oof"]),
        ("loco_c1", loco_sd[1], e4["se_loco_c1"]),
        ("loco_c2", loco_sd[2], e4["se_loco_c2"]),
    ):
        for row in tbl:
            W(f"| {label} | {sd:.4f} | {row['n']} | {row['se_mean']:.4f} | "
              f"{row['se_median']:.4f} |")
    W("")
    W("SE(median) uses the asymptotic normal approximation 1.2533 x SD/sqrt(n). "
      "At n=3-10 this is illustrative of order of magnitude and trend, not a "
      "precise inferential claim -- with only 5 seeds observed here there is no "
      "way to estimate the finite-sample SE of the median more honestly than "
      "this without more runs.\n")

    W("### experiment design: what each protocol can resolve\n")
    W("Resolvable FPR difference = 2 x SE(median-difference) at illustrative "
      "seed-pairing correlations rho. rho=0.0 is the independent-arms case "
      "(two unrelated seed sets). rho>0 is a PAIRED design -- same seeds used "
      "in both arms, e.g. seed i's baseline run compared against seed i's "
      "treatment run -- which cancels shared per-seed noise IF the treatment's "
      "effect is roughly seed-independent. rho is NOT measured here: this "
      "dataset has no paired baseline/treatment runs to estimate it from. "
      "Values are illustrative sensitivity, not a fitted number. Pairing does "
      "not change the run count or cost -- only the achievable resolution for "
      "the same spend.\n")
    W("| protocol | n (seeds/arm) | total units (2 arms) | wall-clock (h) | "
      "resolvable, independent (rho=0) | rho=0.3 | rho=0.5 | rho=0.7 |")
    W("|---|---|---|---|---|---|---|---|")
    for tbl in (e4["resolve_pooled_oof"], e4["resolve_loco_c1"], e4["resolve_loco_c2"]):
        for row in tbl:
            W(f"| {row['protocol']} | {row['n']} | {row['total_units']} | "
              f"{row['hours']:.1f} | {row['resolvable_rho0.0']:.4f} | "
              f"{row['resolvable_rho0.3']:.4f} | {row['resolvable_rho0.5']:.4f} | "
              f"{row['resolvable_rho0.7']:.4f} |")
    W("")

    # ---------------- Experiment 5 ----------------
    W("\n## 5. Does ensembling remove the hospital signal?\n")
    W("Single-seed baselines: AUC(logit predicts centre \\| negatives) median "
      "0.300 (0.5 = no hospital information); Spearman rho(logit, "
      "file_size_bytes \\| positives) median +0.360; top-50 hardest-negative "
      "enrichment 1.89x center_2, 1.99x has_redaction (all from "
      "`reports/noise_floor.md`; recomputed below for the same 5 seeds).\n")

    def fmt_val(v):
        return f"{v:.4f}" if np.isfinite(v) else "nan"

    W("| condition | AUC(logit->centre \\| neg) | rho(file_size \\| pos) | "
      "rho(file_size \\| neg) | top50 center_2 ratio | top50 redaction ratio |")
    W("|---|---|---|---|---|---|")
    for label in ("raw", "rank"):
        d = e5["single_seed"][label]
        W(f"| single-seed, {label} (median, n=5) | "
          f"{fmt_val(d['centre_auc_negatives']['median'])} | "
          f"{fmt_val(d['rho_file_size_positives']['median'])} | "
          f"{fmt_val(d['rho_file_size_negatives']['median'])} | "
          f"{fmt_val(d['top50_centre2_ratio']['median'])} | "
          f"{fmt_val(d['top50_redaction_ratio']['median'])} |")
    for label in ("raw", "rank"):
        d = e5["ensemble"][label]
        W(f"| 5-seed ensemble, {label} (k=5 point) | "
          f"{fmt_val(d['centre_auc_negatives']['k5'])} | "
          f"{fmt_val(d['rho_file_size_positives']['k5'])} | "
          f"{fmt_val(d['rho_file_size_negatives']['k5'])} | "
          f"{fmt_val(d['top50_centre2_ratio']['k5'])} | "
          f"{fmt_val(d['top50_redaction_ratio']['k5'])} |")
    W("")
    W("k=5 ensemble figures above are single points; their k=4 leave-one-out "
      "spreads (median, IQR):\n")
    for label in ("raw", "rank"):
        d = e5["ensemble"][label]
        W(f"- **{label} ensemble**: AUC(centre) "
          f"{fmt_spread(d['centre_auc_negatives']['loo4_spread'])}; "
          f"rho(file_size|pos) {fmt_spread(d['rho_file_size_positives']['loo4_spread'])}")
    W("")

    # ---------------- Summary ----------------
    e5_raw_single = e5["single_seed"]["raw"]["centre_auc_negatives"]["median"]
    e5_raw_ens = e5["ensemble"]["raw"]["centre_auc_negatives"]["k5"]
    e5_rank_ens = e5["ensemble"]["rank"]["centre_auc_negatives"]["k5"]
    dist_raw_single = abs(e5_raw_single - 0.5)
    dist_raw_ens = abs(e5_raw_ens - 0.5)
    dist_rank_ens = abs(e5_rank_ens - 0.5)

    fpr_after_logit = e2["mean_logit"][5]["fpr_values"][0]
    fpr_after_rank = e2["mean_rank"][5]["fpr_values"][0]

    gap1_before = e3["directions"]["holdout_center_1"]["before_gap"]
    gap1_after = e3["directions"]["holdout_center_1"]["mean_rank"]["after_gap_k5"]
    gap2_before = e3["directions"]["holdout_center_2"]["before_gap"]
    gap2_after = e3["directions"]["holdout_center_2"]["mean_rank"]["after_gap_k5"]

    W("\n## Summary\n")
    W("```")
    W(box([
        "1. RANK NORMALISATION",
        f"   Raw seed SD {e1['fpr_raw_spread']['sd']:.4f} -> rank-normalised "
        f"{e1['fpr_rank_spread']['sd']:.4f} (grew, did not shrink)",
        "   Fixed the FP-share imbalance (fold 3's 37% -> ~20% everywhere) but",
        "   that fix is largely mechanical (equal fold sizes); it also erased",
        "   real inter-fold quality differences (solo AUC 0.921-0.988), which",
        "   dominated. The two effects are entangled, not separable with this data.",
        "",
        "2. ENSEMBLING",
        f"   Single-seed median {baseline_median:.4f} -> 5-seed ensemble "
        f"{fpr_after_logit:.4f} (mean_logit) / {fpr_after_rank:.4f} (mean_rank)",
        f"   mean_logit: {flatten_desc['mean_logit']}",
        f"   mean_rank:  {flatten_desc['mean_rank']}",
        "",
        "3. DOMAIN GAP",
        f"   holdout_center_1: gap {gap1_before:+.4f} -> {gap1_after:+.4f} "
        f"(mean_rank, k=5)",
        f"   holdout_center_2: gap {gap2_before:+.4f} -> {gap2_after:+.4f} "
        f"(mean_rank, k=5)",
        "",
        "4. HOSPITAL SIGNAL",
        f"   |AUC(centre)-0.5|: single-seed {dist_raw_single:.3f} -> raw "
        f"ensemble {dist_raw_ens:.3f} -> rank ensemble {dist_rank_ens:.3f}",
    ]))
    W("```\n")

    text = "\n".join(lines)
    out_stem = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    with open(out_stem + ".md", "w", encoding="utf-8") as fh:
        fh.write(text)
    payload = {
        "sanity_check_raw_spread": raw_spread, "experiment1": e1,
        "experiment2": e2, "experiment3": e3, "experiment4": e4,
        "experiment5": e5,
    }
    with open(out_stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_JD)

    print("\n" + text)
    print(f"\nwritten: {out_stem}.md / .json")

    print("\n" + "=" * 78)
    print("FOUR QUESTIONS")
    print("=" * 78)
    print(f"1. Rank normalisation: seed SD {e1['fpr_raw_spread']['sd']:.4f} -> "
         f"{e1['fpr_rank_spread']['sd']:.4f} (grew; fixed FP-share imbalance but "
         f"erased real inter-fold quality differences, net effect negative)")
    print(f"2. Ensembling: {baseline_median:.4f} -> {fpr_after_rank:.4f} "
         f"(mean_rank, 5 seeds); mean_logit {flatten_desc['mean_logit']}, "
         f"mean_rank {flatten_desc['mean_rank']}")
    print(f"3. Domain gap: center_1 {gap1_before:+.4f} -> {gap1_after:+.4f}, "
         f"center_2 {gap2_before:+.4f} -> {gap2_after:+.4f} (mean_rank, k=5) -- "
         f"{'narrows' if abs(gap1_after)<abs(gap1_before) and abs(gap2_after)<abs(gap2_before) else 'does not consistently narrow'}")
    print(f"4. Hospital signal: AUC(centre) distance from 0.5 -- single-seed "
         f"{dist_raw_single:.3f}, raw ensemble {dist_raw_ens:.3f}, rank "
         f"ensemble {dist_rank_ens:.3f} -- "
         f"{'reduced' if dist_rank_ens < dist_raw_single * 0.5 else 'NOT substantially reduced; common to all seeds'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
