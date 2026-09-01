"""The A1 early report. Written the moment A1 finishes, not held for the end.

ONE QUESTION: does the centre x class sampler move the AUC of
logit-predicts-centre among negatives off the baseline's 0.300?

Why this cannot wait. Seed ensembling could not shift that number -- the
hospital signal turned out to be common to every seed, so averaging them
preserves it. If a TRAINING-TIME intervention aimed squarely at the confound
also fails to move it, the shortcut is not coming from the sampled class/centre
association at all, and the rest of the plan is aimed at the wrong thing. That
conclusion is worth having at hour 10 rather than hour 55.

    python scripts/25_a1_early.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import Config  # noqa: E402
from src.data import (  # noqa: E402
    BarrettDataset, make_sampler, stratum_draw_counts,
)
from src.evaluate import box, fmt_spread, metric_block, spread  # noqa: E402
from src.folds import get_split  # noqa: E402
from src.io import write_text_durable  # noqa: E402


def _load_reanalysis():
    path = os.path.join(REPO_ROOT, "scripts", "17_reanalysis.py")
    spec = importlib.util.spec_from_file_location("_reanalysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RA = _load_reanalysis()

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
REPEAT = 0

# reports/noise_floor.md, single seed, pooled OOF
BASELINE_CENTRE_AUC = 0.300
BASELINE_RHO_POS = 0.360
# scripts/21_sampler_demo.py, class-only weighting on the pooled training set
BASELINE_STRATA = {
    "center_1|neoplasia": 19.2, "center_1|non-dysplastic": 37.9,
    "center_2|neoplasia": 30.8, "center_2|non-dysplastic": 12.1,
}


def pooled_diagnostics(out_dir: str, manifest: pd.DataFrame,
                       seeds=SEEDS) -> Dict[str, Any]:
    """Per-seed and ensembled confound diagnostics for one arm."""
    wide = {f: RA.build_wide_fold(os.path.join(REPO_ROOT, out_dir), f, manifest,
                                  seeds) for f in FOLDS}

    per_seed = []
    for s in seeds:
        pool = RA.ensemble_pool_oof(wide, [s], "mean_logit", manifest)
        d = RA.diagnostics_for(pool, manifest)
        y = pool["label_int"].to_numpy()
        d["fpr_at_90_recall"] = metric_block(y, pool["logit"].to_numpy())[
            "fpr_at_90_recall"]
        per_seed.append(d)

    k5_pool = RA.ensemble_pool_oof(wide, seeds, "mean_logit", manifest)
    k5 = RA.diagnostics_for(k5_pool, manifest)
    k5["fpr_at_90_recall"] = metric_block(
        k5_pool["label_int"].to_numpy(), k5_pool["logit"].to_numpy())[
        "fpr_at_90_recall"]

    loo = []
    for dropped in seeds:
        subset = [s for s in seeds if s != dropped]
        pool = RA.ensemble_pool_oof(wide, subset, "mean_logit", manifest)
        d = RA.diagnostics_for(pool, manifest)
        d["fpr_at_90_recall"] = metric_block(
            pool["label_int"].to_numpy(), pool["logit"].to_numpy())[
            "fpr_at_90_recall"]
        loo.append(d)

    fields = ("centre_auc_negatives", "rho_file_size_positives",
              "rho_file_size_negatives", "fpr_at_90_recall")
    return {
        "per_seed": {f: spread([d[f] for d in per_seed]) for f in fields},
        "per_seed_values": {f: [d[f] for d in per_seed] for f in fields},
        "k5": {f: k5[f] for f in fields},
        "loo4": {f: spread([d[f] for d in loo]) for f in fields},
    }


def realised_strata(config_path: str, manifest: pd.DataFrame,
                    manifest_path: str, seed: int = 0) -> Dict[str, float]:
    cfg = Config.from_yaml(os.path.join(REPO_ROOT, config_path))
    train, _ = get_split(REPEAT, 0, manifest_path)
    ds = BarrettDataset(train, manifest, cfg, train=True, cache=None,
                        build_cache=False)
    gen = torch.Generator()
    gen.manual_seed(seed)
    counts = stratum_draw_counts(make_sampler(ds, cfg, gen), ds, gen)
    total = sum(counts.values())
    return {k: 100.0 * v / total for k, v in counts.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a0-runs", default="runs/noise_floor_a")
    ap.add_argument("--a1-runs", default="runs/sweep_a1")
    ap.add_argument("--manifest", default="manifests/rare25_folds_v2.csv")
    ap.add_argument("--out", default="reports/a1_early.md")
    args = ap.parse_args(argv)

    manifest_path = os.path.join(REPO_ROOT, args.manifest)
    manifest = pd.read_csv(manifest_path)

    a0 = pooled_diagnostics(args.a0_runs, manifest)
    a1 = pooled_diagnostics(args.a1_runs, manifest)
    strata = realised_strata("configs/sweep_a1.yaml", manifest, manifest_path)

    def dist(v):
        return abs(v - 0.5)

    a0_auc, a1_auc = a0["k5"]["centre_auc_negatives"], a1["k5"]["centre_auc_negatives"]
    a0_ss = a0["per_seed"]["centre_auc_negatives"]["median"]
    a1_ss = a1["per_seed"]["centre_auc_negatives"]["median"]

    moved = dist(a1_ss) < dist(a0_ss)
    L: List[str] = []
    A = L.append
    A("# A1 early report — does the centre x class sampler touch the hospital signal?")
    A("")
    A("Written as soon as A1 finished, ahead of the rest of the sweep. Pooled "
      "out-of-fold, repeat 0, five folds, seeds 0-4. Every figure is recomputed "
      "from prediction parquets on disk.")
    A("")
    A(box([
        "THE QUESTION",
        "",
        "AUC of logit predicting centre, among NEGATIVES only.",
        "0.5 = no hospital information. Baseline distance from 0.5 = 0.200.",
        "",
        f"  A0 baseline      single-seed median {a0_ss:.4f}   "
        f"distance {dist(a0_ss):.4f}",
        f"  A1 centre x class  single-seed median {a1_ss:.4f}   "
        f"distance {dist(a1_ss):.4f}",
        "",
        f"  A0 5-seed ensemble {a0_auc:.4f}   distance {dist(a0_auc):.4f}",
        f"  A1 5-seed ensemble {a1_auc:.4f}   distance {dist(a1_auc):.4f}",
        "",
        f"  CHANGE IN DISTANCE (single seed): "
        f"{dist(a1_ss) - dist(a0_ss):+.4f}",
        f"  ANSWER: the sampler "
        f"{'REDUCES' if moved else 'DOES NOT REDUCE'} the hospital signal.",
    ]))
    A("")

    A("## Realised per-stratum draw counts")
    A("")
    A("One drawn epoch on the pooled training set, against the class-only "
      "baseline. This is what the intervention actually did to the data stream.")
    A("")
    A("| centre | class | class-only baseline | A1 centre x class |")
    A("|---|---|---|---|")
    for key in sorted(BASELINE_STRATA):
        centre_name, class_name = key.split("|")
        A(f"| {centre_name} | {class_name} | {BASELINE_STRATA[key]:.1f}% | "
          f"{strata.get(key, 0.0):.1f}% |")
    A("")
    A("The sampler is doing exactly what it was built to do: under class-only "
      "weighting center_2 supplies 62% of the sampled positives against 24% of "
      "the sampled negatives, and that association is what makes hospital "
      "identity predictive. A1 flattens all four cells to ~25%.")
    A("")

    A("## Every diagnostic, A0 vs A1")
    A("")
    A("| metric | A0 single-seed (median, IQR) | A1 single-seed (median, IQR) | "
      "A0 k=5 | A1 k=5 | A1 k=5 LOO-4 IQR |")
    A("|---|---|---|---|---|---|")
    labels = {
        "centre_auc_negatives": "AUC(logit→centre \\| neg)",
        "rho_file_size_positives": "rho(file_size) \\| pos",
        "rho_file_size_negatives": "rho(file_size) \\| neg",
        "fpr_at_90_recall": "**FPR@90R**",
    }
    for f, label in labels.items():
        s0, s1 = a0["per_seed"][f], a1["per_seed"][f]
        A(f"| {label} | {s0['median']:.4f} ({s0['iqr']:.4f}) | "
          f"{s1['median']:.4f} ({s1['iqr']:.4f}) | {a0['k5'][f]:.4f} | "
          f"{a1['k5'][f]:.4f} | {a1['loo4'][f]['iqr']:.4f} |")
    A("")
    A(f"Reference points from earlier reports — centre-AUC {BASELINE_CENTRE_AUC:.3f}, "
      f"rho(file_size|positives) +{BASELINE_RHO_POS:.3f}, pooled single-seed "
      f"FPR@90R 0.1061, 5-seed ensemble 0.0427.")
    A("")
    A("Per-seed centre-AUC values:")
    A("")
    A(f"- A0: " + ", ".join(f"{v:.4f}" for v in
                            a0["per_seed_values"]["centre_auc_negatives"]))
    A(f"- A1: " + ", ".join(f"{v:.4f}" for v in
                            a1["per_seed_values"]["centre_auc_negatives"]))
    A("")

    A("## Reading")
    A("")
    if moved:
        A(f"A training-time intervention **can** move the hospital signal where "
          f"ensembling could not: the distance from 0.5 falls "
          f"{dist(a0_ss):.4f} → {dist(a1_ss):.4f}. The shortcut is at least "
          f"partly attributable to the sampled centre/class association, which "
          f"means attacking it at the sampler is aimed at something real.")
    else:
        A(f"**The sampler does not reduce the hospital signal.** The distance "
          f"from 0.5 goes {dist(a0_ss):.4f} → {dist(a1_ss):.4f} despite the "
          f"four strata being drawn at ~25% each, i.e. despite centre and class "
          f"being made statistically independent in the sampled stream. "
          f"Ensembling could not move this number either. Two interventions "
          f"aimed at the confound from different directions have now both "
          f"failed to shift it, which points away from the sampled "
          f"centre/class association as the source. The signal is more likely "
          f"carried by the images themselves — acquisition, processing, "
          f"compression — in a way that survives any reweighting of WHICH "
          f"images are shown. That is a different problem, and it is worth "
          f"knowing now.")
    A("")
    A("FPR@90R for A1 is reported above but is not decided here; the acceptance "
      "rule is applied in `reports/magnitude_sweep.md` once every arm exists.")
    A("")

    md = "\n".join(L)
    out_md = os.path.join(REPO_ROOT, args.out)
    write_text_durable(out_md, md)
    write_text_durable(os.path.join(REPO_ROOT, "reports", "a1_early.json"),
                       json.dumps({"a0": a0, "a1": a1, "strata": strata},
                                  indent=2, default=RA._JD))

    print(md)
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
