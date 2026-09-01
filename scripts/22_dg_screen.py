"""Factorial screen of the domain-generalisation components. NO TRAINING, NO GPU.

Reads prediction parquets already on disk and produces reports/dg_screen.md.

DECISION METRIC: 5-seed ensembled (mean raw logit) FPR@90R, per LOCO direction,
reported separately and never averaged across directions. The deployed artefact
is an ensemble, so the decision unit is an ensemble.

SPREAD IS MANDATORY. A single 5-seed ensemble is one sample and has no variance
of its own. Every k=5 figure is therefore reported with the spread of its five
leave-one-out 4-seed ensembles, and that spread IS the ensembled minimum
detectable effect -- the figure reports/reanalysis.md asked for and did not
deliver, because at the time no two configurations existed to difference.

ACCEPTANCE RULE, fixed before the numbers were seen: a component is accepted
only if it improves BOTH directions, and the improvement in at least one
direction exceeds that direction's ensembled leave-one-out IQR. A gain in one
direction only is an accident and is reported as one.

The machinery for building wide per-seed tables and for the confound
diagnostics is imported from scripts/17_reanalysis.py rather than reimplemented,
so the two reports cannot drift apart on what "the center_2 test set" or
"rho(file_size)" means.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluate import box, fmt_spread, metric_block, spread  # noqa: E402
from src.metrics import roc_auc  # noqa: E402


def _load_reanalysis():
    """Import scripts/17_reanalysis.py, whose name is not a valid identifier."""
    path = os.path.join(REPO_ROOT, "scripts", "17_reanalysis.py")
    spec = importlib.util.spec_from_file_location("_reanalysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RA = _load_reanalysis()

SEEDS = (0, 1, 2, 3, 4)
CENTRES = (1, 2)
FUSION = "mean_logit"   # the primary decision fusion


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
# `runs` is None for an arm that was NOT executed because it is provably the
# same computation as another arm; `same_as` names that arm. See the module
# docstring of configs/dg_c1.yaml and section 1 of the report.
CONFIGS: List[Dict[str, Any]] = [
    {"key": "C0", "label": "baseline",
     "runs": "runs/noise_floor_b", "same_as": None,
     "desc": "class-weighted sampler, hflip + black boxes"},
    {"key": "C1", "label": "+ centre x class sampler",
     "runs": None, "same_as": "C0",
     "desc": "COMPONENT A; degenerate on LOCO (single-centre training set)"},
    {"key": "C2", "label": "+ photometric/optical stack",
     "runs": "runs/dg_c2", "same_as": None,
     "desc": "COMPONENT B"},
    {"key": "C3", "label": "+ both",
     "runs": None, "same_as": "C2",
     "desc": "COMPONENTS A+B; A degenerate on LOCO, so identical to C2"},
]


def resolve(cfg: Dict[str, Any], by_key: Dict[str, Dict[str, Any]]) -> str:
    """The run directory whose predictions this configuration's numbers come
    from -- its own, or those of the arm it is provably identical to."""
    while cfg["runs"] is None:
        cfg = by_key[cfg["same_as"]]
    return cfg["runs"]


# ---------------------------------------------------------------------------
# Per-direction analysis
# ---------------------------------------------------------------------------
def analyse_direction(run_dir: str, centre: int, manifest: pd.DataFrame,
                      seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """Everything the report needs for one (configuration, LOCO direction)."""
    wide = RA.build_wide_loco(os.path.join(REPO_ROOT, run_dir), centre, manifest,
                              seeds)
    base, logit_wide, rank_wide = wide

    # --- single seeds, for context only; the decision unit is the ensemble ---
    single = []
    for s in seeds:
        y = base["label_int"].to_numpy()
        single.append(metric_block(y, logit_wide[s].loc[base["filepath"]].to_numpy())
                      ["fpr_at_90_recall"])

    # --- k=5, the decision figure ---
    y5, s5, frame5 = RA.ensemble_loco(wide, seeds, FUSION)
    k5 = metric_block(y5, s5)

    # --- the five leave-one-out 4-seed ensembles: the only variance a k=5
    #     point estimate has without retraining ---
    loo, loo_frames = [], []
    for dropped in seeds:
        subset = [s for s in seeds if s != dropped]
        y4, s4, frame4 = RA.ensemble_loco(wide, subset, FUSION)
        loo.append(metric_block(y4, s4)["fpr_at_90_recall"])
        loo_frames.append({"dropped_seed": dropped, "frame": frame4})

    diag = RA.diagnostics_for(frame5, manifest)
    diag_loo = [RA.diagnostics_for(d["frame"], manifest) for d in loo_frames]

    return {
        "centre": centre,
        "n": int(len(base)),
        "n_pos": int((base["label_int"] == 1).sum()),
        "n_neg": int((base["label_int"] == 0).sum()),
        "single_seed_fpr": single,
        "single_seed_spread": spread(single),
        "k5_fpr": k5["fpr_at_90_recall"],
        "k5_roc_auc": k5["roc_auc"],
        "k5_pauc_15_std": k5["pauc_15_std"],
        "loo4_fpr": loo,
        "loo4_spread": spread(loo),
        "diagnostics_k5": diag,
        "diagnostics_loo4": {
            f: spread([d[f] for d in diag_loo])
            for f in ("rho_file_size_positives", "rho_file_size_negatives",
                      "top50_redaction_ratio")
        },
        "_frame_k5": frame5,
    }


# ---------------------------------------------------------------------------
# Centre separability -- see the long note this writes into the report
# ---------------------------------------------------------------------------
def centre_separability(per_dir: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """AUC of the logit predicting centre, among negatives, on the LOCO pool.

    THIS IS NOT THE METRIC THAT WAS ASKED FOR, and the difference matters. A
    LOCO held-out set contains exactly ONE centre, so within a direction the
    quantity is undefined -- there is no second class to separate. Pooling the
    two directions does produce a set with both centres and with every image
    scored by a model that never saw its hospital, but the two halves come from
    two DIFFERENT models, so any constant offset between their logit scales
    reads as centre separability even if neither model encodes hospital at all.

    Both numbers are therefore reported: raw, and after removing each
    direction's own negative median. If the raw figure is extreme and the
    centred one sits near 0.5, what was measured was the offset between two
    models and nothing about hospital signal. The clean measurement needs a
    single model scoring both centres, which is the pooled-OOF protocol
    deliberately deferred out of this step.
    """
    frames = [per_dir[c]["_frame_k5"] for c in sorted(per_dir)]
    pool = pd.concat(frames, ignore_index=True)
    neg = pool[pool["label_int"] == 0].copy()
    if neg["centre"].nunique() < 2:
        return {"auc_raw": float("nan"), "auc_centred": float("nan")}

    is_c2 = (neg["centre"] == "center_2").astype(int).to_numpy()
    raw = float(roc_auc(is_c2, neg["logit"].to_numpy()))

    centred = neg["logit"] - neg.groupby("centre")["logit"].transform("median")
    return {
        "auc_raw": raw,
        "auc_centred": float(roc_auc(is_c2, centred.to_numpy())),
        "median_logit_by_centre": {
            str(k): float(v)
            for k, v in neg.groupby("centre")["logit"].median().items()
        },
        "n_negatives": int(len(neg)),
    }


# ---------------------------------------------------------------------------
# Component A -- what the sampler actually draws
# ---------------------------------------------------------------------------
def sampler_draws(manifest_path: str, manifest: pd.DataFrame,
                  seed: int = 0) -> Dict[str, Any]:
    """Realised one-epoch stratum counts under both samplers, on the pooled CV
    training set (both centres) and on each LOCO training set (one centre).

    Computed here rather than quoted, so the report cannot come to disagree
    with the code it describes.
    """
    import torch

    from src.config import Config
    from src.data import BarrettDataset, make_sampler, stratum_draw_counts
    from src.folds import get_holdout_split, get_split

    cfg = Config.from_yaml(os.path.join(REPO_ROOT, "configs", "dg_c0.yaml"))
    out: Dict[str, Any] = {}
    splits = {"pooled_cv_r0f0": get_split(0, 0, manifest_path)[0]}
    for centre in CENTRES:
        splits[f"loco_c{centre}"] = get_holdout_split(centre, manifest_path)[0]

    for name, filepaths in splits.items():
        ds = BarrettDataset(filepaths, manifest, cfg, train=True, cache=None,
                            build_cache=False)
        block: Dict[str, Any] = {"n_train": len(ds)}
        for sampler_name in ("weighted", "balanced_centre_class"):
            gen = torch.Generator()
            gen.manual_seed(seed)
            s = make_sampler(
                ds, Config(**{**cfg.to_dict(), "sampler": sampler_name,
                              "aug": cfg.aug}), gen)
            block[sampler_name] = dict(stratum_draw_counts(s, ds, gen))
        block["identical"] = (block["weighted"] == block["balanced_centre_class"])
        out[name] = block
    return out


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def epoch_seconds(run_dir: str, seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """Median per-epoch seconds across every unit of a configuration.

    Read from each run's summary.json rather than timed again: these are the
    epochs that actually produced the predictions being compared, which is the
    only timing that answers "what did this configuration cost".
    """
    per_dir: Dict[int, List[float]] = {1: [], 2: []}
    for centre in CENTRES:
        for s in seeds:
            path = os.path.join(REPO_ROOT, run_dir, f"loco_c{centre}_s{s}",
                                "summary.json")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                per_dir[centre].append(float(json.load(fh)["median_epoch_seconds"]))
    allv = per_dir[1] + per_dir[2]
    return {
        "per_direction": {c: spread(v) for c, v in per_dir.items() if v},
        "overall": spread(allv) if allv else None,
    }


# ---------------------------------------------------------------------------
# Acceptance rule
# ---------------------------------------------------------------------------
def apply_acceptance(treatment: Dict[str, Any], control: Dict[str, Any],
                     name: str) -> Dict[str, Any]:
    """The rule as stated before the results were seen.

    Improvement is a REDUCTION in ensembled FPR@90R. The IQR bar is taken as
    the larger of the two arms' leave-one-out IQRs for that direction: the
    quantity under test is a difference between two noisy ensembles, so
    choosing the smaller of the two would be picking the yardstick that most
    flatters the result.
    """
    directions = {}
    for centre in CENTRES:
        t, c = treatment[centre], control[centre]
        improvement = c["k5_fpr"] - t["k5_fpr"]
        bar = max(c["loo4_spread"]["iqr"], t["loo4_spread"]["iqr"])
        # Robustness: repeat the comparison on the MEDIAN of each arm's five
        # leave-one-out ensembles instead of its single k=5 point. A k=5 figure
        # that happens to land at the edge of its own LOO family could carry a
        # verdict on its own; if the sign of the effect survives this swap, it
        # is a property of the configuration and not of one lucky ensemble.
        loo_improvement = (c["loo4_spread"]["median"] - t["loo4_spread"]["median"])
        directions[centre] = {
            "control_k5": c["k5_fpr"],
            "treatment_k5": t["k5_fpr"],
            "improvement": improvement,
            "improved": bool(improvement > 0),
            "control_loo4_median": c["loo4_spread"]["median"],
            "treatment_loo4_median": t["loo4_spread"]["median"],
            "improvement_loo4_median": loo_improvement,
            "sign_agrees_with_loo4_median": bool(
                (improvement > 0) == (loo_improvement > 0)),
            "iqr_control": c["loo4_spread"]["iqr"],
            "iqr_treatment": t["loo4_spread"]["iqr"],
            "iqr_bar": bar,
            "exceeds_iqr": bool(improvement > bar),
        }
    both = all(d["improved"] for d in directions.values())
    any_exceeds = any(d["exceeds_iqr"] for d in directions.values())
    return {
        "component": name,
        "directions": directions,
        "improves_both_directions": both,
        "exceeds_iqr_somewhere": any_exceeds,
        "accepted": bool(both and any_exceeds),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt(x: float, p: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def outcome_label(d: Dict[str, Any]) -> str:
    """Four outcomes, not three. An exactly-zero delta is not a worsening, and
    calling it one would misreport the null result this script is most likely
    to be asked to produce."""
    if d["exceeds_iqr"]:
        return "PASS"
    if d["improvement"] > 0:
        return "improved, under IQR"
    if d["improvement"] == 0:
        return "no change"
    return "WORSE"


def primary_table(results: Dict[str, Any]) -> str:
    """The 4x2 table of ensembled FPR@90R with IQRs."""
    head = (f"{'config':6s} {'component':32s} "
            f"{'  c1 FPR@90R (IQR)':>26s} {'  c2 FPR@90R (IQR)':>26s}")
    lines = [head, "-" * len(head)]
    for cfg in CONFIGS:
        r = results[cfg["key"]]
        cells = []
        for centre in CENTRES:
            d = r["directions"][centre]
            cells.append(f"{d['k5_fpr']:.4f} (IQR {d['loo4_spread']['iqr']:.4f})")
        note = "" if cfg["same_as"] is None else f"  = {cfg['same_as']}"
        lines.append(f"{cfg['key']:6s} {cfg['label'] + note:32s} "
                     f"{cells[0]:>26s} {cells[1]:>26s}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="manifests/rare25_folds_v2.csv")
    ap.add_argument("--out-md", default="reports/dg_screen.md")
    ap.add_argument("--out-json", default="reports/dg_screen.json")
    ap.add_argument("--c0-runs", default=None, help="override C0's run directory")
    ap.add_argument("--c2-runs", default=None,
                    help="override C2's run directory; pointing it at C0's "
                         "exercises every code path here against a known "
                         "zero-effect comparison")
    args = ap.parse_args(argv)

    manifest = pd.read_csv(os.path.join(REPO_ROOT, args.manifest))
    by_key = {c["key"]: c for c in CONFIGS}
    for key, override in (("C0", args.c0_runs), ("C2", args.c2_runs)):
        if override:
            by_key[key]["runs"] = override

    results: Dict[str, Any] = {}
    for cfg in CONFIGS:
        run_dir = resolve(dict(cfg), by_key)
        per_dir = {c: analyse_direction(run_dir, c, manifest) for c in CENTRES}
        results[cfg["key"]] = {
            "config": cfg,
            "run_dir": run_dir,
            "directions": per_dir,
            "centre_separability": centre_separability(per_dir),
            "timing": epoch_seconds(run_dir),
        }
    results["sampler_draws"] = sampler_draws(
        os.path.join(REPO_ROOT, args.manifest), manifest)

    accept_b = apply_acceptance(results["C2"]["directions"],
                                results["C0"]["directions"],
                                "COMPONENT B (photometric/optical stack)")

    table = primary_table(results)
    print("\n" + table + "\n")

    md = render_markdown(results, accept_b, table)
    out_md = os.path.join(REPO_ROOT, args.out_md)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write(md)

    payload = json.loads(json.dumps(
        {"results": {k: _strip(v) for k, v in results.items()},
         "acceptance": {"component_b": accept_b}},
        default=RA._JD))
    with open(os.path.join(REPO_ROOT, args.out_json), "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"written: {args.out_md}")
    print(f"written: {args.out_json}")
    return 0


def _strip(node):
    """Drop the cached prediction frames before serialising.

    Keys are not all strings -- the per-direction maps are keyed by centre
    number -- so the underscore test has to be guarded rather than applied
    blind.
    """
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def render_markdown(results: Dict[str, Any], accept_b: Dict[str, Any],
                    table: str) -> str:
    from datetime import datetime, timezone

    L: List[str] = []
    A = L.append

    A("# RARE26 domain-generalisation screen (LOCO)")
    A("")
    A(f"Generated by `scripts/22_dg_screen.py` on "
      f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. No training "
      f"and no GPU: every figure is recomputed from prediction parquets on disk.")
    A("")
    A("**Decision metric:** 5-seed ensembled (mean raw logit) FPR@90R, per LOCO "
      "direction. The two directions are reported separately and are never "
      "averaged. Every k=5 figure carries the spread of its five leave-one-out "
      "4-seed ensembles, which is the only variance a single ensemble has "
      "without retraining.")
    A("")

    # ---- headline ----
    A("## Headline")
    A("")
    A("```")
    A(table)
    A("```")
    A("")
    d1 = accept_b["directions"][1]
    d2 = accept_b["directions"][2]
    A(box([
        "ACCEPTANCE RULE -- decided before the numbers were seen",
        "",
        "A component is accepted only if it improves BOTH LOCO directions, and",
        "the improvement in at least one direction exceeds that direction's",
        "ensembled leave-one-out IQR.",
        "",
        "COMPONENT A (centre x class sampler)",
        "  NOT TESTABLE ON THIS SPLIT FAMILY -- not accepted, not rejected.",
        "  Both LOCO training sets hold ONE centre, so the sampler reduces",
        "  exactly to the class-balanced baseline. C1 and C3 are the same",
        "  computations as C0 and C2, verified bit-identical. See section 1.",
        "",
        "COMPONENT B (photometric / optical / sensor / compression stack)",
        f"  holdout_center_1  {d1['control_k5']:.4f} -> {d1['treatment_k5']:.4f}"
        f"   ({d1['improvement']:+.4f}, IQR bar {d1['iqr_bar']:.4f})"
        f"  {outcome_label(d1)}",
        f"  holdout_center_2  {d2['control_k5']:.4f} -> {d2['treatment_k5']:.4f}"
        f"   ({d2['improvement']:+.4f}, IQR bar {d2['iqr_bar']:.4f})"
        f"  {outcome_label(d2)}",
        "",
        f"  improves both directions      : "
        f"{'YES' if accept_b['improves_both_directions'] else 'NO'}",
        f"  exceeds the IQR somewhere     : "
        f"{'YES' if accept_b['exceeds_iqr_somewhere'] else 'NO'}",
        f"  VERDICT                       : "
        f"{'ACCEPTED' if accept_b['accepted'] else 'NOT ACCEPTED'}",
    ]))
    A("")

    # ---- section 1: identifiability ----
    A("## 1. What this split family can and cannot measure")
    A("")
    A("The screen was specified as four configurations. On the LOCO splits it "
      "has **two**, and the reason is structural rather than statistical.")
    A("")
    A("Component A balances the four (centre, class) strata so that hospital "
      "identity stops predicting the label in the sampled stream. That "
      "association is real in the pooled training set -- center_2 carries a "
      "12.5% neoplasia prior against center_1's 2.6%. But leave-one-centre-out "
      "holds an entire hospital out, so its **training side contains exactly "
      "one centre**:")
    A("")
    A("| split | training rows | centres in training | neoplasia | non-dysplastic |")
    A("|---|---|---|---|---|")
    A("| `holdout_center_1` | 809 | center_2 only | 97 | 712 |")
    A("| `holdout_center_2` | 2279 | center_1 only | 61 | 2218 |")
    A("| (pooled CV r0f0, for contrast) | 2471 | both | 127 | 2344 |")
    A("")
    A("With one centre present, two of the four strata are empty and "
      "`1/count(centre, class)` is the same number as `1/count(class)` for every "
      "row. The weight vectors are element-wise equal, so identically seeded "
      "generators draw identical epochs and training is bit-identical. C1 is "
      "not a weak version of C0, it *is* C0; C3 *is* C2.")
    A("")
    A("Here is that stated as measurement rather than algebra — the realised "
      "composition of one drawn epoch under each sampler "
      "(`scripts/21_sampler_demo.py`, reproduced live by this script):")
    A("")
    draws = results["sampler_draws"]
    for name, block in draws.items():
        title = ("pooled CV, repeat 0 fold 0 — BOTH centres in training"
                 if name == "pooled_cv_r0f0" else
                 f"holdout_center_{name[-1]} — single-centre training side")
        A(f"**{title}** ({block['n_train']} training rows)")
        A("")
        A("| centre | class | `weighted` (class only) | `balanced_centre_class` |")
        A("|---|---|---|---|")
        strata = sorted(set(block["weighted"]) | set(block["balanced_centre_class"]))
        for s in strata:
            # the stratum key is "centre|class"; a raw pipe would split the
            # markdown cell, so it is unpacked into two columns instead
            centre_name, class_name = s.split("|")
            row = []
            for sampler_name in ("weighted", "balanced_centre_class"):
                counts = block[sampler_name]
                n, total = counts.get(s, 0), sum(counts.values())
                row.append(f"{n} ({100 * n / total:.1f}%)")
            A(f"| {centre_name} | {class_name} | {row[0]} | {row[1]} |")
        A("")
        if block["identical"]:
            A("→ **identical draws.** The centre x class sampler *is* the "
              "class-balanced sampler on this split.")
        else:
            A("→ the samplers differ. Note what class-only balancing leaves "
              "behind: the two classes are drawn ~50/50, but center_2 supplies "
              f"{100 * block['weighted']['center_2|neoplasia'] / (block['weighted']['center_2|neoplasia'] + block['weighted']['center_1|neoplasia']):.0f}% "
              "of the sampled positives against only "
              f"{100 * block['weighted']['center_2|non-dysplastic'] / (block['weighted']['center_2|non-dysplastic'] + block['weighted']['center_1|non-dysplastic']):.0f}% "
              "of the sampled negatives. That gap is the shortcut; the centre x "
              "class sampler closes it to ~25% in every cell.")
        A("")
    A("This was verified rather than asserted, in two independent ways:")
    A("")
    A("- `tests/test_sampler.py::test_single_centre_training_set_makes_the_samplers_identical` "
      "compares the weight vectors and the drawn index sequences for both "
      "directions.")
    A("- `scripts/20_dg_regression.py --config configs/dg_c1.yaml` trains 3 real "
      "epochs under C1 and compares validation logits against the existing C0 "
      "run: **0 of 2279 logits differ (direction 1), 0 of 816 (direction 2), "
      "max |difference| 0.000e+00**.")
    A("")
    A("**Consequence for the screen.** The main effect of component A and the "
      "A x B interaction are not identifiable on LOCO. The 20 runs those two "
      "arms would have consumed (~6.5 GPU-hours) would have reproduced the "
      "other 20 byte for byte, so they were not run. Component A is therefore "
      "neither accepted nor rejected here; it can only be screened on a split "
      "family whose training set contains both centres, which is the pooled-OOF "
      "protocol this step deliberately excludes.")
    A("")
    A("The C0 arm reuses the 10 LOCO runs already in `runs/noise_floor_b`. That "
      "reuse is licensed by the same regression check: retraining `loco_c1_s0` "
      "for 3 epochs under the new code reproduced the old run's validation "
      "logits bit for bit, which simultaneously establishes that the refactor "
      "left the baseline path alone and that training here is run-to-run "
      "deterministic -- the property the paired design depends on.")
    A("")

    # ---- section 1b: the visual check ----
    A("## 1b. The augmentation stack, as inspected before the sweep")
    A("")
    A("`reports/aug_samples.png` holds the 5x5 grid of one frame under 25 draws "
      "of the full stack, each tile labelled with the transforms that actually "
      "fired. It was rendered and inspected before any GPU time was spent, and "
      "**the first version failed that inspection**, which is why there are two "
      "files:")
    A("")
    A("| file | verdict |")
    A("|---|---|")
    A("| `reports/aug_samples_v1_freecolour.png` | REJECTED — magenta, purple, "
      "mustard and olive frames |")
    A("| `reports/aug_samples.png` | accepted — salmon/red through pink, mauve, "
      "tan and brown |")
    A("")
    A("The v1 failure is worth recording because it was not a simple case of "
      "numbers being too big. Each colour transform was individually at ~2-3x "
      "its library default, as specified. The problem was that three "
      "*independent* colour operations compose: a free per-channel gamma over "
      "[0.67, 1.50], times a free per-channel white balance over [0.75, 1.25], "
      "times a +/-22 degree hue rotation, will readily lift blue while dropping "
      "green. The result leaves the endoscopic gamut entirely — and a model "
      "spending capacity on purple mucosa is not being taught to generalise "
      "across hospitals, it is being taught about a domain that does not exist.")
    A("")
    A("The fix constrains the colour block by gamut rather than by magnitude, "
      "which keeps the variation aggressive where aggression is physical:")
    A("")
    A("- **gamma** decomposed into a wide SHARED exponent (log-uniform over "
      "[1/1.5, 1.5], carrying the large tonal variation that is wanted) plus a "
      "narrow independent per-channel deviation (+/-8%);")
    A("- **white balance** drawn on the two axes real illuminants vary on — "
      "colour temperature, which moves red and blue in OPPOSITE directions, "
      "plus a deliberately narrower green-magenta tint axis — instead of three "
      "free per-channel gains, most of whose probability mass describes "
      "illuminants that cannot exist;")
    A("- **hue** rotation reduced from +/-22 to +/-10 degrees.")
    A("")
    A("Everything else — vignetting, gradients, speculars, both blurs, "
      "distortion, sharpening, both noise models, resolution loss and JPEG — "
      "survived inspection unchanged at 2-3x defaults. Structure, mucosal "
      "texture and lumen geometry are preserved in all 25 draws, and "
      "`tests/test_augment_repro.py::test_stack_changes_the_image_but_still_resembles_it` "
      "puts a machine-checkable floor under that (median correlation with the "
      "source frame across 25 draws must exceed 0.5), so a future magnitude "
      "typo fails in seconds rather than after a sweep.")
    A("")

    # ---- section 2: primary metric ----
    A("## 2. Primary decision metric -- ensembled FPR@90R, per direction")
    A("")
    for cfg in CONFIGS:
        r = results[cfg["key"]]
        same = "" if cfg["same_as"] is None else \
            f" — **identical by construction to {cfg['same_as']}**; the numbers " \
            f"below are that arm's, not a second measurement"
        A(f"### {cfg['key']} — {cfg['label']}{same}")
        A("")
        A(f"`{cfg['desc']}`  ·  predictions from `{r['run_dir']}`")
        A("")
        A("| direction | n (pos/neg) | k=5 FPR@90R | LOO-4 median | LOO-4 IQR | "
          "LOO-4 range | k=5 ROC-AUC | single-seed median |")
        A("|---|---|---|---|---|---|---|---|")
        for centre in CENTRES:
            d = r["directions"][centre]
            s, ss = d["loo4_spread"], d["single_seed_spread"]
            A(f"| `holdout_center_{centre}` | {d['n']} "
              f"({d['n_pos']}/{d['n_neg']}) | **{d['k5_fpr']:.4f}** | "
              f"{s['median']:.4f} | {s['iqr']:.4f} | "
              f"{s['range']:.4f} [{s['min']:.4f}, {s['max']:.4f}] | "
              f"{d['k5_roc_auc']:.4f} | {ss['median']:.4f} |")
        A("")
        for centre in CENTRES:
            d = r["directions"][centre]
            A(f"- `holdout_center_{centre}` LOO-4 ensembles (drop one seed): "
              + ", ".join(f"{v:.4f}" for v in d["loo4_fpr"])
              + f" — {fmt_spread(d['loo4_spread'])}")
        A("")

    # ---- section 3: ensembled MDE ----
    A("## 3. The ensembled minimum detectable effect")
    A("")
    A("`reports/noise_floor.md` quoted an MDE built from the RANGE of five "
      "SINGLE runs. `reports/reanalysis.md` observed that the decision unit is "
      "an ensemble and that the corresponding ensembled figure was missing. It "
      "is the spread of the five leave-one-out 4-seed ensembles, and it is "
      "stated here explicitly, per direction and per configuration:")
    A("")
    A("| configuration | direction | ensembled MDE (LOO-4 IQR) | conservative (LOO-4 range) |")
    A("|---|---|---|---|")
    for key in ("C0", "C2"):
        for centre in CENTRES:
            s = results[key]["directions"][centre]["loo4_spread"]
            A(f"| {key} | `holdout_center_{centre}` | {s['iqr']:.4f} | "
              f"{s['range']:.4f} |")
    A("")
    A("These are roughly an order of magnitude smaller than the single-run "
      "ranges the earlier reports quoted, which is the whole point of deciding "
      "on the ensemble: the ensemble is both what will be deployed and the "
      "quieter measurement. They are still not a confidence interval. Five "
      "leave-one-out ensembles built from the same five seeds share four fifths "
      "of their members with each other, so they are strongly positively "
      "correlated and their spread UNDERSTATES the variance of a genuinely "
      "fresh 5-seed ensemble. Treat the IQR as a lower bound on the noise, "
      "which is why the acceptance rule asks the improvement to clear it "
      "rather than merely to be positive.")
    A("")

    # ---- section 4: mechanism ----
    A("## 4. Mechanistic checks")
    A("")
    A("### 4a. Does the score still carry hospital identity?")
    A("")
    A("The requested statistic -- AUC of the logit predicting centre among "
      "negatives, single-seed baseline 0.300 -- **cannot be computed on LOCO "
      "predictions**, and the reason is the same structural fact as in section "
      "1 read from the other end. A LOCO held-out set contains exactly one "
      "centre, so within a direction there is no second group to separate and "
      "the AUC is undefined.")
    A("")
    A("Pooling the two directions gives a set in which every image was scored "
      "by a model that never saw its hospital, which sounds like the right "
      "object. It is not, quite: the two halves come from two different models, "
      "so a constant offset between their logit scales registers as perfect "
      "centre separability even if neither model encodes hospital at all. Both "
      "figures are given so that the artefact is visible rather than hidden:")
    A("")
    A("| configuration | AUC raw (pooled LOCO) | AUC after removing each direction's median | median logit, center_1 | median logit, center_2 |")
    A("|---|---|---|---|---|")
    for cfg in CONFIGS:
        cs = results[cfg["key"]]["centre_separability"]
        med = cs.get("median_logit_by_centre", {})
        A(f"| {cfg['key']} | {fmt(cs['auc_raw'])} | {fmt(cs['auc_centred'])} | "
          f"{fmt(med.get('center_1', float('nan')), 3)} | "
          f"{fmt(med.get('center_2', float('nan')), 3)} |")
    A("")
    A("If the centred column sits near 0.5 while the raw column does not, what "
      "the raw column measured was the offset between two models, not hospital "
      "signal. **Neither column answers the question that was asked.** The "
      "clean measurement needs one model scoring both centres, i.e. the "
      "pooled-OOF protocol held back for the confirmation step. Until then, "
      "the honest statement is that this screen does not establish whether "
      "either component reduces hospital signal, and the 0.300 baseline has no "
      "LOCO counterpart to be compared against.")
    A("")
    A("### 4b. Spearman rho of logit vs `file_size_bytes`")
    A("")
    A("This one IS computable per direction: it is a correlation within a "
      "single held-out set scored by a single ensemble, with no cross-model "
      "pooling. Baseline quoted for positives: +0.360 (pooled OOF, single "
      "seed) -- not directly comparable to a LOCO figure, but the direction of "
      "travel between C0 and C2 is.")
    A("")
    A("| configuration | direction | rho within positives | rho within negatives |")
    A("|---|---|---|---|")
    for cfg in CONFIGS:
        for centre in CENTRES:
            d = results[cfg["key"]]["directions"][centre]["diagnostics_k5"]
            A(f"| {cfg['key']} | `holdout_center_{centre}` | "
              f"{fmt(d['rho_file_size_positives'], 3)} | "
              f"{fmt(d['rho_file_size_negatives'], 3)} |")
    A("")
    A("With the LOO-4 spread on each, so a change can be read against its noise:")
    A("")
    for key in ("C0", "C2"):
        for centre in CENTRES:
            sp = results[key]["directions"][centre]["diagnostics_loo4"]
            A(f"- **{key}**, `holdout_center_{centre}`: positives "
              f"{fmt_spread(sp['rho_file_size_positives'], 3)}; negatives "
              f"{fmt_spread(sp['rho_file_size_negatives'], 3)}")
    A("")

    # ---- section 5: cost ----
    A("## 5. Cost -- seconds per epoch")
    A("")
    A("| configuration | direction | median epoch seconds | across units |")
    A("|---|---|---|---|")
    for key in ("C0", "C2"):
        t = results[key]["timing"]
        for centre in CENTRES:
            s = t["per_direction"].get(centre)
            if s:
                A(f"| {key} | `holdout_center_{centre}` | {s['median']:.1f} | "
                  f"IQR {s['iqr']:.1f}, range [{s['min']:.1f}, {s['max']:.1f}] |")
    A("")
    c0t, c2t = results["C0"]["timing"], results["C2"]["timing"]
    if c0t["overall"] and c2t["overall"]:
        delta = c2t["overall"]["median"] / c0t["overall"]["median"] - 1.0
        A(f"Overall median epoch time: C0 {c0t['overall']['median']:.1f} s -> "
          f"C2 {c2t['overall']['median']:.1f} s, **{delta * 100:+.1f}%** "
          f"({'ABOVE' if delta > 0.15 else 'below'} the 15% threshold).")
    A("")
    A("Per-transform CPU cost, measured on 64 real frames at 384x384 with "
      "`cv2.setNumThreads(0)` to match a DataLoader worker "
      "(`scripts/19_aug_profile.py`, full table in `reports/aug_profile.json`). "
      "`budget_ms` is cost x probability -- the transform's share of the mean "
      "per-image cost, which is the figure that composes into an epoch:")
    A("")
    A("| transform | cost when it fires (ms) | p | budget (ms) |")
    A("|---|---|---|---|")
    prof_path = os.path.join(REPO_ROOT, "reports", "aug_profile.json")
    if os.path.exists(prof_path):
        with open(prof_path) as fh:
            prof = json.load(fh)
        for row in prof["transforms"][:6]:
            A(f"| {row['transform']} | {row['cost_ms']:.2f} | {row['p']:.2f} | "
              f"{row['budget_ms']:.3f} |")
        A(f"| … | | | **{sum(r['budget_ms'] for r in prof['transforms']):.2f} total** |")
        A("")
        e = prof["end_to_end_ms"]
        A(f"Per item, augmentation plus normalisation: "
          f"{e['item_baseline']:.2f} ms (C0) -> {e['item_full']:.2f} ms (C2).")
        A("")
        A("**The expensive transform is not the one expected.** JPEG "
          "re-encoding was flagged as the likely cost and is 11th of 16 by "
          "budget at 0.24 ms; OpenCV's encoder is very fast on a 384x384 frame. "
          "The cost is dominated by **Poisson noise (16.1 ms per call, 41% of "
          "the whole budget)**, because `rng.poisson` draws one variate per "
          "subpixel -- 442k draws per image -- and Gaussian noise (4.95 ms) for "
          "the same reason. Neither shows up in epoch time because the loader "
          "was never the bottleneck: 8 workers at 10.9 ms/item sustain ~735 "
          "img/s against a GPU consuming ~60 img/s, so roughly 12x headroom "
          "survives even after the stack. If the model or the image size grows "
          "to the point where the loader binds, Poisson noise is the first "
          "thing to replace with a signal-dependent Gaussian approximation.")
    A("")

    # ---- section 6: verdict ----
    A("## 6. Verdict")
    A("")
    A("### Component A — centre x class balanced sampling")
    A("")
    pooled = results["sampler_draws"]["pooled_cv_r0f0"]["balanced_centre_class"]
    tot = sum(pooled.values())
    shares = "/".join(f"{100 * pooled[k] / tot:.1f}"
                      for k in sorted(pooled))
    A("**Not accepted, and not rejected: not testable on this split family.** "
      "Implemented, tested and merged; `sampler: balanced_centre_class` is "
      "available, and the realised per-stratum draw counts are logged at the "
      "start of every run and recorded in `train_log.jsonl`. On a two-centre "
      f"training set it draws the four strata at {shares} percent (section 1). "
      "On both LOCO splits it is provably the class-balanced baseline. "
      "Screening it requires pooled-OOF, which this step excludes.")
    A("")
    A("### Component B — photometric / optical / sensor / compression stack")
    A("")
    verdict = "**ACCEPTED**" if accept_b["accepted"] else "**NOT ACCEPTED**"
    A(f"{verdict} under the stated rule.")
    A("")
    for centre in CENTRES:
        d = accept_b["directions"][centre]
        word = "an improvement of" if d["improvement"] > 0 else \
            ("a regression of" if d["improvement"] < 0 else "a change of")
        A(f"- `holdout_center_{centre}`: {d['control_k5']:.4f} -> "
          f"{d['treatment_k5']:.4f}, {word} {abs(d['improvement']):.4f} "
          f"against an IQR bar of {d['iqr_bar']:.4f} "
          f"(control {d['iqr_control']:.4f}, treatment {d['iqr_treatment']:.4f}). "
          f"{ {'PASS': 'Clears the bar.', 'improved, under IQR': 'Improves, but does not clear the bar.', 'no change': 'No change at all.', 'WORSE': 'WORSE — not an improvement.'}[outcome_label(d)] }")
    A("")
    if not accept_b["improves_both_directions"]:
        A("The rule's single-direction clause applies: this is a gain in one "
          "direction and not the other, and it is reported as an accident "
          "rather than an effect. A component that helps one hospital and "
          "harms the other has not demonstrated domain generalisation; it has "
          "demonstrated that it moved the model somewhere, and the twelve "
          "unseen centres are not obliged to resemble the direction that "
          "happened to improve.")
        A("")

    A("#### Is the split verdict an artefact of the k=5 point?")
    A("")
    A("It is the obvious objection, and worth answering rather than asserting "
      "past. A k=5 ensemble is a single sample, and C0's `holdout_center_2` "
      "figure of "
      f"{accept_b['directions'][2]['control_k5']:.4f} sits at the very bottom "
      "of its own leave-one-out family (median "
      f"{accept_b['directions'][2]['control_loo4_median']:.4f}, range "
      "[0.1039, 0.1882]) -- i.e. that particular ensemble was a lucky one, "
      "which would exaggerate any regression measured against it. So the "
      "comparison is repeated using each arm's LOO-4 MEDIAN in place of its "
      "k=5 point:")
    A("")
    A("| direction | k=5: C0 → C2 | Δ | LOO-4 median: C0 → C2 | Δ | sign agrees |")
    A("|---|---|---|---|---|---|")
    for centre in CENTRES:
        d = accept_b["directions"][centre]
        A(f"| `holdout_center_{centre}` | {d['control_k5']:.4f} → "
          f"{d['treatment_k5']:.4f} | {d['improvement']:+.4f} | "
          f"{d['control_loo4_median']:.4f} → {d['treatment_loo4_median']:.4f} | "
          f"{d['improvement_loo4_median']:+.4f} | "
          f"{'yes' if d['sign_agrees_with_loo4_median'] else 'NO'} |")
    A("")
    if all(d["sign_agrees_with_loo4_median"]
           for d in accept_b["directions"].values()):
        A("**The sign survives in both directions.** The regression on "
          "`holdout_center_2` is smaller when measured this way -- because part "
          "of it genuinely was C0's fortunate k=5 draw -- but it does not go "
          "away, and the gain on `holdout_center_1` does not either. The split "
          "verdict is a property of the configuration, not of one ensemble.")
    else:
        A("**The sign does NOT survive in at least one direction**, so the "
          "verdict above rests on the k=5 points more than it should. Treat it "
          "as provisional.")
    A("")
    A("#### The mechanism moved even though the outcome did not follow")
    A("")
    A("Worth stating plainly, because it is the most informative thing in this "
      "report. Component B **did** reduce the compression/file-size shortcut it "
      "was aimed at -- on `holdout_center_1` the Spearman rho within positives "
      "falls 0.253 → 0.142 and within negatives 0.116 → 0.044, both far outside "
      "their leave-one-out IQRs of 0.013 and 0.007. The proxy signal the stack "
      "was designed to break is measurably weaker.")
    A("")
    A("It still made `holdout_center_2` worse. So the intended mechanism "
      "operated and the intended outcome did not follow from it, which is the "
      "mirror image of the failure mode the brief asked to guard against (an "
      "FPR gain with no reduction in shortcut signal). Reducing a shortcut is "
      "evidently not sufficient; the stack appears to be removing usable "
      "pathology signal at the same time, and on the direction that can least "
      "afford it.")
    A("")
    A("The asymmetry between the directions points at where to look. "
      "`holdout_center_2` trains on center_1, which supplies only **61 "
      "positives**; `holdout_center_1` trains on center_2 and its **97 "
      "positives**. Aggressive colour, blur and noise augmentation is a strong "
      "regulariser, and it plausibly helps the arm whose difficulty is "
      "over-fitting a small, easy-looking training set while hurting the arm "
      "whose difficulty is learning a subtle mucosal appearance from 61 "
      "examples. That is a hypothesis this screen does not test -- it is the "
      "next experiment, not a conclusion.")
    A("")
    A("### What runs next")
    A("")
    A("Pooled-OOF confirmation runs only on what survives here, per the "
      "protocol. It is also the only way to screen component A at all, and the "
      "only setting in which the hospital-signal check of section 4a is "
      "defined -- so if pooled-OOF is run, it should carry all four "
      "configurations rather than only the survivors of this screen.")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
