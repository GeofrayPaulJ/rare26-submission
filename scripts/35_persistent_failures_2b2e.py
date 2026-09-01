"""JOB 2b-2e -- dossier, rescue matrix, counterfactual, contact sheet for the
persistent-failure positive set. NO TRAINING, NO GPU.

The persistent-failure set (JOB 2a, reports/persistent_failures_2a.json,
PRIMARY reading, A0 alone, 3 repeats = 15 groups) is a fixed set of
positives that stayed in the bottom-16-of-158 threshold-setting group in
every repeat x seed combination tested. This script looks up that set
(by relative image path, supplied via --targets or the default manifest
file) and computes, per arm, each image's rank, whether any arm rescues
it out of the bottom 16, and the counterfactual FPR@90R effect of
excluding it.

The target list is derived from the challenge dataset and is not
distributed with this release.

This script:
  2b -- one dossier row per image: manifest metadata + per-arm (k=5
        seed-ensembled, repeat 0) rank and percentile among that arm's 158
        pooled-OOF positives, for A0, A1, A2, A4, G1, G2, G3.
  2c -- rescue matrix: which arms move each image out of the bottom-16
        (the 90%-recall threshold-setting group).
  2d -- counterfactual: A4's and A0's pooled-OOF FPR@90R (k=5 ensemble,
        repeat 0) recomputed with each image excluded singly and all three
        excluded together.
  2e -- contact sheet: the 3 images at full FOV from 00_source/, annotated,
        with the inscribed-crop box overlaid from the manifest.

Reuses scripts/17_reanalysis.py's build_wide_fold / ensemble_pool_oof (the
k=5 seed-ensemble primitive, FUSION="mean_logit") rather than reinventing
ensembling -- the same pattern scripts/30_gastronet.py's k5_pool uses.

    python scripts/35_persistent_failures_2b2e.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.evaluate import metric_block  # noqa: E402
from src.metrics import TARGET_RECALL, fpr_at_recall  # noqa: E402


def _load(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RA = _load("_reanalysis", "17_reanalysis.py")  # build_wide_fold, ensemble_pool_oof

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
REPEAT = 0
FUSION = "mean_logit"
N_POSITIVES = 158
BOTTOM_N = 16

DEFAULT_TARGETS_PATH = os.path.join(REPO_ROOT, "manifests", "persistent_failure_targets.txt")


def load_targets(path: str) -> List[str]:
    """One relative image path per line. The target list is derived from
    the challenge dataset and is not distributed with this release --
    supply your own copy of manifests/persistent_failure_targets.txt (or
    pass --targets) to reproduce JOB 2b-2e locally."""
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]

ARMS = {
    "A0": "runs/noise_floor_a",
    "A1": "runs/sweep_a1",
    "A2": "runs/sweep_a2",
    "A4": "runs/sweep_a4",
    "G1": "runs/g1_rn50_swsl",
    "G2": "runs/g2_vitb_dinov2",
    "G3": "runs/g3_rn50_gastronet",
}

# Known baselines to verify against (from the brief).
KNOWN_BASELINE_FPR90 = {"A4": 0.0529, "A0": 0.0427}

MANIFEST_PATH = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
POSITIVE_REVIEW_PATH = os.path.join(REPO_ROOT, "manifests", "positive_review.csv")
SOURCE_ROOT = os.path.join(REPO_ROOT, "00_source")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")


def k5_pool(out_dir_rel: str, manifest: pd.DataFrame) -> pd.DataFrame:
    """k=5 seed-ensembled pooled-OOF predictions for one arm, repeat 0.

    Same primitive as scripts/30_gastronet.py's k5_pool: RA.build_wide_fold
    per fold, then RA.ensemble_pool_oof with mean_logit fusion across the
    5 seeds.
    """
    out_dir = os.path.join(REPO_ROOT, out_dir_rel)
    wide = {f: RA.build_wide_fold(out_dir, f, manifest, SEEDS, repeat=REPEAT)
            for f in FOLDS}
    return RA.ensemble_pool_oof(wide, list(SEEDS), FUSION, manifest, repeat=REPEAT)


def positive_ranks(pool: pd.DataFrame) -> pd.DataFrame:
    """Rank positives ascending by (ensembled) logit: rank_asc=1 is the
    single worst-detected positive, rank_asc=158 the best. percentile_from
    _bottom = rank_asc / 158 * 100, so <= 16/158*100 = 10.13% is exactly the
    bottom-16 threshold-setting group at 90% recall (0.9*158=142.2 -> 16
    given up)."""
    pos = pool[pool["label_int"] == 1].copy()
    if len(pos) != N_POSITIVES:
        raise AssertionError(f"expected {N_POSITIVES} positives, got {len(pos)}")
    pos = pos.sort_values("logit", ascending=True, kind="mergesort").reset_index(drop=True)
    pos["rank_asc"] = np.arange(1, len(pos) + 1)
    pos["percentile_from_bottom"] = pos["rank_asc"] / N_POSITIVES * 100.0
    pos["in_bottom16"] = pos["rank_asc"] <= BOTTOM_N
    return pos.set_index("filepath")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=DEFAULT_TARGETS_PATH,
                        help="path to a file of relative image paths, one per line "
                             "(default: manifests/persistent_failure_targets.txt)")
    args = parser.parse_args()
    TARGETS = load_targets(args.targets)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    pos_review = pd.read_csv(POSITIVE_REVIEW_PATH)

    # ------------------------------------------------------------------
    # Per-arm k=5 pools and positive ranks
    # ------------------------------------------------------------------
    pools: Dict[str, pd.DataFrame] = {}
    pos_ranks: Dict[str, pd.DataFrame] = {}
    for arm, out_dir in ARMS.items():
        print(f"[{arm}] pooling k=5 ensemble from {out_dir} ...", flush=True)
        pool = k5_pool(out_dir, manifest)
        pools[arm] = pool
        pos_ranks[arm] = positive_ranks(pool)
        print(f"[{arm}] n={len(pool)} n_pos={int((pool['label_int'] == 1).sum())}",
              flush=True)

    # ------------------------------------------------------------------
    # 2b -- dossier
    # ------------------------------------------------------------------
    man_idx = manifest.set_index("filepath")
    rev_idx = pos_review.set_index("filepath")["visibility"]

    dossier_rows: List[Dict[str, Any]] = []
    for fp in TARGETS:
        m = man_idx.loc[fp]
        vis_manifest = m.get("visibility")
        vis_review = rev_idx.get(fp)
        row: Dict[str, Any] = {
            "filepath": fp,
            "centre": m["centre"],
            "group_id_v2": m["group_id_v2"],
            "visibility_manifest": vis_manifest,
            "visibility_positive_review": vis_review,
            "visibility_agree": (str(vis_manifest) == str(vis_review)),
            "width": int(m["width"]), "height": int(m["height"]),
            "file_size_bytes": int(m["file_size_bytes"]),
            "rg_ratio": float(m["rg_ratio"]),
            "has_redaction": bool(m["has_redaction"]),
            "fov_radius": float(m["fov_radius"]),
            "inner_left": float(m["inner_left"]), "inner_top": float(m["inner_top"]),
            "inner_right": float(m["inner_right"]), "inner_bottom": float(m["inner_bottom"]),
        }
        for arm in ARMS:
            pr = pos_ranks[arm]
            if fp not in pr.index:
                row[f"{arm}_rank_asc"] = None
                row[f"{arm}_percentile_from_bottom"] = None
                row[f"{arm}_in_bottom16"] = None
                row[f"{arm}_logit"] = None
                continue
            r = pr.loc[fp]
            row[f"{arm}_rank_asc"] = int(r["rank_asc"])
            row[f"{arm}_percentile_from_bottom"] = float(r["percentile_from_bottom"])
            row[f"{arm}_in_bottom16"] = bool(r["in_bottom16"])
            row[f"{arm}_logit"] = float(r["logit"])
        dossier_rows.append(row)

    dossier_df = pd.DataFrame(dossier_rows)

    group_ids = dossier_df["group_id_v2"].tolist()
    n_unique_groups = dossier_df["group_id_v2"].nunique()
    groups_collapse = n_unique_groups < len(TARGETS)

    # ------------------------------------------------------------------
    # 2c -- rescue matrix
    # ------------------------------------------------------------------
    rescue_rows = []
    for fp in TARGETS:
        row = {"filepath": fp}
        for arm in ARMS:
            pr = pos_ranks[arm]
            r = pr.loc[fp]
            rescued = not bool(r["in_bottom16"])
            row[arm] = {
                "rescued": rescued,
                "rank_asc": int(r["rank_asc"]),
                "percentile_from_bottom": round(float(r["percentile_from_bottom"]), 2),
            }
        rescue_rows.append(row)

    rescue_counts = {
        arm: sum(1 for row in rescue_rows if row[arm]["rescued"])
        for arm in ARMS
    }

    # ------------------------------------------------------------------
    # 2d -- counterfactual FPR@90R with images excluded
    # ------------------------------------------------------------------
    def fpr90(pool: pd.DataFrame, exclude: Sequence[str] = ()) -> float:
        sub = pool[~pool["filepath"].isin(exclude)] if exclude else pool
        y = sub["label_int"].to_numpy()
        s = sub["logit"].to_numpy()
        return fpr_at_recall(y, s, TARGET_RECALL)

    counterfactual: Dict[str, Any] = {}
    for arm in ("A4", "A0"):
        pool = pools[arm]
        baseline = fpr90(pool)
        entry: Dict[str, Any] = {
            "baseline_fpr90_recomputed": baseline,
            "baseline_fpr90_known": KNOWN_BASELINE_FPR90[arm],
            "baseline_matches_known": abs(baseline - KNOWN_BASELINE_FPR90[arm]) < 1e-3,
            "single_exclusions": {},
        }
        for fp in TARGETS:
            v = fpr90(pool, exclude=[fp])
            entry["single_exclusions"][fp] = {
                "fpr90": v, "delta_vs_baseline": v - baseline,
            }
        v_all = fpr90(pool, exclude=TARGETS)
        entry["all_three_excluded"] = {
            "fpr90": v_all, "delta_vs_baseline": v_all - baseline,
        }
        counterfactual[arm] = entry

    # ------------------------------------------------------------------
    # write dossier json
    # ------------------------------------------------------------------
    dossier_json = {
        "targets": TARGETS,
        "group_id_v2_by_image": dict(zip(TARGETS, group_ids)),
        "n_unique_group_id_v2": int(n_unique_groups),
        "groups_collapse": bool(groups_collapse),
        "dossier": dossier_rows,
        "rescue_matrix": rescue_rows,
        "rescue_counts_out_of_3": rescue_counts,
        "counterfactual_fpr90": counterfactual,
    }
    dossier_json_path = os.path.join(REPORTS_DIR, "persistent_failures_dossier.json")
    with open(dossier_json_path, "w") as fh:
        json.dump(dossier_json, fh, indent=2, default=lambda o: (
            int(o) if isinstance(o, np.integer) else
            float(o) if isinstance(o, np.floating) else
            bool(o) if isinstance(o, np.bool_) else str(o)))
    print(f"wrote {dossier_json_path}")

    # ------------------------------------------------------------------
    # 2e -- contact sheet
    # ------------------------------------------------------------------
    a4_pct = {fp: dossier_json["dossier"][i]["A4_percentile_from_bottom"]
              for i, fp in enumerate(TARGETS)}

    n = len(TARGETS)
    panel_in = 7.0
    fig, axes = plt.subplots(1, n, figsize=(panel_in * n, panel_in * 512 / 637 + 1.4))
    if n == 1:
        axes = [axes]

    for ax, fp in zip(axes, TARGETS):
        img_path = os.path.join(SOURCE_ROOT, fp.replace("/", os.sep))
        img = Image.open(img_path).convert("RGB")
        ax.imshow(np.asarray(img))

        m = man_idx.loc[fp]
        il, it, ir, ib = m["inner_left"], m["inner_top"], m["inner_right"], m["inner_bottom"]
        rect = mpatches.Rectangle((il, it), ir - il, ib - it,
                                  linewidth=2.5, edgecolor="#00e5ff",
                                  facecolor="none", linestyle="--")
        ax.add_patch(rect)

        vis = m.get("visibility")
        pct = a4_pct[fp]
        title = (f"{os.path.basename(fp)}\n"
                 f"{m['centre']}  |  visibility: {vis}\n"
                 f"A4 percentile-from-bottom: {pct:.1f}%  "
                 f"({'BOTTOM-16' if pct <= BOTTOM_N / N_POSITIVES * 100 else 'above threshold'})")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Persistent-failure positives (A0, 3 repeats, bottom-16-of-158 in every "
        "repeat x seed group) -- full FOV, dashed box = inscribed training crop",
        fontsize=12, y=1.02)
    fig.tight_layout()
    png_path = os.path.join(REPORTS_DIR, "persistent_failures.png")
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    png_size = os.path.getsize(png_path)
    print(f"wrote {png_path} ({png_size} bytes)")

    # ------------------------------------------------------------------
    # write markdown report
    # ------------------------------------------------------------------
    lines: List[str] = []
    W = lines.append
    W("# JOB 2b-2e -- persistent-failure dossier, rescue matrix, counterfactual, contact sheet\n")
    W("Generated by `scripts/35_persistent_failures_2b2e.py`. No training, no GPU: "
      "recomputed from prediction parquets on disk and manifests already on disk. "
      "Ensembling uses the project's k=5 seed convention (mean_logit fusion), reusing "
      "`scripts/17_reanalysis.py`'s `build_wide_fold`/`ensemble_pool_oof` (the same "
      "primitive `scripts/30_gastronet.py`'s `k5_pool` wraps).\n")
    W(f"Set under study (JOB 2a PRIMARY reading, A0 alone, 3 repeats = 15 groups): "
      f"{len(TARGETS)} images:\n")
    for fp in TARGETS:
        W(f"- `{fp}`")
    W("")

    W("## group_id_v2 check\n")
    if groups_collapse:
        W(f"**FLAG: the {len(TARGETS)} images share group_id_v2 values, collapsing to "
          f"{n_unique_groups} independent group(s).** The persistent-failure count of "
          f"{len(TARGETS)} images does NOT mean {len(TARGETS)} independent "
          f"lesions/patients.\n")
    else:
        W(f"All {len(TARGETS)} images have distinct `group_id_v2` values "
          f"({', '.join(str(g) for g in group_ids)}) -- **no collapse**. The set "
          f"genuinely represents {n_unique_groups} independent groups (lesions/patients), "
          f"not fewer.\n")

    W("## 2b. Dossier\n")
    dossier_cols = [
        "filepath", "centre", "group_id_v2", "visibility_manifest",
        "visibility_positive_review", "visibility_agree", "width", "height",
        "file_size_bytes", "rg_ratio", "has_redaction", "fov_radius",
        "inner_left", "inner_top", "inner_right", "inner_bottom",
    ]
    header = "| " + " | ".join(dossier_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(dossier_cols)) + " |"
    W(header)
    W(sep)
    for row in dossier_rows:
        W("| " + " | ".join(str(row[c]) for c in dossier_cols) + " |")
    W("")

    W("### Per-arm rank / percentile among 158 positives (k=5 seed ensemble, repeat 0)\n")
    W("`rank_asc`: 1 = worst-detected positive (lowest ensembled logit), 158 = best. "
      "`pct_from_bottom` = rank_asc/158*100; the bottom-16 threshold-setting group "
      "(given up at 90% recall) is pct_from_bottom <= 10.13%.\n")
    arm_cols = list(ARMS.keys())
    header = "| filepath | " + " | ".join(f"{a} rank (pct%, bottom16?)" for a in arm_cols) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(arm_cols)) + " |"
    W(header)
    W(sep)
    for row in dossier_rows:
        cells = []
        for a in arm_cols:
            rk = row[f"{a}_rank_asc"]
            pc = row[f"{a}_percentile_from_bottom"]
            b16 = row[f"{a}_in_bottom16"]
            cells.append(f"{rk} ({pc:.1f}%, {'YES' if b16 else 'no'})")
        W(f"| `{row['filepath']}` | " + " | ".join(cells) + " |")
    W("")

    W("## 2c. Rescue matrix\n")
    W("Cell = rescued (rank among that arm's 158 positives is OUTSIDE the worst 16) "
      "or NOT RESCUED (still in the bottom 16). Rank shown is rank_asc "
      "(1=worst, 158=best) and percentile-from-bottom.\n")
    header = "| filepath | " + " | ".join(arm_cols) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(arm_cols)) + " |"
    W(header)
    W(sep)
    for row in rescue_rows:
        cells = []
        for a in arm_cols:
            c = row[a]
            tag = "RESCUED" if c["rescued"] else "not rescued"
            cells.append(f"{tag} (rank {c['rank_asc']}, {c['percentile_from_bottom']:.1f}%)")
        W(f"| `{row['filepath']}` | " + " | ".join(cells) + " |")
    W("")
    W("Rescue counts (out of 3 images) per arm:\n")
    for a in arm_cols:
        W(f"- {a}: {rescue_counts[a]}/3 rescued")
    W("")

    W("## 2d. Counterfactual: A4 / A0 pooled-OOF FPR@90R with images excluded\n")
    for arm in ("A4", "A0"):
        e = counterfactual[arm]
        match_tag = "MATCHES" if e["baseline_matches_known"] else "DOES NOT MATCH"
        W(f"### {arm}\n")
        W(f"- baseline FPR@90R recomputed here: **{e['baseline_fpr90_recomputed']:.4f}** "
          f"(known baseline: {e['baseline_fpr90_known']:.4f}, {match_tag})\n")
        W("| excluded | FPR@90R | delta vs baseline |")
        W("| --- | --- | --- |")
        for fp in TARGETS:
            se = e["single_exclusions"][fp]
            W(f"| `{fp}` | {se['fpr90']:.4f} | {se['delta_vs_baseline']:+.4f} |")
        at = e["all_three_excluded"]
        W(f"| **all 3 excluded** | **{at['fpr90']:.4f}** | **{at['delta_vs_baseline']:+.4f}** |")
        W("")

    W("## 2e. Contact sheet\n")
    W(f"See `reports/persistent_failures.png` ({png_size:,} bytes) -- the 3 images at "
      f"full FOV loaded directly from `00_source/`, each annotated with filename, "
      f"centre, visibility rating, and A4 percentile-from-bottom, with the inscribed-"
      f"square training-crop boundary (from `inner_left/top/right/bottom`) overlaid "
      f"as a dashed cyan rectangle.\n")

    text = "\n".join(lines) + "\n"
    md_path = os.path.join(REPORTS_DIR, "persistent_failures.md")
    with open(md_path, "w") as fh:
        fh.write(text)
    print(f"wrote {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
