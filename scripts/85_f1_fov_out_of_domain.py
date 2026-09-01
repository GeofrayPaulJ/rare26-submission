"""F1 -- FOV fallback on out-of-domain endoscopy (EVC, 100 images, TOP PRIORITY).

Every fallback figure on record so far (1.00% at N=200, 1.00% at
N=3000) is RARE25-only: two Dutch centres, one Olympus Exera III
platform. The real test set is twelve BONSAI centres. This runs the
SHIPPING container's FOV fitter (`submission.rare26_infer.fov`,
unmodified import, not reimplemented) over the 100 EVC images -- an
out-of-domain measurement set only (EVC is not RARE25/RARE26 training
data; this does not touch the EVC-paste arm or reopen the component
freeze).

RARE25's fit_quality distribution is NOT recomputed here -- it is
already on disk (`manifests/rare25_folds_v2.csv`'s own `fit_quality`
column), produced by `scripts/05_fov_crop.py`, which `fov.py`'s own
docstring states is verbatim-ported into the shipping container. A spot
check against the shipping code confirms this before trusting the
column wholesale.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/85_f1_fov_out_of_domain.py'
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = "/workspace/RARE26"
sys.path.insert(0, os.path.join(REPO_ROOT, "submission"))
from rare26_infer.fov import FIT_QUALITY_FLOOR, detect_crop_box  # noqa: E402

EVC_MANIFEST = os.path.join(REPO_ROOT, "manifests", "evc_inventory.csv")
EVC_ROOT = os.path.join(REPO_ROOT, "02_evc")
RARE25_MANIFEST = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
REPORT_JSON = os.path.join(REPO_ROOT, "reports", "f1_fov_out_of_domain.json")
PLOT_PATH = os.path.join(REPO_ROOT, "reports", "f1_fit_quality_distributions.png")


def load_rgb(path: str) -> np.ndarray:
    """PIL, matching scripts/05_fov_crop.py's own loader exactly -- the
    pipeline that produced manifests/rare25_folds_v2.csv's fit_quality
    column, so the EVC-vs-RARE25 comparison is apples-to-apples."""
    return np.asarray(Image.open(path).convert("RGB"))


def spot_check_rare25(df: pd.DataFrame, n: int = 15) -> dict:
    """Confirm the shipping fov.py reproduces the manifest's own fit_quality
    column, so trusting that column wholesale (rather than recomputing all
    3,095 rows) is verified, not assumed."""
    rng = np.random.default_rng(0)
    has_fit = df.dropna(subset=["fit_quality"])
    sample = has_fit.sample(n=min(n, len(has_fit)), random_state=0)
    diffs = []
    for _, row in sample.iterrows():
        arr = load_rgb(os.path.join(REPO_ROOT, "00_source", row["filepath"]))
        box, used_fallback, fit_quality, reason = detect_crop_box(arr)
        if fit_quality is not None:
            # manifests/rare25_folds_v2.csv stores fit_quality rounded to 4dp
            # (scripts/05_fov_crop.py: `round(fit_quality, 4)`) -- round the
            # shipping value the same way before comparing, or every check
            # here reads as a false mismatch from rounding alone.
            diffs.append(abs(round(fit_quality, 4) - row["fit_quality"]))
    return {"n_checked": len(sample), "max_abs_diff": float(max(diffs)) if diffs else None,
           "all_match_1e-6": bool(diffs and max(diffs) < 1e-6)}


def run_evc() -> pd.DataFrame:
    man = pd.read_csv(EVC_MANIFEST)
    rows = []
    for _, row in tqdm(man.iterrows(), total=len(man), desc="EVC FOV", file=sys.stderr):
        arr = load_rgb(os.path.join(EVC_ROOT, row["filepath"]))
        box, used_fallback, fit_quality, reason = detect_crop_box(arr)
        rows.append({
            "filepath": row["filepath"], "class_label": row["class_label"],
            "used_fallback": used_fallback,
            "fit_quality": fit_quality if fit_quality is not None else np.nan,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def summarize(name: str, df: pd.DataFrame, fallback_col: str, fq_col: str) -> dict:
    n = len(df)
    n_fb = int(df[fallback_col].sum())
    fq = df[fq_col].dropna()
    return {
        "name": name, "n": n, "n_fallback": n_fb, "fallback_frac": n_fb / n,
        "fit_quality_n": len(fq),
        "fit_quality_mean": float(fq.mean()) if len(fq) else None,
        "fit_quality_median": float(fq.median()) if len(fq) else None,
        "fit_quality_min": float(fq.min()) if len(fq) else None,
        "fit_quality_p1": float(fq.quantile(0.01)) if len(fq) else None,
        "fit_quality_p5": float(fq.quantile(0.05)) if len(fq) else None,
        "floor_percentile_within_this_distribution": (
            float((fq < FIT_QUALITY_FLOOR).mean()) if len(fq) else None),
    }


def main() -> int:
    df25 = pd.read_csv(RARE25_MANIFEST)

    print("[f1] spot-checking shipping fov.py against manifest's own fit_quality column...")
    spot = spot_check_rare25(df25)
    print(f"[f1] spot check: {spot}")
    assert spot["all_match_1e-6"], "shipping fov.py does NOT reproduce the manifest's fit_quality -- do not trust the column"

    print("[f1] running shipping FOV fitter over 100 EVC images...")
    evc = run_evc()

    df25["used_fallback"] = df25["fit_quality"].isna() | (df25["fit_quality"] < FIT_QUALITY_FLOOR)
    # a detector that found no content at all (fit is None) also has NaN
    # fit_quality in the manifest -- both count as fallback, matching
    # detect_crop_box's own two failure branches.

    overall25 = summarize("RARE25 (all)", df25, "used_fallback", "fit_quality")
    c1 = summarize("RARE25 centre_1", df25[df25["centre"] == "center_1"], "used_fallback", "fit_quality")
    c2 = summarize("RARE25 centre_2", df25[df25["centre"] == "center_2"], "used_fallback", "fit_quality")
    evc_summary = summarize("EVC (out-of-domain)", evc, "used_fallback", "fit_quality")

    reason_counts = evc["reason"].value_counts().to_dict()
    print(f"\n[f1] EVC fallback: {evc_summary['n_fallback']}/{evc_summary['n']} "
         f"({evc_summary['fallback_frac']*100:.1f}%)")
    print(f"[f1] EVC reasons: {reason_counts}")
    print(f"[f1] RARE25 overall fallback: {overall25['fallback_frac']*100:.2f}%")
    print(f"[f1] RARE25 centre_1 fallback: {c1['fallback_frac']*100:.2f}%")
    print(f"[f1] RARE25 centre_2 fallback: {c2['fallback_frac']*100:.2f}%")

    # --- plot: same axes, EVC vs RARE25 fit_quality distributions ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(df25["fit_quality"].dropna(), bins=bins, alpha=0.5, density=True,
           label=f"RARE25 (n={overall25['fit_quality_n']})", color="#4C72B0")
    ax.hist(evc["fit_quality"].dropna(), bins=bins, alpha=0.5, density=True,
           label=f"EVC, out-of-domain (n={evc_summary['fit_quality_n']})", color="#C44E52")
    ax.axvline(FIT_QUALITY_FLOOR, color="black", linestyle="--",
              label=f"fallback floor ({FIT_QUALITY_FLOOR})")
    ax.set_xlabel("fit_quality")
    ax.set_ylabel("density")
    ax.set_title("FOV fit_quality: RARE25 (training domain) vs EVC (out-of-domain)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"[f1] plot -> {PLOT_PATH}")

    result = {
        "spot_check": spot,
        "rare25_overall": overall25, "rare25_centre_1": c1, "rare25_centre_2": c2,
        "evc": evc_summary, "evc_reason_counts": reason_counts,
    }
    with open(REPORT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[f1] written: {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
