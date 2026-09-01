"""L5 -- redo the arithmetic against real data.

L4's JOB B is retracted: it applied a class-AGNOSTIC common offset per
synthetic centre, which shifts positives and negatives together and
therefore preserves within-centre ranking -- weak by construction, and
inconsistent with L4's own JOB A finding that the real centre_1/centre_2
offset is almost entirely a POSITIVES phenomenon (-3.62 logits vs -0.08
for negatives). This script redoes it as a class-differential sweep
against the REAL pooled-OOF logits (no parametric negative model), using
the challenge's own resampling protocol (scripts/08_score.py,
`official_score`: 100:1 imbalance, 1000 bootstrap draws, median).

JOB A: shift positives only, down by d in [0, 10], step 0.25.
JOB B: control -- inflate negative spread around the negative mean by
       multiplier m in [1, 10], leave positives untouched.
JOB C: calibrate against the real EVC-minus-RARE25 shift (from
       reports/l4_evc_scores.csv, empirical distributions, not
       parametric summary stats).

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/109_l5_sink_sweep.py'
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from importlib import import_module
score_mod = import_module("08_score")
official_score = score_mod.official_score

FOLD_TEMPLATE = "runs/a4_checkpointed/r0_f{f}_s0/val_r0_f{f}_s0.parquet"
TARGET_AUROC = 0.6466
TARGET_PPV = 0.0113
N_ITER = 1000
SEED = 20260821


def load_pooled():
    frames = []
    for f in range(5):
        df = pd.read_parquet(FOLD_TEMPLATE.format(f=f))
        frames.append(df[["filepath", "label_int", "logit"]])
    pooled = pd.concat(frames, ignore_index=True)
    return pooled


def sweep_positive_shift(pooled: pd.DataFrame, ds: np.ndarray) -> list:
    neg_logit = pooled.loc[pooled["label_int"] == 0, "logit"].to_numpy()
    pos_logit = pooled.loc[pooled["label_int"] == 1, "logit"].to_numpy()
    n_neg, n_pos = len(neg_logit), len(pos_logit)
    y = np.r_[np.zeros(n_neg), np.ones(n_pos)]

    rows = []
    for d in ds:
        shifted_pos = pos_logit - d
        p = np.r_[1 / (1 + np.exp(-neg_logit)), 1 / (1 + np.exp(-shifted_pos))]
        r = official_score(y, p, n_iterations=N_ITER, imbalance_ratio=100, seed=SEED)
        rows.append({"d": round(float(d), 2), "auroc": float(r["AUROC"]), "ppv90": float(r["Score"])})
        print(f"  d={d:5.2f}  AUROC={r['AUROC']:.4f}  PPV@90R={r['Score']:.4f}", flush=True)
    return rows


def sweep_negative_spread(pooled: pd.DataFrame, ms: np.ndarray) -> list:
    neg_logit = pooled.loc[pooled["label_int"] == 0, "logit"].to_numpy()
    pos_logit = pooled.loc[pooled["label_int"] == 1, "logit"].to_numpy()
    neg_mean = neg_logit.mean()
    n_neg, n_pos = len(neg_logit), len(pos_logit)
    y = np.r_[np.zeros(n_neg), np.ones(n_pos)]

    rows = []
    for m in ms:
        shifted_neg = neg_mean + m * (neg_logit - neg_mean)
        p = np.r_[1 / (1 + np.exp(-shifted_neg)), 1 / (1 + np.exp(-pos_logit))]
        r = official_score(y, p, n_iterations=N_ITER, imbalance_ratio=100, seed=SEED)
        rows.append({"m": round(float(m), 2), "auroc": float(r["AUROC"]), "ppv90": float(r["Score"])})
        print(f"  m={m:5.2f}  AUROC={r['AUROC']:.4f}  PPV@90R={r['Score']:.4f}", flush=True)
    return rows


def interp_crossing(rows: list, key: str, target: float, x_key: str) -> dict:
    """Linear-interpolate the x value where `key` crosses `target`, using the
    grid points that bracket it (key assumed monotonic in x over the tested range)."""
    xs = [r[x_key] for r in rows]
    ys = [r[key] for r in rows]
    # find first bracket where ys crosses target (works for monotonic decreasing or increasing)
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - target) == 0:
            return {"x": xs[i], "y_at_x": y0, "bracket": [xs[i], xs[i]]}
        if (y0 - target) * (y1 - target) < 0:
            x0, x1 = xs[i], xs[i + 1]
            frac = (target - y0) / (y1 - y0)
            x_cross = x0 + frac * (x1 - x0)
            return {"x": round(float(x_cross), 4), "bracket": [x0, x1], "y_bracket": [y0, y1]}
    return {"x": None, "note": f"{key} never crosses {target} over tested range "
                                f"[{xs[0]}, {xs[-1]}] (range: [{min(ys):.4f}, {max(ys):.4f}])"}


def other_metric_at_x(rows: list, x_target: float, x_key: str, other_key: str) -> float:
    xs = [r[x_key] for r in rows]
    ys = [r[other_key] for r in rows]
    for i in range(len(xs) - 1):
        if xs[i] <= x_target <= xs[i + 1]:
            frac = (x_target - xs[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] != xs[i] else 0
            return float(ys[i] + frac * (ys[i + 1] - ys[i]))
    return float(np.interp(x_target, xs, ys))


def job_c_evc_calibration(pooled: pd.DataFrame) -> dict:
    evc = pd.read_csv("reports/l4_evc_scores.csv")
    evc_logit = np.log(evc["prob"] / (1 - evc["prob"]))
    evc_pos_logit = evc_logit[evc["y"] == 1].to_numpy()
    evc_neg_logit = evc_logit[evc["y"] == 0].to_numpy()

    r25_pos_logit = pooled.loc[pooled["label_int"] == 1, "logit"].to_numpy()
    r25_neg_logit = pooled.loc[pooled["label_int"] == 0, "logit"].to_numpy()

    return {
        "evc_pos": {"n": len(evc_pos_logit), "median": round(float(np.median(evc_pos_logit)), 4),
                    "iqr": [round(float(np.percentile(evc_pos_logit, 25)), 4),
                            round(float(np.percentile(evc_pos_logit, 75)), 4)]},
        "r25_pos": {"n": len(r25_pos_logit), "median": round(float(np.median(r25_pos_logit)), 4),
                    "iqr": [round(float(np.percentile(r25_pos_logit, 25)), 4),
                            round(float(np.percentile(r25_pos_logit, 75)), 4)]},
        "evc_neg": {"n": len(evc_neg_logit), "median": round(float(np.median(evc_neg_logit)), 4),
                    "iqr": [round(float(np.percentile(evc_neg_logit, 25)), 4),
                            round(float(np.percentile(evc_neg_logit, 75)), 4)]},
        "r25_neg": {"n": len(r25_neg_logit), "median": round(float(np.median(r25_neg_logit)), 4),
                    "iqr": [round(float(np.percentile(r25_neg_logit, 25)), 4),
                            round(float(np.percentile(r25_neg_logit, 75)), 4)]},
        "positive_shift_median_diff": round(float(np.median(evc_pos_logit) - np.median(r25_pos_logit)), 4),
        "negative_shift_median_diff": round(float(np.median(evc_neg_logit) - np.median(r25_neg_logit)), 4),
    }


def main() -> int:
    pooled = load_pooled()
    print(f"pooled OOF: n_neg={int((pooled.label_int==0).sum())} n_pos={int((pooled.label_int==1).sum())}\n")

    print("=== JOB A: positive-only shift sweep ===")
    ds = np.arange(0.0, 10.01, 0.25)
    a_rows = sweep_positive_shift(pooled, ds)

    print("\n=== JOB B: negative-spread sweep (control) ===")
    ms = np.arange(1.0, 10.01, 1.0)
    b_rows = sweep_negative_spread(pooled, ms)

    d_at_auroc_target = interp_crossing(a_rows, "auroc", TARGET_AUROC, "d")
    d_at_ppv_target = interp_crossing(a_rows, "ppv90", TARGET_PPV, "d")
    ppv_at_d_auroc = (other_metric_at_x(a_rows, d_at_auroc_target["x"], "d", "ppv90")
                       if d_at_auroc_target["x"] is not None else None)
    auroc_at_d_ppv = (other_metric_at_x(a_rows, d_at_ppv_target["x"], "d", "auroc")
                       if d_at_ppv_target["x"] is not None else None)

    print("\n=== JOB A crossings ===")
    print(f"d where AUROC = {TARGET_AUROC}: {d_at_auroc_target}")
    print(f"  -> PPV@90R at that d: {ppv_at_d_auroc}")
    print(f"d where PPV@90R = {TARGET_PPV}: {d_at_ppv_target}")
    print(f"  -> AUROC at that d: {auroc_at_d_ppv}")
    if d_at_auroc_target["x"] is not None and d_at_ppv_target["x"] is not None:
        disagreement = abs(d_at_auroc_target["x"] - d_at_ppv_target["x"])
        print(f"disagreement between the two d values: {disagreement:.4f} logits "
              f"({'FLAG: >1.0 logit' if disagreement > 1.0 else 'within 1.0 logit'})")

    m_where_auroc_below_080 = None
    for r in b_rows:
        if r["auroc"] < 0.80:
            m_where_auroc_below_080 = r["m"]
            break
    min_auroc_jobb = min(r["auroc"] for r in b_rows)
    max_ppv_drop_jobb = max(r["ppv90"] for r in b_rows[:1] + b_rows) - min(r["ppv90"] for r in b_rows)

    print(f"\n=== JOB B check ===")
    print(f"min AUROC observed across m=1..10: {min_auroc_jobb:.4f}")
    print(f"m at which AUROC first drops below 0.80: {m_where_auroc_below_080}")
    print(f"PPV@90R range across sweep: [{min(r['ppv90'] for r in b_rows):.4f}, {max(r['ppv90'] for r in b_rows):.4f}]")

    print("\n=== JOB C: EVC calibration ===")
    c = job_c_evc_calibration(pooled)
    print(json.dumps(c, indent=2))

    required_d = d_at_auroc_target["x"]
    observed_evc_shift = abs(c["positive_shift_median_diff"])
    ratio = (required_d / observed_evc_shift) if (required_d is not None and observed_evc_shift) else None
    print(f"\nrequired d (AUROC->0.6466) = {required_d}")
    print(f"observed EVC positive-class median shift = {c['positive_shift_median_diff']} "
          f"(magnitude {observed_evc_shift:.4f})")
    print(f"ratio required-d / observed-EVC-shift = {ratio}")

    out = {
        "job_a_rows": a_rows,
        "job_b_rows": b_rows,
        "job_a_crossings": {
            "d_at_auroc_target": d_at_auroc_target, "ppv90_at_that_d": ppv_at_d_auroc,
            "d_at_ppv_target": d_at_ppv_target, "auroc_at_that_d": auroc_at_d_ppv,
        },
        "job_b_check": {"min_auroc": min_auroc_jobb, "m_where_auroc_below_0.80": m_where_auroc_below_080,
                         "ppv90_range": [min(r["ppv90"] for r in b_rows), max(r["ppv90"] for r in b_rows)]},
        "job_c_evc_calibration": c,
        "required_d_over_observed_evc_shift": ratio,
    }
    with open("reports/l5_sink_sweep_raw.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote reports/l5_sink_sweep_raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
