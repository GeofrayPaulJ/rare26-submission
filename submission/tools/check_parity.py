"""Validation 4: does the container reproduce the training harness's logits?

If the container and src/train.py disagree on the same images, the container is
wrong -- the harness is the reference, because it is what the reported
validation metrics were computed from. A disagreement here means the shipped
preprocessing has drifted from the trained-on preprocessing, and the model is
being fed something it was never fitted to.

Two comparisons, and they answer different questions:

  * bf16 container vs harness -- both run the same arithmetic, so anything
    beyond floating-point noise is a PREPROCESSING difference. This is the
    sharp test. It is the one that catches a dropped 431 -> 384 intermediate
    resize, a BGR/RGB flip, or a changed interpolation rule.

  * fp16 container vs harness -- the configuration that actually ships. The
    difference here is dominated by fp16 rounding, and the question is only
    whether it stays inside fp16 tolerance and leaves the RANKING intact. The
    metric is a ranking metric, so rank correlation and ROC-AUC matter more
    than absolute agreement.

FALLBACK IMAGES ARE SCORED SEPARATELY, and that is not a way of excusing bad
numbers. The harness cropped every training image with the geometry the
detector produced, however poor the fit; the container refuses that geometry
below the 1st-percentile floor and takes a centred square instead. On those
images the two are SUPPOSED to disagree -- that is the entire purpose of the
fallback. Pooling them would hide a real preprocessing bug behind a handful of
intentional differences, so the detector path is held to a near-exact tolerance
and the fallback images are reported as a separate, expected-nonzero delta.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

# fp16 carries a 10-bit mantissa (relative eps ~9.8e-4). Accumulated through an
# 88M-parameter ConvNeXt the per-logit deviation lands well inside 0.25 in
# practice; anything beyond that is not rounding, it is a bug.
LOGIT_TOL_FP16 = 0.25
# bf16 on both sides removes the precision difference but NOT the kernel
# difference: the container ships torch 2.7.1+cu128 while the checkpoint was
# trained under NGC's torch 2.10, and the two pick different cuDNN algorithms
# and accumulate in a different order. Same-torch, the detector path agrees to
# 4.7e-13; cross-torch it lands at 2.3e-2. So this is a numerics gate, not the
# preprocessing gate.
#
# THE PREPROCESSING GATE IS tools/check_preprocess_parity.py, which compares
# the CHW tensors themselves (cv2 and numpy only, no GPU) and requires them to
# be bit-identical. That is the test that catches a dropped 431 -> 384 resize;
# this one cannot, and should not be relied on for it.
LOGIT_TOL_BF16 = 0.05
AUC_TOL = 0.005
RANK_TOL = 0.995


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1.0 - 1e-16)
    return np.log(p / (1.0 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-json", required=True, help="container's likelihoods JSON")
    ap.add_argument("--order-csv", required=True, help="stack_order.csv from make_test_stack")
    ap.add_argument("--reference-parquet", required=True, help="harness val dump")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--stats-json", default=None,
                    help="rare26_run_stats.json, to identify fallback slices")
    args = ap.parse_args()

    with open(args.output_json) as fh:
        probs = np.asarray(json.load(fh), dtype=np.float64)
    order = pd.read_csv(args.order_csv)
    ref = pd.read_parquet(args.reference_parquet)

    if len(probs) != len(order):
        raise SystemExit(f"length mismatch: {len(probs)} probs vs {len(order)} stack rows")

    # Align the positional container output to the harness dump by filepath.
    ref_by_fp = ref.set_index("filepath")
    missing = [fp for fp in order["filepath"] if fp not in ref_by_fp.index]
    if missing:
        raise SystemExit(f"{len(missing)} stack filepaths absent from the reference dump")

    ref_logit = ref_by_fp.loc[order["filepath"], "logit"].to_numpy(dtype=np.float64)
    ref_label = ref_by_fp.loc[order["filepath"], "label_int"].to_numpy(dtype=np.int64)
    ref_prob = 1.0 / (1.0 + np.exp(-ref_logit))
    got_logit = _logit(probs)

    d_logit = np.abs(got_logit - ref_logit)
    d_prob = np.abs(probs - ref_prob)

    # Split the detector path from the fallback path; they are held to
    # different standards for the reason given in the module docstring.
    fallback_idx: list = []
    if args.stats_json and os.path.isfile(args.stats_json):
        with open(args.stats_json) as fh:
            fallback_idx = json.load(fh).get("fallback_indices", []) or []
    det = np.ones(len(probs), dtype=bool)
    det[np.asarray(fallback_idx, dtype=int)] = False

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    rank = float(spearmanr(probs, ref_logit).statistic)
    auc_got = float(roc_auc_score(ref_label, probs))
    auc_ref = float(roc_auc_score(ref_label, ref_logit))

    tol = {"fp16": LOGIT_TOL_FP16, "bf16": LOGIT_TOL_BF16, "fp32": LOGIT_TOL_BF16}[args.precision]

    print(f"precision under test : {args.precision}")
    print(f"images compared      : {len(probs)}")
    print(f"reference            : {os.path.basename(args.reference_parquet)}")
    print(f"FOV fallback slices  : {len(fallback_idx)} "
          f"({100.0 * len(fallback_idx) / len(probs):.2f}%) -- scored separately")
    print()
    print(f"--- DETECTOR PATH (n={int(det.sum())}) -- must match the harness ---")
    print(f"logit  max |diff|    : {d_logit[det].max():.3e}   (tolerance {tol:g})")
    print(f"logit  mean |diff|   : {d_logit[det].mean():.3e}")
    print(f"prob   max |diff|    : {d_prob[det].max():.3e}")
    if len(fallback_idx):
        print()
        print(f"--- FALLBACK PATH (n={len(fallback_idx)}) -- expected to differ by design ---")
        print(f"logit  max |diff|    : {d_logit[~det].max():.6f}")
        print(f"logit  mean |diff|   : {d_logit[~det].mean():.6f}")
        print(f"prob   max |diff|    : {d_prob[~det].max():.3e}")
    print()
    print(f"--- ALL {len(probs)} SLICES (ranking is what the metric sees) ---")
    print(f"spearman rank corr   : {rank:.8f}   (tolerance >{RANK_TOL})")
    print(f"ROC-AUC container    : {auc_got:.6f}")
    print(f"ROC-AUC harness      : {auc_ref:.6f}")
    print(f"ROC-AUC |diff|       : {abs(auc_got - auc_ref):.6f}   (tolerance {AUC_TOL})")
    print(f"distinct probs       : {len(np.unique(probs))}/{len(probs)}")

    failures = []
    if d_logit[det].max() > tol:
        failures.append(f"detector-path max logit diff {d_logit[det].max():.3e} > {tol:g}")
    if rank < RANK_TOL:
        failures.append(f"rank correlation {rank:.6f} < {RANK_TOL}")
    if abs(auc_got - auc_ref) > AUC_TOL:
        failures.append(f"ROC-AUC diff {abs(auc_got - auc_ref):.6f} > {AUC_TOL}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nPASS: container reproduces the harness within tolerance.")


if __name__ == "__main__":
    main()
