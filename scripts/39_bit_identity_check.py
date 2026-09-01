"""Bit-identity check between a checkpointed retrain and its screening-run
reference. NO TRAINING, NO GPU: reads canonical val_*.parquet files only.

Doubles as the reproducibility audit: if configs/sweep_a4_checkpointed.yaml
(or g3_checkpointed.yaml) really is the reference config verbatim plus only
save_checkpoint/out_dir, and training on this machine is deterministic run to
run, then every unit's logits must match EXACTLY -- not approximately. A
mismatch means either the config drifted from "verbatim", the code changed
underneath it, or determinism does not hold, and any of those three is a
finding serious enough to halt on rather than paper over.

    python scripts/39_bit_identity_check.py --new runs/a4_checkpointed \
        --reference runs/sweep_a4 --mode cv --label A4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from src.io import read_predictions  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)


def unit_names(mode: str, repeats: List[int] = (0,)) -> List[str]:
    if mode == "loco":
        return [f"loco_c{c}_s{s}" for c in (1, 2) for s in SEEDS]
    return [f"r{r}_f{f}_s{s}" for r in repeats for f in FOLDS for s in SEEDS]


def compare_unit(new_dir: str, ref_dir: str, name: str) -> dict:
    new_path = os.path.join(new_dir, name, f"val_{name}.parquet")
    ref_path = os.path.join(ref_dir, name, f"val_{name}.parquet")
    if not os.path.exists(new_path):
        return {"unit": name, "status": "MISSING_NEW", "path": new_path}
    if not os.path.exists(ref_path):
        return {"unit": name, "status": "MISSING_REFERENCE", "path": ref_path}

    new = read_predictions(new_path).set_index("filepath").sort_index()
    ref = read_predictions(ref_path).set_index("filepath").sort_index()

    if set(new.index) != set(ref.index):
        return {"unit": name, "status": "FILEPATH_SET_MISMATCH",
               "n_new": len(new), "n_ref": len(ref)}

    ref = ref.loc[new.index]
    label_match = bool((new["label_int"].to_numpy() == ref["label_int"].to_numpy()).all())
    logit_exact = bool(np.array_equal(new["logit"].to_numpy(), ref["logit"].to_numpy()))
    max_abs_diff = float(np.max(np.abs(new["logit"].to_numpy() - ref["logit"].to_numpy())))
    n_diff = int((new["logit"].to_numpy() != ref["logit"].to_numpy()).sum())

    status = "EXACT_MATCH" if (label_match and logit_exact) else "MISMATCH"
    return {"unit": name, "status": status, "label_match": label_match,
           "logit_exact": logit_exact, "max_abs_logit_diff": max_abs_diff,
           "n_rows_differ": n_diff, "n_rows": len(new)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True, help="checkpointed retrain out_dir")
    ap.add_argument("--reference", required=True, help="original screening-run out_dir")
    ap.add_argument("--mode", default="cv", choices=["cv", "loco"])
    ap.add_argument("--repeats", default="0", help="cv mode: comma-separated")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    new_dir = os.path.join(REPO_ROOT, args.new)
    ref_dir = os.path.join(REPO_ROOT, args.reference)
    repeats = [int(x) for x in args.repeats.split(",") if x.strip() != ""]

    results = [compare_unit(new_dir, ref_dir, n)
              for n in unit_names(args.mode, repeats)]
    n_exact = sum(1 for r in results if r["status"] == "EXACT_MATCH")
    n_total = len(results)
    all_exact = n_exact == n_total

    label = args.label or args.new
    print(f"\n{'=' * 78}")
    print(f"BIT-IDENTITY CHECK: {label}  ({args.new}  vs  {args.reference})")
    print(f"{'=' * 78}")
    for r in results:
        mark = "OK" if r["status"] == "EXACT_MATCH" else "!!"
        print(f"  [{mark}] {r['unit']:16s} {r['status']}"
             + (f"  max_abs_diff={r.get('max_abs_logit_diff'):.3e}"
                f"  n_rows_differ={r.get('n_rows_differ')}"
                if r["status"] == "MISMATCH" else ""))
    print(f"\n{n_exact}/{n_total} units EXACT_MATCH.")
    if all_exact:
        print(f"PASS: {label} is bit-identical to its reference across all "
             f"{n_total} units. Reproducibility confirmed.")
    else:
        print(f"HALT: {label} is NOT bit-identical against its reference. "
             f"{n_total - n_exact} unit(s) mismatched or missing -- see above. "
             f"This is a finding, not a warning: report it, do not proceed to "
             f"use these weights or continue the chain past this point.")
    print(f"{'=' * 78}\n")

    out = {"label": label, "new_dir": args.new, "reference_dir": args.reference,
          "mode": args.mode, "n_total": n_total, "n_exact_match": n_exact,
          "all_exact": all_exact, "results": results}
    out_json = args.out_json or os.path.join(
        REPO_ROOT, "reports", f"bit_identity_{label.lower().replace(' ', '_')}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"written: {out_json}")

    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
