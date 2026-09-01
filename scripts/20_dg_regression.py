"""Prove a reused arm really is the arm it claims to be.

Two arms of the magnitude sweep reuse predictions that were produced by earlier
code:

    A0 (baseline)                 reuses runs/noise_floor_a  (pooled OOF)
    A1 (centre x class sampler)   reuses runs/noise_floor_b  (LOCO only, where
                                  the sampler provably degenerates to A0's)

Reuse saves ~13 GPU-hours and is only legitimate if the current code reproduces
those runs exactly. "The new transforms are gated off" is an argument, not
evidence, so this retrains one unit for a few epochs and compares its
per-image validation logits against the same epoch of the run on disk.

Bit-identical logits establish two things at once, and the sweep needs both:

  * the code changes did not disturb the path this arm exercises;
  * training on this machine is deterministic run to run, which is the
    assumption every paired comparison in the sweep rests on.

Anything short of bit-identical fails. A near-match is not a pass -- it would
mean the reused predictions came from a different computation than the one the
sweep is about to attribute them to.

    python scripts/20_dg_regression.py --mode cv   --repeat 0 --fold 0 --seed 0 \
        --config configs/sweep_a0.yaml --reference runs/noise_floor_a
    python scripts/20_dg_regression.py --mode loco --centre 1 --seed 0 \
        --config configs/sweep_a1.yaml --reference runs/noise_floor_b
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from src.io import read_predictions  # noqa: E402


def unit_name(mode: str, centre: int, repeat: int, fold: int, seed: int) -> str:
    """The run-directory stem, matching src.train.split_name."""
    if mode == "loco":
        return f"loco_c{centre}_s{seed}"
    return f"r{repeat}_f{fold}_s{seed}"


def epoch_parquet(run_dir: str, name: str, epoch: int) -> str:
    return os.path.join(run_dir, "preds", f"val_{name}_e{epoch:03d}.parquet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="loco", choices=["loco", "cv"])
    ap.add_argument("--centre", type=int, default=1, choices=[1, 2])
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3,
                    help="epochs to retrain and compare")
    ap.add_argument("--reference", required=True,
                    help="run directory holding the predictions being reused")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "runs",
                                                      "regression_check"))
    args = ap.parse_args(argv)

    name = unit_name(args.mode, args.centre, args.repeat, args.fold, args.seed)
    ref_dir = args.reference if os.path.isabs(args.reference) else \
        os.path.join(REPO_ROOT, args.reference)
    reference = epoch_parquet(os.path.join(ref_dir, name), name, args.epochs)
    if not os.path.exists(reference):
        print(f"FAIL: no reference predictions at {reference}")
        return 2

    # --max-epochs caps COMPLETED epochs without touching the schedule, so
    # epoch N here rides the same LR curve as epoch N of the 30-epoch reference.
    # Capping cfg.epochs instead would compare two different curves.
    argv_train = [
        sys.executable, "-u", "-m", "src.train",
        "--config", args.config,
        "--seed", str(args.seed),
        "--out-dir", args.out_dir,
        "--max-epochs", str(args.epochs),
        "--no-progress",
    ]
    if args.mode == "loco":
        argv_train += ["--holdout-centre", str(args.centre)]
    else:
        argv_train += ["--repeat", str(args.repeat), "--fold", str(args.fold)]

    print(f"[regression] {' '.join(argv_train)}\n", flush=True)
    rc = subprocess.run(argv_train, cwd=REPO_ROOT).returncode
    if rc != 0:
        print(f"FAIL: training exited {rc}")
        return 2

    candidate = epoch_parquet(os.path.join(args.out_dir, name), name, args.epochs)
    ref = read_predictions(reference).set_index("filepath").sort_index()
    new = read_predictions(candidate).set_index("filepath").sort_index()

    if list(ref.index) != list(new.index):
        print("FAIL: the two runs scored different images")
        return 2

    a = ref["logit"].to_numpy()
    b = new["logit"].to_numpy()
    identical = bool(np.array_equal(a, b))
    max_abs = float(np.max(np.abs(a - b))) if len(a) else 0.0
    n_diff = int((a != b).sum())

    print(f"\n{'=' * 74}")
    print(f"REGRESSION -- {name}, epoch {args.epochs}, {len(a)} validation images")
    print(f"  config               : {args.config}")
    print(f"  reference (on disk)  : {os.path.relpath(reference, REPO_ROOT)}")
    print(f"  candidate (new code) : {os.path.relpath(candidate, REPO_ROOT)}")
    print(f"  logits differing     : {n_diff} / {len(a)}")
    print(f"  max |difference|     : {max_abs:.3e}")
    print(f"{'=' * 74}")
    if identical:
        print(f"PASS -- bit-identical. Reusing {args.reference} for this arm is\n"
              f"       legitimate, and training is run-to-run deterministic.")
        return 0
    print(f"FAIL -- the logits differ. Either the code changed the path this arm\n"
          f"       exercises, or training is not deterministic on this machine.\n"
          f"       Do NOT reuse {args.reference}; the arm must be run in full.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
