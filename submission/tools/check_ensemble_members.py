"""Validation 4, ensemble edition: does each member reproduce ITS OWN fold's
harness logits inside the containerised forward pass?

WHY THIS EXISTS AND NOT A DIRECT "ENSEMBLE PARITY" CHECK. The 5-fold
single-seed A4 ensemble is a k-fold OOF design: member r0_f{k}_s0 was
evaluated by the harness only on fold k's held-out images -- the other 4
folds were in that member's training set. There is therefore no harness
artefact anywhere in this project that records a genuine "5-model average"
logit for any image; averaging 4 non-held-out members with 1 held-out member
is exactly what happens at real (unseen) test time, and nothing in the
training run ever computed that quantity for a comparison. Building a
synthetic reference by averaging the 5 members' harness dumps would silently
mix held-out and non-held-out predictions and call the result "ground truth,"
which it is not.

What IS testable, and what this script checks: for every image in the pooled
validation set, exactly one member had it held out. That member's individual
(pre-average) logit inside the container -- dumped via
predict_stack(..., dump_member_logits_path=...) -- should reproduce the
harness dump for that same image to the same fp16 tolerance
tools/check_parity.py uses for the single-checkpoint container. This
validates preprocessing parity AND that the ensemble's per-member forward
pass and averaging code did not corrupt any individual member's output; it
does not (and cannot) validate the average itself against a reference,
because no such reference exists.

Usage (from repo root; do_test_run.sh remaps RARE26_DUMP_MEMBER_LOGITS to
/output/<basename>, so the readable file ends up at <output_dir>/<basename>):

    python submission/tools/make_test_stack.py parity-pooled \\
      --out-dir runs/submission_test/parity_pooled/interface_0
    cd submission && RARE26_DUMP_MEMBER_LOGITS=member_logits.npy \\
      ./do_test_run.sh ../runs/submission_test/parity_pooled/interface_0 \\
      ../runs/submission_test/parity_pooled/output fp16 && cd ..
    python submission/tools/check_ensemble_members.py \\
      --member-logits runs/submission_test/parity_pooled/output/member_logits.npy \\
      --order-csv runs/submission_test/parity_pooled/interface_0/stack_order.csv \\
      --reference-dir runs/a4_checkpointed \\
      --manifest submission/resources/ensemble/manifest.txt
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# Same fp16 tolerance as tools/check_parity.py -- fp16 has a 10-bit mantissa
# (relative eps ~9.8e-4); through an 88M-parameter ConvNeXt the per-logit
# deviation lands well inside 0.25 in practice.
LOGIT_TOL_FP16 = 0.25


def _fold_of_manifest_name(name: str) -> int:
    """``a4_r0_f{k}_s0.pth`` -> k. Raises if the naming convention drifts."""
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    fold_part = next((p for p in parts if p.startswith("f") and p[1:].isdigit()), None)
    if fold_part is None:
        raise ValueError(f"cannot parse fold index out of checkpoint name {name!r}")
    return int(fold_part[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--member-logits", required=True,
                    help="(n_images, n_members) .npy from predict_stack's dump_member_logits_path")
    ap.add_argument("--order-csv", required=True,
                    help="stack_order.csv from make_test_stack.py (index -> filepath)")
    ap.add_argument("--reference-dir", default="runs/a4_checkpointed",
                    help="dir holding val_r0_f{k}_s0.parquet per fold")
    ap.add_argument("--manifest", default="submission/resources/ensemble/manifest.txt",
                    help="lists checkpoint filenames in the same order as member-logits columns")
    args = ap.parse_args()

    member_logits = np.load(args.member_logits)
    order = pd.read_csv(args.order_csv)
    if len(order) != member_logits.shape[0]:
        raise SystemExit(
            f"length mismatch: {len(order)} stack rows vs {member_logits.shape[0]} "
            f"logit rows"
        )

    with open(args.manifest) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    if len(names) != member_logits.shape[1]:
        raise SystemExit(
            f"manifest lists {len(names)} checkpoints but the dump has "
            f"{member_logits.shape[1]} member columns"
        )
    fold_of_column = [_fold_of_manifest_name(n) for n in names]

    # Pull every fold's held-out harness dump once, keyed by filepath.
    ref_by_fold = {}
    for fold in sorted(set(fold_of_column)):
        path = os.path.join(args.reference_dir, f"r0_f{fold}_s0", f"val_r0_f{fold}_s0.parquet")
        if not os.path.isfile(path):
            raise SystemExit(f"missing harness dump for fold {fold}: {path}")
        ref_by_fold[fold] = pd.read_parquet(path).set_index("filepath")

    rows = []
    for pos, fp in enumerate(order["filepath"]):
        owning_fold = None
        for fold, ref in ref_by_fold.items():
            if fp in ref.index:
                owning_fold = fold
                break
        if owning_fold is None:
            continue  # image held out by no fold in this manifest -- not comparable
        col = fold_of_column.index(owning_fold)
        got_logit = float(member_logits[pos, col])
        ref_logit = float(ref_by_fold[owning_fold].loc[fp, "logit"])
        rows.append((fp, owning_fold, got_logit, ref_logit, abs(got_logit - ref_logit)))

    if not rows:
        raise SystemExit(
            "no stack image matched any fold's held-out set -- wrong stack "
            "for this check (use make_test_stack.py parity-pooled or equivalent)"
        )

    df = pd.DataFrame(rows, columns=["filepath", "own_fold", "container_logit",
                                      "harness_logit", "abs_diff"])
    print(f"images compared      : {len(df)} (each scored by its OWN held-out member only)")
    for fold in sorted(df["own_fold"].unique()):
        sub = df[df["own_fold"] == fold]
        print(f"  fold {fold}: n={len(sub)}  max|diff|={sub['abs_diff'].max():.3e}  "
              f"mean|diff|={sub['abs_diff'].mean():.3e}")
    print(f"overall max |diff|   : {df['abs_diff'].max():.3e}   (tolerance {LOGIT_TOL_FP16:g})")
    print(f"overall mean |diff|  : {df['abs_diff'].mean():.3e}")

    worst = df.loc[df["abs_diff"].idxmax()]
    if worst["abs_diff"] > LOGIT_TOL_FP16:
        print(f"\nFAIL: worst per-member disagreement {worst['abs_diff']:.4f} > "
              f"{LOGIT_TOL_FP16:g} (image {worst['filepath']!r}, fold {int(worst['own_fold'])})")
        raise SystemExit(1)
    print("\nPASS: every member reproduces its own held-out fold within fp16 tolerance.")


if __name__ == "__main__":
    main()
