"""D2 -- order integrity. Build a stack with slices in a DELIBERATELY
shuffled order (distinct from filename-sorted order, manifest order, or any
other "natural" ordering the container could accidentally fall back to),
keep a separate ground-truth index, run the container, and verify every
output position maps to the correct image.

CONTRACT, stated explicitly (submission/rare26_infer/predict.py:5-8,
docstring): the output JSON is POSITIONAL against the INPUT STACK'S OWN
PAGE ORDER -- "every item carries its own index and results are scattered
into a preallocated array by that index" specifically because "the
DataLoader is unshuffled, but that alone is not a guarantee worth resting
on -- workers complete out of order internally". The stack format itself
(TIFF pages / ITK z-slices) carries no filenames -- ordering is the ONLY
channel that ties an output likelihood back to an image, which is exactly
why this needs to be tested with an order that could not accidentally be
right for the wrong reason (e.g. already-sorted input data would pass even
if the scatter were broken and it silently fell back to file order).

Not locally verifiable: whether this positional contract MATCHES what the
Grand Challenge platform's own scoring harness assumes -- that lives in
the platform's algorithm documentation, not this repo. State this
explicitly rather than assuming agreement.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/72_d2_order_integrity.py'
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission", "tools"))
from make_test_stack import INPUT_DIRNAME, MANIFEST, _load_rgb, write_inputs_json  # noqa: E402

REPO_ROOT = "/workspace/RARE26"
OUT_DIR = os.path.join(REPO_ROOT, "runs/submission_test/d2_order")
SEED = 20260810777  # deliberately distinct from D1/D6's seeds


def main() -> int:
    df = pd.read_csv(MANIFEST)
    usable = df[df["fold_r0"] != -1]
    val = usable[usable["fold_r0"] == 0].reset_index(drop=True)  # fold 0 -- this checkpoint's own held-out set

    # deliberately shuffle, and confirm the shuffle actually differs from
    # both manifest order and filename-sorted order (a shuffle that
    # happened to match one of those would not be a real test)
    rng = np.random.default_rng(SEED)
    shuffled = val.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    is_manifest_order = (shuffled["filepath"].values == val["filepath"].values).all()
    is_filename_sorted = (shuffled["filepath"].values == sorted(val["filepath"].values)).all()
    print(f"[d2] shuffled order == manifest order? {is_manifest_order} "
          f"(must be False for this to be a real test)")
    print(f"[d2] shuffled order == filename-sorted order? {is_filename_sorted} "
          f"(must be False)")
    assert not is_manifest_order and not is_filename_sorted, "shuffle degenerate -- rerun with a different seed"

    img_dir = os.path.join(OUT_DIR, "interface_0", "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, "stack.tif")
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for fp in tqdm(shuffled["filepath"], desc="write shuffled stack", unit="img", file=sys.stderr):
            tw.write(_load_rgb(fp), photometric="rgb", compression=None, contiguous=False)

    order_path = os.path.join(OUT_DIR, "interface_0", "shuffled_ground_truth.csv")
    shuffled[["filepath", "class_label"]].assign(
        stack_position=range(len(shuffled))).to_csv(order_path, index=False)
    write_inputs_json(os.path.join(OUT_DIR, "interface_0"))
    print(f"[d2] {len(shuffled)}-image shuffled stack written: {path}")
    print(f"[d2] ground truth (by stack position): {order_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
