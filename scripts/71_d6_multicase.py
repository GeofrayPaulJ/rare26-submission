"""D6 -- multi-case handling test.

CODE-LEVEL FINDING FIRST, BEFORE ANY EXPERIMENT: `find_stack_file()` in
submission/rare26_infer/stack.py globs every TIFF/ITK file in the mounted
image directory and returns `sorted(candidates)[0]` -- the alphabetically
FIRST one. Its own docstring states the assumption plainly: "Grand
Challenge mounts exactly one file per image socket". If that assumption is
wrong -- if the platform ever mounts more than one stack file per job
(one per case) in the SAME socket directory -- every file except the
first is silently discarded. No error, no warning, no log line naming
which file was picked or how many were ignored.  `inference.py`'s own
`interface_0_handler()` reinforces this: it calls `resolve_image_dir()`
ONCE, `predict_stack()` ONCE, writes ONE output file -- despite printing
`cases_received={len(inputs)}` (which reads the actual case count) and a
`case_0_*`-prefixed report block whose naming implies more cases could
exist. There is no loop anywhere over case 1, 2, 3.

This script builds the experiment that would PROVE this is what happened
in production: 4 separate case stacks (~772 images each, the pooled-OOF
3088-image pool split 4 ways, real images with real labels), placed
together in ONE mounted images/<socket>/ directory -- exactly the layout
`find_stack_file` would see if the platform batches multiple cases into
one job -- then the SUBMITTED container run exactly once, exactly as the
platform would invoke it.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/71_d6_multicase.py'
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission", "tools"))
from make_test_stack import INPUT_DIRNAME, INPUT_SLUG, MANIFEST, _load_rgb, write_inputs_json  # noqa: E402

REPO_ROOT = "/workspace/RARE26"
OUT_DIR = os.path.join(REPO_ROOT, "runs/submission_test/d6_multicase")
N_CASES = 4
SEED = 20260810


def main() -> int:
    df = pd.read_csv(MANIFEST)
    frames = []
    for fold in range(5):
        usable = df[df["fold_r0"] != -1]
        val = usable[usable["fold_r0"] == fold]
        frames.append(val.assign(_own_fold=fold))
    pooled = pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pooled))
    pooled = pooled.iloc[idx].reset_index(drop=True)
    case_id = np.arange(len(pooled)) % N_CASES
    pooled["case"] = case_id

    img_dir = os.path.join(OUT_DIR, "interface_0", "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)

    order_rows = []
    case_names = []
    for c in range(N_CASES):
        sub = pooled[pooled["case"] == c].reset_index(drop=True)
        # sorted() filename order matters -- this IS the mechanism under test
        name = f"case-{c:04d}.tif"
        case_names.append(name)
        path = os.path.join(img_dir, name)
        with tifffile.TiffWriter(path, bigtiff=True) as tw:
            for fp in tqdm(sub["filepath"], desc=f"write {name}", unit="img", file=sys.stderr):
                tw.write(_load_rgb(fp), photometric="rgb", compression=None, contiguous=False)
        for pos, row in sub.iterrows():
            order_rows.append({"case_file": name, "position_in_case": pos,
                               "filepath": row["filepath"], "own_fold": row["_own_fold"],
                               "class_label": row["class_label"]})
        print(f"[d6] wrote {name}: {len(sub)} images, "
              f"{os.path.getsize(path) / 2**20:.0f} MiB")

    order = pd.DataFrame(order_rows)
    order_path = os.path.join(OUT_DIR, "interface_0", "multicase_ground_truth.csv")
    order.to_csv(order_path, index=False)

    write_inputs_json(os.path.join(OUT_DIR, "interface_0"))

    print(f"[d6] {N_CASES} case files written to {img_dir}: {sorted(case_names)}")
    print(f"[d6] sorted()[0] (what find_stack_file will pick) = {sorted(case_names)[0]}")
    print(f"[d6] ground truth for all cases: {order_path}")
    print(f"[d6] total images across all cases: {len(order)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
