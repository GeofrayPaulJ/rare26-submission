"""L4 JOB C -- build the 100-image EVC .mha stack for container scoring.

All 100 EVC images are natively 1600x1200 (confirmed, manifests/evc_inventory.csv)
-- uniform shape, so this is a real .mha volume, no resize/pad needed (unlike
H2's ragged RARE25 native sizes).

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/108_l4_evc_build_stack.py'
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

REPO_ROOT = "/workspace/RARE26"
EVC_ROOT = os.path.join(REPO_ROOT, "02_evc")
INVENTORY = os.path.join(REPO_ROOT, "manifests", "evc_inventory.csv")
OUT_DIR = os.path.join(REPO_ROOT, "runs/submission_test/l4_evc/interface_0")
INPUT_DIRNAME = "stacked-barretts-esophagus-endoscopy"


def write_inputs_json(interface_dir: str) -> None:
    payload = [{
        "interface": {
            "slug": "stacked-barretts-esophagus-endoscopy-images",
            "kind": "Image",
            "relative_path": f"images/{INPUT_DIRNAME}",
        }
    }]
    with open(os.path.join(interface_dir, "inputs.json"), "w") as fh:
        json.dump(payload, fh, indent=4)


def main() -> int:
    df = pd.read_csv(INVENTORY).sort_values("filepath").reset_index(drop=True)
    assert len(df) == 100
    assert (df["width"] == 1600).all() and (df["height"] == 1200).all(), "not uniform -- check inventory"

    frames = []
    for fp in df["filepath"]:
        arr = np.asarray(Image.open(os.path.join(EVC_ROOT, fp)).convert("RGB"))
        frames.append(arr)
    stack_arr = np.stack(frames, axis=0)  # (100, 1200, 1600, 3) uint8
    print(f"stack: shape={stack_arr.shape} dtype={stack_arr.dtype} "
          f"min={stack_arr.min()} max={stack_arr.max()} mean={stack_arr.mean():.2f}")

    img_dir = os.path.join(OUT_DIR, "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    mha_path = os.path.join(img_dir, "stack.mha")
    sitk_img = sitk.GetImageFromArray(stack_arr, isVector=True)
    sitk.WriteImage(sitk_img, mha_path)
    print(f"wrote {mha_path} ({os.path.getsize(mha_path) / 2**20:.0f} MiB) "
          f"size={sitk_img.GetSize()} ncomp={sitk_img.GetNumberOfComponentsPerPixel()}")

    write_inputs_json(OUT_DIR)

    order_path = os.path.join(OUT_DIR, "stack_order.csv")
    df[["filepath", "patient_id", "pathology_code", "class_label"]].assign(
        position=range(len(df))
    ).to_csv(order_path, index=False)
    print(f"order: {order_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
