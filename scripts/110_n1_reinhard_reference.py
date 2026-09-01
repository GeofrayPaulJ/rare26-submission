"""N1 -- compute the FIXED Reinhard LAB reference statistic over the RARE25
training set, ONCE, before any arm runs (hard-coded thereafter, not
per-batch). Uses the exact FOV-crop geometry already on record
(manifests/rare25_folds_v2.csv's inner_left/top/right/bottom), matching the
insertion point N1's pre-registration specifies (immediately after FOV
crop, before the two-stage resize).

Per-image LAB mean/SD computed after downsizing the crop to 64x64 (cheap,
stable estimate of per-image colour statistics -- Reinhard transfer only
needs first/second moments, not full resolution), then averaged across all
3088 keep_for_training==True images (each image weighted equally,
regardless of native crop size).

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/110_n1_reinhard_reference.py'
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

MANIFEST = "manifests/rare25_folds_v2.csv"
IMAGE_ROOT = "00_source"
DOWNSIZE = 64


def main() -> int:
    df = pd.read_csv(MANIFEST)
    train = df[df["keep_for_training"] == True].reset_index(drop=True)
    print(f"{len(train)} keep_for_training images")

    means = np.zeros((len(train), 3))
    stds = np.zeros((len(train), 3))

    for i, row in enumerate(tqdm(train.itertuples(), total=len(train))):
        img = np.asarray(Image.open(f"{IMAGE_ROOT}/{row.filepath}").convert("RGB"))
        left, top, right, bottom = int(row.inner_left), int(row.inner_top), int(row.inner_right), int(row.inner_bottom)
        if right <= left or bottom <= top:
            crop = img
        else:
            crop = img[top:bottom, left:right]
        small = cv2.resize(crop, (DOWNSIZE, DOWNSIZE), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float64)
        means[i] = lab.mean(axis=(0, 1))
        stds[i] = lab.std(axis=(0, 1))

    ref_mean = means.mean(axis=0)
    ref_std = stds.mean(axis=0)

    out = {
        "n_images": len(train),
        "reference_lab_mean": ref_mean.tolist(),
        "reference_lab_std": ref_std.tolist(),
        "note": "L,A,B order, OpenCV COLOR_RGB2LAB convention (L in [0,255], A/B in [0,255] offset-128)",
    }
    print(json.dumps(out, indent=2))
    with open("reports/n1_reinhard_reference.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
