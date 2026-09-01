"""D10.4 continued -- all four constructed .mha variants preserved the
correct slice count only when isVector=True (dim=3, Z=n_frames matches).
Both isVector=True variants (uint8 and float[0,1]) read back CORRECTLY at
slice 0. Testing the remaining named suspect: does ItkStack's STREAMING
extract-based read (SetExtractIndex/SetExtractSize via ImageFileReader,
used for memory-bounded slice-wise access) diverge from a full
non-streaming sitk.ReadImage() read, for a vector-pixel MetaImage,
at slice indices beyond 0?
"""
from __future__ import annotations

import json
import os

import numpy as np
import SimpleITK as sitk

REPO_ROOT = "/workspace/RARE26"
PATH = os.path.join(REPO_ROOT, "runs/submission_test/d10_variants/A_vector_uint8.mha")


def streaming_read(path: str, i: int) -> np.ndarray:
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    size = list(r.GetSize())
    r.SetExtractIndex([0, 0, int(i)])
    r.SetExtractSize([size[0], size[1], 1])
    return np.squeeze(sitk.GetArrayFromImage(r.Execute()))


def full_read(path: str, i: int) -> np.ndarray:
    img = sitk.ReadImage(path)
    full = sitk.GetArrayFromImage(img)  # (Z, Y, X, 3)
    return full[i]


def main() -> int:
    results = {}
    for i in [0, 5, 10, 15, 19]:  # variant file only has 20 frames
        s = streaming_read(PATH, i)
        f = full_read(PATH, i)
        same_shape = s.shape == f.shape
        match = same_shape and np.array_equal(s, f)
        results[f"slice_{i}"] = {
            "streaming_shape": list(s.shape), "streaming_mean": float(s.mean()),
            "streaming_min": float(s.min()), "streaming_max": float(s.max()),
            "full_shape": list(f.shape), "full_mean": float(f.mean()),
            "full_min": float(f.min()), "full_max": float(f.max()),
            "EXACT_MATCH": bool(match),
        }
    print(json.dumps(results, indent=2))
    with open(os.path.join(REPO_ROOT, "reports/d10_streaming_check.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
