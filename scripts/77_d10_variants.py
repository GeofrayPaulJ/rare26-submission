"""D10.4 -- bisect WHICH .mha construction reproduces the black-image
failure. The naive isVector=True construction (76_d10_mha_repro.py) read
back correctly -- does NOT reproduce the bug. Testing named alternative
constructions against the prime suspects list, same 200-image data.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO_ROOT = "/workspace/RARE26"
OUT_DIR = os.path.join(REPO_ROOT, "runs/submission_test/d10_variants")
MODAL_W, MODAL_H = 637, 512


def read_via_itkstack_logic(path: str, i: int = 0):
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    size = list(r.GetSize())
    ncomp = r.GetNumberOfComponents()
    dim = len(size)
    pixel_type = sitk.GetPixelIDValueAsString(r.GetPixelID())
    r2 = sitk.ImageFileReader()
    r2.SetFileName(path)
    r2.ReadImageInformation()
    if dim >= 3:
        extract_size = [size[0], size[1]] + [1] * (dim - 2)
        extract_index = [0, 0] + [int(i)] * (dim - 2)
        # ItkStack literally does SetExtractIndex([0,0,i]) / SetExtractSize([size0,size1,1])
        # -- only sets 3 values regardless of dim. Replicate EXACTLY, not a generalised version.
        r2.SetExtractIndex([0, 0, int(i)])
        r2.SetExtractSize([size[0], size[1], 1])
    arr = sitk.GetArrayFromImage(r2.Execute())
    return {"size": size, "ncomp": ncomp, "dim": dim, "pixel_type": pixel_type,
            "raw_extracted_shape": list(arr.shape), "raw_dtype": str(arr.dtype),
            "raw_min": float(arr.min()), "raw_max": float(arr.max()),
            "raw_mean": float(np.asarray(arr, dtype=np.float64).mean()),
            "squeezed_shape": list(np.squeeze(arr).shape),
            "nonzero_frac": float((np.asarray(arr) != 0).mean())}


def main() -> int:
    gt = pd.read_csv(os.path.join(REPO_ROOT, "runs/submission_test/d3_fov/ground_truth.csv")).sort_values("position")
    frames = []
    for fp in gt["filepath"][:20]:  # 20 is plenty for the bisection; full 200 not needed
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(REPO_ROOT, "00_source", fp)))
        img = cv2.resize(img, (MODAL_W, MODAL_H), interpolation=cv2.INTER_AREA)
        frames.append(img)
    stack_u8 = np.stack(frames, axis=0)  # (Z,Y,X,3) uint8, [0,255]
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}

    # A) baseline (already tested, reads fine) -- isVector, uint8
    pA = os.path.join(OUT_DIR, "A_vector_uint8.mha")
    sitk.WriteImage(sitk.GetImageFromArray(stack_u8, isVector=True), pA)
    results["A_vector_uint8 (known-good baseline)"] = read_via_itkstack_logic(pA)

    # B) NOT marked as vector -- scalar 4D image, the "isVector forgotten" suspect
    pB = os.path.join(OUT_DIR, "B_scalar_4d_uint8.mha")
    sitk.WriteImage(sitk.GetImageFromArray(stack_u8, isVector=False), pB)
    results["B_scalar_4d_uint8 (isVector=False)"] = read_via_itkstack_logic(pB)

    # C) float [0,1] normalised, vector
    stack_f01 = (stack_u8.astype(np.float32) / 255.0)
    pC = os.path.join(OUT_DIR, "C_vector_float01.mha")
    sitk.WriteImage(sitk.GetImageFromArray(stack_f01, isVector=True), pC)
    results["C_vector_float01 (float [0,1], isVector=True)"] = read_via_itkstack_logic(pC)

    # D) float [0,1], NOT marked vector
    pD = os.path.join(OUT_DIR, "D_scalar_4d_float01.mha")
    sitk.WriteImage(sitk.GetImageFromArray(stack_f01, isVector=False), pD)
    results["D_scalar_4d_float01 (float [0,1], isVector=False)"] = read_via_itkstack_logic(pD)

    # E) channel-first per slice: (Z, 3, Y, X), vector flag off (mimics a
    #    component-first writer -- some ITK pipelines put components as a
    #    literal leading axis instead of using vector pixel types)
    stack_chw = np.transpose(stack_u8, (0, 3, 1, 2))  # (Z,3,Y,X)
    pE = os.path.join(OUT_DIR, "E_channel_first_uint8.mha")
    sitk.WriteImage(sitk.GetImageFromArray(stack_chw, isVector=False), pE)
    results["E_channel_first_uint8 (Z,C,Y,X, isVector=False)"] = read_via_itkstack_logic(pE)

    print(json.dumps(results, indent=2))
    with open(os.path.join(REPO_ROOT, "reports/d10_variants.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
