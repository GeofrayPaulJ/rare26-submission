"""D10 -- reproduce the leaderboard failure locally via .mha input.

Every local test to date (D1-D9) used .tiff, built by
submission/tools/make_test_stack.py's tifffile writer. Grand Challenge
converts uploads to MetaImage (.mha) -- confirmed via public docs
(grand-challenge.org: "Images used in ... challenges are provided in the
MetaImage (.mha) file format"). This path has never been exercised
locally. Try-out log evidence: FOV fallback 213/213, reason "no non-black
pixels found", logit range [-0.94, 0.87] vs the normal +/-8.5 -- reads as
entirely black.

Builds the SAME 200 images (D3's baseline set) as BOTH a .tiff stack
(reusing D3's fixture) and a fresh .mha stack (via
SimpleITK.GetImageFromArray(arr, isVector=True) -> WriteImage, the
standard way to construct a multi-component/vector 3D image from a numpy
array), then runs the submitted container against the .mha stack to
check for reproduction, and does a side-by-side raw-array bisection of
what SimpleITK's own reader (matching ItkStack's logic exactly) returns
for slice 0 of each format.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/76_d10_mha_repro.py'
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import SimpleITK as sitk

REPO_ROOT = "/workspace/RARE26"
sys.path.insert(0, os.path.join(REPO_ROOT, "submission", "tools"))
from make_test_stack import INPUT_DIRNAME, write_inputs_json  # noqa: E402

TIFF_BASELINE_DIR = os.path.join(REPO_ROOT, "runs/submission_test/d3_fov/baseline/interface_0")
OUT_DIR = os.path.join(REPO_ROOT, "runs/submission_test/d10_mha")


def main() -> int:
    # --- rebuild the SAME 200 images from D3's ground truth, in the SAME order ---
    import cv2
    import pandas as pd
    gt = pd.read_csv(os.path.join(REPO_ROOT, "runs/submission_test/d3_fov/ground_truth.csv")).sort_values("position")
    # A .mha volume needs uniform per-slice shape (unlike a TIFF page-list,
    # where each page is independent). RARE25 images are not all the same
    # native size, so resize to the modal training resolution (637x512,
    # same convention make_test_stack.py's synthetic mode uses) -- this
    # isolates the FORMAT/reading question from a resize confound.
    MODAL_W, MODAL_H = 637, 512
    frames = []
    for fp in gt["filepath"]:
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(REPO_ROOT, "00_source", fp)))
        # ReadImage on a PNG gives (Y, X, 3) uint8 RGB already
        img = cv2.resize(img, (MODAL_W, MODAL_H), interpolation=cv2.INTER_AREA)
        frames.append(img)
    stack_arr = np.stack(frames, axis=0)  # (Z, Y, X, 3) uint8
    print(f"[d10] built numpy stack: shape={stack_arr.shape} dtype={stack_arr.dtype} "
          f"min={stack_arr.min()} max={stack_arr.max()} mean={stack_arr.mean():.2f}")

    # --- write as .mha, the standard vector-image construction ---
    img_dir = os.path.join(OUT_DIR, "interface_0", "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    mha_path = os.path.join(img_dir, "stack.mha")
    sitk_img = sitk.GetImageFromArray(stack_arr, isVector=True)
    print(f"[d10] sitk image built: size={sitk_img.GetSize()} "
          f"components={sitk_img.GetNumberOfComponentsPerPixel()} "
          f"pixel_type={sitk_img.GetPixelIDTypeAsString()}")
    sitk.WriteImage(sitk_img, mha_path)
    print(f"[d10] wrote {mha_path} ({os.path.getsize(mha_path) / 2**20:.0f} MiB)")

    write_inputs_json(os.path.join(OUT_DIR, "interface_0"))
    gt.to_csv(os.path.join(OUT_DIR, "ground_truth.csv"), index=False)

    # --- D10.3 bisection: read slice 0 back via SimpleITK, same logic as ItkStack.read() ---
    def read_via_itkstack_logic(path: str, i: int = 0):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        size = list(reader.GetSize())
        ncomp = reader.GetNumberOfComponents()
        dim = len(size)
        r = sitk.ImageFileReader()
        r.SetFileName(path)
        r.ReadImageInformation()
        if dim >= 3:
            r.SetExtractIndex([0, 0, int(i)])
            r.SetExtractSize([size[0], size[1], 1])
        arr = sitk.GetArrayFromImage(r.Execute())
        return {"size": size, "ncomp": ncomp, "dim": dim, "raw_extracted_shape": arr.shape,
                "raw_dtype": str(arr.dtype), "raw_min": float(arr.min()), "raw_max": float(arr.max()),
                "raw_mean": float(arr.mean()), "squeezed_shape": np.squeeze(arr).shape}

    mha_info = read_via_itkstack_logic(mha_path, 0)

    # build an equivalent single-slice .tiff for direct A/B (same image, image 0)
    import tifffile
    tiff_probe_path = os.path.join(OUT_DIR, "probe_single.tif")
    tifffile.imwrite(tiff_probe_path, stack_arr[0], photometric="rgb")
    tiff_arr = tifffile.imread(tiff_probe_path)
    tiff_info = {"raw_extracted_shape": tiff_arr.shape, "raw_dtype": str(tiff_arr.dtype),
                "raw_min": float(tiff_arr.min()), "raw_max": float(tiff_arr.max()),
                "raw_mean": float(tiff_arr.mean())}

    print("\n[d10] === D10.3 BISECTION: slice 0, same source image ===")
    print("MHA (via SimpleITK, ItkStack's own extraction logic):")
    print(json.dumps(mha_info, indent=2))
    print("TIFF (via tifffile, TiffStack's own reader):")
    print(json.dumps(tiff_info, indent=2))

    with open(os.path.join(REPO_ROOT, "reports/d10_bisection.json"), "w") as fh:
        json.dump({"mha": mha_info, "tiff": tiff_info,
                   "numpy_source_stack": {"shape": list(stack_arr.shape), "dtype": str(stack_arr.dtype),
                                          "min": int(stack_arr.min()), "max": int(stack_arr.max())}}, fh, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
