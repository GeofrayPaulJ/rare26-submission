"""F3 -- corruption signature battery.

Domain shift cannot produce AUROC 0.7051 on the RARE25-validation
subset (the training domain itself, where this checkpoint reads 0.9358
held-out). Something degrades in-domain and out-of-domain alike. This
builds fold 0's 617 held-out images (the REAL held-out set, not the
200-image training-manifest probe, which scores 1.0 and can't show
degradation) as a baseline + 7 corrupted variants, runs each through the
UNMODIFIED shipping container, and reports AUROC / n=1 FPR@90R for each
against the baseline of 0.9358.

Same construction as P1 throughout (resize to 637x512 modal, RGB uint8
vector .mha) except where the transform itself changes axis/channel
structure -- those differences ARE the transform.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/86_f3_corruption_battery.py'
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm

REPO_ROOT = "/workspace/RARE26"
IMAGE_ROOT = os.path.join(REPO_ROOT, "00_source")
VAL_PARQUET = os.path.join(REPO_ROOT, "runs", "a4_checkpointed", "r0_f0_s0", "val_r0_f0_s0.parquet")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "submission_test", "f3_battery")
MANIFEST_OUT = os.path.join(REPO_ROOT, "reports", "f3_reference_manifest.csv")

INPUT_SLUG = "stacked-barretts-esophagus-endoscopy-images"
INPUT_DIRNAME = "stacked-barretts-esophagus-endoscopy"
MODAL_W, MODAL_H = 637, 512


def write_inputs_json(interface_dir: str) -> None:
    payload = [{"interface": {"slug": INPUT_SLUG, "kind": "Image",
                              "relative_path": f"images/{INPUT_DIRNAME}"}}]
    os.makedirs(interface_dir, exist_ok=True)
    with open(os.path.join(interface_dir, "inputs.json"), "w") as fh:
        json.dump(payload, fh, indent=4)


def load_resize(fp: str) -> np.ndarray:
    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(IMAGE_ROOT, fp)))
    return cv2.resize(img, (MODAL_W, MODAL_H), interpolation=cv2.INTER_AREA)


def write_variant(name: str, build_fn) -> str:
    """build_fn(frames: List[HWC uint8]) -> (array, isVector, extra_write_kwargs)"""
    case_dir = os.path.join(OUT_DIR, name, "interface_0")
    img_dir = os.path.join(case_dir, "images", INPUT_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    mha_path = os.path.join(img_dir, "stack.mha")
    write_inputs_json(case_dir)
    return mha_path


def main() -> int:
    val = pd.read_parquet(VAL_PARQUET)
    assert len(val) == 617, f"expected 617 held-out rows, got {len(val)}"
    val = val.reset_index(drop=True)

    man_out = val[["filepath", "centre", "class_label", "label_int"]].copy()
    man_out.insert(0, "slice_index", range(len(man_out)))
    man_out.to_csv(MANIFEST_OUT, index=False)
    print(f"[f3] reference manifest -> {MANIFEST_OUT} ({len(man_out)} rows, "
         f"{int(val['label_int'].sum())} positive)")

    print("[f3] loading + resizing 617 held-out images (shared base for every variant)...")
    frames = [load_resize(fp) for fp in tqdm(val["filepath"], desc="load+resize",
                                             unit="img", file=sys.stderr)]
    base = np.stack(frames, axis=0)  # (Z, Y, X, 3) uint8, correct
    print(f"[f3] base stack: shape={base.shape} dtype={base.dtype}")

    variants = {}

    # 0. BASELINE -- correct construction, same as P1
    variants["baseline"] = (base.copy(), "vector_zyxc")

    # 1. channel order BGR instead of RGB
    variants["1_bgr"] = (base[..., ::-1].copy(), "vector_zyxc")

    # 2. channel axis first (Z,C,Y,X) instead of last -- D10's variant E,
    #    already known to misread slice count; built anyway for completeness
    chw_first = np.transpose(base, (0, 3, 1, 2))  # (Z, C, Y, X)
    variants["2_channel_first"] = (chw_first.copy(), "scalar_asis")

    # 3a. vertical flip (Y axis)
    variants["3a_vflip"] = (base[:, ::-1, :, :].copy(), "vector_zyxc")
    # 3b. horizontal flip (X axis)
    variants["3b_hflip"] = (base[:, :, ::-1, :].copy(), "vector_zyxc")

    # 4. transpose Y/X (per-slice)
    yx_t = np.transpose(base, (0, 2, 1, 3))  # (Z, X, Y, 3) -- Y/X swapped
    variants["4_transpose_yx"] = (yx_t.copy(), "vector_zyxc")

    # 5. source float32 [0,1] interpreted as uint8 [0,255] (no rescale)
    as_float01 = base.astype(np.float32) / 255.0
    misread_uint8 = as_float01.astype(np.uint8)  # truncates -- almost all zero
    variants["5_float01_as_uint8"] = (misread_uint8.copy(), "vector_zyxc")

    # 6. source 16-bit rescaled by observed max instead of the fixed /257
    src16 = (base.astype(np.uint16)) * 257  # simulate a genuine 16-bit source
    rescaled_by_max = np.empty_like(base)
    for i in range(src16.shape[0]):
        mx = src16[i].max()
        rescaled_by_max[i] = (src16[i].astype(np.float32) / max(mx, 1) * 255.0).astype(np.uint8)
    variants["6_16bit_rescaled_by_max"] = (rescaled_by_max.copy(), "vector_zyxc")

    # 7. slice axis transposed into a spatial axis (Z <-> Y)
    z_into_y = np.transpose(base, (1, 0, 2, 3))  # (Y, Z, X, 3)
    variants["7_slice_axis_transposed"] = (z_into_y.copy(), "vector_zyxc")

    paths = {}
    for name, (arr, mode) in variants.items():
        mha_path = write_variant(name, None)
        if mode == "vector_zyxc":
            img = sitk.GetImageFromArray(arr, isVector=True)
        else:  # scalar_asis -- write exactly as constructed, no vector reinterpretation
            img = sitk.GetImageFromArray(arr, isVector=False)
        sitk.WriteImage(img, mha_path)
        paths[name] = mha_path
        print(f"[f3] {name}: array shape={arr.shape} -> sitk size={img.GetSize()} "
             f"components={img.GetNumberOfComponentsPerPixel()} -> {mha_path}")

    with open(os.path.join(REPO_ROOT, "reports", "f3_variant_paths.json"), "w") as fh:
        json.dump(paths, fh, indent=2)
    print(f"\n[f3] all variants written. paths -> reports/f3_variant_paths.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
