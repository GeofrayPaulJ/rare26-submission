"""T1 -- profile the data path, per-stage, on the real 200-slice platform probe.

REVISED after the first pass revealed a confound: `/proc/<pid>/io` showed
13.4 GiB read (1.7M read syscalls) against a ~335 MB pair of .mha files
while profiling from `/workspace` (this dev container's D:\-backed mount,
filesystem type `9p` -- the WSL2/Docker-Desktop bridge into Windows'
NTFS, confirmed via `df -T`). That bridge is a LOCAL DEV ARTIFACT, not
present on the real A10G evaluation host (native Linux storage) -- so
before blaming SimpleITK's reader, this profiles the SAME file from a
native path (this container's own overlay filesystem, `/root`, NOT
9p-backed) and reports both, so the bridge's own contribution is
isolated rather than baked silently into "the" answer.

  a) SimpleITK slice extraction, bridged vs native, uncompressed vs
     compressed -- read amplification (bytes read / file size) is the
     real discriminator for "does per-slice access decompress/reread
     the whole volume", not wall-clock alone (wall-clock alone cannot
     distinguish a slow filesystem from an inefficient reader).
  b) FOV geometry fitting per image -- cost per call, and whether the
     200 boxes are similar enough that fitting once per stack would be
     a valid shortcut.
  c) Resize / colour conversion / normalisation.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/83_t1_data_path_profile.py'
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

import numpy as np

REPO_ROOT = "/workspace/RARE26"
sys.path.insert(0, os.path.join(REPO_ROOT, "submission"))

from rare26_infer.fov import detect_crop_box  # noqa: E402
from rare26_infer.preprocess import (  # noqa: E402
    CACHE_SIZE, IMAGE_SIZE, crop_box, normalize_imagenet, resize_square)
from rare26_infer.stack import ItkStack  # noqa: E402

BRIDGED_MHA = os.path.join(
    REPO_ROOT, "runs", "submission_test", "platform_probe", "mha",
    "interface_0", "images", "stacked-barretts-esophagus-endoscopy", "stack.mha")
NATIVE_DIR = "/root/t1_native"
NATIVE_UNCOMPRESSED = os.path.join(NATIVE_DIR, "stack.mha")
NATIVE_COMPRESSED = os.path.join(NATIVE_DIR, "stack_compressed.mha")
REPORT_JSON = os.path.join(REPO_ROOT, "reports", "t1_data_path_profile.json")


def _self_io() -> dict:
    with open("/proc/self/io") as fh:
        stat = dict(ln.split(": ") for ln in fh.read().splitlines())
    return {k: int(v) for k, v in stat.items()}


def prepare_native() -> None:
    os.makedirs(NATIVE_DIR, exist_ok=True)
    if not os.path.exists(NATIVE_UNCOMPRESSED):
        shutil.copyfile(BRIDGED_MHA, NATIVE_UNCOMPRESSED)
    if not os.path.exists(NATIVE_COMPRESSED):
        import SimpleITK as sitk
        img = sitk.ReadImage(NATIVE_UNCOMPRESSED)
        w = sitk.ImageFileWriter()
        w.SetFileName(NATIVE_COMPRESSED)
        w.UseCompressionOn()
        w.Execute(img)


def profile_read(path: str, label: str, n: int) -> dict:
    io0 = _self_io()
    t0 = time.perf_counter()
    stack = ItkStack(path)
    open_s = time.perf_counter() - t0
    assert len(stack) == n, f"{label}: expected {n} slices, got {len(stack)}"

    per_slice = []
    for i in range(n):
        t = time.perf_counter()
        stack.read(i)
        per_slice.append(time.perf_counter() - t)
    per_slice = np.array(per_slice)
    io1 = _self_io()

    file_size = os.path.getsize(path)
    bytes_read = io1["rchar"] - io0["rchar"]
    syscalls = io1["syscr"] - io0["syscr"]

    return {
        "label": label, "path": path,
        "file_size_mb": file_size / 2**20,
        "open_header_seconds": open_s,
        "n_slices": n,
        "total_read_seconds": float(per_slice.sum()),
        "mean_read_seconds": float(per_slice.mean()),
        "median_read_seconds": float(np.median(per_slice)),
        "first_slice_seconds": float(per_slice[0]),
        "last_slice_seconds": float(per_slice[-1]),
        "bytes_read_total_mb": bytes_read / 2**20,
        "read_amplification_x": bytes_read / max(file_size, 1),
        "read_syscalls": syscalls,
        "read_syscalls_per_slice": syscalls / n,
    }


def profile_fov(arrs: list) -> dict:
    boxes, per_call = [], []
    for arr in arrs:
        t = time.perf_counter()
        box, used_fallback, fit_quality, reason = detect_crop_box(arr)
        per_call.append(time.perf_counter() - t)
        boxes.append(box)
    per_call = np.array(per_call)
    boxes_arr = np.array(boxes, dtype=float)
    box_std = boxes_arr.std(axis=0)
    box_mean = boxes_arr.mean(axis=0)
    cv = (box_std / np.maximum(box_mean, 1.0)).tolist()
    return {
        "n": len(arrs), "total_seconds": float(per_call.sum()),
        "mean_seconds": float(per_call.mean()), "median_seconds": float(np.median(per_call)),
        "std_seconds": float(per_call.std()),
        "box_mean_left_top_right_bottom": box_mean.tolist(),
        "box_std_left_top_right_bottom": box_std.tolist(),
        "box_coefficient_of_variation": cv,
        "boxes_nearly_identical": bool(np.all(np.array(cv) < 0.05)),
        "boxes": boxes,
    }


def profile_preprocess(arrs: list, boxes: list) -> dict:
    crop_t, cache_resize_t, final_resize_t, norm_t = [], [], [], []
    for arr, box in zip(arrs, boxes):
        t = time.perf_counter(); cropped = crop_box(arr, box); crop_t.append(time.perf_counter() - t)
        t = time.perf_counter(); cached = resize_square(cropped, CACHE_SIZE); cache_resize_t.append(time.perf_counter() - t)
        t = time.perf_counter(); final = resize_square(cached, IMAGE_SIZE); final_resize_t.append(time.perf_counter() - t)
        t = time.perf_counter(); normalize_imagenet(final); norm_t.append(time.perf_counter() - t)

    def stats(x):
        x = np.array(x)
        return {"total_seconds": float(x.sum()), "mean_seconds": float(x.mean())}

    return {"n": len(arrs), "crop": stats(crop_t),
           "cache_resize_637_to_431": stats(cache_resize_t),
           "final_resize_431_to_384": stats(final_resize_t),
           "normalize_imagenet": stats(norm_t)}


def main() -> int:
    n = 200
    print(f"[t1] preparing native (non-bridged) copies in {NATIVE_DIR} ...", flush=True)
    prepare_native()

    results = {}
    for path, label in [
        (BRIDGED_MHA, "bridged_uncompressed"),
        (NATIVE_UNCOMPRESSED, "native_uncompressed"),
        (NATIVE_COMPRESSED, "native_compressed"),
    ]:
        print(f"[t1] profiling {label} ({n} slices) ...", flush=True)
        r = profile_read(path, label, n)
        results[label] = r
        print(f"    mean={r['mean_read_seconds']*1000:.2f} ms/slice  "
             f"read_amp={r['read_amplification_x']:.1f}x  "
             f"syscalls/slice={r['read_syscalls_per_slice']:.0f}", flush=True)

    bridge_slowdown = (results["bridged_uncompressed"]["mean_read_seconds"] /
                       max(results["native_uncompressed"]["mean_read_seconds"], 1e-9))
    compression_slowdown = (results["native_compressed"]["mean_read_seconds"] /
                            max(results["native_uncompressed"]["mean_read_seconds"], 1e-9))
    print(f"\n[t1] bridge (9p) slowdown vs native: {bridge_slowdown:.2f}x")
    print(f"[t1] compression slowdown, native vs native: {compression_slowdown:.2f}x")

    # --- FOV + preprocess, from the NATIVE uncompressed copy (representative
    #     of a real Linux host's storage, not this dev box's 9p bridge) ---
    stack = ItkStack(NATIVE_UNCOMPRESSED)
    arrs = [stack.read(i) for i in range(len(stack))]
    fov = profile_fov(arrs)
    print(f"\n[t1] FOV: {fov['mean_seconds']*1000:.2f} ms/image mean, "
         f"boxes_nearly_identical={fov['boxes_nearly_identical']}")
    prep = profile_preprocess(arrs, fov["boxes"])
    print(f"[t1] preprocess stages (mean ms/image): "
         f"crop={prep['crop']['mean_seconds']*1000:.3f} "
         f"cache_resize={prep['cache_resize_637_to_431']['mean_seconds']*1000:.3f} "
         f"final_resize={prep['final_resize_431_to_384']['mean_seconds']*1000:.3f} "
         f"normalize={prep['normalize_imagenet']['mean_seconds']*1000:.3f}")

    fov_for_json = dict(fov)
    fov_for_json.pop("boxes")  # not JSON-critical, keeps the file small

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "filesystem_note": "bridged path is /workspace (9p, WSL2->Windows D:\\); "
                           "native path is /root (container overlay, ext4-class)",
        "stack_read": results,
        "bridge_slowdown_x": bridge_slowdown,
        "compression_slowdown_x": compression_slowdown,
        "fov": fov_for_json,
        "preprocess": prep,
    }
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[t1] written: {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
