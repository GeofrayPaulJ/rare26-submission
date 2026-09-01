"""N1 -- run all 4 arms (control + 3 normalisation variants) over both
cohorts, on the frozen shipping k=5 ensemble. GPU, real forward passes,
no shortcuts. Insertion point, arm definitions, reference statistic, and
acceptance rule are all pre-registered in reports/n1_pre_registration.md
-- written and locked before this script runs.

FOV crop is computed ONCE per source image (identical across arms -- only
the post-crop colour transform differs), via the exact production
detector (submission.rare26_infer.fov.detect_crop_box), matching what the
shipping container does on both cohorts.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/111_n1_run_arms.py'
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, ".")
from submission.rare26_infer.fov import detect_crop_box
from submission.rare26_infer.preprocess import resize_square, normalize_imagenet, CACHE_SIZE, IMAGE_SIZE
from submission.rare26_infer.model import build_ensemble

WEIGHTS = [f"runs/a4_checkpointed/r0_f{f}_s0/checkpoints/weights_fp32.pt" for f in range(5)]
BATCH = 32
DEVICE = "cuda"

REF_LAB_MEAN = np.array([109.6186117854143, 156.364550417569, 146.53343682956202])
REF_LAB_STD = np.array([40.47698684956508, 8.046969246363606, 5.645149344343892])


# --- arm transforms: FOV-cropped uint8 RGB -> uint8 RGB, same shape ---

def arm_control(img: np.ndarray) -> np.ndarray:
    return img


def arm_reinhard(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    src_mean = lab.mean(axis=(0, 1))
    src_std = lab.std(axis=(0, 1)) + 1e-6
    out = (lab - src_mean) * (REF_LAB_STD / src_std) + REF_LAB_MEAN
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def arm_greyworld(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float64)
    means = f.mean(axis=(0, 1))
    grey = means.mean()
    gain = np.clip(grey / np.maximum(means, 1e-6), 0.5, 2.0)
    out = np.clip(f * gain, 0, 255).astype(np.uint8)
    return out


def arm_zscore(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float64)
    mean = f.mean(axis=(0, 1))
    std = f.std(axis=(0, 1)) + 1e-6
    z = (f - mean) / std
    rescaled = np.clip(z * (255.0 / 6.0) + 127.5, 0, 255).astype(np.uint8)
    return rescaled


ARMS = {"N1-0": arm_control, "N1-a": arm_reinhard, "N1-b": arm_greyworld, "N1-c": arm_zscore}


def load_and_crop(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"))
    box, _used_fallback, _fq, _reason = detect_crop_box(arr)
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        return arr
    return arr[top:bottom, left:right, :]


def preprocess_final(cropped: np.ndarray, arm_fn) -> np.ndarray:
    transformed = arm_fn(cropped)
    cached = resize_square(transformed, CACHE_SIZE)
    final = resize_square(cached, IMAGE_SIZE)
    return normalize_imagenet(final)  # CHW float32


def run_cohort(paths: list, models: list, out_prefix: str) -> dict:
    print(f"[{out_prefix}] loading + cropping {len(paths)} images (FOV detect, once)...", flush=True)
    crops = []
    for p in tqdm(paths, desc=f"{out_prefix} crop", file=sys.stderr):
        crops.append(load_and_crop(p))

    results = {}
    for arm_name, arm_fn in ARMS.items():
        t0 = time.perf_counter()
        logits_sum = np.zeros(len(paths), dtype=np.float64)
        for bstart in range(0, len(paths), BATCH):
            batch_crops = crops[bstart:bstart + BATCH]
            chw = np.stack([preprocess_final(c, arm_fn) for c in batch_crops])
            x = torch.from_numpy(chw).to(DEVICE, non_blocking=True)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                member_out = [m(x).squeeze(-1).float() for m in models]
                out = torch.stack(member_out, dim=0).mean(dim=0)
            logits_sum[bstart:bstart + len(batch_crops)] = out.cpu().numpy().astype(np.float64)
        el = time.perf_counter() - t0
        results[arm_name] = logits_sum.tolist()
        print(f"[{out_prefix}] {arm_name}: {len(paths)} images in {el:.1f}s ({len(paths)/el:.1f} img/s)", flush=True)
    return results


def main() -> int:
    print("loading frozen 5-member ensemble...", flush=True)
    models = build_ensemble(WEIGHTS, device=DEVICE)
    print(f"loaded {len(models)} members\n", flush=True)

    # --- RARE25 pooled OOF cohort (3088 images) ---
    frames = []
    for f in range(5):
        frames.append(pd.read_parquet(f"runs/a4_checkpointed/r0_f{f}_s0/val_r0_f{f}_s0.parquet"))
    r25 = pd.concat(frames, ignore_index=True)
    r25_paths = [f"00_source/{fp}" for fp in r25["filepath"]]
    r25_labels = r25["label_int"].to_numpy().tolist()

    r25_logits = run_cohort(r25_paths, models, "RARE25")

    # --- EVC cohort (100 images) ---
    evc_df = pd.read_csv("manifests/evc_inventory.csv").sort_values("filepath").reset_index(drop=True)
    evc_paths = [f"02_evc/{fp}" for fp in evc_df["filepath"]]
    evc_labels = (evc_df["class_label"] == "cancer").astype(int).tolist()

    evc_logits = run_cohort(evc_paths, models, "EVC")

    out = {
        "rare25": {"filepaths": r25["filepath"].tolist(), "labels": r25_labels, "logits_by_arm": r25_logits},
        "evc": {"filepaths": evc_df["filepath"].tolist(), "labels": evc_labels, "logits_by_arm": evc_logits},
    }
    with open("reports/n1_raw_logits.json", "w") as fh:
        json.dump(out, fh)
    print("\nwrote reports/n1_raw_logits.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
