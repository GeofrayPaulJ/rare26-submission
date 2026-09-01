"""STAGE 2 -- TTA sweep, as amended 2026-08-11.

Candidates (unchanged from pre-registration): identity / h-flip /
h+v-flip / h+v+2 rotations / 5-crop. Logit averaging. Metric: n=1
FPR@90R, both LOCO directions. Existing checkpoints only (LOCO seeds
0-9, both directions -- `runs/a4_checkpointed_loco/`), inference only.

EFFICIENCY: the flip-family candidates (identity/h-flip/h+v-flip/
h+v+2rotations) are NESTED subsets of one 6-view set
{orig, hflip, vflip, hvflip, rot90, rot270} computed from the final
384x384 tensor. 5-crop is a disjoint 5-view set computed from the
431x431 cache (4 corners + centre, matching preprocess.py's own
cache_size=431 > image_size=384 relationship -- this is the classic
5-crop-from-a-larger-canvas pattern already latent in this pipeline).
So every image needs exactly 11 forward passes, not 1+2+4+6+5=18, and
those 11 are shared across all 5 candidates by slicing+averaging in
logit space.

FURTHER EFFICIENCY: FOV crop + resize-to-431-cache does not depend on
model weights, so it is computed ONCE PER DIRECTION (not once per seed)
and reused across all 10 seeds for that direction -- the expensive
per-image work happens 3,095 times total, not 30,950.

TWO PROTOCOLS, per Axis 1 (T3 restored k=5):
  (a) n=1, per checkpoint, as originally pre-registered.
  (b) k=5, seeds 0-4 logit-averaged per LOCO direction, TTA views
      averaged in logit space WITHIN each member before members fuse.
Reported separately, never collapsed into one verdict.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/87_stage2_tta_sweep.py'
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

REPO_ROOT = "/workspace/RARE26"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "submission"))

from rare26_infer.fov import detect_crop_box  # noqa: E402
from rare26_infer.model import build_model  # noqa: E402
from rare26_infer.preprocess import (  # noqa: E402
    CACHE_SIZE, IMAGE_SIZE, crop_box, normalize_imagenet, resize_square)
from src.folds import get_holdout_split  # noqa: E402

MANIFEST = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
IMAGE_ROOT = os.path.join(REPO_ROOT, "00_source")
LOCO_DIR = os.path.join(REPO_ROOT, "runs", "a4_checkpointed_loco")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "stage2_tta")
REPORT_JSON = os.path.join(REPO_ROOT, "reports", "f_stage2_tta_results.json")

SEEDS = list(range(10))
K5_SEEDS = [0, 1, 2, 3, 4]
DIRECTIONS = [1, 2]

CANDIDATES = ["identity", "h-flip", "h+v-flip", "h+v+2 rotations", "5-crop"]
# which of the 11 raw view indices each candidate averages.
# indices 0-5: flip-family (orig, hflip, vflip, hvflip, rot90, rot270)
# indices 6-10: crop-family (TL, TR, BL, BR, centre)
CANDIDATE_VIEW_IDX = {
    "identity": [0],
    "h-flip": [0, 1],
    "h+v-flip": [0, 1, 2, 3],
    "h+v+2 rotations": [0, 1, 2, 3, 4, 5],
    "5-crop": [6, 7, 8, 9, 10],
}
N_VIEWS = 11
BATCH_IMAGES = 4  # x11 views = 44 tensors/batch, comfortable at 384px


def load_rgb(fp: str) -> np.ndarray:
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(IMAGE_ROOT, fp)))


def build_cache(fp: str) -> np.ndarray:
    """FOV crop -> resize to the 431 cache. Model-independent; computed once
    per direction and reused across all 10 seeds."""
    arr = load_rgb(fp)
    box, used_fallback, fit_quality, reason = detect_crop_box(arr)
    cropped = crop_box(arr, box)
    return resize_square(cropped, CACHE_SIZE)  # 431x431x3 uint8


def make_11_views(cached: np.ndarray) -> np.ndarray:
    """cached: 431x431x3 uint8 -> (11, 3, 384, 384) float32, normalised."""
    final = resize_square(cached, IMAGE_SIZE)  # 384x384x3, the normal path
    flip_views = [
        final,
        final[:, ::-1, :],
        final[::-1, :, :],
        final[::-1, ::-1, :],
        np.rot90(final, k=1),
        np.rot90(final, k=3),
    ]
    margin = CACHE_SIZE - IMAGE_SIZE  # 47
    crop_offsets = [(0, 0), (0, margin), (margin, 0), (margin, margin),
                    (margin // 2, margin // 2)]
    crop_views = [cached[t:t + IMAGE_SIZE, l:l + IMAGE_SIZE, :] for t, l in crop_offsets]

    views = flip_views + crop_views
    out = np.empty((N_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, v in enumerate(views):
        out[i] = normalize_imagenet(np.ascontiguousarray(v))
    return out


def fpr_at_90_recall(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    recall = tp / n_pos
    idx = np.searchsorted(recall, 0.90)
    if idx >= len(recall):
        idx = len(recall) - 1
    return float(fp[idx] / n_neg)


def prepare_direction(centre: int) -> tuple:
    """Held-out filepaths/labels + the 431-cache array, computed once."""
    man = pd.read_csv(MANIFEST)
    _, test_fps = get_holdout_split(centre, MANIFEST)
    sub = man.set_index("filepath").loc[test_fps].reset_index()
    labels = (sub["class_label"] == "neoplasia").astype(int).to_numpy()

    cache_path = os.path.join(OUT_DIR, f"cache_c{centre}.npy")
    fp_path = os.path.join(OUT_DIR, f"cache_c{centre}_filepaths.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(cache_path) and os.path.exists(fp_path):
        with open(fp_path) as fh:
            saved_fps = json.load(fh)
        if saved_fps == test_fps:
            cache = np.load(cache_path)
            print(f"[stage2] c{centre}: reusing cached 431-crops ({len(test_fps)} images)")
            return test_fps, labels, cache

    print(f"[stage2] c{centre}: building 431-crops for {len(test_fps)} images "
         f"(model-independent, computed once)...")
    cache = np.empty((len(test_fps), CACHE_SIZE, CACHE_SIZE, 3), dtype=np.uint8)
    for i, fp in enumerate(tqdm(test_fps, desc=f"c{centre} cache", unit="img", file=sys.stderr)):
        cache[i] = build_cache(fp)
    np.save(cache_path, cache)
    with open(fp_path, "w") as fh:
        json.dump(test_fps, fh)
    return test_fps, labels, cache


def run_checkpoint(weights_path: str, cache: np.ndarray, device: str) -> np.ndarray:
    """-> (n_images, 5 candidates) logit array for this checkpoint."""
    model = build_model(weights_path, device=device)
    n = len(cache)
    candidate_logits = np.zeros((n, len(CANDIDATES)), dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, n, BATCH_IMAGES):
            end = min(start + BATCH_IMAGES, n)
            batch_views = np.stack([make_11_views(cache[i]) for i in range(start, end)], axis=0)
            bsz = end - start
            flat = torch.from_numpy(batch_views.reshape(bsz * N_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(flat).squeeze(-1).float()
            out = out.cpu().numpy().reshape(bsz, N_VIEWS)  # raw per-view logits
            for ci, cname in enumerate(CANDIDATES):
                idx = CANDIDATE_VIEW_IDX[cname]
                candidate_logits[start:end, ci] = out[:, idx].mean(axis=1)

    del model
    torch.cuda.empty_cache()
    return candidate_logits


def apply_bar(per_seed_results: dict, direction_labels: dict) -> dict:
    """per_seed_results: {centre: {seed: (n_images, 5) logit array}}
    Protocol (a): candidate beats identity per seed per direction; bar is
    >=8/10 seeds per direction, BOTH directions."""
    verdict = {}
    for ci, cname in enumerate(CANDIDATES):
        if cname == "identity":
            continue
        wins = {1: 0, 2: 0}
        for centre in DIRECTIONS:
            labels = direction_labels[centre]
            for seed in SEEDS:
                logits = per_seed_results[centre][seed]
                fpr_cand = fpr_at_90_recall(labels, logits[:, ci])
                fpr_id = fpr_at_90_recall(labels, logits[:, 0])
                if fpr_cand < fpr_id:
                    wins[centre] += 1
        bar_c1 = wins[1] >= 8
        bar_c2 = wins[2] >= 8
        if bar_c1 and bar_c2:
            v = "ACCEPT"
        elif bar_c1 != bar_c2:
            v = "REJECT (mixed)"
        else:
            v = "REJECT"
        verdict[cname] = {"wins_c1": wins[1], "wins_c2": wins[2], "verdict": v}
    return verdict


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    per_seed_results = {}   # {centre: {seed: (n,5) array}}
    direction_labels = {}
    direction_fps = {}

    for centre in DIRECTIONS:
        fps, labels, cache = prepare_direction(centre)
        direction_labels[centre] = labels
        direction_fps[centre] = fps
        per_seed_results[centre] = {}
        for seed in SEEDS:
            wp = os.path.join(LOCO_DIR, f"loco_c{centre}_s{seed}", "checkpoints", "weights_fp32.pt")
            print(f"[stage2] c{centre} seed{seed}: running 11-view inference "
                 f"over {len(cache)} images...", flush=True)
            t_ckpt = time.perf_counter()
            logits = run_checkpoint(wp, cache, device)
            per_seed_results[centre][seed] = logits
            print(f"[stage2] c{centre} seed{seed}: done in {time.perf_counter()-t_ckpt:.1f}s", flush=True)
            np.save(os.path.join(OUT_DIR, f"logits_c{centre}_s{seed}.npy"), logits)

    # --- protocol (a): n=1 ---
    verdict_a = apply_bar(per_seed_results, direction_labels)
    table_a = []
    for centre in DIRECTIONS:
        labels = direction_labels[centre]
        for seed in SEEDS:
            logits = per_seed_results[centre][seed]
            row = {"direction": centre, "seed": seed}
            for ci, cname in enumerate(CANDIDATES):
                row[cname] = fpr_at_90_recall(labels, logits[:, ci])
            table_a.append(row)

    # --- protocol (b): k=5, seeds 0-4 logit-averaged per direction ---
    table_b = []
    k5_logits = {}
    for centre in DIRECTIONS:
        stacked = np.stack([per_seed_results[centre][s] for s in K5_SEEDS], axis=0)  # (5, n, 5cand)
        ens = stacked.mean(axis=0)  # (n, 5cand) -- logit-averaged across members
        k5_logits[centre] = ens
        labels = direction_labels[centre]
        row = {"direction": centre}
        for ci, cname in enumerate(CANDIDATES):
            row[cname] = fpr_at_90_recall(labels, ens[:, ci])
        table_b.append(row)

    verdict_b = {}
    for ci, cname in enumerate(CANDIDATES):
        if cname == "identity":
            continue
        beats = {}
        for centre in DIRECTIONS:
            labels = direction_labels[centre]
            fpr_cand = fpr_at_90_recall(labels, k5_logits[centre][:, ci])
            fpr_id = fpr_at_90_recall(labels, k5_logits[centre][:, 0])
            beats[centre] = fpr_cand < fpr_id
        if beats[1] and beats[2]:
            v = "ACCEPT"
        elif beats[1] != beats[2]:
            v = "REJECT (mixed)"
        else:
            v = "REJECT"
        verdict_b[cname] = {"beats_identity_c1": beats[1], "beats_identity_c2": beats[2], "verdict": v}

    result = {
        "protocol_a_n1": {"per_seed_fpr90": table_a, "verdict": verdict_a},
        "protocol_b_k5": {"per_direction_fpr90": table_b, "verdict": verdict_b},
        "wall_seconds": time.perf_counter() - t0,
    }
    with open(REPORT_JSON, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\n[stage2] written: {REPORT_JSON}")
    print(f"[stage2] total wall time: {(time.perf_counter()-t0)/60:.1f} min")
    print(f"\nprotocol (a) n=1 verdicts: {json.dumps(verdict_a, indent=2)}")
    print(f"\nprotocol (b) k=5 verdicts: {json.dumps(verdict_b, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
