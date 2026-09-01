"""EVC halo-ring-delta null distributions. CPU ONLY, NO GPU.

The halo_ring_delta_lab metric (scripts/53_evc_grid_check.py) measures Lab
colour discontinuity at a mask boundary. On its own, a raw threshold on
that number (8.0 pre-specified, or a post-hoc mean+2std) says nothing about
whether pasted composites are actually more discontinuous at their seam
than ANY boundary in a real, untouched endoscopy image is -- tissue itself
has natural colour discontinuities (specular highlights, vessel edges, fold
shadows). Two null distributions, same metric, same RING_WIDTH:

  NULL A -- boundary of a REAL lesion, in situ, never composited. Reuses
  the exact same 50 expert-consensus lesion masks scripts/52_evc_paste.py
  extracts, measured at their ORIGINAL location in their ORIGINAL EVC
  source image (no clone, no paste) -- "what does a genuine lesion
  boundary look like under this metric."

  NULL B -- an arbitrary blob-shaped boundary at a RANDOM location on a
  REAL, UNTOUCHED RARE25 negative image (no lesion, no paste). One draw
  per paste tile (n=40, matched), mask shape/size resampled from the same
  50-lesion pool so area is comparable to NULL A and to the actual pastes.
  "How much ring delta do you get from ANY blob boundary in real tissue,
  with nothing pasted."

If the 40 pasted-composite deltas (already computed by 53_evc_grid_check.py)
sit inside these two null ranges, a threshold flagging them is flagging
normal image structure, not a paste artefact. If they sit clearly above
both, that is real evidence the seam is more discontinuous than real
tissue boundaries ever are.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/55_evc_halo_null.py'
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVC_ROOT = os.path.join(REPO_ROOT, "02_evc")
RARE_ROOT = os.path.join(REPO_ROOT, "00_source")
GRID_CHECK_JSON = os.path.join(REPO_ROOT, "reports", "evc_grid_check.json")
OUT_MD = os.path.join(REPO_ROOT, "reports", "evc_halo_null_check.md")
OUT_JSON = os.path.join(REPO_ROOT, "reports", "evc_halo_null_check.json")

CONSENSUS_MIN_EXPERTS = 3
N_EXPERTS = 5
MIN_LESION_PX = 2000
RING_WIDTH = 5
N_NULL_B = 40
SEED = 20260808


def consensus_mask(stem: str) -> np.ndarray | None:
    votes = None
    for e in range(1, N_EXPERTS + 1):
        p = os.path.join(EVC_ROOT, "annotations_bmp", f"{stem}_exp{e}.bmp")
        if not os.path.exists(p):
            return None
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        v = (m > 127).astype(np.uint8)
        votes = v if votes is None else votes + v
    return ((votes >= CONSENSUS_MIN_EXPERTS) * 255).astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == biggest) * 255).astype(np.uint8)


def ring_delta(img_bgr: np.ndarray, full_mask: np.ndarray) -> float | None:
    k = np.ones((RING_WIDTH * 2 + 1, RING_WIDTH * 2 + 1), np.uint8)
    dilated = cv2.dilate(full_mask, k, iterations=1)
    eroded = cv2.erode(full_mask, k, iterations=1)
    ring_out = (dilated > 0) & (full_mask == 0)
    ring_in = (full_mask > 0) & (eroded == 0)
    if not ring_out.any() or not ring_in.any():
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = lab[ring_out].mean(axis=0) - lab[ring_in].mean(axis=0)
    return float(np.sqrt((delta ** 2).sum()))


def main() -> int:
    rng = np.random.default_rng(SEED)

    inv = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "evc_inventory.csv"))
    cancer = inv[inv["class_label"] == "cancer"].reset_index(drop=True)
    rare = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))
    neg = rare[(rare["class_label"] == "non-neoplastic")
               & (rare["keep_for_training"] == True)  # noqa: E712
               & rare["inner_left"].notna()].reset_index(drop=True)
    if len(neg) == 0:
        labels = rare["class_label"].unique().tolist()
        neg_label = [l for l in labels if "non" in l.lower()][0]
        neg = rare[(rare["class_label"] == neg_label)
                   & rare["inner_left"].notna()].reset_index(drop=True)

    # --- NULL A: real lesion, in situ, original EVC image, no compositing ---
    null_a = []
    lesion_pool = []  # (mask) for NULL B's shape resampling
    for _, row in tqdm(cancer.iterrows(), total=len(cancer), desc="NULL A: in-situ lesions", unit="img"):
        stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
        mask = consensus_mask(stem)
        if mask is None:
            continue
        mask = largest_component(mask)
        if int((mask > 0).sum()) < MIN_LESION_PX:
            continue
        img = cv2.imread(os.path.join(EVC_ROOT, row["filepath"]))
        if img is None or img.shape[:2] != mask.shape[:2]:
            continue
        d = ring_delta(img, mask)
        if d is not None:
            null_a.append(d)
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        lesion_pool.append(mask[y0:y1, x0:x1])

    # --- NULL B: arbitrary blob shape, random location, real untouched negative ---
    null_b = []
    for _ in tqdm(range(N_NULL_B), desc="NULL B: random blob on real negatives", unit="draw"):
        drow = neg.sample(n=1, random_state=int(rng.integers(0, 2**31))).iloc[0]
        dst = cv2.imread(os.path.join(RARE_ROOT, drow["filepath"]))
        if dst is None:
            continue
        il, it = int(drow["inner_left"]), int(drow["inner_top"])
        ir, ib = int(drow["inner_right"]), int(drow["inner_bottom"])
        inner_w, inner_h = ir - il, ib - it
        if inner_w < 64 or inner_h < 64:
            continue
        shape = lesion_pool[int(rng.integers(len(lesion_pool)))]
        frac = float(rng.uniform(0.20, 0.45))
        target_max = frac * min(inner_w, inner_h)
        ph, pw = shape.shape[:2]
        s = target_max / max(ph, pw)
        nw, nh = max(8, int(pw * s)), max(8, int(ph * s))
        if inner_w - nw < 2 or inner_h - nh < 2:
            continue
        mask_r = cv2.resize(shape, (nw, nh), interpolation=cv2.INTER_NEAREST)
        cx = il + nw // 2 + int(rng.integers(0, inner_w - nw))
        cy = it + nh // 2 + int(rng.integers(0, inner_h - nh))
        full_mask = np.zeros(dst.shape[:2], dtype=np.uint8)
        full_mask[cy - nh // 2:cy - nh // 2 + nh, cx - nw // 2:cx - nw // 2 + nw] = mask_r
        d = ring_delta(dst, full_mask)
        if d is not None:
            null_b.append(d)

    with open(GRID_CHECK_JSON) as fh:
        grid = json.load(fh)
    pasted = [r["halo_ring_delta_lab"] for r in grid["rows"] if r.get("halo_ring_delta_lab") is not None]

    def summary(vals):
        a = np.array(vals)
        return {"n": len(a), "mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
                "median": round(float(np.median(a)), 3), "p95": round(float(np.percentile(a, 95)), 3),
                "max": round(float(a.max()), 3)}

    s_paste, s_a, s_b = summary(pasted), summary(null_a), summary(null_b)

    # where do the pasted values fall relative to each null's distribution?
    frac_above_a_p95 = float(np.mean(np.array(pasted) > s_a["p95"]))
    frac_above_b_p95 = float(np.mean(np.array(pasted) > s_b["p95"]))

    out = {"pasted": s_paste, "null_a_in_situ_lesion": s_a, "null_b_random_blob": s_b,
           "frac_pasted_above_null_a_p95": round(frac_above_a_p95, 3),
           "frac_pasted_above_null_b_p95": round(frac_above_b_p95, 3),
           "primary_threshold": 8.0, "secondary_threshold": grid.get("halo_flag_threshold_secondary_lab")}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)

    lines = ["# EVC halo ring-delta -- null distributions",
             "",
             "`scripts/55_evc_halo_null.py`, CPU only, zero GPU. Same "
             "`halo_ring_delta_lab` metric as `scripts/53_evc_grid_check.py`, "
             "computed on two null cases instead of pasted composites.",
             "",
             "| distribution | n | mean | std | median | p95 | max |",
             "|---|---|---|---|---|---|---|",
             f"| pasted composites (the 40 tiles) | {s_paste['n']} | {s_paste['mean']} | {s_paste['std']} | {s_paste['median']} | {s_paste['p95']} | {s_paste['max']} |",
             f"| NULL A: real lesion, in situ, no paste | {s_a['n']} | {s_a['mean']} | {s_a['std']} | {s_a['median']} | {s_a['p95']} | {s_a['max']} |",
             f"| NULL B: random blob, real untouched negative | {s_b['n']} | {s_b['mean']} | {s_b['std']} | {s_b['median']} | {s_b['p95']} | {s_b['max']} |",
             "",
             f"**{round(frac_above_a_p95 * 100)}% of pasted tiles exceed NULL A's 95th percentile "
             f"({s_a['p95']}); {round(frac_above_b_p95 * 100)}% exceed NULL B's "
             f"({s_b['p95']}).**",
             "",
             f"PRIMARY threshold (8.0, pre-specified): "
             + ("inside both null ranges -- flags tiles a real tissue boundary "
                "could also produce, not distinctively a paste artefact"
                if 8.0 <= max(s_a["p95"], s_b["p95"]) else
                "above both nulls' 95th percentile -- tiles flagged at this bar "
                "are more discontinuous than real tissue boundaries typically are"),
             ".",
             "",
             f"SECONDARY threshold ({out['secondary_threshold']}, mean+2std of the "
             f"pasted grid's own values): "
             + ("also inside both null ranges -- confirms it is a relative-outlier "
                "detector (finds the most halo-like tiles in any grid of 40) rather "
                "than an absolute paste-artefact detector"
                if out['secondary_threshold'] and out['secondary_threshold'] <= max(s_a["p95"], s_b["p95"])
                else "above both nulls' 95th percentile"),
             "."]

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[halo-null] pasted={s_paste} null_a={s_a} null_b={s_b}")
    print(f"[halo-null] wrote {OUT_MD} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
