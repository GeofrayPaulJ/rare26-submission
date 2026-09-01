"""STEP -- automated EVC paste grid checks. CPU ONLY, NO TRAINING, NO GPU.

Replaces human review of reports/evc_paste_grid.png with objective per-tile
metrics, deterministically re-deriving scripts/52_evc_paste.py's exact 40
pastes (same SEED, same RNG call order) and checking each one for:

  - HALO RING DELTA: mean colour difference (Lab) between a thin ring just
    outside the paste mask and a thin ring just inside it, in the composited
    output. Targets the "faint blue-white halo" the human reviewer flagged
    on 2-3 tiles in reports/overnight_20260807_report.md (STEP 7) -- rg_ratio
    colour-matches the mean of the pasted patch, but says nothing about the
    seam itself.
  - PASTE-TO-FOV AREA RATIO: mask pixel count / the destination's own
    inscribed-square area (manifests/rare25_folds_v2.csv inner_* columns).
  - DESTINATION PATCH LUMINANCE/SATURATION: mean L (Lab) and S (HSV) of the
    original destination pixels under the paste footprint, BEFORE pasting --
    characterises what kind of tissue background each paste landed on.
  - CROP CONTAINMENT: whether the placed patch's bounding box is fully
    inside the destination's inscribed square (should hold by construction
    per 52_evc_paste.py's placement math; checked, not assumed).
  - LESION-ID REPEAT COUNTS: how many of the 40 tiles reuse each lesion
    (50 usable lesions, 40 tiles, so repeats are structurally possible from
    rng.integers(len(lesions)) sampling with replacement).

Every recomputed (lesion, dest, scale_frac, centre_xy) tuple is checked
against runs/evc_paste_preview/paste_manifest.json's recorded values; a
mismatch means the RNG replay drifted from the original run and is reported
as a HALT-worthy finding, not silently trusted.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/53_evc_grid_check.py'
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVC_ROOT = os.path.join(REPO_ROOT, "02_evc")
RARE_ROOT = os.path.join(REPO_ROOT, "00_source")
PREVIEW_DIR = os.path.join(REPO_ROOT, "runs", "evc_paste_preview")
MANIFEST_JSON = os.path.join(PREVIEW_DIR, "paste_manifest.json")
OUT_MD = os.path.join(REPO_ROOT, "reports", "evc_grid_check.md")
OUT_JSON = os.path.join(REPO_ROOT, "reports", "evc_grid_check.json")
OUT_GRID_OUTLINED = os.path.join(REPO_ROOT, "reports", "evc_paste_grid_outlined.png")
TILE = 320
GRID_COLS, GRID_ROWS = 8, 5

CONSENSUS_MIN_EXPERTS = 3
N_EXPERTS = 5
N_TILES = 40
SCALE_RANGE = (0.20, 0.45)
SEED = 20260807
MIN_LESION_PX = 2000
RING_WIDTH = 5              # px, dilate/erode radius for the halo ring
# PRIMARY threshold, pre-specified before any halo_ring_delta_lab value was
# read (matches the human reviewer's own qualitative "faint" bar) -- kept
# fixed regardless of what this grid's own data looks like, precisely so it
# cannot be tuned post hoc to match a desired flag count.
HALO_FLAG_DELTA_PRIMARY = 8.0
# SECONDARY, reported not gating: mean + 2*std of this grid's own 40 values.
# A relative-outlier detector, not an absolute halo detector -- it will
# flag the 3 most halo-like tiles out of any 40, whether or not a halo is
# present at all, which is exactly why it is secondary, not primary.
HALO_FLAG_Z = 2.0


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


def rg_match(patch_bgr: np.ndarray, mask: np.ndarray, target_rg: float) -> np.ndarray:
    sel = mask > 0
    if not sel.any():
        return patch_bgr
    mean_g = float(patch_bgr[..., 1][sel].mean())
    mean_r = float(patch_bgr[..., 2][sel].mean())
    if mean_g < 1e-6 or mean_r < 1e-6 or not np.isfinite(target_rg):
        return patch_bgr
    scale = target_rg / (mean_r / mean_g)
    out = patch_bgr.astype(np.float32)
    out[..., 2] *= scale
    return np.clip(out, 0, 255).astype(np.uint8)


def ring_stats(composite_bgr: np.ndarray, full_mask: np.ndarray, cx: int, cy: int,
                nw: int, nh: int) -> dict:
    """Mean Lab of a RING_WIDTH-px ring just outside vs just inside the paste
    mask boundary, evaluated in the composited output. full_mask is the paste
    mask placed into the full destination frame's coordinate system."""
    k = np.ones((RING_WIDTH * 2 + 1, RING_WIDTH * 2 + 1), np.uint8)
    dilated = cv2.dilate(full_mask, k, iterations=1)
    eroded = cv2.erode(full_mask, k, iterations=1)
    ring_out = (dilated > 0) & (full_mask == 0)
    ring_in = (full_mask > 0) & (eroded == 0)
    lab = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if not ring_out.any() or not ring_in.any():
        return {"halo_ring_delta_lab": None, "ring_out_px": int(ring_out.sum()),
                "ring_in_px": int(ring_in.sum())}
    out_mean = lab[ring_out].mean(axis=0)
    in_mean = lab[ring_in].mean(axis=0)
    delta = out_mean - in_mean
    mag = float(np.sqrt((delta ** 2).sum()))
    return {"halo_ring_delta_lab": round(mag, 3),
            "halo_dL": round(float(delta[0]), 3),
            "halo_da": round(float(delta[1]), 3),
            "halo_db": round(float(delta[2]), 3),
            "ring_out_px": int(ring_out.sum()), "ring_in_px": int(ring_in.sum())}


def dest_patch_stats(dst_bgr: np.ndarray, full_mask: np.ndarray) -> dict:
    sel = full_mask > 0
    if not sel.any():
        return {"dest_luminance_L": None, "dest_saturation_S": None}
    lab = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return {"dest_luminance_L": round(float(lab[..., 0][sel].mean()), 2),
            "dest_saturation_S": round(float(hsv[..., 1][sel].mean()), 2)}


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

    with open(MANIFEST_JSON) as fh:
        recorded = json.load(fh)
    recorded_by_tile = {r["tile"]: r for r in recorded["pastes"]}

    lesions = []
    for _, row in tqdm(cancer.iterrows(), total=len(cancer),
                       desc="extract lesions", unit="img", file=sys.stderr):
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
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        lesions.append({"stem": stem, "patch": img[y0:y1, x0:x1],
                        "mask": mask[y0:y1, x0:x1]})

    dest_rows = neg.sample(n=N_TILES, random_state=SEED).reset_index(drop=True)
    rows = []
    drift = []
    lesion_counts: dict[str, int] = {}
    outline_tiles: list[np.ndarray | None] = [None] * N_TILES

    for i, drow in enumerate(tqdm(dest_rows.itertuples(), total=N_TILES,
                                  desc="replay + check", unit="tile", file=sys.stderr)):
        les = lesions[int(rng.integers(len(lesions)))]
        dst = cv2.imread(os.path.join(RARE_ROOT, drow.filepath))
        rec = recorded_by_tile.get(i, {})
        row_out = {"tile": i, "lesion": les["stem"], "dest": drow.filepath,
                   "dest_centre": str(drow.centre)}

        if dst is None:
            row_out["error"] = "dest image unreadable"
            rows.append(row_out)
            continue

        il, it = int(drow.inner_left), int(drow.inner_top)
        ir, ib = int(drow.inner_right), int(drow.inner_bottom)
        inner_w, inner_h = ir - il, ib - it
        fov_area = inner_w * inner_h
        if inner_w < 64 or inner_h < 64:
            row_out["error"] = "inner square too small"
            rows.append(row_out)
            continue

        frac = float(rng.uniform(*SCALE_RANGE))
        target_max = frac * min(inner_w, inner_h)
        ph, pw = les["patch"].shape[:2]
        s = target_max / max(ph, pw)
        nw, nh = max(8, int(pw * s)), max(8, int(ph * s))
        patch = cv2.resize(les["patch"], (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(les["mask"], (nw, nh), interpolation=cv2.INTER_NEAREST)
        patch = rg_match(patch, mask, float(drow.rg_ratio))

        if inner_w - nw < 2 or inner_h - nh < 2:
            row_out["error"] = "placement window degenerate"
            rows.append(row_out)
            continue
        cx = il + nw // 2 + int(rng.integers(0, inner_w - nw))
        cy = it + nh // 2 + int(rng.integers(0, inner_h - nh))

        # --- manifest drift check (a priori, before any metric is computed) ---
        if rec:
            exp_scale, exp_xy = rec.get("scale_frac"), rec.get("centre_xy")
            if (rec.get("lesion") != les["stem"] or rec.get("dest") != drow.filepath
                    or abs(exp_scale - round(frac, 3)) > 1e-6
                    or exp_xy != [cx, cy]):
                drift.append(i)
                row_out["manifest_drift"] = True

        # --- crop containment: bbox fully inside inner square? ---
        x0, x1 = cx - nw // 2, cx - nw // 2 + nw
        y0, y1 = cy - nh // 2, cy - nh // 2 + nh
        contained = (x0 >= il) and (x1 <= ir) and (y0 >= it) and (y1 <= ib)
        row_out["crop_contained"] = bool(contained)
        row_out["containment_margin_px"] = int(min(x0 - il, ir - x1, y0 - it, ib - y1))

        # --- paste-to-FOV area ratio ---
        mask_area = int((mask > 0).sum())
        row_out["mask_area_px"] = mask_area
        row_out["fov_inner_area_px"] = int(fov_area)
        row_out["paste_to_fov_area_ratio"] = round(mask_area / fov_area, 5)

        # --- composite for halo + dest-patch stats ---
        try:
            composite = cv2.seamlessClone(patch, dst, mask, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error as exc:
            row_out["error"] = f"clone failed: {exc}"
            rows.append(row_out)
            continue

        full_mask = np.zeros(dst.shape[:2], dtype=np.uint8)
        my0, my1 = cy - nh // 2, cy - nh // 2 + nh
        mx0, mx1 = cx - nw // 2, cx - nw // 2 + nw
        full_mask[my0:my1, mx0:mx1] = mask

        row_out.update(ring_stats(composite, full_mask, cx, cy, nw, nh))
        row_out.update(dest_patch_stats(dst, full_mask))

        # --- outlined tile: paste boundary contour + containment box, for the
        # regenerated review grid (replaces the un-annotated 52_evc_paste.py one) ---
        outlined = composite.copy()
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outlined, contours, -1, (0, 255, 0), 2)   # green = paste boundary
        cv2.rectangle(outlined, (il, it), (ir, ib), (0, 165, 255), 1)  # orange = inscribed square
        tile_img = cv2.resize(outlined, (TILE, TILE), interpolation=cv2.INTER_AREA)
        cv2.putText(tile_img, f"#{i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        outline_tiles[i] = tile_img

        lesion_counts[les["stem"]] = lesion_counts.get(les["stem"], 0) + 1
        rows.append(row_out)

    for r in rows:
        r["lesion_repeat_count"] = lesion_counts.get(r.get("lesion"), 0)

    # --- halo flagging: PRIMARY is the fixed, pre-specified HALO_FLAG_DELTA_PRIMARY
    # (set before any value below was read). SECONDARY (mean + Z*std of this
    # grid's own values) is reported alongside, not used to gate anything --
    # recalibrating a threshold after seeing the data it will be applied to is
    # post hoc and was reverted for that reason. ---
    halo_vals = [r["halo_ring_delta_lab"] for r in rows if r.get("halo_ring_delta_lab") is not None]
    if halo_vals:
        halo_mean = float(np.mean(halo_vals))
        halo_std = float(np.std(halo_vals))
        halo_threshold_secondary = round(halo_mean + HALO_FLAG_Z * halo_std, 3)
    else:
        halo_mean = halo_std = halo_threshold_secondary = None
    for r in rows:
        if r.get("halo_ring_delta_lab") is None:
            continue
        if r["halo_ring_delta_lab"] >= HALO_FLAG_DELTA_PRIMARY:
            r["halo_flagged"] = True
        if halo_threshold_secondary is not None and r["halo_ring_delta_lab"] >= halo_threshold_secondary:
            r["halo_flagged_secondary"] = True

    # --- outlined grid, same 8x5 layout as 52_evc_paste.py's reports/evc_paste_grid.png ---
    blank = np.zeros((TILE, TILE, 3), dtype=np.uint8)
    filled_tiles = [t if t is not None else blank for t in outline_tiles]
    grid_img = np.vstack([np.hstack(filled_tiles[r * GRID_COLS:(r + 1) * GRID_COLS])
                          for r in range(GRID_ROWS)])
    cv2.imwrite(OUT_GRID_OUTLINED, grid_img)

    # --- write JSON ---
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump({"seed": SEED, "n_tiles": N_TILES, "ring_width_px": RING_WIDTH,
                   "halo_flag_threshold_primary_lab": HALO_FLAG_DELTA_PRIMARY,
                   "halo_flag_z_secondary": HALO_FLAG_Z, "halo_mean_lab": halo_mean,
                   "halo_std_lab": halo_std,
                   "halo_flag_threshold_secondary_lab": halo_threshold_secondary,
                   "manifest_drift_tiles": drift, "rows": rows}, fh, indent=2)

    # --- write markdown per-tile table ---
    cols = ["tile", "lesion", "dest_centre", "mask_area_px", "fov_inner_area_px",
            "paste_to_fov_area_ratio", "dest_luminance_L", "dest_saturation_S",
            "halo_ring_delta_lab", "crop_contained", "containment_margin_px",
            "lesion_repeat_count"]
    lines = ["# EVC paste grid -- automated checks (replaces human grid review)",
             "",
             f"Generated by `scripts/53_evc_grid_check.py`, deterministic replay of "
             f"`scripts/52_evc_paste.py` (SEED {SEED}), CPU only, zero GPU. "
             f"{len(rows)}/{N_TILES} tiles processed"
             + (f", **{len(drift)} tile(s) show manifest drift from the recorded "
                f"paste_manifest.json (RNG replay did not reproduce the original run "
                f"exactly -- treat downstream metrics for those tiles with caution)**"
                if drift else " -- RNG replay matched `paste_manifest.json` exactly on "
                "every tile, so recomputed geometry (mask area, placement) is trusted.")
             + "", ""]

    n_halo = sum(1 for r in rows if r.get("halo_flagged"))
    n_halo_secondary = sum(1 for r in rows if r.get("halo_flagged_secondary"))
    n_uncontained = sum(1 for r in rows if r.get("crop_contained") is False)
    n_errors = sum(1 for r in rows if "error" in r)
    lines += [f"**Summary: {n_halo} tile(s) flagged PRIMARY (halo ring delta >= "
              f"{HALO_FLAG_DELTA_PRIMARY}, pre-specified before any value below was "
              f"read); {n_halo_secondary} tile(s) flagged SECONDARY, reported not "
              f"gating (>= mean {round(halo_mean, 3)} + {HALO_FLAG_Z}*std "
              f"{round(halo_std, 3)} = {halo_threshold_secondary}, a relative-outlier "
              f"threshold recalibrated to this grid's own data -- see "
              f"`reports/evc_halo_null_check.md` for whether either threshold is "
              f"distinguishable from real-tissue boundary noise); "
              f"{n_uncontained} crop-containment violation(s), "
              f"{n_errors} tile(s) errored.**", ""]

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c == "halo_ring_delta_lab" and r.get("halo_flagged"):
                v = f"**{v}**"
            cells.append(str(v) if v is not None else "n/a")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Lesion-ID repeat counts across the 40 tiles", ""]
    lines.append("| lesion stem | times used |")
    lines.append("|---|---|")
    for stem, n in sorted(lesion_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {stem} | {n} |")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[evc-grid-check] wrote {OUT_MD} and {OUT_JSON}")
    print(f"[evc-grid-check] halo flagged={n_halo} uncontained={n_uncontained} "
          f"errors={n_errors} drift={len(drift)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
