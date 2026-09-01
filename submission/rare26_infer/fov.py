"""FOV circle detection, ported from scripts/05_fov_crop.py.

``fit_fov_circle`` and ``inner_square`` below are a VERBATIM port of the
functions in scripts/05_fov_crop.py -- same maths, same rounding, same clamping.
They are copied rather than imported because the container must not depend on
the training repo, and rather than reimplemented because the manifest geometry
every training crop was taken from came out of exactly this code. Any
"improvement" here silently desynchronises inference from training.

WHAT IS NEW HERE is the fallback, and only the fallback. The detector was tuned
on two Dutch centres; the test set comes from twelve unseen centres and will
present FOV geometry it has never seen. When detection fails outright, or
returns a circle whose fit_quality falls below the 1st percentile observed
across the 3,095 training images, we stop trusting it and take a centred square
instead. Every such image is counted and logged -- a fallback rate that is
higher than a few percent is a signal the test geometry is genuinely different,
and that is something you want printed in the run log rather than silently
absorbed into the predictions.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

# Brightness above which a pixel counts as content rather than letterbox black.
# scripts/05_fov_crop.py default; the manifest was built with it.
DEFAULT_THRESHOLD = 15

# 1st percentile of fit_quality over all 3,095 training images with detected
# geometry (manifests/rare25_folds_v2.csv). Distribution for reference:
#   min 0.6468 | p0.5 0.8838 | p1 0.9040 | p2 0.9267 | p5 0.9482 | median 0.9959
# A circle fitting its own disk worse than 99% of the training set is not a
# circle we are willing to crop on.
FIT_QUALITY_FLOOR = 0.9040

# Side of the fallback crop, as a fraction of the shorter image side. Chosen to
# sit inside the FOV circle for a typical frame (the circle's diameter equals
# the frame height, so an inscribed square would be ~0.707 * height); 0.75 is
# slightly more generous and keeps a little peripheral tissue.
FALLBACK_SIDE_FRAC = 0.75


# ---------------------------------------------------------------------------
# VERBATIM from scripts/05_fov_crop.py -- do not edit
# ---------------------------------------------------------------------------
def fit_fov_circle(mask: np.ndarray):
    """Largest connected component -> (centre_x, centre_y, radius, fit_quality, area_pct)."""
    h, w = mask.shape
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return None

    sizes = ndimage.sum(mask, labeled, index=range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    comp_mask = labeled == largest_label
    area = float(sizes[largest_label - 1])

    ys, xs = np.nonzero(comp_mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    radius = math.sqrt(area / math.pi)

    yy, xx = np.ogrid[:h, :w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    disk_area = int(disk.sum())
    fit_quality = float((mask & disk).sum() / disk_area) if disk_area else 0.0

    return cx, cy, radius, fit_quality, 100.0 * area / (w * h)


def inner_square(cx: float, cy: float, radius: float, width: int, height: int):
    """Largest axis-aligned square inscribed in the fitted circle, clamped to bounds."""
    side = math.floor(radius * math.sqrt(2))
    left = round(cx - side / 2)
    top = round(cy - side / 2)
    right = left + side
    bottom = top + side

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    return left, top, right, bottom
# ---------------------------------------------------------------------------
# end verbatim block
# ---------------------------------------------------------------------------


def centred_square(width: int, height: int,
                   frac: float = FALLBACK_SIDE_FRAC) -> Tuple[int, int, int, int]:
    """Centred square of side ``frac * min(H, W)``. The fallback crop."""
    side = int(math.floor(frac * min(width, height)))
    side = max(1, side)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def detect_crop_box(
    arr: np.ndarray,
    threshold: int = DEFAULT_THRESHOLD,
    fit_quality_floor: float = FIT_QUALITY_FLOOR,
) -> Tuple[Tuple[int, int, int, int], bool, Optional[float], str]:
    """Crop box for one HWC uint8 RGB frame.

    Returns ``(box, used_fallback, fit_quality, reason)`` where box is
    ``(left, top, right, bottom)``. ``reason`` is "" on the detected path and
    names the failure mode otherwise, so the run log can distinguish "no
    content found" from "circle fit was poor".
    """
    h, w = arr.shape[:2]
    mask = arr.mean(axis=2) > threshold

    fit = fit_fov_circle(mask)
    if fit is None:
        return centred_square(w, h), True, None, "no non-black pixels found"

    cx, cy, radius, fit_quality, _area_pct = fit
    if fit_quality < fit_quality_floor:
        return (centred_square(w, h), True, fit_quality,
                f"fit_quality {fit_quality:.4f} < floor {fit_quality_floor:.4f}")

    left, top, right, bottom = inner_square(cx, cy, radius, w, h)
    # A collapsed box would make the crop empty; the training loader guarded
    # this by falling back to the whole frame, but a centred square is the
    # better answer here and keeps the aspect ratio the model was trained on.
    if right <= left or bottom <= top:
        return (centred_square(w, h), True, fit_quality,
                f"degenerate inner box ({right - left}x{bottom - top})")

    return (left, top, right, bottom), False, fit_quality, ""
