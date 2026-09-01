"""Pixel pipeline, replicating src/data.py exactly.

THE TWO-STEP RESIZE IS NOT REDUNDANT. Training decoded each image once into a
RAM cache at cache_size=431, then resized that cache entry to image_size=384 at
read time. Both steps used cv2 with INTER_AREA on downscale. Going straight
from the native crop to 384 produces visibly different pixels -- a different
low-pass -- and therefore different logits. The model was trained on the output
of the 431 -> 384 path, so inference reproduces that path, intermediate and all.
tools/check_parity.py is what pins this: it is the test that fails if anyone
"simplifies" the double resize away.

Everything else here is likewise a copy of the training behaviour: the same
INTER_AREA/INTER_LINEAR selection rule, the same ImageNet constants, the same
CHW float32 output.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# cv2's own thread pool oversubscribes the CPU inside DataLoader workers, and
# single-threaded resize is both faster in aggregate and deterministic.
cv2.setNumThreads(0)

# src/config.py -- fixed convention, not tunable
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# src/config.py defaults, matching the trained checkpoint
CACHE_SIZE = 431
IMAGE_SIZE = 384

# src/data.py logs a warning below this; kept for parity of behaviour
_DEGENERATE_PX = 64


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """src/data.py::_resize_square -- INTER_AREA to downscale, INTER_LINEAR up."""
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (size <= h and size <= w) else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def crop_box(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Apply a (left, top, right, bottom) crop, clamped like src/data.py."""
    h, w = img.shape[:2]
    left, top, right, bottom = box
    left = int(max(0, min(w, left)))
    top = int(max(0, min(h, top)))
    right = int(max(0, min(w, right)))
    bottom = int(max(0, min(h, bottom)))
    if right <= left or bottom <= top:
        return img  # fully collapsed: whole frame, as the training loader did
    return img[top:bottom, left:right, :]


def normalize_imagenet(img: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32, [0,1] then ImageNet-standardised."""
    x = img.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    cache_size: int = CACHE_SIZE,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """HWC uint8 RGB + crop box -> CHW float32, exactly as training did it."""
    cropped = crop_box(img, box)
    cached = resize_square(cropped, cache_size)   # the RAM-cache step
    final = resize_square(cached, image_size)     # the read-time step
    return normalize_imagenet(final)


def to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce whatever one stack slice decoded to into HWC uint8 RGB.

    Test data comes from twelve unseen centres via a format we do not control,
    so a slice may arrive greyscale, RGBA, or already-RGB, and at a dtype the
    writer chose rather than the one we expect. Everything downstream assumes
    HWC uint8 RGB; this is the single place that assumption is enforced.
    """
    a = arr
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3:
        c = a.shape[2]
        if c == 1:
            a = np.repeat(a, 3, axis=2)
        elif c == 4:
            a = a[:, :, :3]     # drop alpha
        elif c != 3:
            raise ValueError(f"unsupported channel count {c} in slice of shape {arr.shape}")
    else:
        raise ValueError(f"unsupported slice shape {arr.shape}")

    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            # floats may be [0,1] or already [0,255]; decide on the observed range
            hi = float(np.nanmax(a)) if a.size else 0.0
            a = a * 255.0 if hi <= 1.0 + 1e-6 else a
        elif a.dtype == np.uint16:
            a = a.astype(np.float32) / 257.0
        a = np.clip(a, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(a)
