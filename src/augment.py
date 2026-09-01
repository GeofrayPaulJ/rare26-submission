"""Augmentation: baseline transforms plus the domain-generalisation stack.

The pipeline is five ordered stages, each behind a master gate in AugConfig and
each transform inside it behind its own probability:

  baseline      horizontal flip; synthetic redaction boxes        (ON)
  colour        per-channel gamma; white balance; hue/saturation  (`photometric`)
  illumination  vignette; linear & radial ramps; speculars        (`photometric`)
  optical       distortion; motion blur; defocus; sharpening      (`optical`)
  sensor        Gaussian noise; Poisson noise; resolution loss    (`sensor`)
  compression   JPEG re-encode                                    (`compression`)
  normalise     ImageNet standardisation                          (ON)

WHY THE ORDER IS WHAT IT IS. It follows the physical path light takes to a
stored file: the mucosa is lit and reflects (illumination, speculars), the lens
and its motion smear it (optical), the sensor quantises it and adds shot and
read noise (sensor), and the processor re-encodes it (compression). Applying
JPEG before noise, or sharpening before blur, would produce artefacts in an
order no endoscope stack can produce. Colour comes first because it stands in
for the illuminant and the sensor's colour response, both upstream of the lens.

WHY THE MAGNITUDES ARE LARGE. Defaults sit at roughly 2-3x what the common
libraries ship. The evaluation set is twelve unseen hospitals: different scope
manufacturers, processors, light sources and export pipelines. Jitter tuned to
keep ImageNet photographs recognisable is not calibrated to that. See
reports/aug_samples.png for what these settings actually look like -- if the
images stop reading as endoscopy, the magnitudes are wrong.

RNG CONTRACT -- LOAD-BEARING, DO NOT BREAK.
Every random decision here draws from the single ``rng`` argument, which
src/data.py seeds from (config.seed, epoch, item index) and nothing else. No
transform may touch numpy's global RNG, ``random``, or torch. That is what
makes a given (seed, epoch, index) produce a byte-identical tensor in the main
process, in a fresh DataLoader worker, and in a worker that has already been
alive for ten epochs under persistent_workers -- which in turn is what lets a
resumed run retrace an uninterrupted one. tests/test_augment_repro.py pins it.

Corollary, equally load-bearing: a transform whose probability check FAILS must
consume exactly one draw, and a whole stage whose master gate is OFF must
consume NONE. The baseline transforms are drawn before any of the new stages,
so switching this stack on cannot shift the flip or box draws of a run that had
it off. That is what makes the baseline arm of a factorial screen bit-identical
to the runs made before this file existed.

Everything between decode and normalisation operates on an HWC uint8 RGB array.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .config import AugConfig, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Empirical redaction-box distribution
# ---------------------------------------------------------------------------
# The test set carries no de-identification bars, but 765 training images do,
# and their presence correlates with hospital (center_2 54.4% vs center_1
# 14.1%). Stamping synthetic boxes on *every* image with equal probability
# makes "box present" uninformative, so the model cannot use it as a shortcut.
#
# We never invent box parameters: the count distribution and the per-box
# width/height (as fractions of the image side) are read straight from
# redaction_check.csv, from the geometry of the largest detected box on each
# redacted image.
@dataclass(frozen=True)
class RedactionDist:
    counts: np.ndarray        # candidate box counts, e.g. [1, 2, 3]
    count_probs: np.ndarray   # matching probabilities, sums to 1
    w_frac: np.ndarray        # empirical box widths  as fraction of image W
    h_frac: np.ndarray        # empirical box heights as fraction of image H

    def __post_init__(self) -> None:
        assert self.counts.shape == self.count_probs.shape
        assert abs(self.count_probs.sum() - 1.0) < 1e-9
        assert self.w_frac.shape == self.h_frac.shape
        assert len(self.w_frac) > 0


@lru_cache(maxsize=8)
def load_redaction_dist(csv_path: str, max_boxes: int = 3) -> RedactionDist:
    """Build the empirical box distribution from redaction_check.csv.

    Counts above `max_boxes` are folded into `max_boxes` (the spec draws 1-3
    boxes). Box geometry comes from largest_{left,top,right,bottom} normalised
    by image width/height. Cached per path so the file is read once per process.
    """
    df = pd.read_csv(csv_path)
    r = df[df["has_redaction"] == True].copy()  # noqa: E712 (explicit bool match)
    if len(r) == 0:
        raise ValueError(f"no redacted rows found in {csv_path}")

    # --- count distribution, clamped to [1, max_boxes] ---
    counts = r["redaction_count"].clip(lower=1, upper=max_boxes).astype(int)
    vc = counts.value_counts().sort_index()
    count_vals = vc.index.to_numpy()
    count_probs = (vc.to_numpy() / vc.to_numpy().sum()).astype(np.float64)

    # --- per-box size as a fraction of the image side (largest box) ---
    w_frac = ((r["largest_right"] - r["largest_left"]) / r["width"]).to_numpy()
    h_frac = ((r["largest_bottom"] - r["largest_top"]) / r["height"]).to_numpy()
    keep = np.isfinite(w_frac) & np.isfinite(h_frac) & (w_frac > 0) & (h_frac > 0)
    w_frac, h_frac = w_frac[keep], h_frac[keep]

    return RedactionDist(
        counts=count_vals.astype(np.int64),
        count_probs=count_probs,
        w_frac=w_frac.astype(np.float64),
        h_frac=h_frac.astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _norm_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised pixel coordinates (x, y in [-1, 1]) and radius from centre.

    Cached per image size and returned READ-ONLY. Every illumination and
    distortion transform reads these grids on every call; rebuilding a pair of
    384x384 float arrays per image would cost more than several of the
    transforms combined. The writeable flag is cleared so a caller cannot
    corrupt the cache for every subsequent image in the process.
    """
    ys = ((np.arange(h, dtype=np.float32) + 0.5) / h) * 2.0 - 1.0
    xs = ((np.arange(w, dtype=np.float32) + 0.5) / w) * 2.0 - 1.0
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    r = np.sqrt(gx * gx + gy * gy)
    for a in (gx, gy, r):
        a.flags.writeable = False
    return gx, gy, r


def _apply_gain(img: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Multiply an HWC uint8 image by an HW float gain field, with clipping."""
    out = img.astype(np.float32) * gain[:, :, None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Baseline transforms (HWC uint8 in, HWC uint8 out)
# ---------------------------------------------------------------------------
def horizontal_flip(img: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    if rng.random() < p:
        return np.ascontiguousarray(img[:, ::-1, :])
    return img


def random_black_boxes(
    img: np.ndarray,
    rng: np.random.Generator,
    dist: RedactionDist,
    p: float,
) -> np.ndarray:
    """With probability `p`, stamp 1-N filled black rectangles at random
    positions. Sizes and count come from the empirical `dist`. Boxes may fall
    anywhere and may overlap tissue -- that is the point: the distribution of
    box presence must not depend on the label or the hospital.
    """
    if rng.random() >= p:
        return img

    h, w = img.shape[:2]
    out = img.copy()
    n = int(rng.choice(dist.counts, p=dist.count_probs))
    for _ in range(n):
        idx = rng.integers(0, len(dist.w_frac))
        bw = max(1, int(round(dist.w_frac[idx] * w)))
        bh = max(1, int(round(dist.h_frac[idx] * h)))
        bw = min(bw, w)
        bh = min(bh, h)
        x0 = int(rng.integers(0, w - bw + 1))
        y0 = int(rng.integers(0, h - bh + 1))
        out[y0:y0 + bh, x0:x0 + bw, :] = 0
    return out


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
def channel_gamma_wb_lut(
    gamma: Optional[np.ndarray], gain: Optional[np.ndarray]
) -> np.ndarray:
    """Compose per-channel gamma and per-channel gain into ONE 256x1x3 LUT.

    Both are pure per-channel point operations, so applying them as two uint8
    passes would quantise twice and cost twice. Composing in float and
    quantising once is both faster and slightly more faithful.
    """
    ramp = np.arange(256, dtype=np.float32)[:, None] / 255.0  # (256, 1)
    out = np.repeat(ramp, 3, axis=1)                          # (256, 3)
    if gamma is not None:
        out = np.power(out, gamma[None, :])
    if gain is not None:
        out = out * gain[None, :]
    out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return out.reshape(256, 1, 3)


def per_channel_gamma_draw(
    rng: np.random.Generator, m: float, channel_m: float
) -> np.ndarray:
    """Gamma exponent per channel: a wide SHARED axis plus a narrow per-channel one.

    Log-uniform rather than uniform so darkening and brightening are equally
    likely: m=0.5 spans [1/1.5, 1.5], and the reciprocal of any drawn value is
    as probable as the value itself. A uniform draw on [1-m, 1+m] would be
    biased toward brightening.

    The shared/per-channel split is the part that matters. Three fully
    independent channel exponents at +/-50% will happily lift blue while
    dropping green, and the frame comes out magenta -- a colour no endoscope
    produces, so the model spends capacity on a domain that does not exist. The
    shared component carries the large tonal variation that IS wanted; the
    narrow per-channel component keeps the residual colour cast plausible.
    """
    lo = math.log(1.0 + m)
    shared = math.exp(float(rng.uniform(-lo, lo)))
    lo_c = math.log(1.0 + channel_m)
    per_channel = np.exp(rng.uniform(-lo_c, lo_c, size=3))
    return (shared * per_channel).astype(np.float32)


def white_balance_draw(
    rng: np.random.Generator, m: float, tint_frac: float
) -> np.ndarray:
    """Illuminant shift along the two axes real light sources actually vary on.

    Colour temperature moves red and blue in OPPOSITE directions -- warmer
    light means more red and less blue, and no illuminant raises both at once.
    Tint (the green-magenta axis) is a real but much smaller effect, mostly from
    fluorescent-type sources and from sensor filter differences, so it gets a
    fraction of the temperature range.

    Drawing three free per-channel gains instead would spend most of its
    probability mass on illuminants that cannot exist, which is how the first
    version of this stack produced purple mucosa.
    """
    temp = float(rng.uniform(-m, m))
    tint = float(rng.uniform(-m * tint_frac, m * tint_frac))
    return np.asarray(
        [1.0 + temp, 1.0 + tint, 1.0 - temp], dtype=np.float32
    )


def hue_saturation(
    img: np.ndarray, rng: np.random.Generator, hue_m: float, sat_m: float
) -> np.ndarray:
    """Rotate hue and scale saturation in HSV.

    OpenCV's uint8 HSV puts H in 0..179 (degrees/2), so the drawn degree shift
    is halved before it is added, and the addition is done in int16 to let the
    wraparound be an explicit modulo rather than a uint8 overflow.
    """
    hue_deg = float(rng.uniform(-hue_m, hue_m))
    sat_gain = float(rng.uniform(1.0 - sat_m, 1.0 + sat_m))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.int16) + int(round(hue_deg / 2.0))
    hsv[:, :, 0] = np.mod(h, 180).astype(np.uint8)
    s = hsv[:, :, 1].astype(np.float32) * sat_gain
    hsv[:, :, 1] = np.clip(s, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _fired(trace: Optional[list], name: str) -> None:
    """Record that a transform fired.

    Exists so reports/aug_samples.png can label each tile with what actually
    happened to it. A picture showing that the magnitudes are wrong is only
    actionable if it also says which transform to turn down, and re-deriving
    that from the draw sequence outside this module would be a second copy of
    the ordering, free to drift away from the real one.
    """
    if trace is not None:
        trace.append(name)


def colour_stack(
    img: np.ndarray, aug: AugConfig, rng: np.random.Generator,
    trace: Optional[list] = None,
) -> np.ndarray:
    gamma = (per_channel_gamma_draw(rng, aug.gamma_m, aug.gamma_channel_m)
             if rng.random() < aug.gamma_p else None)
    gain = (white_balance_draw(rng, aug.white_balance_m,
                               aug.white_balance_tint_frac)
            if rng.random() < aug.white_balance_p else None)
    if gamma is not None:
        _fired(trace, "gamma")
    if gain is not None:
        _fired(trace, "white_balance")
    if gamma is not None or gain is not None:
        img = cv2.LUT(img, channel_gamma_wb_lut(gamma, gain))
    if rng.random() < aug.hue_sat_p:
        _fired(trace, "hue_sat")
        img = hue_saturation(img, rng, aug.hue_m, aug.sat_m)
    return img


# ---------------------------------------------------------------------------
# Illumination
# ---------------------------------------------------------------------------
def vignette_field(h: int, w: int, strength: float) -> np.ndarray:
    """Radial falloff: 1.0 at the centre, (1 - strength) at the corners."""
    _, _, r = _norm_grid(h, w)
    r_max = math.sqrt(2.0)
    return 1.0 - strength * (r / r_max) ** 2


def linear_gradient_field(
    h: int, w: int, amplitude: float, angle_rad: float
) -> np.ndarray:
    """Brightness ramp of +/- amplitude across the frame in a given direction."""
    gx, gy, _ = _norm_grid(h, w)
    proj = gx * math.cos(angle_rad) + gy * math.sin(angle_rad)  # ~[-1.41, 1.41]
    return 1.0 + amplitude * (proj / math.sqrt(2.0))


def radial_gradient_field(
    h: int, w: int, amplitude: float, cx: float, cy: float
) -> np.ndarray:
    """Bowl (amplitude<0) or dome (amplitude>0) about an off-centre point.

    Models a light guide that is not aligned with the optical axis -- common,
    and a strong per-scope signature.
    """
    gx, gy, _ = _norm_grid(h, w)
    d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2) / (2.0 * math.sqrt(2.0))
    return 1.0 + amplitude * (1.0 - 2.0 * d)


def specular_highlights(
    img: np.ndarray, rng: np.random.Generator, m: float
) -> np.ndarray:
    """Blend 1-4 soft anisotropic blobs toward white.

    Wet mucosa under a point source produces specular reflections; their number,
    size and shape vary with the scope's light guide geometry, so they are a
    per-hospital signature worth randomising rather than a nuisance. Each blob
    is evaluated only inside its own 3-sigma window -- computing the Gaussian
    over the whole frame for a blob a few percent of the frame wide costs
    roughly thirty times more for the same result.
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    n = int(rng.integers(1, 5))
    for _ in range(n):
        rad = float(rng.uniform(0.015, max(0.016, m))) * min(h, w)
        aspect = float(rng.uniform(0.6, 1.6))
        intensity = float(rng.uniform(0.55, 1.0))
        cx = float(rng.uniform(0.08, 0.92)) * w
        cy = float(rng.uniform(0.08, 0.92)) * h
        rx, ry = max(1.0, rad * aspect), max(1.0, rad)

        x0, x1 = max(0, int(cx - 3 * rx)), min(w, int(cx + 3 * rx) + 1)
        y0, y1 = max(0, int(cy - 3 * ry)), min(h, int(cy + 3 * ry) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy = (np.arange(y0, y1, dtype=np.float32) - cy) / ry
        xx = (np.arange(x0, x1, dtype=np.float32) - cx) / rx
        weight = intensity * np.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2))

        patch = out[y0:y1, x0:x1, :]
        patch += (255.0 - patch) * weight[:, :, None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def illumination_stack(
    img: np.ndarray, aug: AugConfig, rng: np.random.Generator,
    trace: Optional[list] = None,
) -> np.ndarray:
    """Vignette and the two gradients are all multiplicative gain fields, so
    they are accumulated into ONE field and applied in a single pass. Three
    separate uint8 round-trips would clip three times and cost three times."""
    h, w = img.shape[:2]
    gain: Optional[np.ndarray] = None

    def compose(field: np.ndarray) -> np.ndarray:
        return field if gain is None else gain * field

    if rng.random() < aug.vignette_p:
        _fired(trace, "vignette")
        strength = float(rng.uniform(0.25 * aug.vignette_m, aug.vignette_m))
        gain = compose(vignette_field(h, w, strength))
    if rng.random() < aug.linear_gradient_p:
        _fired(trace, "linear_gradient")
        amp = float(rng.uniform(0.3 * aug.linear_gradient_m, aug.linear_gradient_m))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        gain = compose(linear_gradient_field(h, w, amp, angle))
    if rng.random() < aug.radial_gradient_p:
        _fired(trace, "radial_gradient")
        amp = float(rng.uniform(0.3 * aug.radial_gradient_m, aug.radial_gradient_m))
        if rng.random() < 0.5:
            amp = -amp
        cx = float(rng.uniform(-0.7, 0.7))
        cy = float(rng.uniform(-0.7, 0.7))
        gain = compose(radial_gradient_field(h, w, amp, cx, cy))

    if gain is not None:
        img = _apply_gain(img, gain)
    if rng.random() < aug.specular_p:
        _fired(trace, "specular")
        img = specular_highlights(img, rng, aug.specular_m)
    return img


def photometric_stack(
    img: np.ndarray, aug: AugConfig, rng: np.random.Generator,
    trace: Optional[list] = None,
) -> np.ndarray:
    """Colour then illumination: what reaches the lens."""
    img = colour_stack(img, aug, rng, trace)
    return illumination_stack(img, aug, rng, trace)


# ---------------------------------------------------------------------------
# Optical
# ---------------------------------------------------------------------------
def motion_blur_kernel(k: int, angle_rad: float) -> np.ndarray:
    """Normalised line PSF of length `k` at `angle_rad`."""
    kern = np.zeros((k, k), dtype=np.float32)
    c = (k - 1) / 2.0
    dx, dy = math.cos(angle_rad) * c, math.sin(angle_rad) * c
    cv2.line(
        kern,
        (int(round(c - dx)), int(round(c - dy))),
        (int(round(c + dx)), int(round(c + dy))),
        1.0, 1,
    )
    total = kern.sum()
    if total <= 0:  # degenerate at k=1; fall back to identity
        kern[int(c), int(c)] = 1.0
        total = 1.0
    return kern / total


@lru_cache(maxsize=64)
def defocus_kernel(radius_q: int) -> np.ndarray:
    """Normalised antialiased disc PSF -- the actual shape of defocus.

    A Gaussian is the usual stand-in and is wrong in a way that matters here:
    a defocused point source images as a filled disc with a hard edge, which is
    what turns a specular highlight into a bright ring-edged blob rather than a
    soft smear. Cached on radius quantised to 1/4 px, which is finer than the
    kernel's own pixel grid can express.
    """
    radius = radius_q / 4.0
    k = int(2 * math.ceil(radius) + 1)
    coords = np.arange(k, dtype=np.float32) - (k - 1) / 2.0
    d = np.sqrt(coords[:, None] ** 2 + coords[None, :] ** 2)
    kern = np.clip(radius + 0.5 - d, 0.0, 1.0).astype(np.float32)
    return kern / kern.sum()


def barrel_distortion(img: np.ndarray, k1: float) -> np.ndarray:
    """Radial distortion: k1 > 0 pincushions, k1 < 0 barrels.

    Border pixels are filled by reflection rather than black. Black would
    introduce a hard synthetic frame that the inscribed-square crop was chosen
    specifically to remove, handing the model a new artefact in exchange for the
    one it was meant to lose.
    """
    h, w = img.shape[:2]
    gx, gy, r = _norm_grid(h, w)
    scale = 1.0 + k1 * (r * r)
    map_x = ((gx * scale + 1.0) * 0.5 * w - 0.5).astype(np.float32)
    map_y = ((gy * scale + 1.0) * 0.5 * h - 0.5).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )


def unsharp_mask(img: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Variable sharpening, as an endoscope processor's edge enhancement.

    Different manufacturers ship very different amounts of it, and it is one of
    the most visible per-vendor signatures in the data.
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = img.astype(np.float32) * (1.0 + alpha) - blur.astype(np.float32) * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def optical_stack(
    img: np.ndarray, aug: AugConfig, rng: np.random.Generator,
    trace: Optional[list] = None,
) -> np.ndarray:
    """Lens geometry, then the two blurs, then edge enhancement.

    Distortion first because it is a property of the lens the light passes
    through before anything smears it; sharpening last because it is applied by
    the processor to an already-formed image.
    """
    if rng.random() < aug.distortion_p:
        _fired(trace, "distortion")
        k1 = float(rng.uniform(-aug.distortion_m, aug.distortion_m))
        img = barrel_distortion(img, k1)
    if rng.random() < aug.motion_blur_p:
        _fired(trace, "motion_blur")
        k_max = max(3, int(aug.motion_blur_m))
        k = int(rng.integers(1, (k_max - 1) // 2 + 1)) * 2 + 1  # odd, 3..k_max
        angle = float(rng.uniform(0.0, math.pi))
        img = cv2.filter2D(img, -1, motion_blur_kernel(k, angle))
    if rng.random() < aug.defocus_p:
        _fired(trace, "defocus")
        radius = float(rng.uniform(0.5, max(0.51, aug.defocus_m)))
        img = cv2.filter2D(img, -1, defocus_kernel(int(round(radius * 4))))
    if rng.random() < aug.sharpen_p:
        _fired(trace, "sharpen")
        alpha = float(rng.uniform(0.15, max(0.16, aug.sharpen_m)))
        sigma = float(rng.uniform(0.8, 2.0))
        img = unsharp_mask(img, alpha, sigma)
    return img


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------
def gaussian_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Signal-independent read noise, one draw per subpixel."""
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)


def poisson_noise(img: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Signal-dependent shot noise: variance proportional to intensity.

    The physical process is Poisson. The SAMPLER is a Gaussian with the
    Poisson variance substituted, which is a different thing and is worth being
    explicit about.

    `scale` is the photon count corresponding to full white, so LARGER scale
    means LESS noise. Writing the Poisson form out: lambda = (x/255)*scale, the
    output is Poisson(lambda)*255/scale, so

        E[out] = x        Var[out] = 255*x/scale

    and this function draws Normal(x, sqrt(255*x/scale)) instead. Above roughly
    20 photons the two are indistinguishable in every moment that survives
    8-bit quantisation, and endoscopic exposure sits far above that. The
    original ``rng.poisson`` call drew one variate per subpixel through a
    rejection sampler and cost 16 ms per image -- 41% of the entire
    augmentation budget -- for an effect the Gaussian reproduces to well within
    a percent. ``standard_normal`` is a Ziggurat draw straight into float32 and
    is roughly an order of magnitude cheaper.

    scripts/24_shot_noise_check.py holds the equivalence evidence: variance
    versus intensity and the radially-averaged noise power spectrum, both
    against the Poisson implementation, both required to agree within 2%.

    KNOWN AND ACCEPTED DEVIATION: the Gaussian is symmetric and continuous
    where the Poisson is skewed and integer-valued. At the noisiest end of the
    range the skew differs measurably. Skew is not what this transform is for
    -- the point is signal-dependent variance, which matches exactly by
    construction -- so the trade is taken deliberately rather than by accident.
    """
    x = np.clip(img.astype(np.float32), 0.0, 255.0)
    sigma = np.sqrt(x * (255.0 / float(scale)), dtype=np.float32)
    noise = rng.standard_normal(size=img.shape, dtype=np.float32)
    return np.clip(x + noise * sigma, 0.0, 255.0).astype(np.uint8)


def downsample_cycle(img: np.ndarray, factor: float) -> np.ndarray:
    """Shrink then restore: throws away real resolution, keeps the tensor shape.

    Stands in for the wide range of native sensor resolutions and export sizes
    across centres. INTER_AREA down (correct decimation) and INTER_LINEAR back
    up, which is what a naive export pipeline does.
    """
    h, w = img.shape[:2]
    sh, sw = max(8, int(round(h * factor))), max(8, int(round(w * factor)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def sensor_stack(
    img: np.ndarray, aug: AugConfig, rng: np.random.Generator,
    trace: Optional[list] = None,
) -> np.ndarray:
    if rng.random() < aug.gaussian_noise_p:
        _fired(trace, "gaussian_noise")
        sigma = float(rng.uniform(2.0, max(2.01, aug.gaussian_noise_m)))
        img = gaussian_noise(img, rng, sigma)
    if rng.random() < aug.poisson_noise_p:
        _fired(trace, "poisson_noise")
        # m is the noisiest end, so the draw runs from m upward
        scale = float(rng.uniform(aug.poisson_noise_m, aug.poisson_noise_m * 10.0))
        img = poisson_noise(img, rng, scale)
    if rng.random() < aug.downsample_p:
        _fired(trace, "downsample")
        factor = float(rng.uniform(aug.downsample_m, aug.downsample_ceiling_m))
        img = downsample_cycle(img, factor)
    return img


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------
def jpeg_recompress(img: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip through JPEG at a given quality.

    The RGB->BGR conversions are not ceremony. JPEG's luma is
    0.299R + 0.587G + 0.114B and its chroma planes are subsampled, so handing
    an RGB array to an encoder that believes it is BGR computes luma from the
    wrong channel and subsamples the wrong chroma. On endoscopy, where the red
    channel carries most of the signal, that is exactly the channel it would
    damage most.
    """
    bgr = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def compression_stack(
    img: np.ndarray, aug: AugConfig, rng: np.random.Generator,
    trace: Optional[list] = None,
) -> np.ndarray:
    if rng.random() < aug.jpeg_p:
        q = int(rng.integers(aug.jpeg_q_min, aug.jpeg_q_max + 1))
        _fired(trace, f"jpeg(q{q})")
        img = jpeg_recompress(img, q)
    return img


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_imagenet(img: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32, scaled to [0,1] then ImageNet-standardised."""
    x = img.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def to_chw_float(img: np.ndarray) -> np.ndarray:
    """HWC uint8 -> CHW float32 in [0,1], no standardisation (normalise off)."""
    x = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def apply_uint8(
    img: np.ndarray,
    aug: AugConfig,
    rng: np.random.Generator,
    train: bool,
    redaction_dist: Optional[RedactionDist],
    trace: Optional[list] = None,
) -> np.ndarray:
    """Every train-time transform, stopping before normalisation.

    Split out from ``apply`` so the sample-grid renderer and the per-transform
    profiler can see what the model sees WITHOUT having to invert ImageNet
    standardisation to look at it. ``apply`` is this plus the final cast, and
    there is no second copy of the ordering to drift out of sync.

    Pass a list as ``trace`` to have each transform that fires append its name.
    Purely diagnostic: it changes no draw and no pixel.
    """
    if not train:
        return img
    if aug.hflip:
        before = img
        img = horizontal_flip(img, rng, aug.hflip_p)
        if img is not before:
            _fired(trace, "hflip")
    if aug.random_black_boxes:
        if redaction_dist is None:
            raise ValueError(
                "random_black_boxes is enabled but no redaction distribution "
                "was provided"
            )
        before = img
        img = random_black_boxes(img, rng, redaction_dist, aug.black_boxes_p)
        if img is not before:
            _fired(trace, "black_boxes")
    if aug.photometric:
        img = photometric_stack(img, aug, rng, trace)
    if aug.optical:
        img = optical_stack(img, aug, rng, trace)
    if aug.sensor:
        img = sensor_stack(img, aug, rng, trace)
    if aug.compression:
        img = compression_stack(img, aug, rng, trace)
    return img


def apply(
    img: np.ndarray,
    aug: AugConfig,
    rng: np.random.Generator,
    train: bool,
    redaction_dist: Optional[RedactionDist],
) -> np.ndarray:
    """Run the pipeline on an HWC uint8 RGB image.

    Returns a CHW float32 array. Every augmentation runs only when `train` is
    True; normalisation is applied in both train and eval so the tensor
    statistics match. Determinism is entirely a function of `rng`.
    """
    img = apply_uint8(img, aug, rng, train, redaction_dist)
    if aug.normalize_imagenet:
        return normalize_imagenet(img)
    return to_chw_float(img)
