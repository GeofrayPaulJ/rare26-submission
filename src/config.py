"""Single source of truth for run configuration.

Every default lives here and nowhere else. A run resolves a `Config`, writes it
back out as YAML next to its outputs, and every other module reads what it needs
from the object it is handed -- no module carries its own hidden defaults.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Constants that are not tunable knobs (they describe the data, not the run).
# ImageNet statistics are a fixed convention, not a default we might sweep.
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VALID_CROP_MODES = ("inscribed_square", "fov_bbox")
VALID_PRECISION = ("bf16", "fp16", "fp32")
# "none" and "random" both mean "plain shuffled epoch over every training row";
# "none" is the spelling the training step uses to say "no sampler object".
#
# "balanced_centre_class" draws the four (centre, class) strata with equal total
# weight instead of the two class strata. It is a strict generalisation of
# "weighted": on a training set holding one centre the two collapse to the same
# weight vector, and src/data.py says so out loud rather than pretending
# otherwise. See make_balanced_centre_class_sampler.
VALID_SAMPLERS = ("random", "none", "weighted", "balanced_centre_class", "sequential")


@dataclass(frozen=True)
class AugConfig:
    """Augmentation switches: four master gates, then one probability and one
    magnitude per transform.

    Everything except the three baseline transforms (hflip, ImageNet normalise,
    synthetic redaction boxes) defaults OFF, so an AugConfig() built with no
    arguments reproduces the step-3 baseline exactly.

    MAGNITUDE CONVENTION. Every ``*_m`` field is the OUTER edge of that
    transform's range, and the range is documented on the transform itself in
    src/augment.py. The defaults here sit at roughly 2-3x the magnitudes the
    common augmentation libraries ship, because the target is twelve unseen
    hospitals rather than mild jitter -- see reports/aug_samples.png for what
    they actually look like at these settings.
    """

    # --- baseline, ON by default ---
    hflip: bool = True
    hflip_p: float = 0.5
    normalize_imagenet: bool = True
    random_black_boxes: bool = True
    black_boxes_p: float = 0.5

    # --- master gates. Each switches a whole family off without touching the
    #     per-transform knobs below, so a family can be ablated in one edit. ---
    photometric: bool = False    # colour + illumination
    optical: bool = False        # lens / focus / motion
    sensor: bool = False         # noise + resolution loss
    compression: bool = False    # JPEG re-encoding

    # --- colour (gated by `photometric`) ---
    # THESE MAGNITUDES ARE CONSTRAINED BY A GAMUT, NOT ONLY BY A MULTIPLE OF A
    # LIBRARY DEFAULT. Three independent colour operations compose, and at 2-3x
    # defaults each their composition leaves the endoscopic gamut entirely --
    # magenta, purple and olive frames, none of which any scope produces. See
    # reports/aug_samples_v1_freecolour.png for what that looked like. The fix
    # is not simply smaller numbers: gamma and white balance are decomposed
    # into a wide shared axis plus a deliberately narrow per-channel one, so
    # brightness and illuminant variation stay aggressive while hue does not
    # run away. src/augment.py holds the reasoning per transform.
    #
    # shared gamma exponent ~ exp(U(-ln(1+m), +ln(1+m))); m=0.5 -> [0.67, 1.50]
    gamma_p: float = 0.5
    gamma_m: float = 0.50
    # extra INDEPENDENT per-channel gamma deviation on top of the shared one
    gamma_channel_m: float = 0.08
    # illuminant shift: colour-temperature half-range (R up / B down together)
    white_balance_p: float = 0.5
    white_balance_m: float = 0.25
    # green-magenta tint axis, as a fraction of white_balance_m
    white_balance_tint_frac: float = 0.34
    # hue shift ~ U(-m_hue, +m_hue) degrees; saturation gain ~ U(1-m_sat, 1+m_sat)
    hue_sat_p: float = 0.5
    hue_m: float = 10.0
    sat_m: float = 0.45

    # --- illumination (gated by `photometric`) ---
    # fraction of brightness lost at the frame corners ~ U(0.25m, m)
    vignette_p: float = 0.5
    vignette_m: float = 0.55
    # linear brightness ramp, peak-to-mean amplitude ~ U(0.3m, m), random direction
    linear_gradient_p: float = 0.5
    linear_gradient_m: float = 0.35
    # radial brightness bowl/dome about a random centre, amplitude ~ U(0.3m, m)
    radial_gradient_p: float = 0.3
    radial_gradient_m: float = 0.35
    # specular highlights: 1-4 blobs, radius fraction of the frame ~ U(0.015, m)
    specular_p: float = 0.4
    specular_m: float = 0.060

    # --- optical (gated by `optical`) ---
    # motion blur, odd kernel length drawn from [3, m] px at the working size
    motion_blur_p: float = 0.3
    motion_blur_m: float = 21.0
    # defocus, disc-PSF radius ~ U(0.5, m) px
    defocus_p: float = 0.3
    defocus_m: float = 3.5
    # barrel (k1>0) / pincushion (k1<0) distortion, k1 ~ U(-m, +m)
    distortion_p: float = 0.4
    distortion_m: float = 0.25
    # unsharp mask, alpha ~ U(0.15, m)
    sharpen_p: float = 0.3
    sharpen_m: float = 0.85

    # --- sensor (gated by `sensor`) ---
    # additive Gaussian noise, sigma ~ U(2, m) in 0-255 units
    gaussian_noise_p: float = 0.4
    gaussian_noise_m: float = 22.0
    # Poisson shot noise; m is the SMALLEST photon scale, so larger m = milder
    poisson_noise_p: float = 0.3
    poisson_noise_m: float = 20.0
    # downsample-upsample cycle, scale factor ~ U(m, ceiling_m)
    downsample_p: float = 0.35
    downsample_m: float = 0.35
    # mild end of the same range. Was a hardcoded 0.85 in src/augment.py until
    # 2026-07-31: identity-anchoring only downsample_m and leaving this fixed
    # meant downsample_m crosses 0.85 below magnitude_scale~0.231 (identity 1.0
    # vs shipped 0.35 scales faster than a fixed ceiling can follow), producing
    # rng.uniform(low > high). Found by the A4 probe at magnitude_scale=0.10.
    # Both ends now share identity 1.0, so downsample_m(s) <= downsample_ceiling_m(s)
    # for every s in [0, 1] -- see MAGNITUDE_IDENTITY.
    downsample_ceiling_m: float = 0.85

    # --- compression (gated by `compression`) ---
    # JPEG quality ~ U{q_min .. q_max}, inclusive
    jpeg_p: float = 0.5
    jpeg_q_min: int = 30
    jpeg_q_max: int = 95

    # --- global magnitude multiplier -------------------------------------
    # One knob that scales EVERY magnitude above while leaving every
    # probability untouched, so an arm can ask "the same stack, a third as
    # strong" without hand-editing eighteen numbers and getting one wrong.
    #
    # Scaling is IDENTITY-ANCHORED, not multiplicative:
    #
    #     value(s) = identity + s * (shipped - identity)
    #
    # where `identity` is the value at which that transform does nothing. A
    # plain multiply would be wrong for four of these, and wrong in the
    # dangerous direction -- it would make them STRONGER. `downsample_m` is a
    # minimum scale factor whose no-op is 1.0, so 0.33 x 0.35 = 0.12 is a
    # harsher downsample, not a gentler one; the JPEG quality bounds are
    # no-ops at 100, so scaling 30 to 10 is far more destructive; and
    # `poisson_noise_m` is a photon count whose no-op is infinity. See
    # MAGNITUDE_IDENTITY below for the full table.
    #
    # Applied exactly once, by Config.__post_init__, which then sets
    # magnitude_scale_applied so that reloading a dumped config cannot scale a
    # second time. The YAML written next to a run therefore holds the EFFECTIVE
    # magnitudes plus the scale that produced them.
    magnitude_scale: float = 1.0
    magnitude_scale_applied: bool = False

    # Probability fields, checked as a group so a typo'd knob fails at config
    # resolution rather than 40 minutes into an epoch.
    _P_FIELDS = (
        "hflip_p", "black_boxes_p", "gamma_p", "white_balance_p", "hue_sat_p",
        "vignette_p", "linear_gradient_p", "radial_gradient_p", "specular_p",
        "motion_blur_p", "defocus_p", "distortion_p", "sharpen_p",
        "gaussian_noise_p", "poisson_noise_p", "downsample_p", "jpeg_p",
    )
    _M_FIELDS = (
        "gamma_m", "gamma_channel_m", "white_balance_m",
        "white_balance_tint_frac", "hue_m", "sat_m", "vignette_m",
        "linear_gradient_m", "radial_gradient_m", "specular_m",
        "motion_blur_m", "defocus_m", "distortion_m", "sharpen_m",
        "gaussian_noise_m", "poisson_noise_m", "downsample_m",
        "downsample_ceiling_m",
    )

    def validate(self) -> None:
        for name in self._P_FIELDS:
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1], got {v}")
        for name in self._M_FIELDS:
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(f"{name} must be non-negative, got {v}")
        if self.white_balance_m >= 1.0:
            raise ValueError(
                f"white_balance_m must be < 1 (gain ~ U(1-m, 1+m) would go "
                f"non-positive), got {self.white_balance_m}"
            )
        if self.sat_m >= 1.0:
            raise ValueError(f"sat_m must be < 1, got {self.sat_m}")
        if self.vignette_m > 1.0:
            raise ValueError(
                f"vignette_m is the fraction of corner brightness REMOVED and "
                f"must be <= 1, got {self.vignette_m}"
            )
        if not 0.0 < self.downsample_m <= 1.0:
            raise ValueError(
                f"downsample_m is a scale factor in (0, 1], got {self.downsample_m}"
            )
        if not 0.0 < self.downsample_ceiling_m <= 1.0:
            raise ValueError(
                f"downsample_ceiling_m is a scale factor in (0, 1], got "
                f"{self.downsample_ceiling_m}"
            )
        if self.downsample_m > self.downsample_ceiling_m:
            raise ValueError(
                f"downsample_m ({self.downsample_m}) must be <= "
                f"downsample_ceiling_m ({self.downsample_ceiling_m}), or "
                f"rng.uniform(low, high) in sensor_stack raises"
            )
        if self.poisson_noise_m <= 0.0:
            raise ValueError(
                f"poisson_noise_m is a photon scale and must be > 0, got "
                f"{self.poisson_noise_m}"
            )
        if not 1 <= self.jpeg_q_min <= self.jpeg_q_max <= 100:
            raise ValueError(
                f"need 1 <= jpeg_q_min <= jpeg_q_max <= 100, got "
                f"{self.jpeg_q_min}..{self.jpeg_q_max}"
            )
        if not 0.0 <= self.magnitude_scale <= 1.0:
            raise ValueError(
                f"magnitude_scale interpolates from no-op (0) to the shipped "
                f"magnitudes (1) and must lie in [0, 1], got "
                f"{self.magnitude_scale}"
            )


# ---------------------------------------------------------------------------
# Magnitude scaling
# ---------------------------------------------------------------------------
# The value of each magnitude parameter at which its transform DOES NOTHING.
# This table is the whole content of the scaling rule: scaling is a straight
# interpolation from here toward the shipped value, so getting an entry wrong
# is the only way the rule can misbehave, and each one is justified below.
#
#   0.0 entries    the magnitude is a half-range or an amplitude about zero
#                  (gamma exponent spread, hue degrees, distortion k1, noise
#                  sigma, blur radius, blob radius); zero means "no spread".
#   motion_blur    a kernel LENGTH; 1 is the identity kernel, not 0.
#   downsample     a minimum SCALE FACTOR; 1.0 means "do not resample". Both
#                  ends of its draw range (downsample_m, downsample_ceiling_m)
#                  share this identity -- see the 2026-07-31 note on
#                  downsample_ceiling_m in AugConfig for why the ceiling has to
#                  scale too, not stay fixed at its shipped value.
#   jpeg quality   100 is near-lossless, so the no-op is the TOP of the range.
#
# Two fields are deliberately absent and must stay absent:
#
#   white_balance_tint_frac   a SHAPE ratio -- it says how the tint axis relates
#                             to the temperature axis, not how strong the shift
#                             is. Strength already scales through
#                             white_balance_m; scaling this too would narrow the
#                             illuminant family's shape as well as its size,
#                             which is a different intervention.
#   poisson_noise_m           a photon COUNT whose no-op is infinity, so linear
#                             interpolation cannot express it. Noise amplitude
#                             goes as 1/sqrt(count), so scaling amplitude by s
#                             means count / s**2. Handled explicitly below.
MAGNITUDE_IDENTITY: Dict[str, float] = {
    "gamma_m": 0.0,
    "gamma_channel_m": 0.0,
    "white_balance_m": 0.0,
    "hue_m": 0.0,
    "sat_m": 0.0,
    "vignette_m": 0.0,
    "linear_gradient_m": 0.0,
    "radial_gradient_m": 0.0,
    "specular_m": 0.0,
    "motion_blur_m": 1.0,
    "defocus_m": 0.0,
    "distortion_m": 0.0,
    "sharpen_m": 0.0,
    "gaussian_noise_m": 0.0,
    "downsample_m": 1.0,
    "downsample_ceiling_m": 1.0,
    "jpeg_q_min": 100.0,
    "jpeg_q_max": 100.0,
}

_INTEGER_MAGNITUDES = ("jpeg_q_min", "jpeg_q_max")


def with_magnitude_scale(aug: AugConfig) -> AugConfig:
    """Return `aug` with every magnitude interpolated toward its no-op identity.

    Idempotent: once applied, the flag is set and a second call is a no-op, so
    a config dumped to YAML and read back describes the same run rather than a
    run scaled twice. Probabilities are never touched -- an arm at a third of
    the magnitude fires the same transforms on the same images with the same
    RNG draws, which is what makes the magnitude the only variable.
    """
    if aug.magnitude_scale_applied:
        return aug
    s = float(aug.magnitude_scale)
    if s == 1.0:
        return dataclasses.replace(aug, magnitude_scale_applied=True)

    updates: Dict[str, Any] = {}
    for field_name, identity in MAGNITUDE_IDENTITY.items():
        shipped = float(getattr(aug, field_name))
        value = identity + s * (shipped - identity)
        updates[field_name] = (int(round(value))
                               if field_name in _INTEGER_MAGNITUDES else value)

    # Photon count: amplitude ~ 1/sqrt(count), so scaling amplitude by s
    # multiplies the count by 1/s**2. s=0 would be infinite photons, i.e. no
    # noise; the probability gate is what switches the transform off there, so
    # the limit is guarded rather than expressed.
    if s > 0.0:
        updates["poisson_noise_m"] = float(aug.poisson_noise_m) / (s * s)

    return dataclasses.replace(aug, magnitude_scale_applied=True, **updates)


@dataclass(frozen=True)
class Config:
    """Frozen, fully-resolved run configuration."""

    # --- reproducibility / split selection ---
    seed: int = 0
    repeat: int = 0            # selects fold_r{repeat} column, 0..9
    fold: int = 0             # validation fold within that repeat, 0..4

    # Leave-one-centre-out. None -> ordinary (repeat, fold) grouped CV. 1 or 2 ->
    # the holdout_center_{n} column is used instead and repeat/fold are ignored
    # for split selection. These are two different split families over the same
    # manifest, not two settings of one knob, which is why this is a separate
    # field rather than a sentinel value of `fold`.
    holdout_centre: Optional[int] = None

    # --- image pipeline ---
    image_size: int = 384
    crop_mode: str = "inscribed_square"   # see VALID_CROP_MODES
    cache_size: int = 431                  # side length of the RAM cache crop
    precision: str = "bf16"                # see VALID_PRECISION

    # --- loader ---
    batch_size: int = 32
    num_workers: int = 8
    sampler: str = "random"                # see VALID_SAMPLERS

    # --- model ---
    # Fully-qualified timm tag on purpose. The bare name "convnext_base" resolves
    # to fb_in22k_ft_in1k -- ImageNet-22k pretraining fine-tuned on 1k, a
    # different and stronger initialisation than the ImageNet-1k one we mean.
    # Swap this string for a ViT tag and nothing else has to change.
    arch: str = "convnext_base.fb_in1k"
    pretrained: bool = True
    grad_checkpointing: bool = False
    # Path to a LOCAL self-supervised checkpoint (GastroNet / DINOv2), relative
    # to the repo root. When set, the timm model is built with pretrained=False
    # and these weights are loaded instead -- see src/backbones.py, which is
    # strict about partial loads. Empty string means "use timm's own weights",
    # which is every ConvNeXt run to date. Part of TRAJECTORY_FIELDS: resuming
    # a run under different initial weights is a different experiment.
    init_weights: str = ""

    # --- optimisation ---
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.03    # linear warmup over the first 3% of total steps
    deterministic: bool = True   # cudnn.benchmark off + deterministic kernels

    # --- pAUC surrogate loss (OFF by default; src/losses.py holds the full
    #     rationale for every knob). With pauc_enabled=False none of these
    #     fields affect anything -- existing configs train byte-identically. ---
    pauc_enabled: bool = False
    pauc_beta: float = 0.15          # target FPR restriction; sweepable (0.05/0.25)
    pauc_pos_fraction: float = 0.25  # bottom-quartile positive restriction
    pauc_temperature: float = 1.0    # surrogate estimator only; hinge ignores it
    pauc_margin: float = 1.0         # squared-hinge margin
    pauc_lambda: float = 0.3         # weight on BCE in lambda*BCE + (1-lambda)*pAUC
    pauc_warmup_epochs: int = 5      # plain BCE for epochs [0, warmup)
    pauc_anneal_epochs: int = 5      # beta 1.0 -> pauc_beta across these epochs

    # --- weight averaging (OFF by default; both are PASSIVE: they observe the
    #     weights and never write them back, so the base run's parquets and
    #     checkpoints stay bit-identical with these on or off). ---
    ema_enabled: bool = False
    ema_decay: float = 0.999         # per optimiser step
    swa_enabled: bool = False
    swa_start_frac: float = 0.75     # average epochs from this fraction onward

    # --- checkpointing ---
    # False (the default) is "screening": no checkpoint survives a successful
    # run, only the predictions parquets. A rolling resume checkpoint is still
    # written every epoch so a killed run can restart, but it is deleted the
    # moment the schedule completes. Set True only for a run explicitly
    # designated as a final ensemble member, which instead keeps a single
    # weights-only fp32 file (no optimiser state) once it completes. See
    # src/train.py's module docstring for the full policy.
    save_checkpoint: bool = False

    # --- augmentation (nested) ---
    aug: AugConfig = field(default_factory=AugConfig)

    # --- paths (relative to the repo root / cwd) ---
    manifest: str = "manifests/rare25_folds_v2.csv"
    image_root: str = "00_source"
    redaction_stats: str = "manifests/redaction_check.csv"
    out_dir: str = "runs/exp"

    def __post_init__(self) -> None:
        if self.crop_mode not in VALID_CROP_MODES:
            raise ValueError(
                f"crop_mode must be one of {VALID_CROP_MODES}, got {self.crop_mode!r}"
            )
        if self.precision not in VALID_PRECISION:
            raise ValueError(
                f"precision must be one of {VALID_PRECISION}, got {self.precision!r}"
            )
        if self.sampler not in VALID_SAMPLERS:
            raise ValueError(
                f"sampler must be one of {VALID_SAMPLERS}, got {self.sampler!r}"
            )
        if not 0 <= self.repeat <= 9:
            raise ValueError(f"repeat must be 0..9, got {self.repeat}")
        if self.holdout_centre not in (None, 1, 2):
            raise ValueError(
                f"holdout_centre must be None, 1 or 2, got {self.holdout_centre!r}"
            )
        if self.image_size <= 0 or self.cache_size <= 0:
            raise ValueError("image_size and cache_size must be positive")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if not 0.0 <= self.warmup_frac < 1.0:
            raise ValueError(f"warmup_frac must be in [0, 1), got {self.warmup_frac}")
        if self.cache_size < self.image_size:
            raise ValueError(
                f"cache_size ({self.cache_size}) must be >= image_size "
                f"({self.image_size}); the cache is the source the final resize "
                f"reads from"
            )
        # Resolve the magnitude multiplier HERE, before validation, so that
        # every consumer of a Config sees effective magnitudes and no caller
        # can forget to apply it. Frozen dataclass, hence object.__setattr__ --
        # the same escape hatch __post_init__ exists to provide.
        self.aug.validate()
        object.__setattr__(self, "aug", with_magnitude_scale(self.aug))
        self.aug.validate()

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Dump the fully-resolved config. This is the artefact written next to
        every run's outputs."""
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = dict(d)  # shallow copy; do not mutate caller's dict
        aug = d.pop("aug", None)
        if aug is None:
            aug_cfg = AugConfig()
        elif isinstance(aug, AugConfig):
            aug_cfg = aug
        else:
            aug_cfg = AugConfig(**aug)
        return cls(aug=aug_cfg, **d)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Precision helpers -- the one place "bf16"/"fp16"/"fp32" turns into behaviour.
# torch is imported lazily so the pure-pandas tools (folds, io) never pull it in.
# ---------------------------------------------------------------------------
def torch_dtype(precision: str):
    """Map the precision string to a torch dtype."""
    import torch

    if precision not in VALID_PRECISION:
        raise ValueError(f"unknown precision {precision!r}")
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]


def autocast(device_type: str, precision: str):
    """Return an autocast context manager honouring `precision`.

    fp32 -> a no-op context (autocast disabled). fp16/bf16 -> autocast in that
    dtype. This is what makes the eval target (T4/A10G, no bf16) work: pick
    "fp16" and every path downstream casts to float16 instead.
    """
    import contextlib
    import torch

    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=torch_dtype(precision))
