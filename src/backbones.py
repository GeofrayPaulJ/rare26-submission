"""Load LOCAL self-supervised backbone checkpoints into timm models.

Everything trained so far in this project came from timm's own hub via
``pretrained=True``. The GastroNet / DINOv2 ensemble members are local ``.pth``
files produced by other codebases, and their state dicts do not line up with
timm's by name or by shape. This module is the one place that reconciles them,
and it is deliberately STRICT: a silently-partial load is the failure mode that
matters here, because it produces a model that trains, converges, reports
plausible numbers, and is quietly missing pretrained weights.

WHAT ACTUALLY DIFFERS, measured from the real files rather than assumed:

  * Nesting. dinov2.pth is ``{'teacher': {'backbone.*': ..., 'dino_head.*': ...}}``.
    The DINO projection head is not part of the encoder and is dropped -- it
    also inflates a naive parameter count from 86.0M to 109.1M, which is what
    first made this checkpoint look like the wrong architecture.

  * pos_embed length AND convention. The checkpoint carries [1, 577, 768]:
    a cls-token slot plus a 24x24 grid, i.e. pretraining at 336px. timm's
    DINOv2 models set ``no_embed_class=True`` and store the grid ONLY, so at
    378px the target is [1, 729, 768] = 27x27 with no prefix slot. Resampling
    without first stripping the cls slot yields 730 and fails by one -- the
    prefix convention has to be handled explicitly, not just the size.

  * Register tokens. The checkpoint calls them ``register_tokens``; timm calls
    them ``reg_token``. Same shape, different name, so a non-strict load
    silently leaves all four RANDOMLY INITIALISED while reporting success.
    This one is the reason ``verify_complete`` exists.

  * mask_token. Used only by masked-image-modelling pretraining; it has no
    role downstream and is dropped on purpose.

IMAGE SIZE. ViT-B/14 needs a size divisible by 14. 384 is not (384/14 = 27.43);
timm accepts it and silently floors to a 27x27 grid, which quietly discards the
last 6 pixels of every image. The ViT arm therefore runs at 378 = 27*14, an
exact fit, and that deviation from the ConvNeXt arms' 384 is recorded in the
report rather than hidden. The ResNet arms are fully convolutional and stay at
384, matching the A4 ConvNeXt runs exactly.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

# Keys that legitimately have no counterpart in a downstream classifier and are
# dropped rather than being treated as a mismatch.
_DROP = ("mask_token",)
# Renames, checkpoint name -> timm name.
_RENAME = {"register_tokens": "reg_token"}
# The only keys allowed to be missing after a load: the fresh task head.
_ALLOWED_MISSING = ("head.weight", "head.bias", "fc.weight", "fc.bias")


def _unwrap(obj: Any) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    notes: List[str] = []
    sd = obj
    for k in ("teacher", "student", "state_dict", "model", "network"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            notes.append(f"unwrapped ['{k}']")
            break
    if not isinstance(sd, dict):
        raise ValueError(f"checkpoint is not a dict: {type(sd).__name__}")
    tensors = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    if not tensors:
        raise ValueError("checkpoint contains no tensors")
    for pref in ("module.", "backbone.", "encoder."):
        matching = {k: v for k, v in tensors.items() if k.startswith(pref)}
        # A DINO checkpoint holds {backbone.*, dino_head.*} together, so
        # requiring EVERY key to share the prefix would strip nothing and leave
        # the projection head in the encoder's parameter count.
        if matching and len(matching) >= 0.5 * len(tensors):
            dropped = len(tensors) - len(matching)
            tensors = {k[len(pref):]: v for k, v in matching.items()}
            notes.append(f"stripped '{pref}'"
                         + (f", dropped {dropped} non-encoder tensors" if dropped else ""))
            break
    return tensors, notes


def _fit_pos_embed(sd: Dict[str, torch.Tensor], model: torch.nn.Module,
                   notes: List[str]) -> None:
    if "pos_embed" not in sd or not hasattr(model, "pos_embed"):
        return
    src, tgt = sd["pos_embed"], model.pos_embed
    if src.shape == tgt.shape:
        return
    from timm.layers import resample_abs_pos_embed

    # How many prefix slots does each side store? timm's DINOv2 configs set
    # no_embed_class=True and store the grid alone; the checkpoint stores a
    # cls slot in front of it. Infer both from perfect-square-ness rather than
    # hardcoding, so a differently-built ViT does not silently misalign.
    def prefix_of(n: int) -> int:
        root = int(round(n ** 0.5))
        return 0 if root * root == n else 1

    src_prefix = prefix_of(src.shape[1])
    tgt_prefix = prefix_of(tgt.shape[1])
    grid = int(round((tgt.shape[1] - tgt_prefix) ** 0.5))

    body = src[:, src_prefix:, :] if src_prefix else src
    out = resample_abs_pos_embed(body, new_size=[grid, grid],
                                 num_prefix_tokens=0, verbose=False)
    if tgt_prefix:
        out = torch.cat([src[:, :1, :], out], dim=1)
    sd["pos_embed"] = out
    notes.append(f"pos_embed {list(src.shape)} (prefix {src_prefix}) -> "
                 f"{list(out.shape)} (grid {grid}x{grid}, prefix {tgt_prefix})")


def load_local_backbone(model: torch.nn.Module, path: str,
                        strict_verify: bool = True) -> Dict[str, Any]:
    """Load ``path`` into ``model`` in place. Returns a load report.

    Raises RuntimeError when anything beyond the fresh classifier head fails to
    line up. That strictness is the point: the alternative is a model that
    silently trains from partially random weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) == 0:
        raise ValueError(f"{path} is 0 bytes (truncated download)")

    sd, notes = _unwrap(torch.load(path, map_location="cpu", weights_only=False))

    for old, new in _RENAME.items():
        if old in sd:
            sd[new] = sd.pop(old)
            notes.append(f"renamed '{old}' -> '{new}'")
    for k in _DROP:
        if sd.pop(k, None) is not None:
            notes.append(f"dropped '{k}'")

    _fit_pos_embed(sd, model, notes)

    n_loaded = sum(v.numel() for v in sd.values())
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys]
    unexpected = [k for k in result.unexpected_keys]

    bad_missing = [k for k in missing if not k.endswith(_ALLOWED_MISSING)]
    report = {
        "path": os.path.basename(path),
        "tensors_loaded": len(sd),
        "params_loaded": n_loaded,
        "params_loaded_millions": round(n_loaded / 1e6, 2),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "notes": notes,
    }
    if strict_verify and (unexpected or bad_missing):
        raise RuntimeError(
            f"incomplete backbone load from {os.path.basename(path)}:\n"
            f"  unexpected (in checkpoint, not in model, so DISCARDED): {unexpected}\n"
            f"  missing (in model, not in checkpoint, so LEFT RANDOM): {bad_missing}\n"
            f"  notes: {notes}\n"
            f"Refusing to train: these weights would be silently partial."
        )
    logger.info("loaded backbone %s: %d tensors, %.2fM params; %s",
                os.path.basename(path), len(sd), n_loaded / 1e6, "; ".join(notes))
    if missing:
        logger.info("  fresh (expected) head params: %s", missing)
    return report
