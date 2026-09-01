"""Model construction. One arch string in, one raw logit out.

The architecture is a config string handed straight to timm, so swapping
ConvNeXt for a ViT is a config edit and not a code edit. The only thing this
module insists on is the head: ``num_classes=1``, a single raw pre-sigmoid
logit. No sigmoid, no softmax, no two-class head. Everything downstream --
BCEWithLogitsLoss, the parquet dump, the metrics -- reads that raw logit, and
the io layer stores it at float64 precisely because it is never squashed.

FP32 CLASSIFIER HEAD.
Under bf16/fp16 autocast, the final classifier Linear's matmul (a 1024-dim dot
product for ConvNeXt-Base) is itself computed in the lower-precision dtype and
its output quantises to that dtype's grid -- bf16 has an 8-bit mantissa, so at
logit magnitude ~8.5 the representable step is ~0.0625. That collapses many
validation rows onto a handful of distinct logit values, which ties the
PPV@90R threshold across dozens of images. Casting the *result* to fp32 after
the fact (as the loss computation already does) does not undo this: the value
was quantised before the cast ever runs. The fix is to make the matmul itself
happen in fp32 -- disable autocast around just this one Linear and feed it a
fp32 input -- which is cheap (one 1024x1 GEMM) and leaves every other op at
its configured precision. See ``_wrap_classifier_fp32`` below.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

import timm
import torch
import torch.nn as nn

from .config import Config

logger = logging.getLogger(__name__)


class Fp32Linear(nn.Module):
    """Wraps a Linear so its matmul always runs in fp32, regardless of an
    enclosing autocast context. Parameters stay fp32 (autocast never touches
    module weights, only activations), so disabling autocast and casting the
    input is sufficient to force a full-precision accumulation."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.linear(x.float())


def _wrap_classifier_fp32(model: nn.Module) -> None:
    """Replace the model's final classifier Linear (as returned by timm's
    ``get_classifier()``) with an ``Fp32Linear`` wrapping it, in place.

    Locates the module generically via ``get_classifier()`` + identity search
    over ``named_modules()`` rather than hard-coding a ConvNeXt attribute
    path, so this keeps working when the arch string is swapped for a ViT.
    """
    fc = model.get_classifier()
    if not isinstance(fc, nn.Linear):
        raise RuntimeError(
            f"expected the classifier head to be a single nn.Linear, got "
            f"{type(fc).__name__}; fp32 head wrapping needs updating for this arch"
        )
    target_name = next(
        (name for name, module in model.named_modules() if module is fc), None
    )
    if target_name is None:
        raise RuntimeError("could not locate the classifier module by identity")

    parent = model
    *parents, leaf = target_name.split(".")
    for p in parents:
        parent = getattr(parent, p)
    setattr(parent, leaf, Fp32Linear(fc))


def build_model(config: Config) -> nn.Module:
    """Create the timm backbone named by ``config.arch`` with a 1-logit head.

    ``img_size`` is forwarded only to architectures that accept it. Convnets
    (ConvNeXt, EfficientNet, ResNet) are fully convolutional and reject the
    kwarg; ViT-family models need it to resize their position embeddings for a
    non-224 input. Trying it and falling back on TypeError is what lets a ViT
    tag drop into the config without touching this file.
    """
    # A local self-supervised checkpoint replaces timm's pretrained weights
    # entirely, so never download them just to overwrite them.
    use_local = bool(getattr(config, "init_weights", ""))
    kwargs: Dict[str, Any] = dict(
        pretrained=False if use_local else config.pretrained,
        num_classes=1,
    )
    try:
        model = timm.create_model(config.arch, img_size=config.image_size, **kwargs)
        logger.info("built %s with img_size=%d", config.arch, config.image_size)
    except TypeError:
        model = timm.create_model(config.arch, **kwargs)
        logger.info("built %s (img_size not accepted; fully convolutional)", config.arch)

    if use_local:
        # Strict by default: raises rather than training from a partially
        # random backbone. See src/backbones.py for what actually differs
        # between these checkpoints and timm's expectations.
        from .backbones import load_local_backbone

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = config.init_weights
        if not os.path.isabs(path):
            path = os.path.join(repo_root, path)
        report = load_local_backbone(model, path)
        logger.info("local backbone weights: %s (%.2fM params, %d tensors)",
                    report["path"], report["params_loaded_millions"],
                    report["tensors_loaded"])

    if config.grad_checkpointing:
        # Recompute activations in the backward pass instead of storing them:
        # roughly 30-40% slower per step, but it buys a much larger batch.
        # Not every timm model implements the hook, so fail loudly rather than
        # silently training without the memory saving the config asked for.
        if not hasattr(model, "set_grad_checkpointing"):
            raise RuntimeError(
                f"{config.arch} does not support gradient checkpointing; "
                f"set grad_checkpointing=false"
            )
        model.set_grad_checkpointing(True)
        logger.info("gradient checkpointing enabled")

    _wrap_classifier_fp32(model)
    logger.info("classifier head wrapped for fp32 accumulation")

    return model


def param_counts(model: nn.Module) -> Dict[str, int]:
    """Total and trainable parameter counts, for the run log."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}
