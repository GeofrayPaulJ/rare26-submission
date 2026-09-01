"""Model construction for inference, key-compatible with the trained checkpoint.

``Fp32Linear`` is copied from src/model.py rather than imported, for the same
reason fov.py copies the detector: the container must stand alone. It has to
stay byte-compatible in one specific way -- the wrapper introduces a ``.linear``
level in the parameter names, so the checkpoint's head keys are
``head.fc.linear.weight`` / ``head.fc.linear.bias``. Building a bare timm model
and loading these would fail on unexpected keys, which is the loud failure we
want rather than a silent partial load.

The fp32 head matters as much at inference as it did in training. Under fp16
autocast a bare 1024-dim classifier matmul quantises its output onto the fp16
grid, which collapses distinct images onto identical logits; on the training
fold that showed up as 53 distinct logits across 617 rows. With 23,176
negatives at test time, every tie sits at the operating threshold and inflates
the false positive count for no reason at all. Forcing this one small GEMM to
fp32 costs nothing measurable and removes the entire failure mode.
"""
from __future__ import annotations

import os
from typing import List, Sequence

import torch
import torch.nn as nn

# RARE26_ARCH overrides the baked-in default -- same env-var convention as
# RARE26_PRECISION / RARE26_BATCH_SIZE elsewhere in this container. Unset in
# every image built before 2026-08-08 (the ConvNeXt-Base ensemble/fallback/
# pinned085 variants), so their behaviour is unchanged; set explicitly when
# building a non-ConvNeXt variant (e.g. "resnet50" for G3).
ARCH = os.environ.get("RARE26_ARCH", "convnext_base.fb_in1k")


class Fp32Linear(nn.Module):
    """Runs the wrapped Linear in fp32 regardless of any enclosing autocast.

    Autocast never casts module parameters, only activations, so disabling
    autocast and casting the input up is enough to force a full-precision
    accumulation.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.linear(x.float())


def _wrap_classifier_fp32(model: nn.Module) -> None:
    """Replace the final classifier Linear with an Fp32Linear, in place."""
    fc = model.get_classifier()
    if not isinstance(fc, nn.Linear):
        raise RuntimeError(
            f"expected a single nn.Linear classifier, got {type(fc).__name__}"
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


def build_model(weights_path: str, device: str = "cuda", arch: str = ARCH) -> nn.Module:
    """Build the backbone and load the baked-in checkpoint.

    ``pretrained=False`` is load-bearing rather than an optimisation: the
    evaluation container has no network, so any attempt by timm to reach the
    Hugging Face hub for the pretrained tag would hang and then fail the run.
    Every weight comes from ``weights_path``, which is baked into the image.
    """
    import timm

    model = timm.create_model(arch, pretrained=False, num_classes=1)
    _wrap_classifier_fp32(model)

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and "stem.0.weight" not in state:
        state = state["model"]           # tolerate a full training checkpoint
    model.load_state_dict(state, strict=True)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_ensemble(
    weights_paths: Sequence[str], device: str = "cuda", arch: str = ARCH
) -> List[nn.Module]:
    """One ``build_model`` call per checkpoint, sharing nothing between members.

    Each fold's model was trained independently (different held-out split,
    same seed), so there is no state to share -- this is deliberately just a
    loop, not a combined module, which keeps per-member logits reachable for
    diagnostics rather than baking the average in at construction time.
    """
    if not weights_paths:
        raise ValueError("build_ensemble requires at least one checkpoint")
    return [build_model(p, device=device, arch=arch) for p in weights_paths]
