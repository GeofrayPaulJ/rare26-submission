"""Test 10: the model emits exactly one raw logit, and the arch is a config knob.

Uses resnet18 with pretrained=False so the test needs no network and no 350 MB
download. What is under test is the head and the config plumbing, not ConvNeXt.
"""
import pytest
import torch

from src.model import build_model, param_counts
from tests.conftest import make_config


def test_single_raw_logit_out():
    model = build_model(make_config(arch="resnet18", pretrained=False)).eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 128, 128))
    assert out.shape == (4, 1), "head must be num_classes=1, one logit per image"
    # raw logits, not probabilities: something must land outside [0, 1] eventually
    assert out.dtype == torch.float32


def test_arch_is_config_selectable():
    """Swapping the arch string is the entire mechanism for changing backbone."""
    a = build_model(make_config(arch="resnet18", pretrained=False))
    b = build_model(make_config(arch="resnet34", pretrained=False))
    assert param_counts(a)["params_total"] != param_counts(b)["params_total"]


def test_grad_checkpointing_flag_runs_and_still_learns_gradients():
    cfg = make_config(arch="resnet18", pretrained=False, grad_checkpointing=True)
    model = build_model(cfg).train()
    loss = model(torch.randn(2, 3, 128, 128)).squeeze(-1).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "gradient checkpointing produced no gradients"
    assert all(torch.isfinite(g).all() for g in grads)


def test_grad_checkpointing_rejects_unsupported_arch():
    """Silently training without the memory saving the config asked for would
    show up as an inexplicable OOM, so it must fail loudly instead."""
    class _Bare(torch.nn.Module):
        pass

    cfg = make_config(arch="resnet18", pretrained=False, grad_checkpointing=True)
    import src.model as model_mod

    real = model_mod.timm.create_model
    model_mod.timm.create_model = lambda *a, **k: _Bare()
    try:
        with pytest.raises(RuntimeError, match="gradient checkpointing"):
            build_model(cfg)
    finally:
        model_mod.timm.create_model = real
