"""Test 8: fp16, bf16 and fp32 tensor paths all produce finite output.

The eval hardware (T4 / A10G) has no bf16, so the fp16 path must be as real as
the bf16 one. We exercise all three through the same autocast helper the run
uses, on GPU when present (falling back to CPU)."""
import torch
import torch.nn as nn

from src.config import autocast, torch_dtype

PRECISIONS = ["fp32", "fp16", "bf16"]


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def test_all_precisions_finite():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _Tiny().to(device).eval()
    x = torch.randn(4, 3, 32, 32, device=device)

    for prec in PRECISIONS:
        assert torch_dtype(prec) in (torch.float32, torch.float16, torch.bfloat16)
        with torch.no_grad(), autocast(device, prec):
            out = model(x)
        assert torch.isfinite(out).all(), f"{prec} produced non-finite output"
        assert out.shape == (4, 1)


def test_only_fp16_gets_a_live_gradient_scaler():
    """fp32 and bf16 must take a genuinely unscaled path, not a scaled one that
    happens to use a factor of 1.0."""
    from src.train import make_scaler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert make_scaler("fp16", device).is_enabled() is (device == "cuda")
    assert make_scaler("bf16", device).is_enabled() is False
    assert make_scaler("fp32", device).is_enabled() is False


def test_fp32_path_is_a_true_no_op():
    """fp32 must disable autocast outright and leave the loss untouched --
    same numbers as running with no AMP machinery at all."""
    import contextlib

    from src.train import make_scaler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert isinstance(autocast(device, "fp32"), contextlib.nullcontext)

    model = _Tiny().to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.zeros(4, device=device)

    with autocast(device, "fp32"):
        out = model(x).squeeze(-1)
        assert out.dtype == torch.float32, "fp32 path must not cast anything"
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)

    scaled = make_scaler("fp32", device).scale(loss)
    assert scaled is loss or torch.equal(scaled, loss), "fp32 loss was rescaled"
