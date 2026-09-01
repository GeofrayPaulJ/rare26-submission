"""Fail loudly if the shipped torch cannot run on the evaluation GPU.

The evaluation hardware is a T4 (sm_75) or an A10G (sm_86). A torch wheel that
was not compiled for those architectures will either JIT-compile from PTX on
first use -- adding minutes of startup to every job -- or refuse to run at all,
and neither failure is visible until the submission is already on the platform.
Checking the compiled arch list takes a second and turns that into a build-time
error.

Run inside the built image with --gpus all.
"""
from __future__ import annotations

import sys

import torch

REQUIRED = ["sm_75", "sm_86"]   # T4, A10G
NICE_TO_HAVE = ["sm_80", "sm_89", "sm_90", "sm_120"]


def main() -> int:
    print(f"torch            : {torch.__version__}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available; run this with --gpus all")
        return 1

    archs = torch.cuda.get_arch_list()
    print(f"compiled archs   : {archs}")
    print(f"local device     : {torch.cuda.get_device_name(0)}")
    print(f"local capability : sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    missing = [a for a in REQUIRED if a not in archs]
    if missing:
        print(f"\nFAIL: missing required arch(es) {missing}.")
        print("The evaluation GPU is a T4 (sm_75) or A10G (sm_86); this wheel "
              "would JIT from PTX or fail outright there.")
        return 1

    print(f"\nOK: required {REQUIRED} all present.")
    print(f"    also present: {[a for a in NICE_TO_HAVE if a in archs]}")

    # Prove fp16 actually executes rather than merely being advertised.
    x = torch.randn(8, 64, device="cuda")
    w = torch.randn(64, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        y = x @ w
    print(f"    fp16 matmul dtype={y.dtype} finite={bool(torch.isfinite(y).all())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
