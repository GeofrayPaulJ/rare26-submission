"""Bake the trained checkpoint into resources/ as a bare state_dict.

A run designated a final ensemble member (``save_checkpoint: true``) already
writes checkpoints/weights_fp32.pt as a slim, weights-only fp32 artefact --
src/train.py drops the optimiser, scheduler, scaler, every RNG state and the
epoch history at write time, since inference has no use for any of it. This
script does the last mile: unwraps that artefact down to a bare state_dict,
which is both smaller and safer to load.

Selection is fixed: the CANONICAL checkpoint is the LAST epoch. This script
refuses anything that is not marked canonical.
"""
from __future__ import annotations

import argparse
import hashlib
import os

import torch

DEFAULT_CKPT = "runs/convnext_b_384/r0_f0_s0_fixed/checkpoints/weights_fp32.pt"
DEFAULT_OUT = "submission/resources/convnext_base_r0f0s0_e30.pth"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--allow-non-canonical", action="store_true",
                    help="escape hatch; never use for a real submission")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if not ck.get("canonical", False) and not args.allow_non_canonical:
        raise SystemExit(
            f"{args.checkpoint} has canonical={ck.get('canonical')!r} "
            f"(selection={ck.get('selection')!r}). The canonical output of a run "
            f"is the LAST epoch; refusing to bake a diagnostic checkpoint."
        )

    state = ck["model"]
    # keep it a plain tensor dict so the container can load with weights_only=True
    state = {k: v.detach().cpu().contiguous() for k, v in state.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(state, args.out)

    n_params = sum(v.numel() for v in state.values())
    size_mb = os.path.getsize(args.out) / 1e6
    with open(args.out, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()

    print(f"source      : {args.checkpoint}")
    print(f"epoch       : {ck['epoch']}  (selection={ck['selection']}, "
          f"canonical={ck['canonical']})")
    if ck.get("metrics"):
        print(f"val metrics : " + ", ".join(f"{k}={v:.4f}" for k, v in ck["metrics"].items()))
    else:
        # Full-data checkpoints train on all images, no holdout -- no
        # validation set exists by construction (confirmed, S1V), so there
        # is no per-epoch val metric to print. Expected, not an error.
        print("val metrics : none (full-data checkpoint, no holdout by construction)")
    print(f"written     : {args.out}")
    print(f"params      : {n_params:,}   size: {size_mb:.1f} MB")
    print(f"sha256      : {digest}")
    print(f"head keys   : {[k for k in state if k.startswith('head.fc')]}")


if __name__ == "__main__":
    main()
