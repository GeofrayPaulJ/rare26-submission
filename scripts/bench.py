"""Scaffold benchmark: cache build cost and one-epoch iteration time per precision.

No training loop and no model -- this measures the data path only. Each epoch
decodes nothing after the cache is warm; the per-batch work is
resize -> augment -> to tensor -> H2D transfer -> cast to the run precision under
autocast. That is exactly the path a real step feeds its forward, minus the
forward itself.

Usage (inside the Prometheus container, from the repo root):
    python -m scripts.bench
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import torch

from src.config import Config, autocast, torch_dtype
from src.data import BarrettDataset, make_loader
from src.folds import get_split
from src.seeding import seed_everything


def human_bytes(n: int) -> str:
    return f"{n / 1e9:.2f} GB ({n / (1024**3):.2f} GiB)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--cache-size", type=int, default=431)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--manifest", default="manifests/rare25_folds_v2.csv")
    ap.add_argument("--image-root", default="00_source")
    ap.add_argument("--redaction", default="manifests/redaction_check.csv")
    args = ap.parse_args()

    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    manifest_df = pd.read_csv(args.manifest)
    train_fps, _ = get_split(args.repeat, args.fold, args.manifest)
    print(f"training split (repeat={args.repeat}, fold={args.fold}): {len(train_fps)} images")

    base = dict(
        seed=0, repeat=args.repeat, fold=args.fold,
        image_size=args.image_size, cache_size=args.cache_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        manifest=args.manifest, image_root=args.image_root,
        redaction_stats=args.redaction,
    )

    # --- cache build (once, shared across precisions) ---
    cfg = Config(precision="fp32", **base)
    ds = BarrettDataset(train_fps, manifest_df, cfg, train=True, build_cache=True)
    print("\n=== CACHE ===")
    print(f"build time      : {ds.cache_build_seconds:.1f} s")
    print(f"resident size   : {human_bytes(ds.cache_bytes())}")
    print(f"shape / dtype   : {tuple(ds.cache.shape)} {ds.cache.dtype}")

    # --- one epoch per precision ---
    print("\n=== ONE EPOCH (data path only) ===")
    print(f"batch_size={args.batch_size}  image_size={args.image_size}  "
          f"num_workers={args.num_workers}\n")
    for prec in ("fp16", "bf16", "fp32"):
        pcfg = Config(precision=prec, **base)
        loader = make_loader(ds, pcfg, shuffle=True)
        dtype = torch_dtype(prec)

        # warm the workers/pipeline with one batch, not timed
        it = iter(loader)
        next(it)
        if device == "cuda":
            torch.cuda.synchronize()

        n_imgs = 0
        t0 = time.time()
        for tensor, _label, _fp in loader:
            tensor = tensor.to(device, non_blocking=True)
            with autocast(device, prec):
                tensor = tensor.to(dtype)  # honour precision on the data tensor
            n_imgs += tensor.shape[0]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"{prec:>4}: {dt:6.2f} s  ({n_imgs} imgs, {n_imgs / dt:6.1f} img/s)")


if __name__ == "__main__":
    main()
