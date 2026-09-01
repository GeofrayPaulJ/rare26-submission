"""Does bf16 inference cost metric resolution? Re-validate one checkpoint at
each precision and compare.

WHY THIS EXISTS. src/io.py stores logits at float64 to keep the ranking exact,
on the grounds that ties at the operating threshold inflate the false-positive
count. That reasoning is correct but it guards the wrong end of the pipe: by the
time a logit reaches io.py it has already been rounded to whatever the forward
pass used. Training run r0_f0_s0 finished with 53 distinct logit values across
617 validation rows, 136 negatives sharing one value, and the 90%-recall
threshold landing on a value shared by 32 images.

Two causes compound. The model overfits to near-zero training loss and saturates
its outputs to about +/-8.5, and bf16 quantises that region onto a 0.0625 grid.
This script separates "the model is saturated" from "bf16 threw away the
ordering" by running the identical weights at bf16, fp16 and fp32.

Usage (inside the Prometheus container, from the repo root):
    python -m scripts.precision_ablation runs/convnext_b_384/r0_f0_s0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import Config, autocast  # noqa: E402
from src.data import BarrettDataset, make_loader  # noqa: E402
from src.folds import get_split  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.model import build_model  # noqa: E402
from src.seeding import seed_everything  # noqa: E402


@torch.no_grad()
def validate_at(model, loader, device: str, precision: str) -> np.ndarray:
    model.eval()
    out = []
    for x, _y, _fp in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model(x).squeeze(-1)
        out.append(logits.float().cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint", default="checkpoints/weights_fp32.pt")
    args = ap.parse_args()

    cfg = Config.from_yaml(os.path.join(args.run_dir, "config.yaml"))
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest_df = pd.read_csv(cfg.manifest)
    _train_fps, val_fps = get_split(cfg.repeat, cfg.fold, cfg.manifest)
    val_ds = BarrettDataset(val_fps, manifest_df, cfg, train=False, build_cache=True)
    loader = make_loader(val_ds, cfg, shuffle=False)
    labels = val_ds.labels()

    model = build_model(cfg).to(device)
    ck = torch.load(os.path.join(args.run_dir, args.checkpoint),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"checkpoint: epoch {ck['epoch']}, selection={ck['selection']}, "
          f"canonical={ck['canonical']}\n")

    results = {}
    for precision in ("bf16", "fp16", "fp32"):
        logits = validate_at(model, loader, device, precision)
        m = evaluate(labels, logits)
        results[precision] = (logits, m)

    ref = results["fp32"][0]
    print(f"{'prec':>5} {'uniq/617':>9} {'ROC-AUC':>9} {'pAUC.15':>9} {'PPV@90R':>9} "
          f"{'ties@thr':>9} {'max|d| vs fp32':>15}")
    print("-" * 72)
    for precision, (logits, m) in results.items():
        pos = np.sort(logits[labels == 1])
        thr = pos[int(np.floor(0.1 * len(pos)))]
        print(f"{precision:>5} {len(np.unique(logits)):>9} {m['roc_auc']:>9.4f} "
              f"{m['pauc_15_std']:>9.4f} {m['ppv_at_90_recall']:>9.4f} "
              f"{int((logits == thr).sum()):>9} "
              f"{np.abs(logits - ref).max():>15.4f}")

    # Spearman: does the low-precision ranking still agree with fp32's?
    print()
    for precision in ("bf16", "fp16"):
        lo = results[precision][0]
        concordant = np.corrcoef(
            pd.Series(lo).rank().to_numpy(), pd.Series(ref).rank().to_numpy()
        )[0, 1]
        print(f"rank correlation {precision} vs fp32: {concordant:.6f}")

    print("\nfp32 costs one extra validation pass per epoch (~4 s of a 43 s epoch) "
          "and\nis the precision the ranking metrics should be computed at.")


if __name__ == "__main__":
    main()
