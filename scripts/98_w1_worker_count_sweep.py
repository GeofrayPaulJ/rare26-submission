"""W1 -- the untested fix: remove the workers entirely. G1 Part A/B proved
ItkStack.read()/StackDataset.__getitem__ flat in a bare single-process
loop (no DataLoader machinery at all). This tests the REAL production
DataLoader (StackDataset, _collate, persistent_workers where applicable)
at num_workers in {0, 2, 8}, all 94 batches of probe C's actual
3,000-image stack, same lean psutil-only instrumentation G1/G2 already
validated (no gc.get_objects()/Counter overhead to contaminate the
measurement itself -- see reports/g1_scaling_defect.md's own account of
why that matters).

Zero GPU, no model built (same convention as G1/G2 -- isolates the data
path signal this test is actually about; the model/GPU-compute cost is
a separate, already-characterised ~1% of wall time and would swamp the
per-batch data-path signal on this box's CPU-only forward-pass speed
if included).

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/98_w1_worker_count_sweep.py'
"""
from __future__ import annotations

import glob
import os
import sys
import time

import pandas as pd
import psutil

sys.path.insert(0, "/workspace/RARE26/submission")

from rare26_infer.predict import StackDataset, _collate  # noqa: E402
from rare26_infer.stack import ItkStack  # noqa: E402

STACK_PATH = (
    "/workspace/RARE26/runs/submission_test/platform_probe_p5/c/interface_0/"
    "images/stacked-barretts-esophagus-endoscopy/stack.mha"
)
LOG_ROOT = "/workspace/RARE26/runs/submission_test/w1_worker_sweep_logs"
REPORT_ROOT = "/workspace/RARE26/reports"
BATCH_SIZE = 32


def run_config(num_workers: int) -> pd.DataFrame:
    from torch.utils.data import DataLoader

    log_dir = os.path.join(LOG_ROOT, f"nw{num_workers}")
    if os.path.isdir(log_dir):
        for f in glob.glob(os.path.join(log_dir, "*.csv")):
            os.remove(f)
    os.makedirs(log_dir, exist_ok=True)

    stack = ItkStack(STACK_PATH)
    n = len(stack)
    ds = StackDataset(stack, n)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
        pin_memory=False, collate_fn=_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    proc = psutil.Process()
    rows = []
    print(f"[w1] num_workers={num_workers}: running {n} images, "
          f"{-(-n // BATCH_SIZE)} batches...", flush=True)
    it = iter(loader)
    t_prev = time.perf_counter()
    bi = 0
    t_run_start = t_prev
    while True:
        try:
            batch = next(it)
        except StopIteration:
            break
        now = time.perf_counter()
        wall = now - t_prev
        t_prev = now
        rows.append({
            "num_workers": num_workers,
            "batch": bi,
            "wall_seconds": wall,
            "n_items": int(batch[1].shape[0]),
            "main_rss_bytes": proc.memory_info().rss,
        })
        if bi % 20 == 0:
            print(f"[w1]   nw={num_workers} batch {bi:3d} wall={wall:6.3f}s "
                  f"rss={rows[-1]['main_rss_bytes']/2**20:7.1f}MiB", flush=True)
        bi += 1
    total_wall = time.perf_counter() - t_run_start
    del it, loader
    time.sleep(1.0)

    df = pd.DataFrame(rows)
    print(f"[w1] num_workers={num_workers}: TOTAL {total_wall:.1f}s for {n} images "
          f"({total_wall/n*1000:.1f} ms/image), {bi} batches", flush=True)
    return df


def main() -> int:
    all_dfs = []
    for nw in (0, 2, 8):
        df = run_config(nw)
        all_dfs.append(df)
        out_path = os.path.join(REPORT_ROOT, f"w1_nw{nw}.parquet")
        df.to_parquet(out_path)
        print(f"[w1] wrote {out_path}\n", flush=True)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(os.path.join(REPORT_ROOT, "w1_worker_sweep_combined.parquet"))

    print("\n[w1] === SUMMARY ===")
    for nw, g in combined.groupby("num_workers"):
        g = g.sort_values("batch")
        total = g["wall_seconds"].sum()
        n_img = g["n_items"].sum()
        rss_delta = (g["main_rss_bytes"].iloc[-1] - g["main_rss_bytes"].iloc[0]) / 2**20
        floor = g[(g["batch"] >= 4) & (g["batch"] <= 8)]["wall_seconds"].mean()
        tail = g[g["batch"] >= g["batch"].max() - 4]["wall_seconds"].mean()
        print(f"  num_workers={nw}: total={total:.1f}s  n={n_img}  "
              f"{total/n_img*1000:.2f} ms/img  rss_delta={rss_delta:+.1f}MiB  "
              f"floor(batches 4-8)={floor:.3f}s  tail(last 5)={tail:.3f}s  "
              f"ratio={tail/max(floor,1e-9):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
