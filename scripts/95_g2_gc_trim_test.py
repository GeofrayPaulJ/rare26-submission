"""G2 -- fourth candidate fix. Three targeted fixes have now failed to
flatten the curve: malloc_trim(0) alone (91, only -15% slope), recycling
the SimpleITK reader every batch (92, no effect), forcing ITK
single-threaded (94, no effect). The isolation in 93 pins the leak to
ItkStack.read(i) alone, present only inside a forked worker. If SimpleITK's
SWIG-wrapped C++ objects hold a reference cycle (self-referential smart
pointers), Python's automatic refcounting would never free them --
malloc_trim(0) alone can't help because the memory is still genuinely
reachable, just uncollected, until the CYCLE detector runs. Testing
gc.collect() (breaks cycles) immediately followed by malloc_trim(0)
(returns whatever that frees) every batch.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/95_g2_gc_trim_test.py'
"""
from __future__ import annotations

import ctypes
import gc
import glob
import os
import sys
import time

import pandas as pd
import psutil

sys.path.insert(0, "/workspace/RARE26/submission")

from rare26_infer.stack import ItkStack  # noqa: E402

STACK_PATH = (
    "/workspace/RARE26/runs/submission_test/platform_probe_p5/c/interface_0/"
    "images/stacked-barretts-esophagus-endoscopy/stack.mha"
)
LOG_DIR = "/workspace/RARE26/runs/submission_test/g2_gctrim_logs"
REPORT_OUT = "/workspace/RARE26/reports/g2_gc_trim_test.parquet"
EVERY = 32  # once per batch


def _collate(batch):
    import torch
    idx = torch.tensor([b[0] for b in batch], dtype=torch.long)
    x = torch.stack([b[1] for b in batch])
    return idx, x


class GcTrimReadDataset:
    def __init__(self, stack, n, log_dir):
        self.stack = stack
        self.n = n
        self.log_dir = log_dir
        self._pid = None
        self._logf = None
        self._proc = None
        self._call_count = 0
        self._libc = None
        self._n_before_after = []

    def __len__(self):
        return self.n

    def _ensure_worker_init(self):
        pid = os.getpid()
        if self._pid != pid:
            self._pid = pid
            self._proc = psutil.Process()
            try:
                self._libc = ctypes.CDLL("libc.so.6")
            except OSError:
                self._libc = None
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"worker_{pid}.csv")
            self._logf = open(path, "a")
            if self._logf.tell() == 0:
                self._logf.write("pid,call_idx,item_i,rss_bytes,gc_collected\n")

    def __getitem__(self, i):
        import torch
        self._ensure_worker_init()
        self._call_count += 1
        self.stack.read(i)
        x = torch.zeros(3, 384, 384, dtype=torch.float32)

        collected = -1
        if self._call_count % EVERY == 0:
            collected = gc.collect()
            if self._libc is not None:
                self._libc.malloc_trim(0)

        rss = self._proc.memory_info().rss
        self._logf.write(f"{self._pid},{self._call_count},{i},{rss},{collected}\n")
        self._logf.flush()
        return i, x


def main() -> int:
    from torch.utils.data import DataLoader

    if os.path.isdir(LOG_DIR):
        for f in glob.glob(os.path.join(LOG_DIR, "*.csv")):
            os.remove(f)
    os.makedirs(LOG_DIR, exist_ok=True)

    stack = ItkStack(STACK_PATH)
    n = len(stack)
    ds = GcTrimReadDataset(stack, n, LOG_DIR)
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=8,
        pin_memory=False, collate_fn=_collate,
        persistent_workers=True, prefetch_factor=2,
    )

    print("[g2-gctrim] running (gc.collect()+malloc_trim(0) every batch)...", flush=True)
    it = iter(loader)
    bi = 0
    while True:
        try:
            next(it)
        except StopIteration:
            break
        bi += 1
    del it, loader
    time.sleep(1.0)

    frames = []
    for path in glob.glob(os.path.join(LOG_DIR, "*.csv")):
        df = pd.read_csv(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(REPORT_OUT)

    print(f"[g2-gctrim] gc.collect() return values seen (nonzero = cycles found): "
          f"{sorted(out[out['gc_collected']>=0]['gc_collected'].unique().tolist())[:20]}")

    deltas = []
    for pid, g in out.groupby("pid"):
        g = g.sort_values("call_idx")
        delta_mb = (g.iloc[-1]["rss_bytes"] - g.iloc[0]["rss_bytes"]) / 2**20
        deltas.append(delta_mb)
    print(f"[g2-gctrim] {bi} batches, {len(deltas)} workers, "
          f"per-worker delta MiB: {[round(d,1) for d in deltas]} "
          f"(mean {sum(deltas)/len(deltas):.1f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
