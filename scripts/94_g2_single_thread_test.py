"""G2 -- third candidate fix. 93 isolated the leak to ItkStack.read(i)
alone, present only inside a forked multiprocessing worker (flat in a
plain single process per G1 Part A). SimpleITK/ITK's default pipeline
uses its own internal multi-threading (itk::PoolMultiThreader), and C++
thread pools are a classic fork() incompatibility -- only the forking
thread survives fork(), so a library that assumes a live thread pool can
misbehave (and leak) across repeated calls in a forked child. Testing
whether forcing ITK single-threaded execution before any read happens in
the worker eliminates the growth.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/94_g2_single_thread_test.py'
"""
from __future__ import annotations

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
LOG_DIR = "/workspace/RARE26/runs/submission_test/g2_singlethread_logs"
REPORT_OUT = "/workspace/RARE26/reports/g2_single_thread_test.parquet"


class SingleThreadReadDataset:
    """read()-only, single-threaded ITK, same shape as stage 'read' in 93."""

    def __init__(self, stack, n, log_dir):
        self.stack = stack
        self.n = n
        self.log_dir = log_dir
        self._pid = None
        self._logf = None
        self._proc = None
        self._call_count = 0
        self._threading_set = False

    def __len__(self):
        return self.n

    def _ensure_worker_init(self):
        pid = os.getpid()
        if self._pid != pid:
            self._pid = pid
            self._proc = psutil.Process()
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"worker_{pid}.csv")
            self._logf = open(path, "a")
            if self._logf.tell() == 0:
                self._logf.write("pid,call_idx,item_i,rss_bytes\n")
        if not self._threading_set:
            import SimpleITK as sitk
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
            self._threading_set = True

    def __getitem__(self, i):
        import torch
        self._ensure_worker_init()
        self._call_count += 1
        self.stack.read(i)
        x = torch.zeros(3, 384, 384, dtype=torch.float32)
        rss = self._proc.memory_info().rss
        self._logf.write(f"{self._pid},{self._call_count},{i},{rss}\n")
        self._logf.flush()
        return i, x


def _collate(batch):
    import torch
    idx = torch.tensor([b[0] for b in batch], dtype=torch.long)
    x = torch.stack([b[1] for b in batch])
    return idx, x


def main() -> int:
    from torch.utils.data import DataLoader

    if os.path.isdir(LOG_DIR):
        for f in glob.glob(os.path.join(LOG_DIR, "*.csv")):
            os.remove(f)
    os.makedirs(LOG_DIR, exist_ok=True)

    stack = ItkStack(STACK_PATH)
    n = len(stack)
    ds = SingleThreadReadDataset(stack, n, LOG_DIR)
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=8,
        pin_memory=False, collate_fn=_collate,
        persistent_workers=True, prefetch_factor=2,
    )

    print("[g2-1t] running (SetGlobalDefaultNumberOfThreads(1) in each worker)...",
          flush=True)
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

    deltas = []
    for pid, g in out.groupby("pid"):
        g = g.sort_values("call_idx")
        delta_mb = (g.iloc[-1]["rss_bytes"] - g.iloc[0]["rss_bytes"]) / 2**20
        deltas.append(delta_mb)
    print(f"[g2-1t] {bi} batches, {len(deltas)} workers, "
          f"per-worker delta MiB: {[round(d,1) for d in deltas]} "
          f"(mean {sum(deltas)/len(deltas):.1f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
