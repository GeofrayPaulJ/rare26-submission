"""G2 -- candidate fix test, BEFORE touching predict.py.

G1 (reports/g1_scaling_defect.md) localised the leak to native heap growth
inside persistent DataLoader worker processes (~170 KB/item, uniform
across workers, absent in a single-process control, uncorrelated with fd
or Python object counts) -- the textbook signature of glibc malloc arenas
never returning freed memory to the OS under a long-lived process doing
many size-varying allocate/free cycles. The standard mitigation is a
periodic malloc_trim(0) call, which asks glibc to release what it can.

This tests that BEFORE editing any shipped file: same lean instrumentation
as scripts/90_g1_worker_rss_clean.py, but the Dataset calls malloc_trim(0)
every 32 items (once per batch) inside the worker. If RSS flattens, this
is the fix. If it only slows the slope, G2 must say so rather than ship it.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/91_g2_malloc_trim_test.py'
"""
from __future__ import annotations

import ctypes
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
LOG_DIR = "/workspace/RARE26/runs/submission_test/g2_trim_test_logs"
REPORT_OUT = "/workspace/RARE26/reports/g2_malloc_trim_test.parquet"
TRIM_EVERY = 32  # once per batch


class TrimTestDataset(StackDataset):
    def __init__(self, stack, n, log_dir):
        super().__init__(stack, n)
        self.log_dir = log_dir
        self._pid = None
        self._logf = None
        self._proc = None
        self._call_count = 0
        self._libc = None

    def _ensure_log(self):
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
                self._logf.write("pid,call_idx,item_i,rss_bytes,trimmed\n")

    def __getitem__(self, i):
        self._ensure_log()
        self._call_count += 1
        result = super().__getitem__(i)
        trimmed = 0
        if self._libc is not None and self._call_count % TRIM_EVERY == 0:
            self._libc.malloc_trim(0)
            trimmed = 1
        rss = self._proc.memory_info().rss
        self._logf.write(f"{self._pid},{self._call_count},{i},{rss},{trimmed}\n")
        self._logf.flush()
        return result


def main() -> int:
    from torch.utils.data import DataLoader

    if os.path.isdir(LOG_DIR):
        for f in glob.glob(os.path.join(LOG_DIR, "*.csv")):
            os.remove(f)
    os.makedirs(LOG_DIR, exist_ok=True)

    stack = ItkStack(STACK_PATH)
    n = len(stack)
    ds = TrimTestDataset(stack, n, LOG_DIR)
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=8,
        pin_memory=False, collate_fn=_collate,
        persistent_workers=True, prefetch_factor=2,
    )

    it = iter(loader)
    bi = 0
    while True:
        try:
            next(it)
        except StopIteration:
            break
        if bi % 20 == 0:
            print(f"[g2-trim] batch {bi}", flush=True)
        bi += 1

    del it, loader
    time.sleep(1.0)

    frames = []
    for path in glob.glob(os.path.join(LOG_DIR, "*.csv")):
        df = pd.read_csv(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(REPORT_OUT)
    print(f"[g2-trim] wrote {REPORT_OUT} ({len(out)} rows)")

    for pid, g in out.groupby("pid"):
        g = g.sort_values("call_idx")
        first, last = g.iloc[0], g.iloc[-1]
        mn, mx = g["rss_bytes"].min(), g["rss_bytes"].max()
        delta_mb = (last["rss_bytes"] - first["rss_bytes"]) / 2**20
        range_mb = (mx - mn) / 2**20
        print(f"    pid={pid} n={len(g)} "
              f"first->last: {first['rss_bytes']/2**20:.1f}->{last['rss_bytes']/2**20:.1f} MiB "
              f"(net {delta_mb:+.1f} MiB, range {range_mb:.1f} MiB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
