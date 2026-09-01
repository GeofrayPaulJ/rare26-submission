"""G2 -- second candidate fix test. malloc_trim(0) every batch only cut
the growth ~15% (63->~55MB, scripts/91), not a flatten -- meaning most of
the growing RSS is still REFERENCED memory, not free-but-fragmented heap.
That redirects suspicion at rare26_infer.stack.ItkStack._handle(): it
builds ONE SimpleITK.ImageFileReader per worker process and reuses that
SAME reader object for every read() call for the worker's entire
lifetime (only rebuilt if os.getpid() changes, which never happens
mid-run). If ITK's pipeline caches per-call state inside that persistent
reader/IO object across repeated Execute() calls, this is exactly the
localised, worker-lifetime-scoped, gc-invisible signature G1 found.

Test: subclass ItkStack so the reader is recycled (closed and rebuilt)
every 32 reads (once per batch) instead of held for the worker's whole
life, and rerun the same lean RSS instrumentation. This does not touch
any shipped file.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/92_g2_reader_recycle_test.py'
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
LOG_DIR = "/workspace/RARE26/runs/submission_test/g2_recycle_test_logs"
REPORT_OUT = "/workspace/RARE26/reports/g2_reader_recycle_test.parquet"
RECYCLE_EVERY = 32  # once per batch


class RecyclingItkStack(ItkStack):
    """Same read() contract as ItkStack, except the SimpleITK reader is
    rebuilt every RECYCLE_EVERY calls instead of held for the process's
    entire life."""

    def __init__(self, path):
        super().__init__(path)
        self._reads_since_rebuild = 0

    def _handle(self):
        pid = os.getpid()
        if (self._reader is None or self._owner_pid != pid
                or self._reads_since_rebuild >= RECYCLE_EVERY):
            r = self._sitk.ImageFileReader()
            r.SetFileName(self.path)
            r.ReadImageInformation()
            self._reader = r
            self._owner_pid = pid
            self._reads_since_rebuild = 0
        return self._reader

    def read(self, i):
        out = super().read(i)
        self._reads_since_rebuild += 1
        return out


class RecycleTestDataset(StackDataset):
    def __init__(self, stack, n, log_dir):
        super().__init__(stack, n)
        self.log_dir = log_dir
        self._pid = None
        self._logf = None
        self._proc = None
        self._call_count = 0

    def _ensure_log(self):
        pid = os.getpid()
        if self._pid != pid:
            self._pid = pid
            self._proc = psutil.Process()
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"worker_{pid}.csv")
            self._logf = open(path, "a")
            if self._logf.tell() == 0:
                self._logf.write("pid,call_idx,item_i,rss_bytes\n")

    def __getitem__(self, i):
        self._ensure_log()
        self._call_count += 1
        result = super().__getitem__(i)
        rss = self._proc.memory_info().rss
        self._logf.write(f"{self._pid},{self._call_count},{i},{rss}\n")
        self._logf.flush()
        return result


def main() -> int:
    from torch.utils.data import DataLoader

    if os.path.isdir(LOG_DIR):
        for f in glob.glob(os.path.join(LOG_DIR, "*.csv")):
            os.remove(f)
    os.makedirs(LOG_DIR, exist_ok=True)

    stack = RecyclingItkStack(STACK_PATH)
    n = len(stack)
    ds = RecycleTestDataset(stack, n, LOG_DIR)
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
            print(f"[g2-recycle] batch {bi}", flush=True)
        bi += 1

    del it, loader
    time.sleep(1.0)

    frames = []
    for path in glob.glob(os.path.join(LOG_DIR, "*.csv")):
        df = pd.read_csv(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(REPORT_OUT)
    print(f"[g2-recycle] wrote {REPORT_OUT} ({len(out)} rows)")

    for pid, g in out.groupby("pid"):
        g = g.sort_values("call_idx")
        first, last = g.iloc[0], g.iloc[-1]
        mn, mx = g["rss_bytes"].min(), g["rss_bytes"].max()
        delta_mb = (last["rss_bytes"] - first["rss_bytes"]) / 2**20
        range_mb = (mx - mn) / 2**20
        # slope of second half vs first half -- does growth continue or stop?
        half = len(g) // 2
        first_half_delta = (g.iloc[half]["rss_bytes"] - g.iloc[0]["rss_bytes"]) / 2**20
        second_half_delta = (g.iloc[-1]["rss_bytes"] - g.iloc[half]["rss_bytes"]) / 2**20
        print(f"    pid={pid} n={len(g)} "
              f"first->last: {first['rss_bytes']/2**20:.1f}->{last['rss_bytes']/2**20:.1f} MiB "
              f"(net {delta_mb:+.1f} MiB, range {range_mb:.1f} MiB) "
              f"1st-half {first_half_delta:+.1f} MiB / 2nd-half {second_half_delta:+.1f} MiB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
