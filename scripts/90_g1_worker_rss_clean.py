"""G1 re-check, worker side -- 87's Part C worker log showed each of the 8
workers' RSS climbing ~70MB over ~350-380 __getitem__ calls, but that
measurement was taken from inside a wrapper that also called
gc.get_objects() (a full walk + Counter over ~233k live objects) and
os.readlink on up to 200 fds every 8th call. 89 already showed the
MAIN process's apparent fd/RSS growth in 87 was an artifact of exactly
that kind of heavy per-batch introspection, not a real production
behaviour. This settles the same question for the WORKER side: same
lean approach (psutil RSS + num_fds only, no gc introspection, no fd
target walk), sampled every item in every worker, over the full 3,000
image / 94 batch run.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/90_g1_worker_rss_clean.py'
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
LOG_DIR = "/workspace/RARE26/runs/submission_test/g1_worker_logs_clean"
REPORT_OUT = "/workspace/RARE26/reports/g1_worker_rss_clean.parquet"


class LeanInstrumentedStackDataset(StackDataset):
    """Same idea as 87's InstrumentedStackDataset but psutil-only: no
    gc.get_objects(), no Counter, no /proc/pid/fd walk. Logs EVERY call
    (not every 8th) since the whole point is to see if RSS/fd tick up
    on every single item once the gc/fd-walk overhead is removed."""

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
                self._logf.write("pid,call_idx,item_i,rss_bytes,num_fds\n")

    def __getitem__(self, i):
        self._ensure_log()
        self._call_count += 1
        result = super().__getitem__(i)
        rss = self._proc.memory_info().rss
        try:
            nfd = self._proc.num_fds()
        except Exception:
            nfd = -1
        self._logf.write(f"{self._pid},{self._call_count},{i},{rss},{nfd}\n")
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
    ds = LeanInstrumentedStackDataset(stack, n, LOG_DIR)
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
            print(f"[g1-90] batch {bi}", flush=True)
        bi += 1

    del it, loader
    time.sleep(1.0)

    frames = []
    for path in glob.glob(os.path.join(LOG_DIR, "*.csv")):
        df = pd.read_csv(path)
        df["source_pid"] = os.path.basename(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(REPORT_OUT)
    print(f"[g1-90] wrote {REPORT_OUT} ({len(out)} rows, {len(frames)} workers)")

    for pid, g in out.groupby("pid"):
        g = g.sort_values("call_idx")
        first, last = g.iloc[0], g.iloc[-1]
        delta_mb = (last["rss_bytes"] - first["rss_bytes"]) / 2**20
        print(f"    pid={pid} n_calls={len(g)} "
              f"rss: {first['rss_bytes']/2**20:.1f} -> {last['rss_bytes']/2**20:.1f} MiB "
              f"(delta {delta_mb:+.1f} MiB)  "
              f"fds: {first['num_fds']} -> {last['num_fds']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
