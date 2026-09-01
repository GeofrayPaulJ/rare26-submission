"""W1 supplement -- does num_workers=2 still leak worker-side, the way
num_workers=8 did (scripts/90_g1_worker_rss_clean.py)? num_workers=0 has
no forked worker at all (main process IS the reader), already confirmed
flat by scripts/98's main-process RSS tracking -- that test is complete
for nw=0 by construction. nw=2 and nw=8 both fork, and G1's own
root-cause localisation was "fork-specific," not "8-worker-specific," so
the prior is that nw=2 leaks too, just with 4x fewer processes sharing
the same total image count. Confirming rather than assuming.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/99_w1_nw2_worker_rss_check.py'
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
LOG_DIR = "/workspace/RARE26/runs/submission_test/w1_nw2_worker_logs"
REPORT_OUT = "/workspace/RARE26/reports/w1_nw2_worker_rss.parquet"


class LeanInstrumentedStackDataset(StackDataset):
    def __init__(self, stack, n, log_dir):
        super().__init__(stack, n)
        self.log_dir = log_dir
        self._pid = None
        self._logf = None
        self._proc = None

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
        result = super().__getitem__(i)
        rss = self._proc.memory_info().rss
        self._logf.write(f"{self._pid},{i},{i},{rss}\n")
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
        ds, batch_size=32, shuffle=False, num_workers=2,
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
            print(f"[w1-nw2] batch {bi}", flush=True)
        bi += 1
    del it, loader
    time.sleep(1.0)

    frames = []
    for path in glob.glob(os.path.join(LOG_DIR, "*.csv")):
        df = pd.read_csv(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(REPORT_OUT)
    print(f"[w1-nw2] wrote {REPORT_OUT} ({len(out)} rows, {len(frames)} workers)")

    for pid, g in out.groupby("pid"):
        g = g.sort_values("call_idx")
        delta_mb = (g.iloc[-1]["rss_bytes"] - g.iloc[0]["rss_bytes"]) / 2**20
        print(f"    pid={pid} n={len(g)} "
              f"rss: {g.iloc[0]['rss_bytes']/2**20:.1f} -> {g.iloc[-1]['rss_bytes']/2**20:.1f} MiB "
              f"(delta {delta_mb:+.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
