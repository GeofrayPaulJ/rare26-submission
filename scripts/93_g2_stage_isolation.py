"""G2 -- differential test. Neither malloc_trim(0) every batch (91, ~15%
slope cut) nor recycling the SimpleITK reader every batch (92, no effect
at all) flattened the curve. Single-process is flat (G1 Part B); the real
8-worker DataLoader is not. This isolates WHICH stage of __getitem__,
running inside an actual persistent multiprocessing worker, is
responsible: three variants, same DataLoader config, same 3,000 images,
same lean RSS-only instrumentation, differing only in how much of
__getitem__ runs:

  stage=read       -- self.stack.read(i) only, return a dummy zero tensor
  stage=read_fov    -- + hashlib digest + detect_crop_box (FOV fit)
  stage=full        -- + preprocess() (the real StackDataset.__getitem__,
                        unmodified -- the baseline, same as G1's finding)

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/93_g2_stage_isolation.py'
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys
import time

import pandas as pd
import psutil
import torch
from torch.utils.data import Dataset

sys.path.insert(0, "/workspace/RARE26/submission")

from rare26_infer.fov import detect_crop_box  # noqa: E402
from rare26_infer.preprocess import CACHE_SIZE, IMAGE_SIZE, preprocess  # noqa: E402
from rare26_infer.stack import ItkStack  # noqa: E402

STACK_PATH = (
    "/workspace/RARE26/runs/submission_test/platform_probe_p5/c/interface_0/"
    "images/stacked-barretts-esophagus-endoscopy/stack.mha"
)
LOG_ROOT = "/workspace/RARE26/runs/submission_test/g2_stage_logs"
REPORT_ROOT = "/workspace/RARE26/reports"


def _collate(batch):
    idx = torch.tensor([b[0] for b in batch], dtype=torch.long)
    x = torch.stack([b[1] for b in batch])
    return idx, x


class StageDataset(Dataset):
    def __init__(self, stack, n, stage, log_dir):
        self.stack = stack
        self.n = n
        self.stage = stage
        self.log_dir = log_dir
        self._pid = None
        self._logf = None
        self._proc = None
        self._call_count = 0

    def __len__(self):
        return self.n

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

        arr = self.stack.read(i)
        if self.stage == "read":
            x = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
        else:
            _digest = hashlib.blake2b(arr.tobytes(), digest_size=16).digest()
            box, _used_fb, _fq, _reason = detect_crop_box(arr)
            if self.stage == "read_fov":
                x = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
            else:  # full
                chw = preprocess(arr, box, cache_size=CACHE_SIZE, image_size=IMAGE_SIZE)
                x = torch.from_numpy(chw)

        rss = self._proc.memory_info().rss
        self._logf.write(f"{self._pid},{self._call_count},{i},{rss}\n")
        self._logf.flush()
        return i, x


def run_stage(stage: str) -> None:
    from torch.utils.data import DataLoader

    log_dir = os.path.join(LOG_ROOT, stage)
    if os.path.isdir(log_dir):
        for f in glob.glob(os.path.join(log_dir, "*.csv")):
            os.remove(f)
    os.makedirs(log_dir, exist_ok=True)

    stack = ItkStack(STACK_PATH)
    n = len(stack)
    ds = StageDataset(stack, n, stage, log_dir)
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=8,
        pin_memory=False, collate_fn=_collate,
        persistent_workers=True, prefetch_factor=2,
    )

    print(f"[g2-stage:{stage}] running...", flush=True)
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
    for path in glob.glob(os.path.join(log_dir, "*.csv")):
        df = pd.read_csv(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(REPORT_ROOT, f"g2_stage_{stage}.parquet")
    out.to_parquet(out_path)

    deltas = []
    for pid, g in out.groupby("pid"):
        g = g.sort_values("call_idx")
        delta_mb = (g.iloc[-1]["rss_bytes"] - g.iloc[0]["rss_bytes"]) / 2**20
        deltas.append(delta_mb)
    print(f"[g2-stage:{stage}] {bi} batches, {len(deltas)} workers, "
          f"per-worker delta MiB: {[round(d,1) for d in deltas]} "
          f"(mean {sum(deltas)/len(deltas):.1f})", flush=True)


def main() -> int:
    for stage in ("read", "read_fov", "full"):
        run_stage(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
