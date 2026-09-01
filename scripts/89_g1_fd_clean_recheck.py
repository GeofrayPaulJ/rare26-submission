"""G1 re-check -- 87's Part C showed main-process fds climbing 44->226 over
93 batches; 88's leaner script (same DataLoader config, no per-worker CSV
logging, no gc.get_objects()/Counter per batch) sampled only 3 points and
saw fds DROP (47->49->37) over the same batch range. That is a real
contradiction, not a rounding difference, and it matters: if the fd growth
is an artifact of 87's heavier instrumentation (opening log files, walking
gc.get_objects() every 8th worker call) rather than of the shipped
DataLoader/predict.py code itself, G1 must not report it as a production
defect. This settles it: same lean approach as 88, but samples psutil
num_fds()/RSS EVERY batch (not 3 snapshots) so a real trend, if any, is
visible at full resolution, and prints every value rather than a summary.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/89_g1_fd_clean_recheck.py'
"""
from __future__ import annotations

import sys
import time

import psutil

sys.path.insert(0, "/workspace/RARE26/submission")

from rare26_infer.predict import StackDataset, _collate  # noqa: E402
from rare26_infer.stack import ItkStack  # noqa: E402

STACK_PATH = (
    "/workspace/RARE26/runs/submission_test/platform_probe_p5/c/interface_0/"
    "images/stacked-barretts-esophagus-endoscopy/stack.mha"
)
N_BATCHES = 94


def main() -> int:
    from torch.utils.data import DataLoader

    stack = ItkStack(STACK_PATH)
    n = len(stack)
    ds = StackDataset(stack, n)  # UNWRAPPED, no profiling side-channel at all
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=8,
        pin_memory=False, collate_fn=_collate,
        persistent_workers=True, prefetch_factor=2,
    )

    proc = psutil.Process()
    it = iter(loader)
    print("batch,num_fds,rss_bytes", flush=True)
    for bi in range(N_BATCHES):
        try:
            next(it)
        except StopIteration:
            break
        print(f"{bi},{proc.num_fds()},{proc.memory_info().rss}", flush=True)

    del it, loader
    time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
