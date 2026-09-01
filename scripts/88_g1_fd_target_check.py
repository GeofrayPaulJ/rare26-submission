"""G1 supplement -- what are the main process's growing file descriptors
actually pointing at? scripts/87_g1_scaling_defect.py found main_num_fds
climbing 44->226 over 93 batches in lockstep with main_rss (658->818 MiB),
in discrete jumps every ~8 batches (matching num_workers=8) -- but only
logged the COUNT, not the targets. This reruns the same production
DataLoader for 40 batches and dumps the full /proc/self/fd target list
(readlink of every fd) at batch 0 and batch 39, so the diff shows exactly
what kind of object each new fd is (shared-memory segment, pipe, socket,
regular file...).

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/88_g1_fd_target_check.py'
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

sys.path.insert(0, "/workspace/RARE26/submission")

from rare26_infer.predict import StackDataset, _collate  # noqa: E402
from rare26_infer.stack import ItkStack  # noqa: E402

STACK_PATH = (
    "/workspace/RARE26/runs/submission_test/platform_probe_p5/c/interface_0/"
    "images/stacked-barretts-esophagus-endoscopy/stack.mha"
)
N_BATCHES = 40


def fd_targets():
    out = {}
    fd_dir = "/proc/self/fd"
    for entry in os.listdir(fd_dir):
        try:
            out[int(entry)] = os.readlink(os.path.join(fd_dir, entry))
        except OSError:
            pass
    return out


def main() -> int:
    from torch.utils.data import DataLoader

    stack = ItkStack(STACK_PATH)
    n = len(stack)
    ds = StackDataset(stack, n)
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=8,
        pin_memory=False, collate_fn=_collate,
        persistent_workers=True, prefetch_factor=2,
    )

    it = iter(loader)
    snapshots = {}
    for bi in range(N_BATCHES):
        next(it)
        if bi in (0, 1, N_BATCHES - 1):
            snapshots[bi] = fd_targets()
            print(f"[g1-fd] batch {bi}: {len(snapshots[bi])} fds open", flush=True)

    before = snapshots[1]
    after = snapshots[N_BATCHES - 1]
    new_fds = {k: v for k, v in after.items() if k not in before}
    print(f"\n[g1-fd] fds present at batch {N_BATCHES-1} but NOT at batch 1: "
          f"{len(new_fds)}")
    kind = Counter()
    for fd, target in sorted(new_fds.items()):
        if target.startswith("/dev/shm") or "shm" in target or "memfd" in target:
            k = "SHARED_MEMORY"
        elif target.startswith("pipe:"):
            k = "pipe"
        elif target.startswith("socket:"):
            k = "socket"
        elif target.startswith("anon_inode:"):
            k = f"anon_inode:{target.split(':',1)[1]}"
        else:
            k = "other:" + target.rsplit("/", 1)[-1][:40]
        kind[k] += 1
    print("[g1-fd] new-fd targets by kind:")
    for k, v in kind.most_common(20):
        print(f"    {v:4d}  {k}")

    print("\n[g1-fd] sample of 15 new fd targets, verbatim:")
    for fd, target in list(new_fds.items())[:15]:
        print(f"    fd={fd}  ->  {target}")

    del it, loader
    time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
