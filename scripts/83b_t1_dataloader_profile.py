"""T1 addendum -- the three named stages (read/FOV/preprocess) sum to
~12.4 ms/image (2.5s for 200 images), a 44x gap from the try-out's
observed 110.9s data time. The remaining lead is DataLoader
multiprocessing overhead: num_workers=4, persistent_workers=True,
prefetch_factor=2, matching predict.py exactly. This reproduces that
DataLoader (CPU-only, no model, no GPU) against the native-path probe
and times it batch by batch, to check for the same "batch 1 slow,
batches 2-6 still non-trivial" signature the platform log showed
(b1 61.55s, b2-b6 8-15s).

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/83b_t1_dataloader_profile.py'
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join("/workspace/RARE26", "submission"))

from rare26_infer.predict import StackDataset, _collate  # noqa: E402
from rare26_infer.stack import ItkStack  # noqa: E402

NATIVE_MHA = "/root/t1_native/stack.mha"
REPORT_JSON = "/workspace/RARE26/reports/t1_dataloader_profile.json"


def main() -> int:
    from torch.utils.data import DataLoader

    stack = ItkStack(NATIVE_MHA)
    n = len(stack)
    ds = StackDataset(stack, n)

    for batch_size, num_workers in [(32, 4), (32, 0)]:
        print(f"\n[t1b] batch_size={batch_size} num_workers={num_workers}", flush=True)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=False, collate_fn=_collate,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        t_loop = time.perf_counter()
        batch_times = []
        t_prev = t_loop
        for bi, batch in enumerate(loader):
            now = time.perf_counter()
            batch_times.append(now - t_prev)
            t_prev = now
        total = time.perf_counter() - t_loop
        print(f"    total={total:.2f}s  n_batches={len(batch_times)}  "
             f"per-batch={[round(t, 2) for t in batch_times]}")
        yield_data = {"batch_size": batch_size, "num_workers": num_workers,
                      "total_seconds": total, "per_batch_seconds": batch_times}
        globals().setdefault("_results", []).append(yield_data)

    with open(REPORT_JSON, "w") as fh:
        json.dump(_results, fh, indent=2)
    print(f"\n[t1b] written: {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
