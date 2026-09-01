"""G1 -- find what accumulates. Zero GPU, production code paths.

Three parts, each strictly a superset of the previous one's code path, so a
growth signature can be pinned to the earliest part that shows it:

  PART A -- raw ItkStack.read(i) timing, single process, no dataset wrapper,
            no FOV, no preprocess. Isolates the file-read stage alone.
  PART B -- StackDataset.__getitem__(i) timing, single process, no
            DataLoader, no multiprocessing. Adds FOV fit + preprocess to A.
  PART C -- the REAL production DataLoader (StackDataset, batch_size=32,
            num_workers=8, persistent_workers=True, prefetch_factor=2,
            matching predict.py exactly) against probe C's actual 3,000-image
            stack.mha. Per-batch: wall seconds, main-process RSS/fd/gc, and
            (via a profiling-only Dataset subclass -- NOT a change to any
            shipped file) each worker's own RSS/fd/gc, logged from inside the
            worker process itself since gc.get_objects() cannot be read
            cross-process.

No model is built and CUDA is hidden (CUDA_VISIBLE_DEVICES=) -- the
suspects are all in the data path, and skipping the model keeps the whole
run inside the zero-GPU, ~90 min budget instead of the hours a CPU forward
pass through a 5-member ConvNeXt-Base ensemble over 3,000 images would cost.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && \
      CUDA_VISIBLE_DEVICES= python scripts/87_g1_scaling_defect.py'
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, "/workspace/RARE26/submission")

from rare26_infer.predict import StackDataset, _collate  # noqa: E402
from rare26_infer.stack import ItkStack  # noqa: E402
import rare26_infer.preprocess as pp_mod  # noqa: E402
import rare26_infer.fov as fov_mod  # noqa: E402
import rare26_infer.stack as stack_mod  # noqa: E402

REPO_ROOT = "/workspace/RARE26"
STACK_PATH = os.path.join(
    REPO_ROOT, "runs", "submission_test", "platform_probe_p5", "c", "interface_0",
    "images", "stacked-barretts-esophagus-endoscopy", "stack.mha",
)
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
WORKER_LOG_DIR = os.path.join(REPO_ROOT, "runs", "submission_test", "g1_worker_logs")
BATCH_SIZE = 32
NUM_WORKERS = 8  # min(8, os.cpu_count()) with cpu_count()=28, matches predict.py default
WORKER_LOG_EVERY = 8  # every 8th __getitem__ call inside a worker gets a full gc/rss sample


def module_level_state(mod, name):
    """Any dict/list/set defined at module scope -- where an unbounded cache would live."""
    hits = {}
    for k, v in vars(mod).items():
        if k.startswith("__"):
            continue
        if isinstance(v, (dict, list, set)):
            hits[k] = {"type": type(v).__name__, "len": len(v)}
    return {name: hits}


def bucket_mean(values, bucket_size):
    arr = np.asarray(values, dtype=np.float64)
    n_buckets = int(np.ceil(len(arr) / bucket_size))
    out = []
    for b in range(n_buckets):
        chunk = arr[b * bucket_size:(b + 1) * bucket_size]
        out.append(float(chunk.mean()))
    return out


# ---------------------------------------------------------------------------
# PART A -- raw read(i) timing
# ---------------------------------------------------------------------------
def part_a():
    print("[g1-A] raw ItkStack.read(i) timing, all slices, single process", flush=True)
    stack = ItkStack(STACK_PATH)
    n = len(stack)
    times = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter()
        stack.read(i)
        times[i] = time.perf_counter() - t0
    per_batch = bucket_mean(times, BATCH_SIZE)
    print(f"[g1-A] n={n} mean={times.mean()*1000:.3f}ms "
          f"first-bucket={per_batch[0]*1000:.3f}ms last-bucket={per_batch[-1]*1000:.3f}ms",
          flush=True)
    return {"n": n, "per_index_seconds": times.tolist(), "per_batch_mean_seconds": per_batch}


# ---------------------------------------------------------------------------
# PART B -- StackDataset.__getitem__(i) timing, single process
# ---------------------------------------------------------------------------
def part_b():
    print("[g1-B] StackDataset.__getitem__(i) timing, all slices, single process", flush=True)
    stack = ItkStack(STACK_PATH)
    n = len(stack)
    ds = StackDataset(stack, n)
    times = np.empty(n, dtype=np.float64)
    rss = np.empty(n, dtype=np.float64)
    proc = psutil.Process()
    for i in range(n):
        t0 = time.perf_counter()
        ds[i]
        times[i] = time.perf_counter() - t0
        if i % BATCH_SIZE == 0:
            rss[i] = proc.memory_info().rss
        else:
            rss[i] = np.nan
    per_batch = bucket_mean(times, BATCH_SIZE)
    print(f"[g1-B] n={n} mean={times.mean()*1000:.3f}ms "
          f"first-bucket={per_batch[0]*1000:.3f}ms last-bucket={per_batch[-1]*1000:.3f}ms",
          flush=True)
    return {
        "n": n, "per_index_seconds": times.tolist(), "per_batch_mean_seconds": per_batch,
        "rss_bytes_at_bucket_starts": [float(x) for x in rss if not np.isnan(x)],
    }


# ---------------------------------------------------------------------------
# PART C -- real production DataLoader, instrumented
# ---------------------------------------------------------------------------
class InstrumentedStackDataset(StackDataset):
    """StackDataset + a profiling side-channel. Behaviourally IDENTICAL to
    the shipped class (does not override __getitem__'s return value or
    control flow) -- only wraps it to log RSS/fd/gc from inside whichever
    process (main or a persistent worker) actually executes each call."""

    def __init__(self, stack, n, log_dir, log_every):
        super().__init__(stack, n)
        self.log_dir = log_dir
        self.log_every = log_every
        self._call_count = 0
        self._pid = None
        self._logf = None

    def _ensure_log(self):
        pid = os.getpid()
        if self._pid != pid:
            self._pid = pid
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, f"worker_{pid}.csv")
            self._logf = open(path, "a")
            if self._logf.tell() == 0:
                self._logf.write(
                    "pid,call_idx,item_i,t,rss_bytes,num_fds,n_gc_objects,top_types,"
                    "fd_targets_sample\n"
                )

    def __getitem__(self, i):
        self._ensure_log()
        self._call_count += 1
        result = super().__getitem__(i)
        if self._call_count % self.log_every == 0 or self._call_count <= 3:
            p = psutil.Process()
            rss = p.memory_info().rss
            try:
                nfd = p.num_fds()
            except Exception:
                nfd = -1
            fd_targets = []
            try:
                fd_dir = f"/proc/{os.getpid()}/fd"
                for entry in os.listdir(fd_dir)[:200]:
                    try:
                        fd_targets.append(os.readlink(os.path.join(fd_dir, entry)))
                    except OSError:
                        pass
            except OSError:
                pass
            fd_kind = Counter(
                t.rsplit("/", 1)[-1].split(".")[-1] if "." in t.rsplit("/", 1)[-1] else t
                for t in fd_targets
            )
            objs = gc.get_objects()
            n_obj = len(objs)
            types = Counter(type(o).__name__ for o in objs)
            top = ";".join(f"{k}:{v}" for k, v in types.most_common(5))
            fd_sample = ";".join(f"{k}:{v}" for k, v in fd_kind.most_common(5))
            self._logf.write(
                f"{self._pid},{self._call_count},{i},{time.perf_counter()},"
                f"{rss},{nfd},{n_obj},\"{top}\",\"{fd_sample}\"\n"
            )
            self._logf.flush()
        return result


def part_c():
    from torch.utils.data import DataLoader

    print(f"[g1-C] production DataLoader: batch_size={BATCH_SIZE} "
          f"num_workers={NUM_WORKERS} persistent_workers=True prefetch_factor=2", flush=True)

    if os.path.isdir(WORKER_LOG_DIR):
        for f in glob.glob(os.path.join(WORKER_LOG_DIR, "*.csv")):
            os.remove(f)
    os.makedirs(WORKER_LOG_DIR, exist_ok=True)

    stack = ItkStack(STACK_PATH)
    n = len(stack)
    ds = InstrumentedStackDataset(stack, n, WORKER_LOG_DIR, WORKER_LOG_EVERY)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=False, collate_fn=_collate,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    main_proc = psutil.Process()
    rows = []
    it = iter(loader)
    t_prev = time.perf_counter()
    bi = 0
    while True:
        try:
            batch = next(it)
        except StopIteration:
            break
        now = time.perf_counter()
        wall = now - t_prev
        t_prev = now

        objs = gc.get_objects()
        types = Counter(type(o).__name__ for o in objs)
        try:
            nfd_main = main_proc.num_fds()
        except Exception:
            nfd_main = -1

        rows.append({
            "batch": bi,
            "wall_seconds": wall,
            "n_items": int(batch[0].shape[0]) if hasattr(batch[0], "shape") else len(batch[0]),
            "main_rss_bytes": main_proc.memory_info().rss,
            "main_num_fds": nfd_main,
            "main_n_gc_objects": len(objs),
            "main_top_gc_types": ";".join(f"{k}:{v}" for k, v in types.most_common(5)),
        })
        if bi % 10 == 0:
            print(f"[g1-C]   batch {bi:3d}  wall={wall:6.3f}s  "
                  f"main_rss={rows[-1]['main_rss_bytes']/2**20:7.1f}MiB  "
                  f"main_fds={nfd_main}  main_gc_objs={rows[-1]['main_n_gc_objects']}",
                  flush=True)
        bi += 1

    # persistent_workers=True keeps workers alive until the loader/iterator is
    # garbage-collected; drop the reference so their processes exit and their
    # log files are flushed/closed before we read them back.
    del it, loader
    time.sleep(1.0)

    df = pd.DataFrame(rows)

    worker_logs = glob.glob(os.path.join(WORKER_LOG_DIR, "*.csv"))
    worker_frames = []
    for path in worker_logs:
        try:
            wdf = pd.read_csv(path)
            wdf["source_pid"] = os.path.basename(path)
            worker_frames.append(wdf)
        except Exception as exc:  # noqa: BLE001
            print(f"[g1-C] WARNING: could not parse {path}: {exc}")
    worker_df = pd.concat(worker_frames, ignore_index=True) if worker_frames else pd.DataFrame()

    print(f"[g1-C] {len(worker_logs)} worker log files, {len(worker_df)} sampled worker calls",
          flush=True)

    return df, worker_df


def main() -> int:
    module_state = {}
    module_state.update(module_level_state(pp_mod, "rare26_infer.preprocess"))
    module_state.update(module_level_state(fov_mod, "rare26_infer.fov"))
    module_state.update(module_level_state(stack_mod, "rare26_infer.stack"))
    print("[g1] module-level dict/list/set state (candidate unbounded caches):")
    print(json.dumps(module_state, indent=2))

    a = part_a()
    b = part_b()
    c_df, worker_df = part_c()

    out_json = os.path.join(REPORT_DIR, "g1_scaling_defect_raw.json")
    with open(out_json, "w") as fh:
        json.dump({"module_level_state": module_state, "part_a": a, "part_b": b}, fh)
    print(f"[g1] wrote {out_json}")

    c_parquet = os.path.join(REPORT_DIR, "g1_scaling_defect_partc.parquet")
    c_df.to_parquet(c_parquet)
    print(f"[g1] wrote {c_parquet} ({len(c_df)} rows)")

    if len(worker_df):
        w_parquet = os.path.join(REPORT_DIR, "g1_scaling_defect_workers.parquet")
        worker_df.to_parquet(w_parquet)
        print(f"[g1] wrote {w_parquet} ({len(worker_df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
