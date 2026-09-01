"""Read-only tqdm progress view for a run_cv.py-style out_dir, e.g. the
currently-running runs/a4_pinned085 (item 5 pinned-arm chain).

DOES NOT TOUCH THE RUNNING TRAINING PROCESS. Polls only two things that
process is already writing on its own: run_index.json (for which units are
done) and each unit's train_log.jsonl (for which epoch the in-progress unit
is on). No signals sent, no files written, no subprocess started -- safe to
run alongside a live training run without any risk of disturbing it.

    python scripts/watch_progress.py --out-dir runs/a4_pinned085
    python scripts/watch_progress.py --out-dir runs/a4_pinned085 --once   # one snapshot, no loop
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

from tqdm.auto import tqdm


def units_done(out_dir: str) -> set:
    idx_path = os.path.join(out_dir, "run_index.json")
    if not os.path.exists(idx_path):
        return set()
    try:
        with open(idx_path) as fh:
            units = json.load(fh).get("units", {})
    except (OSError, json.JSONDecodeError):
        return set()
    return {name for name, rec in units.items()
           if rec.get("status") in ("ok", "skipped")}


def current_unit_and_epoch(out_dir: str, done: set) -> tuple:
    """Most-recently-written train_log.jsonl among not-yet-done units, and
    the highest epoch number logged in it."""
    candidates = []
    for d in glob.glob(os.path.join(out_dir, "r*_f*_s*")):
        name = os.path.basename(d)
        if name in done:
            continue
        log_path = os.path.join(d, "train_log.jsonl")
        if os.path.exists(log_path):
            candidates.append((os.path.getmtime(log_path), name, log_path))
    if not candidates:
        return None, 0
    candidates.sort()
    _, name, log_path = candidates[-1]
    last_epoch = 0
    try:
        with open(log_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "epoch":
                    last_epoch = rec.get("epoch", last_epoch)
    except OSError:
        pass
    return name, last_epoch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--total-units", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--once", action="store_true",
                    help="print one snapshot and exit, no live bars")
    args = ap.parse_args()

    if args.once:
        done = units_done(args.out_dir)
        cur, ep = current_unit_and_epoch(args.out_dir, done)
        print(f"{len(done)}/{args.total_units} units done. "
             f"current: {cur or 'n/a'} epoch {ep}/{args.epochs}")
        return 0

    outer = tqdm(total=args.total_units, desc="units", position=0,
                dynamic_ncols=True, unit="unit")
    inner = tqdm(total=args.epochs, desc="epoch", position=1, leave=False,
                dynamic_ncols=True, unit="ep")

    seen_done: set = set()
    last_current = None
    last_epoch_val = 0
    try:
        while len(seen_done) < args.total_units:
            done = units_done(args.out_dir)
            newly = done - seen_done
            if newly:
                outer.update(len(newly))
                seen_done |= newly

            cur, ep = current_unit_and_epoch(args.out_dir, seen_done)
            if cur != last_current:
                inner.reset(total=args.epochs)
                inner.set_description(f"epoch [{cur or 'waiting'}]")
                last_current, last_epoch_val = cur, 0
            if ep > last_epoch_val:
                inner.update(ep - last_epoch_val)
                last_epoch_val = ep

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        inner.close()
        outer.close()
    if len(seen_done) >= args.total_units:
        print(f"all {args.total_units} units done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
