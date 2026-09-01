"""Orchestrate N repeats x 5 folds x M seeds, or the two LOCO splits, from one config.

WHAT THIS IS. A sequential driver for one GPU. It owns no training logic at all
-- every unit of work is a fresh ``python -m src.train`` subprocess. That
isolation is deliberate and load-bearing:

  * a VRAM spill guard trip, a CUDA OOM or a segfault kills one run, not the
    night's remaining 24;
  * GPU memory is returned to the driver between units by process exit, which is
    the only way to be sure of it;
  * the orchestrator's own address space stays tiny, so it is not itself a
    candidate for the OOM killer.

DETACHED-SAFE. The host has already killed a container mid-run once. Every
durable write here goes through a temp file plus ``os.replace`` (atomic on both
NTFS and ext4), so a kill at any instant leaves either the old state or the new
one, never a torn file. Re-invoking with the same arguments re-derives the plan
from what is on disk and continues; there is no state held only in memory and no
"resume from" flag to remember. Killing this process and running the identical
command again is the supported recovery path.

RESUMPTION IS THREE-WAY, per unit:
    valid canonical parquet          -> SKIP, no GPU time spent
    checkpoint but no valid parquet  -> RESUME from checkpoints/last.pt
    neither                          -> train from scratch
A unit that completed all its epochs but lost its canonical parquet is repaired
by the resume path without retraining: src/train.py re-copies the dump from the
last epoch's prediction file when the loaded history already satisfies the
schedule.

EXECUTION ORDER is seed-major, fold-minor -- all 5 folds of seed 0, then all 5
of seed 1. Level 2 (pooled out-of-fold) needs every fold of one seed to exist
before it can be computed at all, so this ordering means an interrupted night
yields K complete Level-2 estimates rather than 25 fragments of none.

VALIDITY IS CHECKED, NOT ASSUMED. "The parquet exists" is not the same claim as
"the run finished and its predictions are usable". See validate_predictions:
schema, row count against the split definition, label counts, seed/fold
provenance columns, finite logits, and a non-degenerate logit distribution are
all verified before a unit is allowed to be skipped.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import Config  # noqa: E402
from src.folds import get_holdout_split, get_split  # noqa: E402
from src.io import durable_replace, fsync_dir, read_predictions  # noqa: E402
from src.train import split_name  # noqa: E402

# run_index.json doubles as the heartbeat: its updated_utc advances after every
# unit, so an outside observer can tell a working runner from a wedged one.
INDEX_NAME = "run_index.json"
PROVENANCE_NAME = "provenance.json"

# Files whose contents define "the code that produced these numbers". Hashed
# together into one digest -- see source_digest().
PROVENANCE_SOURCES = (
    "src/config.py", "src/data.py", "src/folds.py", "src/io.py",
    "src/metrics.py", "src/model.py", "src/augment.py", "src/seeding.py",
    "src/train.py", "src/evaluate.py", "scripts/run_cv.py", "scripts/08_score.py",
)


# ---------------------------------------------------------------------------
# Atomic durability
# ---------------------------------------------------------------------------
def write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Serialise, fsync, then rename into place.

    The fsync matters as much as the rename: without it the rename can be
    durable while the bytes it points at are still in the page cache, which is
    exactly the state a container kill exposes. durable_replace additionally
    fsyncs the parent directory, which is what makes the new NAME survive a
    host reset and not merely the new bytes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    durable_replace(tmp, path)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_sha() -> Optional[str]:
    """Current commit, or None when this tree is not a git checkout.

    Returns None rather than raising or inventing a placeholder: the honest
    answer to "which commit produced this" is sometimes "there is no commit",
    and a caller that records a fake SHA is worse than one that records nothing.
    source_digest() is what carries provenance in that case.
    """
    try:
        out = subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def source_digest() -> Dict[str, Any]:
    """SHA-256 over the source files that determine a run's numbers.

    This is the provenance record when there is no git SHA, and a cross-check on
    the working tree when there is one -- a dirty checkout has a valid SHA that
    does not describe the code that actually ran. Per-file digests are kept as
    well as the combined one so a later mismatch can be localised instead of
    merely detected.
    """
    per_file: Dict[str, Optional[str]] = {}
    combined = hashlib.sha256()
    for rel in PROVENANCE_SOURCES:
        p = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(p):
            per_file[rel] = None
            combined.update(f"{rel}:ABSENT\n".encode())
            continue
        with open(p, "rb") as fh:
            d = hashlib.sha256(fh.read()).hexdigest()
        per_file[rel] = d
        combined.update(f"{rel}:{d}\n".encode())
    return {"combined_sha256": combined.hexdigest(), "files": per_file}


def environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch

        env.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            gpu_count=torch.cuda.device_count(),
        )
    except Exception as exc:  # torch missing/broken must not stop the plan print
        env["torch_error"] = repr(exc)
    return env


# ---------------------------------------------------------------------------
# Units of work
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Unit:
    """One (repeat, fold, seed) or one (holdout centre, seed)."""

    seed: int
    repeat: int = 0
    fold: int = 0
    holdout_centre: Optional[int] = None

    @property
    def mode(self) -> str:
        return "loco" if self.holdout_centre is not None else "cv"

    @property
    def name(self) -> str:
        if self.holdout_centre is not None:
            return f"loco_c{self.holdout_centre}_s{self.seed}"
        return f"r{self.repeat}_f{self.fold}_s{self.seed}"

    def key(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "seed": self.seed, "repeat": self.repeat,
            "fold": self.fold, "holdout_centre": self.holdout_centre,
            "name": self.name,
        }

    def apply(self, cfg: Config) -> Config:
        return dataclasses.replace(
            cfg, seed=self.seed, repeat=self.repeat, fold=self.fold,
            holdout_centre=self.holdout_centre,
        )

    def pred_fold(self) -> int:
        """The value src/train.py writes into the parquet's `fold` column."""
        return self.fold if self.holdout_centre is None else -self.holdout_centre


def plan_units(
    mode: str,
    repeats: Sequence[int],
    folds: Sequence[int],
    seeds: Sequence[int],
    centres: Sequence[int],
) -> List[Unit]:
    """Enumerate the work, seed-major (see module docstring for why)."""
    units: List[Unit] = []
    if mode == "cv":
        for repeat in repeats:
            for seed in seeds:
                for fold in folds:
                    units.append(Unit(seed=seed, repeat=repeat, fold=fold))
    elif mode == "loco":
        for seed in seeds:
            for centre in centres:
                units.append(Unit(seed=seed, holdout_centre=centre))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return units


def expected_split(unit: Unit, cfg: Config) -> Dict[str, int]:
    """Row and class counts this unit's held-out set must have, read from the
    manifest rather than hardcoded, so a manifest change fails loudly here
    instead of silently shifting what every downstream statistic means."""
    if unit.holdout_centre is not None:
        train, val = get_holdout_split(unit.holdout_centre, cfg.manifest)
    else:
        train, val = get_split(unit.repeat, unit.fold, cfg.manifest)
    df = pd.read_csv(cfg.manifest).set_index("filepath")
    lab = df.loc[list(val), "class_label"]
    return {
        "n_train": len(train),
        "n_val": len(val),
        "n_pos": int((lab == "neoplasia").sum()),
        "n_neg": int((lab != "neoplasia").sum()),
    }


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------
def validate_predictions(
    path: str, unit: Unit, expected: Dict[str, int]
) -> Tuple[bool, str]:
    """Is this parquet a complete, trustworthy dump for this unit?

    Returns (ok, reason). Every failure mode below has been seen in the wild or
    is a direct consequence of a kill at the wrong instant, and each one would
    otherwise corrupt a pooled statistic silently rather than loudly:

      * missing/unreadable      -- killed before or during the dump
      * wrong schema            -- written by an older or different code path
      * wrong row count         -- the split moved under us
      * wrong class balance     -- ditto, and it changes every metric
      * duplicate filepaths     -- would double-weight images in the pool
      * seed/fold mismatch      -- the run directory holds another unit's output
      * non-finite logits       -- poisons the bootstrap silently
      * <=1 distinct logit      -- a collapsed model, not a usable ranking
    """
    if not os.path.exists(path):
        return False, "absent"
    try:
        df = read_predictions(path)
    except Exception as exc:
        return False, f"unreadable ({type(exc).__name__}: {exc})"

    if len(df) != expected["n_val"]:
        return False, f"row count {len(df)} != expected {expected['n_val']}"
    n_pos = int((df["label_int"] == 1).sum())
    n_neg = int((df["label_int"] == 0).sum())
    if n_pos != expected["n_pos"] or n_neg != expected["n_neg"]:
        return False, (f"class counts pos={n_pos}/neg={n_neg} != expected "
                       f"pos={expected['n_pos']}/neg={expected['n_neg']}")
    if df["filepath"].duplicated().any():
        n = int(df["filepath"].duplicated().sum())
        return False, f"{n} duplicate filepaths"

    seeds = df["seed"].unique()
    if len(seeds) != 1 or int(seeds[0]) != unit.seed:
        return False, f"seed column {seeds.tolist()} != unit seed {unit.seed}"
    folds = df["fold"].unique()
    if len(folds) != 1 or int(folds[0]) != unit.pred_fold():
        return False, f"fold column {folds.tolist()} != expected {unit.pred_fold()}"

    logit = df["logit"].to_numpy()
    if not np.isfinite(logit).all():
        return False, "non-finite logits"
    n_unique = int(np.unique(logit).size)
    if n_unique <= 1:
        return False, f"degenerate logits ({n_unique} distinct value)"
    return True, f"ok ({len(df)} rows, {n_unique} distinct logits)"


def run_state(run_dir: str, unit: Unit, cfg: Config,
              expected: Dict[str, int]) -> Tuple[str, str]:
    """Classify a unit as done / resumable / fresh, with the reason."""
    canonical = os.path.join(run_dir, f"val_{split_name(unit.apply(cfg))}.parquet")
    ok, reason = validate_predictions(canonical, unit, expected)
    if ok:
        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as fh:
                    s = json.load(fh)
                done = int(s.get("completed_epochs", 0))
                if done < cfg.epochs:
                    return "resume", (f"parquet valid but summary says "
                                      f"{done}/{cfg.epochs} epochs")
            except (OSError, json.JSONDecodeError) as exc:
                return "resume", f"parquet valid but summary unreadable ({exc})"
        return "done", reason
    if os.path.exists(os.path.join(run_dir, "checkpoints", "last.pt")):
        return "resume", f"checkpoint present; parquet {reason}"
    return "fresh", f"parquet {reason}"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
class Aborted(Exception):
    """SIGINT/SIGTERM reached the orchestrator."""


_stop = {"flag": False}


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        _stop["flag"] = True
        print(f"\n[runner] signal {signum} received -- finishing the current "
              f"unit's cleanup, then stopping. Re-run the same command to "
              f"continue.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass  # not on the main thread / unsupported platform


def train_argv(unit: Unit, cfg_path: str, out_dir: str, epochs: Optional[int],
               resume: bool) -> List[str]:
    argv = [
        sys.executable, "-u", "-m", "src.train",
        "--config", cfg_path,
        "--seed", str(unit.seed),
        "--out-dir", out_dir,
        "--no-progress",
    ]
    if unit.holdout_centre is not None:
        argv += ["--holdout-centre", str(unit.holdout_centre)]
    else:
        argv += ["--repeat", str(unit.repeat), "--fold", str(unit.fold)]
    if epochs is not None:
        argv += ["--epochs", str(epochs)]
    if resume:
        argv += ["--resume"]
    return argv


def write_provenance(run_dir: str, unit: Unit, cfg: Config, argv: List[str],
                     prov_common: Dict[str, Any]) -> None:
    """Record, next to the predictions, everything needed to reproduce them.

    The seeding dict is read back out of train_log.jsonl's run_start record
    rather than recomputed here: what belongs in the provenance file is the
    seeding that was ACTUALLY applied by seed_everything in the training
    process, not this process's prediction of what it would be.
    """
    seeding: Optional[Dict[str, Any]] = None
    log_path = os.path.join(run_dir, "train_log.jsonl")
    if os.path.exists(log_path):
        try:
            with open(log_path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("event") == "run_start" and "seeding" in rec:
                        seeding = rec["seeding"]  # last run_start wins
        except (OSError, json.JSONDecodeError):
            pass
    write_json_atomic(
        os.path.join(run_dir, PROVENANCE_NAME),
        {
            "unit": unit.key(),
            "config": cfg.to_dict(),
            "seeding": seeding,
            "seeding_source": "train_log.jsonl run_start record (as applied)",
            "command": argv,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **prov_common,
        },
    )


def execute(unit: Unit, cfg: Config, cfg_path: str, out_dir: str,
            log_dir: str, epochs: Optional[int], resume: bool,
            timeout_s: float) -> Dict[str, Any]:
    """Run one unit to completion as a subprocess. Never raises on a training
    failure -- the caller decides whether one bad unit ends the night."""
    argv = train_argv(unit, cfg_path, out_dir, epochs, resume)
    log_path = os.path.join(log_dir, f"{unit.name}.log")
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    # append, not truncate: a resumed unit's earlier attempt is part of its
    # history and is what a post-mortem needs
    with open(log_path, "a") as log:
        log.write(f"\n{'=' * 78}\n[runner] {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"{'RESUME' if resume else 'START'} {unit.name}\n"
                  f"[runner] {' '.join(argv)}\n{'=' * 78}\n")
        log.flush()
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=log,
                                stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc, timed_out = -9, True
        log.write(f"[runner] exit={rc} timed_out={timed_out} "
                  f"elapsed={time.time() - t0:.1f}s\n")
    return {
        "returncode": rc,
        "timed_out": timed_out,
        "seconds": time.time() - t0,
        "log": os.path.relpath(log_path, REPO_ROOT),
        "command": argv,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _ints(s: str) -> List[int]:
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sequential CV / LOCO runner. Resumable and detached-safe: "
                    "re-run the identical command to continue after any kill."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="cv", choices=["cv", "loco"])
    ap.add_argument("--repeats", default="0", help="cv mode: comma-separated")
    ap.add_argument("--folds", default="0,1,2,3,4", help="cv mode")
    ap.add_argument("--centres", default="1,2", help="loco mode")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out-dir", default=None,
                    help="overrides the config's out_dir; all runs land here")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--max-runs", type=int, default=None, help="stop after N units")
    ap.add_argument("--timeout-min", type=float, default=180.0,
                    help="per-unit wall clock cap before the child is killed")
    ap.add_argument("--retries", type=int, default=1,
                    help="extra attempts per failed unit (the retry resumes)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and each unit's state, run nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-run units already holding a valid parquet")
    args = ap.parse_args(argv)

    base_cfg = Config.from_yaml(args.config)
    out_dir = args.out_dir or base_cfg.out_dir
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(REPO_ROOT, out_dir)
    log_dir = os.path.join(out_dir, "_logs")
    os.makedirs(out_dir, exist_ok=True)

    units = plan_units(args.mode, _ints(args.repeats), _ints(args.folds),
                       _ints(args.seeds), _ints(args.centres))
    if args.max_runs is not None:
        units = units[: args.max_runs]

    epochs = args.epochs if args.epochs is not None else base_cfg.epochs
    prov_common = {
        "git_sha": git_sha(),
        "git_sha_note": ("this tree is not a git checkout; provenance is carried "
                         "by source_sha256 below"),
        "source_sha256": source_digest(),
        "environment": environment(),
    }

    print(f"[runner] mode={args.mode}  units={len(units)}  epochs={epochs}")
    print(f"[runner] config={args.config}")
    print(f"[runner] out_dir={out_dir}")
    print(f"[runner] source digest={prov_common['source_sha256']['combined_sha256'][:16]}"
          f"  git_sha={prov_common['git_sha']}")

    # STEP 0 (2026-08-07): archive the exact source tree that is about to run
    # into runs/_src/<tree_digest>/ and refresh reports/source_registry.md.
    # Best-effort by contract -- provenance must never cost GPU time.
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_snap", os.path.join(REPO_ROOT, "scripts", "51_source_snapshot.py"))
        _snap = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_snap)
        _snap.snapshot_at_run_start()
    except Exception as _exc:                            # noqa: BLE001
        print(f"[runner] source snapshot failed (non-fatal): {_exc}",
              file=sys.stderr)

    # -- classify everything up front, so the plan is knowable before any GPU
    #    time is spent and --dry-run is a genuine preview of the same decisions
    states: List[Tuple[Unit, str, str, Dict[str, int]]] = []
    for unit in units:
        cfg = unit.apply(dataclasses.replace(base_cfg, out_dir=out_dir, epochs=epochs))
        expected = expected_split(unit, cfg)
        state, reason = run_state(os.path.join(out_dir, unit.name), unit, cfg, expected)
        if args.force and state == "done":
            state, reason = "fresh", "forced by --force"
        states.append((unit, state, reason, expected))

    n_done = sum(1 for _, s, _, _ in states if s == "done")
    print(f"[runner] {n_done}/{len(units)} already complete; "
          f"{len(units) - n_done} to run\n")
    for unit, state, reason, expected in states:
        print(f"  {unit.name:16s} {state:7s} n_val={expected['n_val']:5d} "
              f"pos={expected['n_pos']:4d}  {reason}")
    if args.dry_run:
        print("\n[runner] --dry-run: nothing executed")
        return 0

    _install_signal_handlers()
    index_path = os.path.join(out_dir, INDEX_NAME)
    records: Dict[str, Any] = {}
    if os.path.exists(index_path):
        try:
            with open(index_path) as fh:
                records = json.load(fh).get("units", {})
        except (OSError, json.JSONDecodeError):
            records = {}

    started = time.time()

    def flush_index(current: Optional[str], status: str) -> None:
        write_json_atomic(index_path, {
            "mode": args.mode,
            "config": args.config,
            "out_dir": out_dir,
            "epochs": epochs,
            "seeds": _ints(args.seeds),
            "repeats": _ints(args.repeats),
            "folds": _ints(args.folds),
            "centres": _ints(args.centres) if args.mode == "loco" else [],
            "provenance": prov_common,
            "status": status,
            "current_unit": current,
            "elapsed_seconds": time.time() - started,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "units": records,
        })

    flush_index(None, "running")
    n_ok = n_fail = n_skip = 0

    bar = tqdm(total=len(states), desc="units", unit="unit")
    for i, (unit, state, reason, expected) in enumerate(states, 1):
        if _stop["flag"]:
            bar.write("[runner] stopping before "
                      f"{unit.name} on request; {len(states) - i + 1} units left")
            break
        # 2026-08-10, added after the stage-7a incident: gpu_hold (checked by
        # each chain launcher's OUTER guard, e.g. run_g3pauc_chain.sh) only
        # blocks a NEW invocation from starting -- it does nothing once a
        # chain's inner loop is already running, which is exactly what let
        # stage 7a burn a unit and a half after gpu_hold was set. This is
        # the per-unit boundary EVERY chain already funnels through (one
        # `python -m src.train` subprocess per unit, isolation is the whole
        # design, see module docstring), so checking here covers all of them
        # uniformly rather than duplicating the check into each chain script.
        if os.path.exists(os.path.join(REPO_ROOT, "logs", "gpu_hold_kill")):
            bar.write("[runner] logs/gpu_hold_kill present -- stopping before "
                      f"{unit.name}; {len(states) - i + 1} units left. Re-run "
                      f"the same command to continue once the sentinel is removed.")
            break

        cfg = unit.apply(dataclasses.replace(base_cfg, out_dir=out_dir, epochs=epochs))
        run_dir = os.path.join(out_dir, unit.name)

        if state == "done":
            n_skip += 1
            bar.write(f"[{i}/{len(states)}] {unit.name} SKIP ({reason})")
            records[unit.name] = {**unit.key(), "status": "skipped",
                                  "reason": reason, "expected": expected}
            flush_index(None, "running")
            bar.set_postfix_str(f"ok={n_ok} fail={n_fail} skip={n_skip}")
            bar.update(1)
            continue

        bar.set_postfix_str(f"{unit.name} {state} | ok={n_ok} fail={n_fail} skip={n_skip}")
        bar.write(f"[{i}/{len(states)}] {unit.name} {state.upper()} -- {reason}")
        flush_index(unit.name, "running")

        canonical = os.path.join(run_dir, f"val_{split_name(cfg)}.parquet")
        attempt, result, ok, why = 0, None, False, "not attempted"
        while attempt <= args.retries:
            # IDEMPOTENT RE-CHECK, first thing every iteration, before
            # spending a second on execute(). 2026-08-05: r0_f0_s0 in
            # runs/a4_checkpointed recorded "attempts": 3 with a FINAL
            # attempt that crashed trying to --resume from a checkpoint
            # already deleted by a PRIOR attempt's successful completion --
            # meaning a 3rd execute() ran AFTER validity was already
            # achieved. That is mathematically impossible from this loop's
            # own bound alone (attempt <= args.retries caps at 2 passes when
            # retries=1); the leading theory is a prior process from an
            # interrupted run (tmux kill-session does not reliably reap the
            # full descendant tree -- a real, separate durability gap) still
            # writing to the same run_dir when this loop's own attempt 2
            # already succeeded. Whatever the exact mechanism, this check
            # closes it categorically: if a valid canonical parquet is
            # ALREADY on disk when an iteration begins, there is nothing to
            # gain by executing again, only GPU-hours to lose and a spurious
            # crash to risk (exactly what attempt 3 was -- burned 44s here;
            # on a unit that legitimately takes hours, the same defect burns
            # hours for zero benefit).
            ok, why = validate_predictions(canonical, unit, expected)
            if ok:
                if result is None:
                    result = {"returncode": 0, "timed_out": False, "seconds": 0.0,
                              "log": os.path.join(log_dir, f"{unit.name}.log"),
                              "command": []}
                break

            # after a failed first attempt a checkpoint may now exist, so the
            # retry resumes rather than discarding the epochs already paid for
            resume = (state == "resume") or attempt > 0
            if resume and not os.path.exists(
                os.path.join(run_dir, "checkpoints", "last.pt")
            ):
                resume = False
            result = execute(unit, cfg, args.config, out_dir, log_dir,
                             args.epochs, resume, args.timeout_min * 60.0)
            ok, why = validate_predictions(canonical, unit, expected)
            if ok and result["returncode"] == 0:
                break
            attempt += 1
            if attempt <= args.retries and not _stop["flag"]:
                bar.write(f"    attempt {attempt} failed (rc={result['returncode']}, "
                          f"{why}); retrying with resume")
            if _stop["flag"]:
                break

        write_provenance(run_dir, unit, cfg, result["command"], prov_common)
        records[unit.name] = {
            **unit.key(),
            # STATUS IS VALIDITY, NOT A PROCESS EXIT CODE. Was
            # `"ok" if (ok and result["returncode"] == 0) else "failed"` --
            # ANDing the canonical parquet's own validity with a specific
            # subprocess attempt's raw returncode. These desynchronize
            # whenever the LAST attempt executed (whose returncode this is)
            # is not the attempt that produced the currently-valid canonical
            # parquet -- e.g. a later attempt crashes for an unrelated
            # reason (found 2026-08-05: a checkpoint-write race after a
            # tmux-session restart) while an earlier attempt's already-valid
            # output is still sitting on disk. `ok` is computed FRESH from
            # THAT SAME canonical path every loop iteration -- it already IS
            # the authoritative "does a trustworthy parquet exist right now"
            # answer this whole codebase relies on elsewhere (run_state()'s
            # own skip logic uses nothing else) uses nothing else -- so
            # ANDing in a returncode besides.  A "failed" unit is one whose
            # answers a different, less relevant question. A "failed" unit
            # is one whose ON-DISK OUTPUT does not validate, full stop.
            "status": "ok" if ok else "failed",
            "validation": why,
            "expected": expected,
            "attempts": attempt + 1,
            **{k: result[k] for k in ("returncode", "timed_out", "seconds", "log")},
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        flush_index(None, "running")

        if ok and result["returncode"] == 0:
            n_ok += 1
            bar.write(f"    ok in {result['seconds'] / 60:.1f} min -- {why}")
        else:
            n_fail += 1
            bar.write(f"    FAILED rc={result['returncode']} timed_out="
                      f"{result['timed_out']} -- {why}\n    see {result['log']}")

        bar.set_postfix_str(f"ok={n_ok} fail={n_fail} skip={n_skip}")
        bar.update(1)

    bar.close()
    flush_index(None, "stopped" if _stop["flag"] else "finished")
    total = time.time() - started
    print(f"\n[runner] {n_ok} ok, {n_skip} skipped, {n_fail} failed in "
          f"{total / 3600:.2f} h")
    print(f"[runner] index: {index_path}")
    if n_fail:
        print("[runner] NON-ZERO EXIT: at least one unit failed. Re-running the "
              "same command resumes them.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
