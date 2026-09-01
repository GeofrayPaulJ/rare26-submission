"""Unattended driver for the magnitude sweep. One GPU, sequential, resumable.

Owns the STAGE LIST; it owns no training logic. Every unit of work is a fresh
``python -m src.train`` subprocess, reusing the tested plumbing in
scripts/run_cv.py (Unit, expected_split, validate_predictions, run_state) so
there is no second copy of "is this unit done" to drift out of step.

STAGES, in order:

    prep            shot-noise equivalence check
    a0_regression   is runs/noise_floor_a still bit-reproducible?
    a1_regression   is runs/noise_floor_b still bit-reproducible?
    a1_pooled       centre x class sampler, 25 units
    a1_early        the early report -- written the moment A1 finishes
    a2_pooled       0.33x magnitude, 25 units
    a3_pooled       1.0x magnitude, 25 units
    a2_loco         10 units
    a3_loco         10 units
    interim         interim report, then HALT for the A4 decision

    -- A4 decision made 2026-07-30, see reports/a4_pre_registration.md --

    a4_regression   is runs/sweep_a2_loco still bit-reproducible under
                    configs/sweep_a4.yaml in LOCO mode? (the sampler
                    degeneracy argument that lets A4 reuse it, checked not
                    asserted, same discipline as a1_regression)
    a4_pooled       A1 sampler + A2 stack combined, pooled OOF, 25 units
    a4_probe_loco   stack alone at magnitude_scale 0.10, LOCO only, 10 units
    a4_report       final report, then HALT

HALT CONDITIONS. Each one stops the whole sweep and writes logs/sweep_halt.json
with a plain statement of what went wrong. A sweep that runs on past a broken
invariant spends two days producing numbers nobody can use:

    * the shot-noise equivalence check exceeds its 2% bar
    * either regression check is not bit-identical
    * the VRAM spill guard trips (see src/train.py VramSpillError)
    * any unit produces non-finite logits

The halt file is also the watchdog's stop signal: scripts/sweep_watchdog.ps1
will not relaunch a sweep that halted deliberately, so a broken invariant stays
stopped instead of being restarted into the same wall forever.

DURABILITY. tmux does NOT survive a container restart -- it lives in the
container's PID namespace, and this was tested rather than assumed (the session
and every process under it are gone). Durability therefore comes from three
things together: the container's unless-stopped restart policy, the host-side
watchdog, and the fact that every stage here re-derives its plan from what is
on disk. Relaunching loses at most the in-flight unit, which resumes from its
last epoch checkpoint.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

# scripts/run_cv.py owns the tested definitions of "which units exist", "is this
# unit already done" and "is this parquet trustworthy". Importing them keeps
# this driver from growing a second, subtly different answer to any of those.
from run_cv import (  # noqa: E402
    Unit, expected_split, run_state, source_digest, validate_predictions,
    write_json_atomic,
)
from src.config import Config  # noqa: E402
from src.io import durable_replace, fsync_dir  # noqa: E402
from src.train import split_name  # noqa: E402

LOG_DIR = os.path.join(REPO_ROOT, "logs")
PROGRESS_LOG = os.path.join(LOG_DIR, "sweep_progress.log")
STATUS_JSON = os.path.join(LOG_DIR, "sweep_status.json")
HALT_JSON = os.path.join(LOG_DIR, "sweep_halt.json")
DONE_FILE = os.path.join(LOG_DIR, "sweep_stage_complete")
# Written once the three gates pass. They are properties of the working tree,
# not of the run, so a watchdog relaunch after a container restart should not
# spend GPU time re-proving them -- run_sweep.sh reads this to decide whether
# to pass --skip-prep.
PREP_PASSED = os.path.join(LOG_DIR, "sweep_prep_passed")
# The A4-specific reuse gate: is configs/sweep_a4.yaml's LOCO arm still
# bit-identical to runs/sweep_a2_loco? Independent of PREP_PASSED/--skip-prep
# above (those cover A0/A1 only) and of its own sentinel, so a watchdog
# relaunch does not re-pay for it once it has passed in this working tree.
A4_PREP_PASSED = os.path.join(LOG_DIR, "sweep_a4_prep_passed")
# Liveness is published as a PID, not inferred from a process name.
# `pgrep -f 23_sweep.py` looks like the obvious check and is a trap: the
# watchdog runs it through `docker exec bash -c "...pgrep -f 23_sweep.py..."`,
# whose OWN command line contains the pattern, so pgrep matches the shell asking
# the question and reports the driver alive forever. That was observed, not
# theorised -- with zero driver processes running, the check returned 1, which
# would have left the watchdog permanently convinced there was nothing to do.
PID_FILE = os.path.join(LOG_DIR, "sweep.pid")

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
REPEAT = 0
CENTRES = (1, 2)

# one epoch line from src/train.py, e.g.
#   "epoch   7/30  train 0.1 val 0.2 auc 0.9 ... 42.9s  gpu 88%"
_EPOCH_RE = re.compile(r"^epoch\s+(\d+)/(\d+)\s")
_VRAM_RE = re.compile(r"VRAM spill guard tripped")


class Halt(Exception):
    """A broken invariant. Stops the sweep; the watchdog will not restart it."""

    def __init__(self, stage: str, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.stage, self.reason, self.detail = stage, reason, detail


_stop = {"flag": False}


def _install_signals() -> None:
    def handler(signum, _frame):
        _stop["flag"] = True
        print(f"\n[sweep] signal {signum}: finishing the current unit, then "
              f"stopping. Re-run the same command to continue.", flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Arm:
    key: str
    config: str
    out_dir: str
    mode: str                 # "cv" | "loco"
    label: str
    # cv arms only. Default (REPEAT,) preserves every existing arm's original
    # single-repeat-0 behaviour unchanged; the repeat-gating arms below pass
    # (1, 2) instead. Units() is repeat-major, so a mid-plan stop always lands
    # on a clean repeat boundary rather than half of two repeats at once.
    repeats: Tuple[int, ...] = (REPEAT,)
    # Measure the largest batch that fits before this arm's first unit, instead
    # of inheriting ConvNeXt's 32. Set for the GastroNet arms: a ViT-B/14 at
    # 378px has a very different activation footprint from ConvNeXt-Base at
    # 384px, and this card silently oversubscribes to host RAM rather than
    # raising OOM, so an unprobed batch size does not fail loudly -- it runs
    # ~40x slower and finishes days late. See scripts/vram_probe.py.
    probe_batch: bool = False

    def units(self) -> List[Unit]:
        if self.mode == "cv":
            return [Unit(seed=s, repeat=r, fold=f)
                    for r in self.repeats for s in SEEDS for f in FOLDS]
        return [Unit(seed=s, holdout_centre=c) for s in SEEDS for c in CENTRES]


ARMS: Dict[str, Arm] = {
    "a1_pooled": Arm("a1_pooled", "configs/sweep_a1.yaml", "runs/sweep_a1",
                     "cv", "A1 centre x class sampler, pooled OOF"),
    "a2_pooled": Arm("a2_pooled", "configs/sweep_a2.yaml", "runs/sweep_a2",
                     "cv", "A2 stack at 0.33x, pooled OOF"),
    "a3_pooled": Arm("a3_pooled", "configs/sweep_a3.yaml", "runs/sweep_a3",
                     "cv", "A3 stack at 1.0x, pooled OOF"),
    "a2_loco": Arm("a2_loco", "configs/sweep_a2.yaml", "runs/sweep_a2_loco",
                   "loco", "A2 stack at 0.33x, LOCO stress test"),
    "a3_loco": Arm("a3_loco", "configs/sweep_a3.yaml", "runs/sweep_a3_loco",
                   "loco", "A3 stack at 1.0x, LOCO stress test"),
    "a4_pooled": Arm("a4_pooled", "configs/sweep_a4.yaml", "runs/sweep_a4",
                     "cv", "A4 sampler + stack (0.33x), pooled OOF"),
    "a4_probe_loco": Arm("a4_probe_loco", "configs/sweep_a4_probe.yaml",
                         "runs/sweep_a4_probe", "loco",
                         "A4 probe, stack at 0.10x, LOCO only"),
    # --- 3-repeat gating, requested once A4 was accepted under Rule 2 ---
    # Repeat 0 already exists on disk for all four; only repeats 1-2 are new
    # units here. Priority order A0 -> A4 -> A2 -> A1 (RUN_ORDER_REPEATS
    # below), each arm's repeat 1 run to completion before its repeat 2
    # starts, so a 48h stop always lands on a clean repeat boundary.
    "a0_repeats": Arm("a0_repeats", "configs/sweep_a0.yaml", "runs/sweep_a0",
                      "cv", "A0 control, repeats 1-2 (repeat 0 in runs/noise_floor_a)",
                      repeats=(1, 2)),
    "a4_repeats": Arm("a4_repeats", "configs/sweep_a4.yaml", "runs/sweep_a4",
                      "cv", "A4 sampler + stack (0.33x), repeats 1-2",
                      repeats=(1, 2)),
    "a2_repeats": Arm("a2_repeats", "configs/sweep_a2.yaml", "runs/sweep_a2",
                      "cv", "A2 stack at 0.33x, repeats 1-2", repeats=(1, 2)),
    "a1_repeats": Arm("a1_repeats", "configs/sweep_a1.yaml", "runs/sweep_a1",
                      "cv", "A1 centre x class sampler, repeats 1-2",
                      repeats=(1, 2)),
    # --- GastroNet / DINOv2 ensemble diversity members ---
    # A4 config verbatim (sampler + stack at 0.33x) with the backbone swapped
    # for a local self-supervised checkpoint. Repeat 0 only, 25 units each;
    # repeats come later and only for whatever survives. ConvNeXt-Base stays
    # primary -- these are additional ensemble members, not replacements.
    # Priority order is encoded in RUN_ORDER_GASTRONET below.
    "g1_rn50_swsl": Arm(
        "g1_rn50_swsl", "configs/g1_rn50_swsl.yaml", "runs/g1_rn50_swsl", "cv",
        "G1 RN50 Billion-Scale-SWSL + GastroNet-5M (DINOv1)", probe_batch=True),
    "g2_vitb_dinov2": Arm(
        "g2_vitb_dinov2", "configs/g2_vitb_dinov2.yaml", "runs/g2_vitb_dinov2",
        "cv", "G2 DINOv2 ViT-B/14 @378 (architecture diversity)",
        probe_batch=True),
    # G1's LOCO arm, inserted 2026-08-04 AHEAD of G3. Every G1 figure so far
    # (pooled OOF, the A4+G1 fusion) is repeat-0 pooled-OOF -- the one
    # protocol this project has repeatedly shown cannot resolve domain-
    # generalisation questions (see reports/magnitude_sweep.md section 0).
    # G1 is a candidate for an ensemble slot on evidence from exactly that
    # protocol. G3 answers a control/paper question (does SWSL alone matter);
    # G1's LOCO answers a build question (does G1 actually earn the slot).
    # Build goes first. A4 config verbatim (same file as g1_rn50_swsl, run in
    # loco mode instead of cv) -- both directions x 5 seeds = 10 units.
    "g1_loco": Arm(
        "g1_loco", "configs/g1_rn50_swsl.yaml", "runs/g1_rn50_swsl_loco",
        "loco", "G1 RN50 Billion-Scale-SWSL + GastroNet-5M (DINOv1), LOCO "
        "both directions", probe_batch=True),
    "g3_rn50_gastronet": Arm(
        "g3_rn50_gastronet", "configs/g3_rn50_gastronet.yaml",
        "runs/g3_rn50_gastronet", "cv",
        "G3 RN50 GastroNet-5M (DINOv1) -- SWSL control", probe_batch=True),
    # --- JOB 3 / JOB 4, appended 2026-08-05. GPU and CPU run concurrently now
    # (the "stop before the GPU" gate was the requester's own error, retracted
    # same day): these two arms depend on nothing produced by the CPU-only
    # audits, so there is no reason to hold them behind those reports.
    # G3's LOCO arm: configs/g3_rn50_gastronet.yaml verbatim, mode=loco, both
    # directions x seeds 0-4 = 10 units. Answers whether G3 (the SWSL-ablated
    # GastroNet control) earns a slot on the DG axis, same protocol as G1's
    # own LOCO arm above.
    "g3_loco": Arm(
        "g3_loco", "configs/g3_rn50_gastronet.yaml",
        "runs/g3_rn50_gastronet_loco", "loco",
        "G3 RN50 GastroNet-5M (DINOv1) -- SWSL control, LOCO both directions",
        probe_batch=True),
    # G0 control: identical recipe to G3 (RN50, A4's sampler+stack), ImageNet
    # init instead of the local GastroNet checkpoint. G3 minus G0 isolates
    # GastroNet pretraining at fixed capacity. Pooled OOF only (mode=cv),
    # repeat 0, 5 folds x 5 seeds = 25 units, save_checkpoint left at its
    # default False (see src/config.py) -- same screening-run convention as
    # every other GastroNet arm above.
    "g0_rn50_imagenet": Arm(
        "g0_rn50_imagenet", "configs/g0_rn50_imagenet.yaml",
        "runs/g0_rn50_imagenet", "cv",
        "G0 RN50, ImageNet init -- GastroNet-pretraining control",
        probe_batch=True),
}

RUN_ORDER = ("a1_pooled", "a2_pooled", "a3_pooled", "a2_loco", "a3_loco",
            "a4_pooled", "a4_probe_loco")
RUN_ORDER_REPEATS = ("a0_repeats", "a4_repeats", "a2_repeats", "a1_repeats")
# Priority order, per the brief: strongest configuration, then architecture
# diversity, then G1's own LOCO (build question, ahead of the G3 control),
# then the SWSL-isolating control. G3's own LOCO and the G0 ImageNet control
# were appended 2026-08-05 (JOB 3, JOB 4) in that order, after all three
# backbones' pooled-OOF units -- both are additive to what is already on disk
# and neither is blocked by the other.
RUN_ORDER_GASTRONET = ("g1_rn50_swsl", "g2_vitb_dinov2", "g1_loco",
                       "g3_rn50_gastronet", "g3_loco", "g0_rn50_imagenet")


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
class Progress:
    """Outer bar over every unit in the sweep, inner bar over epochs.

    Both bars are driven from THIS process. The child trains with --no-progress
    and prints one clean line per epoch, which is parsed here -- two processes
    writing tqdm to one terminal fight over the cursor and produce something
    unreadable after a tmux reattach.
    """

    def __init__(self, total_units: int, done_units: int, stream) -> None:
        self.outer = tqdm(
            total=total_units, initial=done_units, unit="run",
            desc="sweep", position=0, dynamic_ncols=True, file=stream,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        self.inner: Optional[tqdm] = None
        self.stream = stream

    def start_unit(self, arm: Arm, unit: Unit, epochs: int) -> None:
        self.outer.set_postfix_str(f"{arm.key} {unit.name}")
        if self.inner is not None:
            self.inner.close()
        self.inner = tqdm(
            total=epochs, unit="ep", desc=f"  {unit.name}", position=1,
            leave=False, dynamic_ncols=True, file=self.stream,
        )

    def epoch(self, n: int) -> None:
        if self.inner is not None:
            self.inner.n = n
            self.inner.refresh()

    def finish_unit(self) -> None:
        if self.inner is not None:
            self.inner.close()
            self.inner = None
        self.outer.update(1)

    def close(self) -> None:
        if self.inner is not None:
            self.inner.close()
        self.outer.close()


class Tee:
    """Write to the terminal and to a file at once.

    tqdm needs a real ``write``/``flush``/``isatty``; giving it only the log
    file would leave a reattached tmux session blank, and giving it only the
    terminal would leave nothing to read after the fact.
    """

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except (ValueError, OSError):
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except (ValueError, OSError):
                pass

    def isatty(self):
        return True


def write_sentinel(path: str, body: Optional[str] = None) -> None:
    """A sentinel file whose EXISTENCE changes behaviour, written durably.

    These are small but load-bearing: sweep_prep_passed decides whether GPU
    gates are re-paid on the next relaunch, sweep_stage_complete and
    sweep_halt.json decide whether the watchdog restarts the sweep at all. A
    sentinel that appears to exist after a host reset but whose directory entry
    never landed would silently flip any of those decisions, so they go through
    the same tmp + fsync + replace + fsync(dir) path as the data files.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(body if body is not None
                 else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    durable_replace(tmp, path)


def append_progress(record: Dict[str, Any]) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    line = (f"{record['utc']}  {record['arm']:<10s} {record['unit']:<14s} "
            f"{record['status']:<7s} wall={record['wall_min']:6.1f}m "
            f"auc={record['roc_auc']} fpr90={record['fpr_at_90_recall']}")
    existed = os.path.exists(PROGRESS_LOG)
    with open(PROGRESS_LOG, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    if not existed:
        fsync_dir(LOG_DIR)


def write_status(state: Dict[str, Any]) -> None:
    write_json_atomic(STATUS_JSON, state)


# ---------------------------------------------------------------------------
# Unit execution
# ---------------------------------------------------------------------------
def unit_metrics(run_dir: str) -> Dict[str, Any]:
    """Final-epoch ROC-AUC and FPR@90R, read from the run's own summary."""
    path = os.path.join(run_dir, "summary.json")
    try:
        with open(path) as fh:
            s = json.load(fh)
        h = s["history"][-1]
        return {"roc_auc": round(float(h["roc_auc"]), 4),
                "fpr_at_90_recall": round(float(h["fpr_at_90_recall"]), 4),
                "median_epoch_seconds": float(s["median_epoch_seconds"])}
    except (OSError, json.JSONDecodeError, KeyError, IndexError):
        return {"roc_auc": "n/a", "fpr_at_90_recall": "n/a",
                "median_epoch_seconds": float("nan")}


def train_argv(arm: Arm, unit: Unit, out_dir: str, epochs: int,
               resume: bool, batch_size: Optional[int] = None) -> List[str]:
    argv = [sys.executable, "-u", "-m", "src.train",
            "--config", arm.config, "--seed", str(unit.seed),
            "--out-dir", out_dir, "--epochs", str(epochs), "--no-progress"]
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    if unit.holdout_centre is not None:
        argv += ["--holdout-centre", str(unit.holdout_centre)]
    else:
        argv += ["--repeat", str(unit.repeat), "--fold", str(unit.fold)]
    if resume:
        argv += ["--resume"]
    return argv


def execute_unit(arm: Arm, unit: Unit, out_dir: str, epochs: int, resume: bool,
                 progress: Progress, stream,
                 batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Run one unit, streaming its output to the unit log and driving the
    inner epoch bar. Raises Halt if the VRAM spill guard trips."""
    log_dir = os.path.join(out_dir, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{unit.name}.log")
    argv = train_argv(arm, unit, out_dir, epochs, resume, batch_size)

    progress.start_unit(arm, unit, epochs)
    t0 = time.time()
    vram_tripped = False

    with open(log_path, "a") as log:
        log.write(f"\n{'=' * 78}\n[sweep] {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"{'RESUME' if resume else 'START'} {arm.key} {unit.name}\n"
                  f"[sweep] {' '.join(argv)}\n{'=' * 78}\n")
        log.flush()
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            m = _EPOCH_RE.match(line)
            if m:
                progress.epoch(int(m.group(1)))
            elif _VRAM_RE.search(line):
                vram_tripped = True
            elif "Traceback" in line or "Error" in line:
                stream.write(line)
        rc = proc.wait()
        log.write(f"[sweep] exit={rc} elapsed={time.time() - t0:.1f}s\n")

    progress.finish_unit()
    if vram_tripped:
        raise Halt(arm.key, "VRAM spill guard tripped",
                   f"unit {unit.name}; see {os.path.relpath(log_path, REPO_ROOT)}")
    return {"returncode": rc, "wall_min": (time.time() - t0) / 60.0,
            "log": os.path.relpath(log_path, REPO_ROOT)}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_shot_noise(stream) -> None:
    stream.write("\n[sweep] stage: shot-noise equivalence check\n")
    rc = subprocess.run(
        [sys.executable, "-u", os.path.join(REPO_ROOT, "scripts",
                                            "24_shot_noise_check.py")],
        cwd=REPO_ROOT).returncode
    if rc != 0:
        raise Halt("prep", "shot-noise equivalence exceeded the 2% bar",
                   "see reports/shot_noise_check.json")
    stream.write("[sweep] shot-noise equivalence: PASS\n")


def scan_for_restarts(stream) -> None:
    """Before any GPU time is spent on the repeat-gating stage this
    invocation, find any repeat-gating unit left mid-flight by whatever ended
    the last process (crash or a planned stop) and record it via
    scripts/29_repeat_gating.py --restart, which appends one timestamped line
    to logs/repeat_gating_restarts.log before regenerating the report. Cheap:
    filesystem checks only via run_state, the same function that decides
    which units run_arm can skip.

    A unit counts as "redone" if it is not done AND train_log.jsonl already
    exists in its run_dir -- not merely if run_state says "resume". A kill
    before epoch 1 finishes leaves no checkpoint at all, so run_state
    correctly calls that unit "fresh" (there is nothing to resume FROM), but
    it is still being redone from a prior attempt, not started for the first
    time: train_log.jsonl (written at the very start of training, well before
    the first checkpoint) is what distinguishes "never touched" from "died
    early last time".
    """
    for key in RUN_ORDER_REPEATS + RUN_ORDER_GASTRONET:
        arm = ARMS[key]
        out_dir = os.path.join(REPO_ROOT, arm.out_dir)
        if not os.path.isdir(out_dir):
            continue  # arm has never started; nothing can have been interrupted
        base_cfg = Config.from_yaml(os.path.join(REPO_ROOT, arm.config))
        # Grouped by unit.repeat (via arm.units(), the same mode-aware source
        # Arm.units() itself uses) rather than by manually reconstructing CV
        # units -- reconstructing them by hand assumed every arm was CV-mode,
        # which broke silently the moment a "loco" arm (Unit(seed, holdout_
        # centre), no fold, repeat always 0) was added to this same run order:
        # the hand-built Unit(seed, repeat, fold) pointed at run_dir paths a
        # LOCO arm never writes, so it would have reported "no restart" no
        # matter what actually happened. arm.units() is mode-aware by
        # construction, so this cannot drift out of sync with it again.
        by_repeat: Dict[int, List] = {}
        for unit in arm.units():
            by_repeat.setdefault(unit.repeat, []).append(unit)
        for repeat, units in by_repeat.items():
            redone = []
            for unit in units:
                cfg = unit.apply(dataclasses.replace(base_cfg, out_dir=out_dir))
                run_dir = os.path.join(out_dir, unit.name)
                expected = expected_split(unit, cfg)
                status, _ = run_state(run_dir, unit, cfg, expected)
                if status == "done":
                    continue
                if (status == "resume"
                        or os.path.exists(os.path.join(run_dir, "train_log.jsonl"))):
                    redone.append(unit.name)
            if redone:
                stream.write(f"[sweep] restart detected: {key} repeat {repeat} -- "
                             f"redoing {redone}\n")
                # Route the restart line to whichever report owns this arm, so
                # each report carries its own restart history.
                script = ("30_gastronet.py" if key in RUN_ORDER_GASTRONET
                          else "29_repeat_gating.py")
                subprocess.run(
                    [sys.executable, "-u",
                     os.path.join(REPO_ROOT, "scripts", script),
                     "--restart", key, str(repeat)] + redone,
                    cwd=REPO_ROOT)


def update_gastronet(arm_key: str, stream, halt: Optional[Dict[str, Any]] = None) -> None:
    """Regenerate reports/gastronet.md at a BACKBONE boundary.

    ``halt`` is passed when the queue is stopping on a broken invariant, so the
    reason lands in the report itself rather than only in logs/sweep_halt.json.
    A halt that is only discoverable by going hunting through logs is a halt
    that gets discovered late.
    """
    stream.write(f"\n[sweep] {arm_key} complete -- updating reports/gastronet.md\n")
    argv = [sys.executable, "-u",
            os.path.join(REPO_ROOT, "scripts", "30_gastronet.py")]
    if halt:
        argv += ["--halt-stage", str(halt.get("stage", "")),
                 "--halt-reason", str(halt.get("reason", "")),
                 "--halt-detail", str(halt.get("detail", ""))]
    subprocess.run(argv, cwd=REPO_ROOT)


def update_repeat_gating(arm_key: str, repeat: int, stream) -> None:
    stream.write(f"\n[sweep] {arm_key} repeat {repeat} complete -- updating "
                 f"reports/repeat_gating.md\n")
    subprocess.run(
        [sys.executable, "-u",
         os.path.join(REPO_ROOT, "scripts", "29_repeat_gating.py")],
        cwd=REPO_ROOT)


def stage_regression(name: str, argv_extra: Sequence[str], stream) -> None:
    stream.write(f"\n[sweep] stage: {name}\n")
    rc = subprocess.run(
        [sys.executable, "-u",
         os.path.join(REPO_ROOT, "scripts", "20_dg_regression.py")]
        + list(argv_extra), cwd=REPO_ROOT).returncode
    if rc != 0:
        raise Halt(name, "regression check was not bit-identical",
                   "the arm cannot reuse the predictions it planned to reuse; "
                   "it must be run in full")
    stream.write(f"[sweep] {name}: PASS (bit-identical)\n")


def probe_batch_size(arm: Arm, stream) -> Optional[int]:
    """Largest batch that genuinely FITS for this arm's arch/image_size.

    Delegates to scripts/vram_probe.py, which already encodes the thing that
    makes this card dangerous: it does not raise OutOfMemoryError when a batch
    overflows, it pages to host RAM over PCIe and runs ~40x slower. The probe
    therefore treats "peak reservation stayed inside physical VRAM" as the fit
    criterion rather than "did not crash". Returning None means the probe could
    not answer and the config's own batch_size stands.
    """
    cfg = Config.from_yaml(os.path.join(REPO_ROOT, arm.config))
    out_json = os.path.join(LOG_DIR, f"vram_probe_{arm.key}.json")
    chosen_file = os.path.join(LOG_DIR, f"batch_size_{arm.key}")

    # A previously chosen batch size is REUSED, never re-probed. batch_size is
    # a TRAJECTORY_FIELD: src/train.py refuses to resume a unit whose batch
    # size changed, because that is a different experiment wearing the old
    # run's directory. Re-probing on every watchdog relaunch could legitimately
    # return a different number (the card's free memory varies), which would
    # turn every restart into "refusing to resume" for every half-finished
    # unit of this arm. Probe once, write it down, honour it thereafter.
    if os.path.exists(chosen_file):
        try:
            with open(chosen_file) as fh:
                cached = int(fh.read().strip())
            stream.write(f"[sweep] {arm.key}: reusing probed batch_size="
                         f"{cached} from {os.path.basename(chosen_file)}\n")
            return cached
        except (OSError, ValueError):
            pass

    stream.write(f"[sweep] probing batch size for {arm.key} "
                 f"({cfg.arch} @ {cfg.image_size}px, {cfg.precision})\n")
    # --image-sizes / --precisions (PLURAL, list-accepting) are what the
    # sweep's cross-product loop actually reads; --image-size / --precision
    # (singular) only set an unused default and were passed here for weeks
    # without scoping anything -- every probe silently ran the full 2x2x2=8
    # cell sweep (both image sizes, both precisions, grad-checkpointing on
    # and off) instead of the 1x1x2=2 this arm's config actually needs.
    # Found while watching G1's probe run ~4x longer than it needed to.
    rc = subprocess.run(
        [sys.executable, "-u", os.path.join(REPO_ROOT, "scripts", "vram_probe.py"),
         "--arch", cfg.arch, "--image-sizes", str(cfg.image_size),
         "--precisions", cfg.precision, "--json-out", out_json],
        cwd=REPO_ROOT).returncode
    if rc != 0 or not os.path.exists(out_json):
        stream.write(f"[sweep] probe failed (rc={rc}); keeping configured "
                     f"batch_size={cfg.batch_size}\n")
        return None
    try:
        with open(out_json) as fh:
            probe = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        stream.write(f"[sweep] probe output unreadable ({exc}); keeping "
                     f"batch_size={cfg.batch_size}\n")
        return None

    # Field names checked against a REAL probe JSON, not assumed: the top-level
    # list is "rows" (not "cells") and each row's field is "batch_size" (not
    # "max_batch") -- see scripts/vram_probe.py's own row-building code. Wrong
    # names here mean cell.get(...) always returns None, the loop never finds
    # a match, and every call silently falls through to the "no usable batch"
    # branch -- which is exactly what happened, undetected, from the first
    # GastroNet unit through G1's full 25 units and G2's first dozen: every
    # one of them trained at the config's bare default (32), the probe having
    # measured a real ceiling and then never being read correctly. Caught by
    # actually reading a completed probe's JSON rather than trusting the
    # schema in this function's own comments.
    best = None
    for row in probe.get("rows", []):
        if (row.get("image_size") == cfg.image_size
                and row.get("precision") == cfg.precision
                and not row.get("grad_checkpointing")
                and row.get("fits")
                and row.get("batch_size")):
            best = int(row["batch_size"])
            break
    if not best:
        stream.write(f"[sweep] probe returned no usable batch; keeping "
                     f"batch_size={cfg.batch_size}\n")
        return None

    # Back off one step from the measured ceiling. The probe measures a peak on
    # an otherwise-idle card; training also holds the image cache and the
    # dataloader's pinned buffers, and the spill guard aborts the unit at 95%
    # of VRAM. Sitting exactly at the ceiling turns a normal fluctuation into a
    # halted arm.
    chosen = max(1, int(best * 0.9))
    stream.write(f"[sweep] {arm.key}: probe fits {best}, using {chosen} "
                 f"(10% headroom under the spill guard)\n")
    write_sentinel(chosen_file, f"{chosen}\n")
    return chosen


def run_arm(arm: Arm, state: Dict[str, Any], progress: Progress, stream,
            epochs: int,
            on_repeat_boundary: Optional[Callable[[int], None]] = None,
            batch_size: Optional[int] = None) -> None:
    """Every unit of one arm, skipping those already complete and valid.

    ``on_repeat_boundary(repeat)`` fires once the LAST unit of that repeat
    (in arm.units()'s repeat-major order) has completed -- a clean point to
    write an incremental report. It is a no-op for every existing arm (all of
    which have exactly one repeat and no callback registered); only the
    repeat-gating arms pass one.
    """
    out_dir = os.path.join(REPO_ROOT, arm.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base_cfg = Config.from_yaml(os.path.join(REPO_ROOT, arm.config))
    if batch_size is not None:
        base_cfg = dataclasses.replace(base_cfg, batch_size=batch_size)

    units = arm.units()
    for i, unit in enumerate(units):
        if _stop["flag"]:
            return
        cfg = unit.apply(dataclasses.replace(base_cfg, out_dir=out_dir,
                                             epochs=epochs))
        expected = expected_split(unit, cfg)
        run_dir = os.path.join(out_dir, unit.name)
        status, reason = run_state(run_dir, unit, cfg, expected)

        if status == "done":
            state["completed"] += 1
            progress.outer.update(1)
            state["skipped"] += 1
            write_status(status_payload(state))
            at_repeat_end = i == len(units) - 1 or units[i + 1].repeat != unit.repeat
            if on_repeat_boundary is not None and at_repeat_end:
                on_repeat_boundary(unit.repeat)
            continue

        result = execute_unit(arm, unit, out_dir, epochs,
                              resume=(status == "resume"), progress=progress,
                              stream=stream, batch_size=batch_size)
        canonical = os.path.join(run_dir, f"val_{split_name(cfg)}.parquet")
        ok, why = validate_predictions(canonical, unit, expected)

        if not ok and "non-finite" in why:
            raise Halt(arm.key, "a unit produced non-finite logits",
                       f"{unit.name}: {why}")
        if not ok or result["returncode"] != 0:
            # one retry, resuming from whatever epochs were already paid for
            result = execute_unit(arm, unit, out_dir, epochs, resume=True,
                                  progress=progress, stream=stream,
                                  batch_size=batch_size)
            ok, why = validate_predictions(canonical, unit, expected)
            if not ok and "non-finite" in why:
                raise Halt(arm.key, "a unit produced non-finite logits",
                           f"{unit.name}: {why}")
            if not ok or result["returncode"] != 0:
                raise Halt(arm.key, "a unit failed twice",
                           f"{unit.name}: rc={result['returncode']}, {why}; "
                           f"see {result['log']}")

        m = unit_metrics(run_dir)
        state["completed"] += 1
        state["unit_minutes"].append(result["wall_min"])
        append_progress({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "arm": arm.key, "unit": unit.name, "status": "ok",
            "wall_min": result["wall_min"], **m,
        })
        write_status(status_payload(state))

        at_repeat_end = i == len(units) - 1 or units[i + 1].repeat != unit.repeat
        if on_repeat_boundary is not None and at_repeat_end:
            on_repeat_boundary(unit.repeat)


def status_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    done = state["completed"]
    total = state["total"]
    mins = state["unit_minutes"]
    per = float(np.median(mins)) if mins else state["assumed_minutes"]
    remaining = max(0, total - done)
    to_next = max(0, state["next_report_at"] - done)
    return {
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_arm": state["current_arm"],
        "current_stage": state["current_stage"],
        "units_completed": done,
        "units_remaining": remaining,
        "units_total": total,
        "units_skipped_already_done": state["skipped"],
        "median_minutes_per_unit": round(per, 2),
        "eta_next_report_hours": round(to_next * per / 60.0, 2),
        "eta_next_report_name": state["next_report_name"],
        "eta_sweep_end_hours": round(remaining * per / 60.0, 2),
        "elapsed_hours": round((time.time() - state["t0"]) / 3600.0, 2),
        "halted": False,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip-prep", action="store_true",
                    help="skip the equivalence and regression gates (they have "
                         "already passed in this working tree)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(LOG_DIR, exist_ok=True)

    all_keys = RUN_ORDER + RUN_ORDER_REPEATS + RUN_ORDER_GASTRONET
    total = sum(len(ARMS[k].units()) for k in all_keys)

    # --dry-run PRINTS THE PLAN AND TOUCHES NOTHING, and returns before any of
    # the shared state below is written. This is not cosmetic. That state
    # belongs to whichever driver is actually running: the sentinel clears
    # below, and PID_FILE, which is published here and REMOVED by _cleanup_pid
    # on exit. Running --dry-run to inspect the plan while an unattended sweep
    # was live therefore deleted that live sweep's PID file -- after which the
    # watchdog would have found no driver, killed the "stale" tmux session (the
    # live one, mid-unit) and started a duplicate. Observed 2026-08-03 and
    # caught before the watchdog's next 60s poll. Inspecting a plan must never
    # be able to disturb a run.
    if args.dry_run:
        print(f"\n[sweep] {total} units across {len(all_keys)} arms, "
              f"{args.epochs} epochs each")
        print(f"[sweep] source digest "
              f"{source_digest()['combined_sha256'][:16]}")
        for k in all_keys:
            print(f"[sweep]   {k:<18s} {len(ARMS[k].units()):3d} units  "
                  f"{ARMS[k].label}")
        print("[sweep] --dry-run: nothing executed, no state touched")
        return 0

    for path in (HALT_JSON, DONE_FILE):
        if os.path.exists(path):
            os.remove(path)
    # Durable like the sentinels, but for the opposite reason: the danger here
    # is a STALE pid outliving a host reset. Container restart resets the PID
    # namespace, so an old pid can be re-issued to an unrelated process, and
    # run_sweep.sh's `kill -0` would then read "driver alive" forever and never
    # relaunch. run_sweep.sh therefore also checks the pid's cmdline; writing
    # the file durably means that check never runs against a half-written pid.
    write_sentinel(PID_FILE, f"{os.getpid()}\n")

    terminal = sys.stderr
    console_log = open(os.path.join(LOG_DIR, "sweep_console.log"), "a",
                       buffering=1)
    stream = Tee(terminal, console_log)

    # A1 finishes after its 25 pooled units; that is when the early report lands
    state: Dict[str, Any] = {
        "t0": time.time(), "total": total, "completed": 0, "skipped": 0,
        "unit_minutes": [], "assumed_minutes": 24.4,
        "current_arm": None, "current_stage": "prep",
        "next_report_at": len(ARMS["a1_pooled"].units()),
        "next_report_name": "a1_early",
    }

    stream.write(f"\n[sweep] {total} units across {len(all_keys)} arms, "
                 f"{args.epochs} epochs each\n")
    stream.write(f"[sweep] source digest "
                 f"{source_digest()['combined_sha256'][:16]}\n")
    for k in all_keys:
        stream.write(f"[sweep]   {k:<18s} {len(ARMS[k].units()):3d} units  "
                     f"{ARMS[k].label}\n")

    _install_signals()
    write_status(status_payload(state))
    progress = Progress(total, 0, stream)

    try:
        if not args.skip_prep:
            stage_shot_noise(stream)
            stage_regression("a0_regression", [
                "--config", "configs/sweep_a0.yaml", "--mode", "cv",
                "--repeat", "0", "--fold", "0", "--seed", "0",
                "--reference", "runs/noise_floor_a"], stream)
            stage_regression("a1_regression", [
                "--config", "configs/sweep_a1.yaml", "--mode", "loco",
                "--centre", "1", "--seed", "0",
                "--reference", "runs/noise_floor_b"], stream)
            write_sentinel(PREP_PASSED)
            stream.write("[sweep] all three gates passed\n")

        if not os.path.exists(A4_PREP_PASSED):
            stage_regression("a4_regression", [
                "--config", "configs/sweep_a4.yaml", "--mode", "loco",
                "--centre", "1", "--seed", "0",
                "--reference", "runs/sweep_a2_loco"], stream)
            write_sentinel(A4_PREP_PASSED)
            stream.write("[sweep] a4_regression gate passed -- runs/sweep_a2_loco "
                         "reuse for A4's LOCO arm is legitimate\n")
        else:
            stream.write("[sweep] a4_regression already passed in this tree; "
                         "skipping\n")

        for key in RUN_ORDER:
            if _stop["flag"]:
                break
            arm = ARMS[key]
            state["current_arm"] = key
            state["current_stage"] = key
            stream.write(f"\n[sweep] === {arm.label} ===\n")
            write_status(status_payload(state))
            run_arm(arm, state, progress, stream, args.epochs)

            if key == "a1_pooled" and not _stop["flag"]:
                # the early answer is worth more now than in the final document
                stream.write("\n[sweep] A1 complete -- writing the early report\n")
                subprocess.run(
                    [sys.executable, "-u",
                     os.path.join(REPO_ROOT, "scripts", "25_a1_early.py")],
                    cwd=REPO_ROOT)
                state["next_report_at"] = total
                state["next_report_name"] = "interim"
                write_status(status_payload(state))

        if not _stop["flag"]:
            stream.write("\n[sweep] all arms complete -- writing the interim "
                         "and A4 reports\n")
            subprocess.run(
                [sys.executable, "-u",
                 os.path.join(REPO_ROOT, "scripts", "26_magnitude_sweep.py")],
                cwd=REPO_ROOT)
            subprocess.run(
                [sys.executable, "-u",
                 os.path.join(REPO_ROOT, "scripts", "28_a4_sweep.py")],
                cwd=REPO_ROOT)
            stream.write("\n[sweep] A4 and its probe are complete. "
                         "reports/magnitude_sweep.md and reports/a4.md are "
                         "written. Continuing into 3-repeat gating "
                         "(A0 -> A4 -> A2 -> A1).\n")

            scan_for_restarts(stream)
            state["next_report_name"] = "repeat_gating"

            for key in RUN_ORDER_REPEATS:
                if _stop["flag"]:
                    break
                arm = ARMS[key]
                state["current_arm"] = key
                state["current_stage"] = key
                stream.write(f"\n[sweep] === {arm.label} ===\n")
                write_status(status_payload(state))
                run_arm(arm, state, progress, stream, args.epochs,
                       on_repeat_boundary=lambda r, k=key: update_repeat_gating(
                           k, r, stream))

        # --- GastroNet ensemble diversity members ---
        # Only after the repeat-gating queue is fully done: these are additive
        # ensemble members, and the accepted-arm gating is the higher priority.
        if not _stop["flag"]:
            stream.write("\n[sweep] 3-repeat gating complete. Continuing into "
                         "GastroNet backbones (G1 -> G2 -> G3).\n")
            state["next_report_name"] = "gastronet"
            for key in RUN_ORDER_GASTRONET:
                if _stop["flag"]:
                    break
                arm = ARMS[key]
                state["current_arm"] = key
                state["current_stage"] = key
                stream.write(f"\n[sweep] === {arm.label} ===\n")
                write_status(status_payload(state))

                bs = probe_batch_size(arm, stream) if arm.probe_batch else None
                run_arm(arm, state, progress, stream, args.epochs, batch_size=bs)

                # Report at the BACKBONE boundary, not per unit: the brief is
                # explicit that a partial backbone is unreported rather than
                # half-reported, so nothing is written until all 25 units of
                # this backbone exist.
                if not _stop["flag"]:
                    update_gastronet(key, stream)

        progress.close()

        if not _stop["flag"]:
            write_sentinel(DONE_FILE)
            final = status_payload(state)
            final["current_stage"] = "gastronet_complete"
            write_status(final)
            stream.write("\n[sweep] HALTED. 3-repeat gating and all three "
                         "GastroNet backbones are complete. "
                         "reports/repeat_gating.md and reports/gastronet.md "
                         "are written.\n")
        return 0

    except Halt as h:
        progress.close()
        payload = {
            "halted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": h.stage, "reason": h.reason, "detail": h.detail,
            "units_completed": state["completed"], "units_total": total,
        }
        write_json_atomic(HALT_JSON, payload)
        bad = status_payload(state)
        bad.update(halted=True, halt_reason=h.reason, halt_stage=h.stage)
        write_status(bad)
        # Surface the halt in the report the reader actually opens, not only in
        # logs/sweep_halt.json.
        if h.stage in RUN_ORDER_GASTRONET:
            try:
                update_gastronet(h.stage, stream, halt=payload)
            except Exception:  # noqa: BLE001 -- reporting must not mask the halt
                stream.write("[sweep] (could not update reports/gastronet.md)\n")
        stream.write(f"\n{'!' * 74}\n[sweep] HALTED at stage {h.stage}\n"
                     f"[sweep] {h.reason}\n[sweep] {h.detail}\n"
                     f"[sweep] Nothing further will run. The watchdog will not "
                     f"restart a deliberate halt.\n{'!' * 74}\n")
        return 3


def _cleanup_pid() -> None:
    """Remove the PID file, but ONLY if it is still ours.

    A stale PID file is worse than none: it tells the watchdog the sweep is
    running when it is not, which is the failure this file exists to prevent.
    But removing someone ELSE'S pid file is worse still, and this ran
    unconditionally in a finally block -- so any second invocation of this
    module (a --dry-run to read the plan, a mistaken double start) deleted the
    live driver's pid on its way out, even after --dry-run stopped writing it.
    The watchdog would then find no driver, kill the "stale" tmux session (the
    live one) and start a duplicate. Ownership is checked by content, so only
    the process that published the pid can retract it.
    """
    try:
        with open(PID_FILE) as fh:
            owner = int(fh.read().strip())
    except (OSError, ValueError):
        return
    if owner != os.getpid():
        return
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        _cleanup_pid()
