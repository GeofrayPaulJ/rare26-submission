"""Runs the full RARE25/RARE26 inventory pipeline end to end (phases 0-4).

Safe to re-run: every phase overwrites its own output files deterministically
and never touches the source dataset.

Usage: python run_pipeline.py [--root PATH] [--threshold N] [--workers N]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent


def run(cmd: list[str], bar: tqdm) -> None:
    bar.write(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    bar.update(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--threshold", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    py = sys.executable
    root_args = ["--root", args.root] if args.root else []
    workers_args = ["--workers", str(args.workers)] if args.workers else []

    phases = [
        ("locate_and_verify", [py, str(SCRIPTS_DIR / "00_locate_and_verify.py"), *root_args]),
        ("build_manifest", [py, str(SCRIPTS_DIR / "01_build_manifest.py"), *root_args, *workers_args]),
        ("duplicate_audit", [py, str(SCRIPTS_DIR / "02_duplicate_audit.py")]),
        ("near_duplicates", [py, str(SCRIPTS_DIR / "03_near_duplicates.py"), "--threshold", str(args.threshold)]),
        ("report", [py, str(SCRIPTS_DIR / "04_report.py"), "--threshold", str(args.threshold)]),
    ]

    with tqdm(total=len(phases), desc="pipeline", unit="phase") as bar:
        for name, cmd in phases:
            bar.set_postfix_str(name)
            run(cmd, bar)


if __name__ == "__main__":
    main()
