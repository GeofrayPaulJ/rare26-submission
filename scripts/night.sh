#!/usr/bin/env bash
# Job A then Job B, sequentially, on one GPU.
#
# RESTARTABLE BY DESIGN. Both jobs are driven by scripts/run_cv.py, which
# re-derives its plan from what is on disk every time it starts. If the
# container is killed mid-night, re-running this exact script continues from
# the last completed unit -- no flags to remember, no state to clean up.
#
#   tmux new -d -s rare26 'bash scripts/night.sh'
#   tmux attach -t rare26
#
# The two jobs are NOT run in parallel: there is one GPU, and two 13 GB
# ConvNeXt-Base runs would either spill to system RAM over PCIe (see the
# VramSpillError note in src/train.py) or halve each other's throughput for no
# net gain.
set -u  # deliberately NOT -e: job B must still run if job A ends with a failed unit

cd "$(dirname "$0")/.." || exit 1
CFG=configs/convnext_b_384.yaml

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
banner() { echo; echo "############ $(stamp)  $* ############"; echo; }

banner "JOB A -- noise floor: repeat 0, folds 0-4, seeds 0-4 (25 units)"
python -u scripts/run_cv.py --config "$CFG" \
    --mode cv --repeats 0 --folds 0,1,2,3,4 --seeds 0,1,2,3,4 \
    --out-dir runs/noise_floor_a
A_RC=$?
banner "JOB A finished rc=$A_RC"

banner "JOB B -- LOCO noise floor: centres 1,2, seeds 0-4 (10 units)"
python -u scripts/run_cv.py --config "$CFG" \
    --mode loco --centres 1,2 --seeds 0,1,2,3,4 \
    --out-dir runs/noise_floor_b
B_RC=$?
banner "JOB B finished rc=$B_RC"

echo "[night] job A rc=$A_RC  job B rc=$B_RC"
echo "[night] a non-zero rc means at least one unit failed; re-run this script to retry it"
echo "[night] ALL DONE $(stamp)"
