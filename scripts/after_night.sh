#!/usr/bin/env bash
# Wait for scripts/night.sh to finish, then produce the report and run the GPU
# tests that had to be deferred while the card was busy.
#
# WHY THIS IS A SEPARATE SCRIPT AND A SEPARATE TMUX SESSION. night.sh is already
# running, and bash reads a script file incrementally as it executes -- editing
# a running shell script can make the interpreter resume at a byte offset that
# is now the middle of a different line. So the post-processing is bolted on
# from outside rather than appended to the script it follows.
#
#   tmux new -d -s rare26_report 'bash scripts/after_night.sh'
#
# The point of this file is that the deliverable survives the loss of whatever
# session started it: the report lands on disk whether or not anyone is
# watching. If the container is killed before night.sh completes, this waits
# forever and writes nothing, which is the correct behaviour -- a report over a
# half-finished Job A would be worse than no report.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG=runs/night.log
MARKER='ALL DONE'

echo "[after] waiting for '$MARKER' in $LOG ..."
until [ -f "$LOG" ] && grep -q "$MARKER" "$LOG"; do
    sleep 120
done
echo "[after] night.sh finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ'); starting post-processing"

# 1. The report. Runs on CPU; needs no GPU and no network.
echo "[after] === evaluate ==="
python -u -m src.evaluate \
    --job-a runs/noise_floor_a \
    --job-b runs/noise_floor_b \
    --out reports/noise_floor 2>&1 | tee runs/report.log
EVAL_RC=${PIPESTATUS[0]}
echo "[after] evaluate rc=$EVAL_RC"

# 2. The GPU tests deferred during the night. These verify that the LOCO wiring
#    added to src/config.py and src/train.py did not disturb the training path
#    (exact-resume, precision, cache, model). They were held back because the
#    card was carrying a 13 GB run and a concurrent test could have tripped the
#    VRAM spill guard and killed a unit.
echo "[after] === deferred GPU tests ==="
python -u -m pytest tests/ -q 2>&1 | tee runs/tests_after.log
TEST_RC=${PIPESTATUS[0]}
echo "[after] pytest rc=$TEST_RC"

echo "[after] evaluate rc=$EVAL_RC  pytest rc=$TEST_RC"
echo "[after] report: reports/noise_floor.md / .json / _level1.csv / _level2.csv / _loco.csv"
echo "[after] POST-PROCESSING COMPLETE $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
