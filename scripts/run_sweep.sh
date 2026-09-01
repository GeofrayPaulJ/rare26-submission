#!/usr/bin/env bash
# Launch the magnitude sweep inside tmux session `rare26_sweep`.
#
# Idempotent and safe to call repeatedly -- that is the whole point, because it
# is what the host-side watchdog calls. If the session already exists and the
# driver is alive inside it, this does nothing. If the sweep halted
# deliberately or finished, this does nothing (and says so). Otherwise it
# starts, and scripts/23_sweep.py re-derives its plan from what is on disk, so
# a relaunch loses at most the unit that was in flight.
#
# NOTE ON DURABILITY: tmux does NOT survive a container restart. It lives in
# the container's PID namespace, so a restart takes the server and every
# process under it. This was tested, not assumed. Durability comes from the
# container's unless-stopped restart policy plus the watchdog plus the sweep's
# own resumability -- not from tmux, which only survives client disconnect.
set -uo pipefail

REPO=/workspace/RARE26
SESSION=rare26_sweep
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

if [ -f "$LOGDIR/sweep_halt.json" ]; then
    log "sweep halted deliberately; not relaunching. See logs/sweep_halt.json:"
    cat "$LOGDIR/sweep_halt.json"
    exit 0
fi

if [ -f "$LOGDIR/sweep_stage_complete" ]; then
    log "sweep already completed its current plan (see logs/sweep_status.json " \
        "current_stage); not relaunching. Delete this sentinel to extend the " \
        "plan and resume."
    exit 0
fi

# Liveness by PID, never by process name. `pgrep -f 23_sweep.py` self-matches
# whenever the caller's own command line carries the pattern -- which it does,
# because the watchdog asks through `docker exec bash -c "...pgrep..."`. Tested:
# with zero driver processes alive the pgrep check still answered "1".
# A container restart resets the PID namespace, so `kill -0 $pid` alone is not
# enough: the recorded pid can be re-issued to a completely unrelated process
# after the restart, and this check would then report the driver alive forever
# and never relaunch it -- the same "confidently wrong liveness" failure the
# pgrep -f trap produced, arriving by a different route. So verify the pid is
# actually OUR driver by reading its cmdline, not merely that something holds
# that pid. Reading /proc/PID/cmdline has no self-match problem: it inspects
# the target process's argv, not the argv of the shell asking.
driver_alive() {
    [ -f "$LOGDIR/sweep.pid" ] || return 1
    local pid
    pid=$(cat "$LOGDIR/sweep.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '23_sweep\.py'
}

if driver_alive; then
    log "driver is alive (pid $(cat "$LOGDIR/sweep.pid")); nothing to do."
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "session $SESSION exists but the driver is gone; killing the stale session."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/sweep.pid"

# --skip-prep is passed on relaunch only: the gates are properties of the
# working tree, not of the run, and re-paying for a GPU regression check on
# every watchdog restart would be waste. The first launch always runs them.
PREP_FLAG=""
if [ -f "$LOGDIR/sweep_prep_passed" ]; then
    PREP_FLAG="--skip-prep"
    log "prep gates already passed in this tree; relaunching with --skip-prep"
fi

log "starting sweep in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "python -u $REPO/scripts/23_sweep.py $PREP_FLAG 2>&1 | tee -a $LOGDIR/sweep_tmux.log"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" \
    || { log "FAILED to start session"; exit 1; }
