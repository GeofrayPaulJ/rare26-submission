#!/usr/bin/env bash
# Launch the overnight chain (scripts/40_overnight_chain.py) inside tmux
# session `rare26_chain`. Mirrors scripts/run_sweep.sh's idempotency exactly
# -- see that file's own header for the full reasoning (pgrep self-match
# trap, PID-namespace reuse after a container restart, tmux not surviving a
# container restart). This file only restates what's different: the session
# name, the driver script, and the PID file, all chain-specific.
#
# WHY THIS EXISTS. scripts/40_overnight_chain.py has its own sentinel-based
# stage skipping and calls run_cv.py, which is itself resumable -- but NONE
# of that fires automatically on a container restart unless something calls
# this script again. scripts/container_start_hook.sh does that, the same way
# it already does for run_sweep.sh.
set -uo pipefail

REPO=/workspace/RARE26
SESSION=rare26_chain
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/chain.pid" ] || return 1
    local pid
    pid=$(cat "$LOGDIR/chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '40_overnight_chain\.py'
}

if driver_alive; then
    log "chain driver is alive (pid $(cat "$LOGDIR/chain.pid")); nothing to do."
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "session $SESSION exists but the driver is gone; killing the stale session."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/chain.pid"

log "starting overnight chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "python -u $REPO/scripts/40_overnight_chain.py 2>&1 | tee -a $LOGDIR/chain_console.log"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" \
    || { log "FAILED to start session"; exit 1; }
