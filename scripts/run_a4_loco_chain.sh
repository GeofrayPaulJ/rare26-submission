#!/usr/bin/env bash
# Launch JOB 1 (scripts/49_a4_loco_chain.py) inside tmux session
# `rare26_a4_loco_chain`. Same idempotency pattern as run_pinned_chain.sh /
# run_sweep.sh / run_chain.sh -- see run_sweep.sh's own header for the full
# reasoning (pgrep self-match trap, PID-namespace reuse after a container
# restart, tmux not surviving a container restart).
set -uo pipefail

REPO=/workspace/RARE26
SESSION=rare26_a4_loco_chain
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/a4_loco_chain.pid" ] || return 1
    local pid
    pid=$(cat "$LOGDIR/a4_loco_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '49_a4_loco_chain\.py'
}

if driver_alive; then
    log "a4-loco-chain driver is alive (pid $(cat "$LOGDIR/a4_loco_chain.pid")); nothing to do."
    exit 0
fi

if [ -f "$LOGDIR/a4_loco_chain_done" ]; then
    log "A4 LOCO chain already completed (sentinel present); not relaunching. Delete logs/a4_loco_chain_done to force a re-run."
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "session $SESSION exists but the driver is gone; killing the stale session."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/a4_loco_chain.pid"

log "starting A4 LOCO chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "python -u $REPO/scripts/49_a4_loco_chain.py 2>&1 | tee -a $LOGDIR/a4_loco_chain_console.log"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" \
    || { log "FAILED to start session"; exit 1; }
