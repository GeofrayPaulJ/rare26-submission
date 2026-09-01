#!/usr/bin/env bash
# Launch the pinned-arm chain (scripts/44_pinned_arm_chain.py) inside tmux
# session `rare26_pinned_chain`. Same idempotency pattern as
# scripts/run_sweep.sh and scripts/run_chain.sh -- see run_sweep.sh's own
# header for the full reasoning (pgrep self-match trap, PID-namespace reuse
# after a container restart, tmux not surviving a container restart).
set -uo pipefail

REPO=/workspace/RARE26
SESSION=rare26_pinned_chain
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/pinned_chain.pid" ] || return 1
    local pid
    pid=$(cat "$LOGDIR/pinned_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '44_pinned_arm_chain\.py'
}

if driver_alive; then
    log "pinned-chain driver is alive (pid $(cat "$LOGDIR/pinned_chain.pid")); nothing to do."
    exit 0
fi

if [ -f "$LOGDIR/pinned_chain_done" ]; then
    log "pinned-arm chain already completed (sentinel present); not relaunching. Delete logs/pinned_chain_done to force a re-run."
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "session $SESSION exists but the driver is gone; killing the stale session."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/pinned_chain.pid"

log "starting pinned-arm chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "python -u $REPO/scripts/44_pinned_arm_chain.py 2>&1 | tee -a $LOGDIR/pinned_chain_console.log"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" \
    || { log "FAILED to start session"; exit 1; }
