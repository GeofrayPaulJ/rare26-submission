#!/usr/bin/env bash
# Launch scripts/63_g3_checkpoint_retrain.py in tmux session `g3ckpt`. Same idempotency
# pattern as run_pinned_chain.sh.
set -uo pipefail
REPO=/workspace/RARE26
SESSION=g3ckpt
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/g3ckpt_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/g3ckpt_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '63_g3_checkpoint_retrain.py'
}
if driver_alive; then log "g3ckpt_chain driver alive; nothing to do."; exit 0; fi
if [ -f "$LOGDIR/g3ckpt_chain_report_done" ]; then log "g3ckpt_chain fully done; not relaunching."; exit 0; fi
if [ -f "$LOGDIR/gpu_hold" ]; then log "gpu_hold present; not starting (avoiding GPU contention)."; exit 0; fi
paucloco_alive() {
    [ -f "$LOGDIR/paucloco_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/paucloco_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null
}
if paucloco_alive; then
    log "paucloco (JOB C) is running; not starting g3ckpt to avoid GPU contention -- will pick this up once paucloco finishes."
    exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "stale session $SESSION; killing."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/g3ckpt_chain.pid"
log "starting g3ckpt_chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO"     "python -u $REPO/scripts/63_g3_checkpoint_retrain.py 2>&1 | tee -a $LOGDIR/g3ckpt_chain_console.log && touch $LOGDIR/g3ckpt_chain_report_done"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" || { log "FAILED"; exit 1; }
