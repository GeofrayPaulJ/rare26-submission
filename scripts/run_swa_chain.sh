#!/usr/bin/env bash
# Launch scripts/53_swa_chain.py in tmux session `swa`. Same idempotency
# pattern as run_pinned_chain.sh (see run_sweep.sh's header for reasoning).
set -uo pipefail
REPO=/workspace/RARE26
SESSION=swa
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/swa_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/swa_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '53_swa_chain.py'
}
if [ -f "$LOGDIR/gpu_hold" ]; then log "gpu_hold present (battery/step5 owns the GPU); not starting."; exit 0; fi
if driver_alive; then log "swa_chain driver alive; nothing to do."; exit 0; fi
if [ -f "$LOGDIR/swa_chain_report_done" ]; then log "swa_chain fully done; not relaunching."; exit 0; fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "stale session $SESSION; killing."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/swa_chain.pid"
log "starting swa_chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO"     "python -u $REPO/scripts/53_swa_chain.py 2>&1 | tee -a $LOGDIR/swa_chain_console.log && touch $LOGDIR/swa_chain_report_done"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" || { log "FAILED"; exit 1; }
