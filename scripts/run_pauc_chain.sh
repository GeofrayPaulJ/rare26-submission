#!/usr/bin/env bash
# Launch scripts/54_pauc_chain.py in tmux session `pauc`. Same idempotency
# pattern as run_pinned_chain.sh (see run_sweep.sh's header for reasoning).
set -uo pipefail
REPO=/workspace/RARE26
SESSION=pauc
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/pauc_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/pauc_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '54_pauc_chain.py'
}
if driver_alive; then log "pauc_chain driver alive; nothing to do."; exit 0; fi
if [ -f "$LOGDIR/pauc_chain_report_done" ]; then log "pauc_chain fully done; not relaunching."; exit 0; fi
# GPU-ordering guard: the pAUC smoke must not start while the SWA arm still
# owns the GPU. The SWA driver invokes this launcher itself on completion.
if [ ! -f "$LOGDIR/swa_chain_done" ]; then
    log "swa arm not finished (no swa_chain_done); refusing to start pauc."
    exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "stale session $SESSION; killing."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/pauc_chain.pid"
log "starting pauc_chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO"     "python -u $REPO/scripts/54_pauc_chain.py 2>&1 | tee -a $LOGDIR/pauc_chain_console.log && touch $LOGDIR/pauc_chain_report_done"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" || { log "FAILED"; exit 1; }
