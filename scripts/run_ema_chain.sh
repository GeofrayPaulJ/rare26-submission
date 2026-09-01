#!/usr/bin/env bash
# Launch scripts/59_full_data_ema.py in tmux session `ema`. Same idempotency
# pattern as run_pinned_chain.sh (see run_sweep.sh's own header for the
# full reasoning: pgrep self-match trap, PID-namespace reuse after a
# container restart, tmux not surviving a container restart).
#
# SELF-GUARDED on step 3 (swaloco) having finished -- see
# run_swaloco_chain.sh's header for the 2026-08-07T~11:00Z incident this
# guards against (watchdog launched this directly, concurrently with swa
# AND swaloco, before either had a self-guard).
set -uo pipefail
REPO=/workspace/RARE26
SESSION=ema
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/ema_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/ema_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '59_full_data_ema.py'
}
if driver_alive; then log "ema_chain driver alive; nothing to do."; exit 0; fi
if [ -f "$LOGDIR/ema_chain_report_done" ]; then log "ema_chain fully done; not relaunching."; exit 0; fi
if [ ! -f "$LOGDIR/swaloco_chain_report_done" ]; then
    log "step 3 (swaloco) not finished yet; not starting step 4."
    exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "stale session $SESSION; killing."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/ema_chain.pid"
log "starting ema_chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO"     "python -u $REPO/scripts/59_full_data_ema.py 2>&1 | tee -a $LOGDIR/ema_chain_console.log && touch $LOGDIR/ema_chain_report_done"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" || { log "FAILED"; exit 1; }
