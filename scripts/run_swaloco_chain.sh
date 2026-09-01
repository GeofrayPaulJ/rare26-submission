#!/usr/bin/env bash
# Launch scripts/58_swaloco_chain.py in tmux session `swaloco`. Same idempotency
# pattern as run_pinned_chain.sh (see run_sweep.sh's own header for the
# full reasoning: pgrep self-match trap, PID-namespace reuse after a
# container restart, tmux not surviving a container restart).
#
# SELF-GUARDED on step 2 (pauc smoke) having finished ATTEMPTING (its own
# report_done fires on success or failure -- step 2 is an independent,
# fall-through stage, so step 3 must not wait for it to have SUCCEEDED,
# only to be OVER). INCIDENT 2026-08-07T~11:00Z: this launcher originally
# had no such guard, relying only on scripts/54_pauc_chain.py's own
# end-of-run chain-call to invoke it in order. The watchdog's independent
# per-driver liveness check (which fires every ~60s, not just on the 5-min
# Task Scheduler cycle) found no swaloco_chain_report_done sentinel and
# launched this directly WHILE swa was still training its last unit --
# three training processes ran concurrently on one GPU (swa, swaloco, and
# step 4's ema, which had the same gap) until caught and killed manually.
# Fixed here and in run_ema_chain.sh / run_pauc_p1_chain.sh: every launcher
# in this chain must independently verify its own prerequisite, not trust
# the self-chaining call alone.
set -uo pipefail
REPO=/workspace/RARE26
SESSION=swaloco
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/swaloco_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/swaloco_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '58_swaloco_chain.py'
}
if driver_alive; then log "swaloco_chain driver alive; nothing to do."; exit 0; fi
if [ -f "$LOGDIR/swaloco_chain_report_done" ]; then log "swaloco_chain fully done; not relaunching."; exit 0; fi
if [ ! -f "$LOGDIR/pauc_chain_report_done" ]; then
    log "step 2 (pauc smoke) not finished attempting yet; not starting step 3."
    exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "stale session $SESSION; killing."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/swaloco_chain.pid"
log "starting swaloco_chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO"     "python -u $REPO/scripts/58_swaloco_chain.py 2>&1 | tee -a $LOGDIR/swaloco_chain_console.log && touch $LOGDIR/swaloco_chain_report_done"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" || { log "FAILED"; exit 1; }
