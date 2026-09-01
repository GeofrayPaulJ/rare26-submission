#!/usr/bin/env bash
# Launch scripts/64_paucloco_chain.py in tmux session `paucloco`. Same idempotency
# pattern as run_pinned_chain.sh.
set -uo pipefail
REPO=/workspace/RARE26
SESSION=paucloco
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/paucloco_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/paucloco_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '64_paucloco_chain.py'
}
if driver_alive; then log "paucloco_chain driver alive; nothing to do."; exit 0; fi
if [ -f "$LOGDIR/paucloco_chain_report_done" ]; then log "paucloco_chain fully done; not relaunching."; exit 0; fi
# SELF-GUARDED on g3ckpt having finished attempting (its own report_done
# fires on success or failure -- g3ckpt is an independent stage, so this
# must not wait for it to have SUCCEEDED, only to be OVER).
if [ ! -f "$LOGDIR/g3ckpt_chain_report_done" ]; then
    log "g3ckpt not finished attempting yet; not starting paucloco."
    exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "stale session $SESSION; killing."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/paucloco_chain.pid"
log "starting paucloco_chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO"     "python -u $REPO/scripts/64_paucloco_chain.py 2>&1 | tee -a $LOGDIR/paucloco_chain_console.log && touch $LOGDIR/paucloco_chain_report_done"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" || { log "FAILED"; exit 1; }
