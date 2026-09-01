#!/usr/bin/env bash
# Launch scripts/82_deploy_a4_full_chain.py (Stage 1: A4-corrected full-data
# deployment checkpoints, seeds 0-4) in tmux session `deploy_a4_full`. Same
# idempotency pattern as run_ema_chain.sh / run_pinned_chain.sh: PID file +
# cmdline match (not a pgrep process-name match, which self-matches its own
# invocation), tmux session recreated if stale.
set -uo pipefail
REPO=/workspace/RARE26
SESSION=deploy_a4_full
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/deploy_a4_full_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/deploy_a4_full_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '82_deploy_a4_full_chain.py'
}
if driver_alive; then log "deploy_a4_full_chain driver alive; nothing to do."; exit 0; fi
if [ -f "$LOGDIR/deploy_a4_full_chain_done" ]; then log "deploy_a4_full_chain fully done; not relaunching."; exit 0; fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "stale session $SESSION; killing."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/deploy_a4_full_chain.pid"
log "starting deploy_a4_full_chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "python -u $REPO/scripts/82_deploy_a4_full_chain.py 2>&1 | tee -a $LOGDIR/deploy_a4_full_chain_console.log"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" || { log "FAILED"; exit 1; }
