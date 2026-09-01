#!/usr/bin/env bash
# Launch scripts/60_pauc_p1_chain.py in tmux session `pauc`. Same idempotency
# pattern as run_pinned_chain.sh (see run_sweep.sh's own header for the
# full reasoning: pgrep self-match trap, PID-namespace reuse after a
# container restart, tmux not surviving a container restart).
#
# SELF-GUARDED on both (a) step 4 (ema) having finished and (b) the pAUC
# smoke having been clean -- checked HERE, not only inside
# scripts/59_full_data_ema.py's own chain-call, so a watchdog cycle that
# invokes this launcher directly (its normal per-driver liveness check,
# independent of the self-chaining sequence) cannot bypass the "ONLY if
# step 2's smoke is clean" gate if step 4 crashed before reaching its own
# check.
set -uo pipefail
REPO=/workspace/RARE26
SESSION=pauc
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/pauc_p1_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/pauc_p1_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q '60_pauc_p1_chain.py'
}
if driver_alive; then log "pauc_p1_chain driver alive; nothing to do."; exit 0; fi
if [ -f "$LOGDIR/pauc_p1_chain_report_done" ]; then log "pauc_p1_chain fully done; not relaunching."; exit 0; fi
if [ ! -f "$LOGDIR/ema_chain_report_done" ]; then
    log "step 4 (ema) not finished yet; not starting step 5."
    exit 0
fi
SMOKE_OK=$(python3 -c "
import json, sys
try:
    with open('$REPO/reports/pauc_smoke.json') as fh:
        s = json.load(fh)
    ok = (not s.get('diverged', True)) and (not s.get('gradient_collapsed', True))
    print('YES' if ok else 'NO')
except Exception:
    print('NO')
" 2>/dev/null)
if [ "$SMOKE_OK" != "YES" ]; then
    log "pAUC smoke was not clean (or reports/pauc_smoke.json missing); step 5 stays skipped."
    touch "$LOGDIR/pauc_p1_chain_report_done"
    exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "stale session $SESSION; killing."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/pauc_p1_chain.pid"
log "starting pauc_p1_chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO"     "python -u $REPO/scripts/60_pauc_p1_chain.py 2>&1 | tee -a $LOGDIR/pauc_p1_chain_console.log && touch $LOGDIR/pauc_p1_chain_report_done"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" || { log "FAILED"; exit 1; }
