#!/usr/bin/env bash
# Weekend queue: G3+pAUC pooled (stage 1, 25 units, runs/g3_pauc) -> G3+pAUC
# LOCO (stage 2, 10 units, runs/g3_pauc_loco) -> G3 full-data deployment
# checkpoint (stage 5, runs/deploy_g3_full, scripts/61_full_data_g3.py) ->
# tighten-the-margin LOCO seeds 5-9 for A4-corrected and G3 (stage 7, 10
# units each, extending runs/a4_checkpointed_loco and
# runs/g3_rn50_gastronet_loco). Sequentially, in tmux session `g3pauc`. Per
# reports/g3_component_pre_registration.md and the weekend-queue instruction.
# Stages 3/4/6/8 (EVC-dependent) are DELIBERATELY NOT in this chain -- no
# training-time EVC-paste integration exists in src/, and building that
# unattended was explicitly ruled out.
#
# Same idempotency pattern as every other run_*_chain.sh in this repo
# (PID-file liveness, `_done` sentinel, stale-tmux cleanup) -- see
# run_g3ckpt_chain.sh's header for the full reasoning (pgrep self-match
# trap, PID-namespace reuse after a container restart, tmux not surviving
# one). Deliberately BASH-only, no separate python chain-driver script for
# stages 1/2/7: scripts/run_cv.py is already unit-level resumable
# (skip-validated) on its own, so calling it in sequence from a shell
# script that is itself safe to re-invoke (this file) is sufficient. Stage
# 5 is its own python script (61_full_data_g3.py) because it has no fold
# structure for run_cv.py to drive -- same reasoning as
# scripts/59_full_data_ema.py being standalone.
#
# COMPLETION SENTINEL RENAMED from the original stages-1+2-only version's
# `g3pauc_chain_report_done` to `g3pauc_full_chain_done`, specifically so
# that name never collides with whatever the ALREADY-RUNNING (at the time
# this file was extended) stage-1/2-only process would touch when it
# finished -- that process's `logs/g3pauc_chain_inner.sh` was already on
# disk and already executing before this edit, so its own hard-coded old
# sentinel name still fires when it completes, but this launcher no longer
# checks for that name, so it correctly relaunches (near-instantly
# skip-validated through the now-complete stages 1/2) to pick up stages 5/7
# rather than mistaking the old sentinel for "fully done."
#
# WIRED INTO scripts/sweep_watchdog.ps1 (added same session this file was
# extended, tested for one poll cycle against the other 11 drivers before
# trusting it -- see that script's own g3pauc block and
# reports/weekend_queue_launch.md for the test record). NOT wired into
# scripts/container_start_hook.sh -- that file is sourced into the
# container's PID-1 entrypoint shell, higher blast radius, left for a
# calmer edit.
set -uo pipefail
REPO=/workspace/RARE26
SESSION=g3pauc
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

driver_alive() {
    [ -f "$LOGDIR/g3pauc_chain.pid" ] || return 1
    local pid; pid=$(cat "$LOGDIR/g3pauc_chain.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'g3pauc_chain_inner'
}
if driver_alive; then log "g3pauc_chain driver alive; nothing to do."; exit 0; fi
if [ -f "$LOGDIR/g3pauc_full_chain_done" ]; then log "g3pauc_chain (full, stages 1/2/5/7) fully done; not relaunching."; exit 0; fi
if [ -f "$LOGDIR/gpu_hold" ]; then log "gpu_hold present; not starting (avoiding GPU contention)."; exit 0; fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "stale session $SESSION; killing."
    tmux kill-session -t "$SESSION" 2>/dev/null
fi
rm -f "$LOGDIR/g3pauc_chain.pid"

cat > "$LOGDIR/g3pauc_chain_inner.sh" <<'INNER'
#!/usr/bin/env bash
set -uo pipefail
REPO=/workspace/RARE26
LOGDIR="$REPO/logs"
echo "$$" > "$LOGDIR/g3pauc_chain.pid"
cd "$REPO"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
halt() {
    log "$1 FAILED rc=$2; halting chain"
    echo "G3PAUC_HALTED $(date -u +%Y-%m-%dT%H:%M:%SZ) $1 rc=$2" >> "$LOGDIR/g3pauc_chain_status"
    exit "$2"
}

log "stage 1: G3+pAUC pooled, 25 units, out_dir runs/g3_pauc"
python -u scripts/run_cv.py --config configs/g3_pauc.yaml --mode cv \
    --repeats 0 --folds 0,1,2,3,4 --seeds 0,1,2,3,4 --out-dir runs/g3_pauc
rc=$?; [ $rc -eq 0 ] || halt stage1 $rc
log "stage 1 complete"

log "stage 2: G3+pAUC LOCO, 10 units, out_dir runs/g3_pauc_loco"
python -u scripts/run_cv.py --config configs/g3_pauc.yaml --mode loco \
    --centres 1,2 --seeds 0,1,2,3,4 --out-dir runs/g3_pauc_loco
rc=$?; [ $rc -eq 0 ] || halt stage2 $rc
log "stage 2 complete"

log "stage 5: G3 full-data deployment checkpoint, out_dir runs/deploy_g3_full"
python -u scripts/61_full_data_g3.py
rc=$?; [ $rc -eq 0 ] || halt stage5 $rc
log "stage 5 complete"

log "stage 7a: tighten margin -- A4-corrected LOCO seeds 5-9, 10 units, extending runs/a4_checkpointed_loco"
python -u scripts/run_cv.py --config configs/sweep_a4_checkpointed.yaml --mode loco \
    --centres 1,2 --seeds 5,6,7,8,9 --out-dir runs/a4_checkpointed_loco
rc=$?; [ $rc -eq 0 ] || halt stage7a $rc
log "stage 7a complete"

log "stage 7b: tighten margin -- G3 LOCO seeds 5-9, 10 units, extending runs/g3_rn50_gastronet_loco"
python -u scripts/run_cv.py --config configs/g3_rn50_gastronet.yaml --mode loco \
    --centres 1,2 --seeds 5,6,7,8,9 --out-dir runs/g3_rn50_gastronet_loco
rc=$?; [ $rc -eq 0 ] || halt stage7b $rc
log "stage 7b complete -- chain done (stages 1/2/5/7)"
echo "G3PAUC_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOGDIR/g3pauc_chain_status"
INNER
chmod +x "$LOGDIR/g3pauc_chain_inner.sh"

log "starting g3pauc_chain in tmux session $SESSION"
tmux new-session -d -s "$SESSION" -c "$REPO" \
    "bash $LOGDIR/g3pauc_chain_inner.sh 2>&1 | tee -a $LOGDIR/g3pauc_chain_console.log && touch $LOGDIR/g3pauc_full_chain_done"
sleep 2
tmux has-session -t "$SESSION" 2>/dev/null && log "session $SESSION started" || { log "FAILED"; exit 1; }
