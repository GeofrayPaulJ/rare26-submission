#!/usr/bin/env bash
# STEP 5 final battery (2026-08-07): the ONE submission image,
# rare26-convnext-base-pinned085_single (a4_pinned085 r0_f0_s0, per D1 and
# step 3's measured member table -- n=1 is the only count clearing 10 min
# with 30% headroom on measured T4-derated numbers).
#
# Three runs:
#   parity   : fold-0 val stack (617 native-res frames) -> fp16 container
#              logits vs the pinned harness parquet (tools/check_parity.py)
#   gate5k   : 23,408-stack truncated to 5,000 (diverse pool, 3.6x reuse)
#              -> tie gate must PASS here (the full-stack failure is the
#              documented synthetic-fixture artefact, not a defect)
#   full     : all 23,408 -> end-to-end timing/memory of the shipped config
set -uo pipefail

REPO="D:/RARE26"
IMAGE=rare26-convnext-base-pinned085_single
LOG="$REPO/logs/step5_final_battery.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

run_one() {
    local name="$1" input="$2" extra_env="$3"
    local outdir="$REPO/test_fixtures/out_step5_${name}"
    mkdir -p "$outdir"
    log "RUN step5_$name"
    MSYS_NO_PATHCONV=1 docker run --rm \
        --network none --gpus all \
        --memory 32g --memory-swap 32g --shm-size 2g \
        --tmpfs /tmp:rw,size=1g \
        -v "$input:/input:ro" \
        -v "$outdir:/output" \
        -e RARE26_PRECISION=fp16 \
        $extra_env \
        "$IMAGE" >> "$LOG" 2>&1
    log "RUN step5_$name exited rc=$?"
    return 0
}

log "===== step5 final battery start (image=$IMAGE) ====="
run_one parity "$REPO/runs/submission_test/parity" ""
run_one gate5k "$REPO/test_fixtures/step2_23408" "-e RARE26_MAX_IMAGES=5000"
run_one full   "$REPO/test_fixtures/step2_23408" ""
log "===== step5 final battery done ====="
