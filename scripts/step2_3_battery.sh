#!/usr/bin/env bash
# STEP 2 + STEP 3 battery (2026-08-07): runs the pinned085 image against the
# 23,408-slice diverse-pool synthetic stack under the platform's own limits
# (--memory 32g, --memory-swap 32g, --network none, --gpus all).
#
# Five runs, sequential (one GPU):
#   step2_full_m5   : 23,408 slices, 5 members  -> STEP 2 memory/timing/ties
#   step2_scale_m5  :  5,000 slices, 5 members  -> memory-SCALING comparison
#   step3_m1/m2/m3  : 23,408 slices, 1/2/3 members (RARE26_MAX_MEMBERS)
#                     -> STEP 3 T(n); the n=5 point reuses step2_full_m5.
# Runs on the HOST (docker CLI), not inside Prometheus.
set -uo pipefail

REPO="D:/RARE26"
IMAGE=rare26-convnext-base-pinned085
INPUT="$REPO/test_fixtures/step2_23408"
OUTROOT="$REPO/test_fixtures"
LOG="$REPO/logs/step2_3_battery.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

run_one() {
    local name="$1" max_members="$2" max_images="$3"
    local outdir="$OUTROOT/out_${name}"
    mkdir -p "$outdir"
    log "RUN $name (members=${max_members:-baked} images=${max_images:-all})"
    MSYS_NO_PATHCONV=1 docker run --rm \
        --network none --gpus all \
        --memory 32g --memory-swap 32g --shm-size 2g \
        --tmpfs /tmp:rw,size=1g \
        -v "$INPUT:/input:ro" \
        -v "$outdir:/output" \
        -e RARE26_PRECISION=fp16 \
        ${max_members:+-e RARE26_MAX_MEMBERS=$max_members} \
        ${max_images:+-e RARE26_MAX_IMAGES=$max_images} \
        "$IMAGE" >> "$LOG" 2>&1
    local rc=$?
    log "RUN $name exited rc=$rc"
    return 0   # continue the battery even if one run trips its tie gate
}

log "===== battery start (image=$IMAGE) ====="
run_one step2_full_m5  ""  ""
run_one step2_scale_m5 ""  5000
run_one step3_m1       1   ""
run_one step3_m2       2   ""
run_one step3_m3       3   ""
log "===== battery done ====="
