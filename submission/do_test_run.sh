#!/usr/bin/env bash
# Run the container the way Grand Challenge will, plus the two limits the
# platform imposes that a naive local run silently ignores.
#
#   --network none    no internet, as on the platform
#   --memory 32g      the host RAM cap. THIS IS NOT OPTIONAL. The development
#                     box has 94 GB, so without an explicit cap a container
#                     that would be OOM-killed during evaluation sails through
#                     locally and the memory result means nothing.
#   --memory-swap 32g swap disabled, so exceeding the cap kills rather than pages
#   --gpus all        GPU, as on the platform
#   --tmpfs /tmp      /tmp starts empty and is not persisted
#
# Usage:
#   ./do_test_run.sh <input_dir> <output_dir> [precision]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-rare26-convnext-base}"

INPUT_DIR="${1:?usage: do_test_run.sh <input_dir> <output_dir> [precision]}"
OUTPUT_DIR="${2:?usage: do_test_run.sh <input_dir> <output_dir> [precision]}"
PRECISION="${3:-fp16}"

MEMORY_LIMIT="${MEMORY_LIMIT:-32g}"
BATCH_SIZE="${RARE26_BATCH_SIZE:-32}"
NUM_WORKERS="${RARE26_NUM_WORKERS:-8}"
# Debug-only, per-member logit dump (see tools/check_ensemble_members.py).
# Must land under /output -- /input is read-only and /tmp is not persisted.
DUMP_MEMBER_LOGITS="${RARE26_DUMP_MEMBER_LOGITS:-}"

mkdir -p "${OUTPUT_DIR}"
chmod -R o+rX "${INPUT_DIR}" 2>/dev/null || true
chmod -R o+rwX "${OUTPUT_DIR}" 2>/dev/null || true

echo "==> input      : ${INPUT_DIR}"
echo "==> output     : ${OUTPUT_DIR}"
echo "==> precision  : ${PRECISION}"
echo "==> memory cap : ${MEMORY_LIMIT} (enforced; eval host caps at 32 GB)"
echo

docker run --rm \
  --platform=linux/amd64 \
  --network none \
  --gpus all \
  --memory "${MEMORY_LIMIT}" \
  --memory-swap "${MEMORY_LIMIT}" \
  --shm-size 2g \
  --tmpfs /tmp:rw,size=1g \
  --volume "${INPUT_DIR}":/input:ro \
  --volume "${OUTPUT_DIR}":/output \
  --env RARE26_PRECISION="${PRECISION}" \
  --env RARE26_BATCH_SIZE="${BATCH_SIZE}" \
  --env RARE26_NUM_WORKERS="${NUM_WORKERS}" \
  ${DUMP_MEMBER_LOGITS:+--env RARE26_DUMP_MEMBER_LOGITS="/output/$(basename "${DUMP_MEMBER_LOGITS}")"} \
  "${DOCKER_IMAGE_TAG}"

echo
echo "==> output files"
ls -la "${OUTPUT_DIR}"
