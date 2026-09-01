#!/usr/bin/env bash
# Build the submission image. Run from submission/.
#
# VARIANT selects which resources/<variant>/ gets baked in:
#   ensemble  (default) -- 5-fold, seed-0, logit-averaged A4 ensemble
#   fallback             -- single checkpoint (r0_f0_s0), for when the
#                            5-member image can't be built or its runtime
#                            budget doesn't fit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${VARIANT:-ensemble}"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-rare26-convnext-base-${VARIANT}}"

MANIFEST="${SCRIPT_DIR}/resources/${VARIANT}/manifest.txt"
if [ ! -f "${MANIFEST}" ]; then
  echo "ERROR: no manifest at ${MANIFEST}. Run first, from the repo root:"
  echo "  python submission/tools/extract_weights.py --checkpoint <ckpt> --out submission/resources/${VARIANT}/<name>.pth"
  exit 1
fi
MISSING=0
while IFS= read -r name; do
  [ -z "${name}" ] && continue
  if [ ! -f "${SCRIPT_DIR}/resources/${VARIANT}/${name}" ]; then
    echo "ERROR: manifest lists ${name} but it is missing under resources/${VARIANT}/"
    MISSING=1
  fi
done < "${MANIFEST}"
[ "${MISSING}" -eq 0 ] || exit 1

N_MEMBERS=$(grep -c . "${MANIFEST}")
TOTAL_SIZE=$(du -ch "${SCRIPT_DIR}/resources/${VARIANT}"/*.pth | tail -1 | cut -f1)
echo "==> building ${DOCKER_IMAGE_TAG} (variant=${VARIANT}, ${N_MEMBERS} checkpoint(s), ${TOTAL_SIZE})"
docker build \
  --platform=linux/amd64 \
  --build-arg VARIANT="${VARIANT}" \
  --tag "${DOCKER_IMAGE_TAG}" \
  "${SCRIPT_DIR}"

echo
echo "==> image size"
docker images "${DOCKER_IMAGE_TAG}" --format '    {{.Repository}}:{{.Tag}}  {{.Size}}'
