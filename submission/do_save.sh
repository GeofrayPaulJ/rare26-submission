#!/usr/bin/env bash
# Export the image as the gzipped tarball Grand Challenge expects.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${VARIANT:-ensemble}"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-rare26-convnext-base-${VARIANT}}"
OUT="${1:-${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}.tar.gz}"

echo "==> saving ${DOCKER_IMAGE_TAG} -> ${OUT}"
echo "    (this takes a few minutes; gzip is the slow part)"
docker save "${DOCKER_IMAGE_TAG}" | gzip -c > "${OUT}"

BYTES=$(stat -c %s "${OUT}" 2>/dev/null || stat -f %z "${OUT}")
echo
echo "==> tarball : ${OUT}"
echo "==> size    : $(awk -v b="${BYTES}" 'BEGIN{printf "%.2f GiB (%d bytes)", b/1073741824, b}')"
echo "==> sha256  : $(sha256sum "${OUT}" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "${OUT}" | cut -d' ' -f1)"
