# RARE26 submission image.
#
# Multi-stage purely for size: the builder resolves and installs the whole
# dependency tree into a venv, strips test suites and bytecode caches, and the
# final stage copies that one directory onto a clean python:3.12-slim. Nothing
# from pip's own cache, no build toolchain and no apt lists survive into the
# shipped layer.
#
# There is deliberately no nvidia/cuda base image. The torch cu128 wheels vendor
# their own CUDA runtime, cuDNN and cuBLAS, so all the container needs from the
# host is the driver, which Grand Challenge injects at run time. Starting from
# nvidia/cuda would add several GB of duplicated runtime for no benefit.

# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    # triton is only reachable through torch.compile, which this container
    # never calls, and it is 538 MB. Everything else nvidia ships is a hard
    # DT_NEEDED of libtorch_cuda -- nccl, cusparse, cusparseLt, cusolver, cufft,
    # curand and cupti were each removed experimentally and each broke
    # `import torch` outright, so they stay.
    && rm -rf /venv/lib/python3.12/site-packages/triton \
    && find /venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /venv -type f -name '*.pyc' -delete
    # NOTE: do NOT blanket-delete directories named "tests". numpy ships
    # numpy/_core/tests/_natype.py, which numpy.testing imports at import time,
    # and scipy does `from numpy import *` -- which triggers it. Stripping
    # "tests" saved ~40 MB and made `import scipy` fail with
    # ModuleNotFoundError: numpy._core.tests. The image built and passed a
    # torch-only smoke test; it died the first time the real entrypoint ran.

# ---------------------------------------------------------------------------
# final
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH" \
    # /tmp is empty and tmpfs at run time, and there is no network, so pin every
    # cache torch/timm might otherwise try to populate or download into.
    TORCH_HOME=/opt/app/.cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OPENCV_IO_ENABLE_OPENEXR=0 \
    # cv2 and the loader workers each grab a full thread pool otherwise, which
    # oversubscribes the CPU the platform actually grants us.
    OMP_NUM_THREADS=1

# opencv-python-headless still links libglib and libgomp; nothing else is needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m -r -g user user

COPY --from=builder /venv /venv

WORKDIR /opt/app
RUN chown -R user:user /opt/app

# VARIANT picks which baked checkpoint set ships: "ensemble" (5-fold,
# seed-0, logit-averaged -- what actually ships) or "fallback" (single
# checkpoint, r0_f0_s0, for when the 5-member image is unavailable or the
# runtime budget forces a single-member run). Each resources/<variant>/
# carries its own manifest.txt read by inference.py at startup, so the
# entrypoint needs no variant-specific code path.
ARG VARIANT=ensemble
COPY --chown=user:user resources/${VARIANT} /opt/app/resources
COPY --chown=user:user rare26_infer         /opt/app/rare26_infer
COPY --chown=user:user inference.py         /opt/app/inference.py

USER user

ENTRYPOINT ["python", "inference.py"]

LABEL rare26.purpose="sanity-check-control-2026-08-24"
