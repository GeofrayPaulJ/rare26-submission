# RARE26 submission. Thin wrapper over the do_*.sh scripts so `make` and the
# shell scripts cannot drift apart -- the scripts are the implementation.
#
# VARIANT selects resources/<variant>/: "ensemble" (5-fold, seed-0,
# logit-averaged -- the default, what ships) or "fallback" (single
# checkpoint, r0_f0_s0).
#
#   make weights   bake the canonical checkpoint(s) into resources/$(VARIANT)/
#   make build     build the image (tag: rare26-convnext-base-$(VARIANT))
#   make arch      verify the shipped torch covers T4 (sm_75) and A10G (sm_86)
#   make parity    617-image validation stack -> compare against the harness
#   make bench     25,000-image synthetic stack under a real 32 GB cap
#   make save      export the upload tarball

SHELL := /bin/bash
VARIANT ?= ensemble
IMAGE ?= rare26-convnext-base-$(VARIANT)
REPO_ROOT := $(shell cd .. && pwd)
TEST_DIR ?= $(REPO_ROOT)/runs/submission_test

.PHONY: all weights build arch parity bench fallback save clean

all: weights build arch

weights:
	@echo "weights are baked per-checkpoint; see submission/tools/extract_weights.py --help"
	@echo "resources/ensemble/ and resources/fallback/ are already populated from runs/a4_checkpointed"

build:
	VARIANT=$(VARIANT) DOCKER_IMAGE_TAG=$(IMAGE) ./do_build.sh

arch:
	docker run --rm --gpus all --platform=linux/amd64 \
	  -v "$(CURDIR)/tools":/tools:ro --entrypoint python $(IMAGE) /tools/check_arch.py

parity:
	cd $(REPO_ROOT) && python submission/tools/make_test_stack.py parity \
	  --out-dir $(TEST_DIR)/parity/interface_0
	DOCKER_IMAGE_TAG=$(IMAGE) ./do_test_run.sh \
	  $(TEST_DIR)/parity/interface_0 $(TEST_DIR)/parity/output fp16

bench:
	cd $(REPO_ROOT) && python submission/tools/make_test_stack.py synthetic \
	  --out-dir $(TEST_DIR)/bench/interface_0 --n 25000
	DOCKER_IMAGE_TAG=$(IMAGE) ./do_test_run.sh \
	  $(TEST_DIR)/bench/interface_0 $(TEST_DIR)/bench/output fp16

fallback:
	cd $(REPO_ROOT) && python submission/tools/fallback_rate.py

save:
	VARIANT=$(VARIANT) DOCKER_IMAGE_TAG=$(IMAGE) ./do_save.sh

clean:
	rm -rf $(TEST_DIR)
