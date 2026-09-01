# RARE26 Grand Challenge submission

**v2**: a 5-fold, single-seed ConvNeXt-Base ensemble (seed 0, folds 0-4),
logit-averaged, no TTA, no multi-crop. Per the throughput budget
(`reports/member_budget.md`, `reports/throughput_bench.md`) this is the
largest ensemble configuration that clears a 60-minute runtime limit; a
1-checkpoint fallback image ships alongside it for a tighter limit or if the
5-member image proves unworkable. The objective is still to be *valid and
correct* — every part of the container is chosen to remove a way the
submission could fail rather than to add a fraction of AUC.

| | |
|---|---|
| Model (ensemble) | `convnext_base.fb_in1k` x 5, 1 logit head each, 87.6 M params/member |
| Checkpoints | repeat 0 / folds 0-4 / seed 0, **last epoch (30)**, `canonical=True`, from `runs/a4_checkpointed` (the checkpointed A4 retrain, item 5 remediation) |
| Combination | logit average across members, pre-sigmoid |
| Fallback image | same pipeline, `resources/fallback/` (fold 0 only) |
| Input socket | `stacked-barretts-esophagus-endoscopy-images` |
| Input path | `/input/images/stacked-barretts-esophagus-endoscopy/` |
| Output | `/output/stacked-neoplastic-lesion-likelihoods.json` — a flat JSON list of floats, one per slice, in stack order |
| Precision | fp16 (T4/A10G have no bf16), fp32 classifier head per member, float64 sigmoid after averaging |

Pooled-OOF ensemble FPR@90R for `runs/a4_checkpointed` (this container's
source checkpoints) is **0.0280** (`reports/gastronet.md`, "A4 new"; ROC-AUC
0.9706, pAUC[0,0.15] 0.9506) — distinct from the **0.0529** figure in
`reports/a4.md`, which is the pre-code-fix `runs/sweep_a4` baseline. Item 5's
isolation run (`reports/a4_pinned085_verdict.md`) has since CONFIRMED
`downsample_ceiling_m` as the parameter explaining the old-vs-new gap (pinning
it to the old value of 0.85 reproduces 0.0529 almost exactly); `runs/a4_checkpointed`
itself uses current, unpinned code, so 0.0280 stands as that run's own figure.

**But 0.0280 is a 5-seed x 5-fold (25-unit) pooled figure, not this container's
own.** This container ships only seed 0 (5 units, the runtime-budget-constrained
configuration). Scored on the identical pooled-OOF images, this container's
own single-seed bf16 pooled FPR@90R is **~0.12** (`reports/a4_fp16_vs_bf16_rank_agreement.md`)
— substantially worse than 0.0280, consistent with this project's established
single-seed-vs-pooled gap (`reports/magnitude_sweep.md`, "1-seed med" column).
Do not expect this container's real submission score to land near 0.0280.

## Validated (2026-08-07), GPU idle, real measurements

Full battery + T4 derate: `reports/container_v2_battery.md`. Headline: real
dev-GPU wall-clock 43.3 img/s (ensemble) / 207.6 img/s (fallback) on a
23,400-image stack; peak host RAM ~3.3 GiB / ~2.9 GiB (both ~90% margin under
the 32 GB cap); peak VRAM 2.68 / 1.37 GiB. T4-derated (x3.0, reused estimate):
ensemble ~27.2 min (clears 60 min, not 30 min), fallback ~5.7 min (clears all
three assumed limits). **The shipped 99% distinct-logit gate FAILS for BOTH
images on the default synthetic stack** — root-caused to `make_test_stack.py`'s
pool selection landing on a manifest region that is 93% one near-duplicate
cluster (not an ensemble or precision defect; a diagnostic stack with a larger,
more diverse pool passes cleanly for both images). See the report for the full
analysis before relying on the default `make bench` stack for anything.

fp16-vs-bf16 rank agreement: `reports/a4_fp16_vs_bf16_rank_agreement.md`.
Headline: Kendall tau 0.9914, Spearman rho 0.9999 (detector path) — fp16 does
not materially reorder images near the threshold; FPR@90R delta +0.0069
(detector path) to +0.0085 (all images), fp16 worse than bf16 as expected from
rounding, not a bug.

Per-member validation fold metrics (`runs/a4_checkpointed/r0_f{0-4}_s0/summary.json`,
this container's own checkpoints — not the pooled-OOF ensemble figure above):

| fold | ROC-AUC | pAUC[0,0.15] | PPV@90R | FPR@90R |
|---|---|---|---|---|
| 0 | 0.9358 | 0.8218 | 0.2471 | 0.1451 |
| 1 | 0.9538 | 0.8015 | 0.2711 | 0.1280 |
| 2 | 0.9614 | 0.8519 | 0.2509 | 0.1468 |
| 3 | 0.9854 | 0.9208 | 0.6288 | 0.0290 |
| 4 | 0.9745 | 0.9183 | 0.7619 | 0.0154 |

## The four things that would have broken this

**1. Loading the stack.** The template does
`SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path))`, which materialises the
whole stack. At 25,000 × 512 × 637 × 3 that is 24.5 GB against a 32 GB cap,
and it does not fail cleanly — it gets OOM-killed with no useful message.
`rare26_infer/stack.py` reads one slice at a time (tifffile page-wise for TIFF,
SimpleITK `SetExtractIndex`/`SetExtractSize` streaming for MetaImage) and never
holds more than a batch.

**2. The socket slug is not the directory name.** `inputs.json` carries
`stacked-barretts-esophagus-endoscopy-images`; the directory the platform mounts
is `images/stacked-barretts-esophagus-endoscopy`, without the trailing
`-images`. `resolve_image_dir()` tries both documented spellings and then falls
back to scanning `/input/images` for any directory containing an image file, so
a platform-side rename costs a log line instead of the submission.

**3. bf16 does not exist on the evaluation GPU.** fp16 is the default and the
tested path. `tools/check_arch.py` fails the build if the shipped torch is not
compiled for sm_75 (T4) and sm_86 (A10G) — otherwise the first job would either
JIT from PTX or refuse to run, and neither is visible until it is too late.

**4. Ties inflate FPR.** Under fp16 autocast a bare classifier matmul quantises
its output onto the fp16 grid and collapses distinct images onto identical
logits. The classifier `Linear` therefore runs in fp32 (`Fp32Linear`), and the
sigmoid is taken in float64 — at logit ±8.5 an fp32 sigmoid is already losing
resolution, while float64 does not saturate until ≈37. The run **fails** if
fewer than 99% of outputs are distinct, because with 23,176 negatives every tie
at the operating threshold costs score for a purely numerical reason.

## Preprocessing must match training exactly

    FOV circle detect → inscribed square crop → resize 431 → resize 384 → ImageNet normalise

The **two-step resize is not redundant**. Training decoded into a RAM cache at
431 px and resized *that* to 384 at read time; both steps used `cv2.INTER_AREA`.
Going straight to 384 is a different low-pass and gives different logits.
`tools/check_parity.py` is what pins this — measured on the 617 validation
images of r0/f0, the detector path agrees with `src/train.py` to **4.7e-13** in
bf16, i.e. bit-identical. That test fails the moment anyone "simplifies" the
intermediate resize away.

`rare26_infer/fov.py` contains a **verbatim** copy of `fit_fov_circle` and
`inner_square` from `scripts/05_fov_crop.py` — same maths, same rounding, same
clamping — because the manifest geometry every training crop came from was
produced by exactly that code.

## The FOV fallback

The detector was tuned on two Dutch centres; the test set comes from twelve
unseen ones. When detection fails, or `fit_quality` falls below **0.9040** (the
1st percentile over the 3,095 training images), the container discards the
fitted circle and takes a centred square of side `0.75 × min(H, W)`. Every
fallback is counted, logged with its reason, and its slice index recorded in
`rare26_run_stats.json`.

Measured trigger rate: **1.35 % on the held-out center_2 split** (11/816),
0.92 % on center_1, 1.03 % across all training images. It is a safety net, not
a load-bearing path.

Note that fallback images are *supposed* to disagree with the training harness:
the harness cropped every image with whatever geometry the detector produced,
however poor the fit. `check_parity.py` therefore scores the detector path to a
near-exact tolerance and reports the fallback images separately — pooling them
would hide a real preprocessing bug behind a few intentional differences.

## Layout

    inference.py              entrypoint; reads resources/<variant>/manifest.txt
    rare26_infer/
      fov.py                  verbatim FOV detector + the fallback
      preprocess.py           crop / two-step resize / ImageNet normalise
      stack.py                slice-wise TIFF and MetaImage readers
      model.py                ConvNeXt-Base + fp32 classifier head; build_ensemble()
      predict.py              batching, fp16, logit-average, float64 sigmoid, tie gate
    resources/
      ensemble/                5 baked checkpoints + manifest.txt (~1.75 GB) — what ships
      fallback/                1 baked checkpoint + manifest.txt (~350 MB)
    tools/                    extract_weights, check_arch, make_test_stack,
                              check_parity, fallback_rate, check_ensemble_members
    Dockerfile                multi-stage, ARG VARIANT selects resources/<variant>/
    do_build.sh / do_test_run.sh / do_save.sh, Makefile

## Usage

```bash
cd submission
make build                   # VARIANT=ensemble (default) -> rare26-convnext-base-ensemble
VARIANT=fallback make build  # -> rare26-convnext-base-fallback
make arch                    # assert sm_75 + sm_86 are compiled in
make parity                  # 617-image stack, compare against src/train.py
make bench                   # 25,000-image stack under a real 32 GB cap
make fallback                # FOV fallback rate by centre
make save                    # export the upload tarball
```

Weights are already baked from `runs/a4_checkpointed` into `resources/ensemble/`
and `resources/fallback/`; re-run `submission/tools/extract_weights.py` per
checkpoint only if those runs are retrained.

`do_test_run.sh` runs with `--network none`, `--gpus all`, `--tmpfs /tmp` and
**`--memory 32g`**. The memory cap is not optional: the development host has
94 GB, so without it a container that would be OOM-killed during evaluation
passes locally and the memory measurement means nothing.

## Runtime knobs

All optional; defaults are what ships.

| Env var | Default | Purpose |
|---|---|---|
| `RARE26_PRECISION` | `fp16` | `fp16` / `bf16` / `fp32` |
| `RARE26_BATCH_SIZE` | `32` | forward batch |
| `RARE26_NUM_WORKERS` | `min(8, cpus)` | preprocessing workers |
| `RARE26_TIE_FRACTION` | `0.99` | distinct-output gate |
| `RARE26_MAX_IMAGES` | unset | truncate the stack (debugging only) |
| `RARE26_WEIGHTS_MANIFEST` | `manifest.txt` | local debugging only; never overridden in the shipped image |
