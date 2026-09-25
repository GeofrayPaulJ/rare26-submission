# RARE26 -- code

Training and inference code for the RARE26 challenge submission
(Barrett's-oesophagus neoplasia classification from endoscopy images:
neoplasia vs non-dysplastic). MIT licensed -- see `LICENSE`.

This is a **code-only** release: no experiment reports, no per-image
manifests, no paper. See "What this release deliberately holds back"
below for why, and what follows it.

## What's in this release

```
src/            training/eval library (config, data, losses, augment, train loop)
scripts/        numbered pipeline scripts -- data prep (0x), analysis (1x-9x),
                sweeps and job orchestration (2-3 digit prefixes)
configs/        YAML configs for every training arm/sweep this code supports
tests/          pytest suite
submission/     inference container source (Dockerfile, rare26_infer/, tools/)
                -- checkpoint weights are NOT included
```

Nothing under `manifests/` or `reports/` is included, and no image data,
checkpoint weights, or per-image scores appear anywhere in this tree,
including in code comments and docstrings.

## Hardware used

- GPU: 1x 16 GB consumer GPU (Blackwell, sm_120).
  Batch sizes throughout are VRAM-probed per config, not assumed --
  regenerate with `scripts/vram_probe.py`.
- The inference container also targets NVIDIA T4 (sm_75) and A10G
  (sm_86), the challenge's evaluation GPUs; the pinned `torch`/CUDA
  build in `submission/requirements.txt` covers all three
  architectures, verified by `submission/tools/check_arch.py` at
  build time.
- CPU/RAM: development ran in a containerised environment with 28
  vCPUs and 94 GB RAM; no code here assumes that specific host, but
  the DataLoader worker-count defaults in the sweep scripts were
  tuned against it (`scripts/98_w1_worker_count_sweep.py` and
  related).

## Install

Two separate dependency sets, because the inference container and
the training/analysis pipeline have different reproducibility
requirements.

**Inference / scoring container** -- exactly what `submission/Dockerfile`
installs, byte-pinned and verified at build time
(`submission/tools/check_arch.py`):

```bash
pip install -r submission/requirements.txt
```

**Training / analysis pipeline** -- everything else under `src/` and
`scripts/` additionally needs the packages in `requirements.txt`
(this file, at the repo root). These versions were captured from the
actual development environment (`pip freeze` on the training
container) rather than guessed; unlike the inference container's
pins, they were not independently re-verified as the exact versions
used for every historical run, since no environment lockfile was
captured at training time.

```bash
pip install -r requirements.txt
```

## Running training

```bash
python -m src.train --config configs/<name>.yaml --seed 0 --fold 0 --out-dir runs/<name>
```

`scripts/run_cv.py` orchestrates a full sweep (N repeats x 5 folds x
M seeds, or the two leave-one-centre-out splits) as a sequence of
isolated `python -m src.train` subprocesses -- one process per unit,
so a CUDA OOM or crash kills one run, not the whole night. It is
resumable: re-running the identical command re-derives what's left
to do from what's already on disk (valid canonical parquet = skip;
checkpoint but no valid parquet = resume; neither = train from
scratch).

```bash
python scripts/run_cv.py --config configs/<name>.yaml --repeats 1 --seeds 0
```

Training requires the RARE25/RARE26 image data and a manifest CSV
(`filepath, centre, class_label, fold_r0, ...`) in the shape
`src/data.py` expects; neither is included in this release -- see
"What this release deliberately holds back."

## Running the official scorer

`scripts/08_score.py` is the organisers' scoring function (median
PPV@90% Recall across 1000 prevalence-matched bootstrap draws),
copied exactly -- not approximated or re-derived. Run its
`noise_floor()` once before comparing any two runs; a score delta
smaller than the noise floor is not a real improvement.

The filename starts with a digit, so it can't be `import`ed
normally -- load it by path, the same way
`src/evaluate.py:load_official_score()` does:

```python
import importlib.util

spec = importlib.util.spec_from_file_location("_official_score", "scripts/08_score.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

result = mod.official_score(y_true, y_pred)
print(result["Score"])
```

## Building the inference container

```bash
cd submission
./do_build.sh                    # builds the default variant
VARIANT=full_k5 ./do_build.sh    # or a named variant, see Makefile
./do_test_run.sh                 # smoke-tests the built image end to end
```

The build expects checkpoint weights under
`submission/resources/<variant>/` (not included in this release) and
produces a self-contained image; no network access or external
weight downloads happen inside the container at inference time.

## What this release deliberately holds back

This is code only. The full development record -- 210 experiment
reports, a reconstructed chronological methods log, and the
per-image manifests (file hashes, fold assignments, QA flags; no
pixel data) that the reports and paper cite -- is held back pending
organiser guidance on EndoVis Rules 1 and 8. It will be released as
a second record once that guidance is confirmed.

## Data

No image data is distributed with this repository. The RARE25
training images and the EVC external validation set are obtained
separately, per the challenge organisers' data access process.

GastroNet-5M weights are not redistributed either; see `NOTICE.md`.
