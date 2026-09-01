"""Grand Challenge entrypoint for RARE26.

Mirrors the contract of the organisers' template (TUE-ARIA/RARE25-Submission):
resolve the interface from /input/inputs.json, run the model over the stacked
input image, and write a bare JSON list of per-slice likelihoods to
/output/stacked-neoplastic-lesion-likelihoods.json.

TWO DEVIATIONS FROM THE TEMPLATE, BOTH DELIBERATE.

1. The template loads the whole stack with SimpleITK.GetArrayFromImage. At the
   scale this challenge ships that is 24.5 GB resident against a 32 GB cap, so
   reading is slice-wise instead (see rare26_infer/stack.py).

2. The socket slug and the directory name do not match, and the mismatch is
   easy to get wrong: the slug in inputs.json is
   "stacked-barretts-esophagus-endoscopy-images" while the directory the
   platform mounts is "images/stacked-barretts-esophagus-endoscopy" -- no
   trailing "-images". The template hardcodes the directory form. This resolves
   the documented path first and then falls back to scanning /input/images, so
   a platform-side rename costs a log line rather than the whole submission.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from glob import glob
from pathlib import Path

INPUT_PATH = Path(os.environ.get("RARE26_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("RARE26_OUTPUT", "/output"))
RESOURCE_PATH = Path(os.environ.get("RARE26_RESOURCES", "resources"))

# The manifest, not a single filename, is what selects the model: it lists
# one baked checkpoint filename per line, relative to RESOURCE_PATH. Which
# variant (5-fold ensemble vs. 1-checkpoint fallback) is baked into the image
# is decided at build time by which resources/<variant>/ the Dockerfile
# copied -- this entrypoint just reads however many lines are there and
# builds that many models. RARE26_WEIGHTS_MANIFEST exists for local
# debugging only; it is never overridden in the shipped image.
WEIGHTS_MANIFEST = os.environ.get("RARE26_WEIGHTS_MANIFEST", "manifest.txt")
OUTPUT_NAME = "stacked-neoplastic-lesion-likelihoods.json"

# The image socket, and the directory the platform actually mounts it under.
INPUT_SLUG = "stacked-barretts-esophagus-endoscopy-images"
INPUT_DIRNAME = "stacked-barretts-esophagus-endoscopy"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("rare26.inference")


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def get_interface_key():
    """Socket slugs present in this job, as the template computes them."""
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def resolve_image_dir() -> Path:
    """Locate the mounted image directory, tolerating a slug/dirname mismatch."""
    images_root = INPUT_PATH / "images"
    for candidate in (images_root / INPUT_DIRNAME, images_root / INPUT_SLUG):
        if candidate.is_dir():
            logger.info("input directory: %s", candidate)
            return candidate

    # Neither documented name exists -- take any subdirectory holding an image.
    if images_root.is_dir():
        for sub in sorted(p for p in images_root.iterdir() if p.is_dir()):
            hits = (glob(str(sub / "*.tif")) + glob(str(sub / "*.tiff"))
                    + glob(str(sub / "*.mha")) + glob(str(sub / "*.mhd")))
            if hits:
                logger.warning(
                    "expected %r or %r under %s but found neither; falling back "
                    "to %r, which contains %d image file(s)",
                    INPUT_DIRNAME, INPUT_SLUG, images_root, sub.name, len(hits),
                )
                return sub

    listing = sorted(os.listdir(images_root)) if images_root.is_dir() else "<missing>"
    raise FileNotFoundError(
        f"no input image directory under {images_root}; contents: {listing}"
    )


def show_torch_cuda_info() -> None:
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current)}")
        # The compiled arch list is what decides whether this image runs at all
        # on the evaluation GPU: T4 is sm_75 and A10G is sm_86.
        print(f"\tbuilt for: {torch.cuda.get_arch_list()}")
    print("=+=" * 10)


def interface_0_handler() -> int:
    from rare26_infer.predict import TIE_FRACTION, predict_stack, write_stats

    show_torch_cuda_info()

    image_dir = resolve_image_dir()

    manifest_path = RESOURCE_PATH / WEIGHTS_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"weights manifest missing at {manifest_path}; the image is built wrong"
        )
    names = [ln.strip() for ln in manifest_path.read_text().splitlines() if ln.strip()]
    if not names:
        raise ValueError(f"{manifest_path} lists no checkpoints")
    weights = [RESOURCE_PATH / n for n in names]
    missing = [str(w) for w in weights if not w.is_file()]
    if missing:
        raise FileNotFoundError(
            f"baked weights missing: {missing}; the image is built wrong"
        )
    logger.info("ensemble: %d member(s) from %s: %s",
                len(weights), manifest_path, [w.name for w in weights])

    # Debug-only member truncation, for the T(n) member-scaling measurement
    # (step 3, 2026-08-07): run the SAME image at n=1,2,3,5 members without
    # rebuilding. Never set in the shipped image -- the shipped member count
    # is whatever the baked manifest lists.
    max_members = os.environ.get("RARE26_MAX_MEMBERS")
    if max_members:
        k = int(max_members)
        if 0 < k < len(weights):
            weights = weights[:k]
            logger.warning("RARE26_MAX_MEMBERS=%d -- truncating to %s "
                           "(debug/measurement only)", k, [w.name for w in weights])

    # Debug-only: dump the pre-average (n_images, n_members) logit matrix so
    # tools/check_ensemble_members.py can verify each member's own held-out
    # agreement with the harness. Never set in the shipped image.
    dump_path = os.environ.get("RARE26_DUMP_MEMBER_LOGITS")

    probabilities, stats = predict_stack(
        str(image_dir), [str(w) for w in weights],
        dump_member_logits_path=dump_path,
    )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_PATH / OUTPUT_NAME
    write_json_file(location=output_path, content=probabilities)

    # THE TIE GATE, for real -- 2026-08-10 (D8, reports/d8_serialization.md).
    # Reads the ACTUAL WRITTEN FILE back off disk and re-derives the distinct
    # count from THAT, not from the in-memory `probabilities` list that
    # rare26_infer.predict already pre-checked. Ties at the operating
    # threshold inflate FPR directly; refusing to ship a degraded submission
    # that looks valid is worth a second, file-level check even though D8
    # found the write/read round trip lossless on every real image tested
    # (max abs diff 2.8e-17) -- "found lossless on what we could test" is not
    # the same claim as "provably lossless for every input", and this is the
    # cheapest possible way to close that gap: read back what actually ships.
    on_disk = load_json_file(location=output_path)
    n_unique_on_disk = len(set(on_disk))
    n_distinct_inputs = stats["n_distinct_inputs"]
    required = stats["tie_gate_required"]
    if len(on_disk) != len(probabilities):
        raise AssertionError(
            f"written file has {len(on_disk)} entries, in-memory list had "
            f"{len(probabilities)} -- the write itself is corrupt, not a "
            f"precision question."
        )
    if not n_unique_on_disk > required:
        raise AssertionError(
            f"ON-DISK distinct probability count {n_unique_on_disk} (file: "
            f"{output_path}) does not exceed {TIE_FRACTION:.0%} of "
            f"{n_distinct_inputs} distinct inputs ({required:.1f}). In-memory "
            f"pre-check said {'PASS' if stats['tie_gate_would_pass_in_memory'] else 'FAIL'} "
            f"({stats['n_unique_probs']} distinct) -- "
            f"{'the write/read path lost precision' if stats['tie_gate_would_pass_in_memory'] else 'consistent with the pre-check, not a serialisation-specific finding'}. "
            f"Ties at the operating threshold inflate FPR directly; refusing "
            f"to leave a degraded submission in place. If this is genuine "
            f"near-duplicate content rather than a precision fault, "
            f"RARE26_TIE_FRACTION lowers the bar."
        )
    logger.info("tie gate PASSED (on-disk): %d distinct probabilities > %.1f "
               "required (%d distinct inputs), read back from %s",
               n_unique_on_disk, required, n_distinct_inputs, output_path)

    write_stats(stats, str(OUTPUT_PATH / "rare26_run_stats.json"))

    logger.info(
        "wrote %d likelihoods to %s in %.1fs (%.1f img/s, %.1f%% FOV fallback)",
        len(probabilities), OUTPUT_PATH / OUTPUT_NAME,
        stats["total_seconds"], stats["images_per_second"],
        100.0 * stats["fallback_frac"],
    )

    # ==CASE REPORT== : one greppable stdout block per case. Only one
    # submission exists in seven days -- it must return information, not just
    # a score, and stdout is the only channel the platform reliably returns.
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    print("==CASE REPORT==", flush=True)
    print(f"cases_received={len(inputs)}", flush=True)
    print(f"case_0_slices={stats['n_images']}", flush=True)
    print(f"case_0_wall_seconds={stats['total_seconds']:.1f}", flush=True)
    print(f"case_0_inference_seconds={stats['inference_seconds']:.1f}", flush=True)
    print(f"case_0_gpu_seconds={stats['gpu_seconds']:.1f}", flush=True)
    print(f"case_0_data_seconds={stats['data_seconds']:.1f}", flush=True)
    print(f"case_0_model_load_seconds={stats['model_load_seconds']:.1f}", flush=True)
    print(f"case_0_images_per_second={stats['images_per_second']:.1f}", flush=True)
    print(f"case_0_n_members={stats['n_members']}", flush=True)
    print(f"case_0_peak_anon_gb={stats.get('cgroup_anon_gb', float('nan')):.2f}", flush=True)
    print(f"case_0_peak_cgroup_gb={stats.get('cgroup_peak_gb', float('nan')):.2f}", flush=True)
    print(f"case_0_vram_peak_alloc_gb={stats.get('vram_peak_alloc_gb', float('nan')):.2f}", flush=True)
    print(f"case_0_fov_fallback_frac={stats['fallback_frac']:.4f}", flush=True)
    print(f"case_0_unique_prob_frac={stats['unique_frac']:.4f}", flush=True)
    print("==END CASE REPORT==", flush=True)

    # ==BUDGET REPORT== : G4/G5 governor outcome, one greppable block per case.
    # A platform-side timeout kill leaves no file at all (reports/g3_timeout_semantics.md);
    # this always leaves a valid one, degraded or not, and this block is the
    # record of which it was.
    print("==BUDGET REPORT==", flush=True)
    print(f"budget_seconds={stats['budget_seconds']:.0f}", flush=True)
    print(f"hard_stop_seconds={stats['hard_stop_seconds']:.0f}", flush=True)
    print(f"budget_elapsed_seconds={stats['budget_elapsed_seconds']:.1f}", flush=True)
    print(f"budget_projected_short_seconds={stats['budget_projected_short_seconds']}", flush=True)
    print(f"budget_projected_long_seconds={stats['budget_projected_long_seconds']}", flush=True)
    print(f"budget_degradations={stats['budget_degradations'] or 'none'}", flush=True)
    print(f"images_scored={stats['images_scored']}", flush=True)
    print(f"images_filled_neutral={stats['images_filled_neutral']}", flush=True)
    print(f"fill_floor_logit={stats['fill_floor_logit']}", flush=True)
    print(f"fill_increment_logit={stats['fill_increment_logit']}", flush=True)
    print("==END BUDGET REPORT==", flush=True)
    return 0


def run() -> int:
    interface_key = get_interface_key()
    handler = {
        (INPUT_SLUG,): interface_0_handler,
    }[interface_key]
    return handler()


if __name__ == "__main__":
    raise SystemExit(run())
