"""JOB 5b/5c -- throughput bench and member-budget ceiling table.

REQUIRES AN IDLE GPU. Refuses to run (exits non-zero) if nvidia-smi reports
any memory in use, because a throughput number measured under training load
is worthless -- this is the brief's own words, not a style choice.

WHAT THIS MEASURES, per backbone, fp16 path:
  - model construction + weight-load time (cold, once)
  - per-image forward-pass throughput at the resolution the brief specifies:
    512px for ConvNeXt-Base and RN50, 378px for ViT-B/14. NOTE: production
    inference (submission/rare26_infer) resizes to 384 (ConvNeXt/RN50) / 378
    (ViT) before the model ever sees the tensor -- 512 is NOT the model's
    production input size, it is the raw per-slice frame size from the
    submission README's own benchmark stack (25,000 x 512 x 637 x 3). This
    script benchmarks at the literal resolution the brief specified rather
    than silently substituting 384, and flags the discrepancy here so it is
    not lost.

RUNTIME LIMIT. Searched the entire repo (submission/, reports/, manifests/,
scripts/, root -- README, do_test_run.sh, Makefile, every *.md) for any
recorded organiser time/runtime limit. FOUND NONE. do_test_run.sh documents a
32 GB RAM cap and no-network constraint, never a wall-clock limit. This
script does NOT assume five minutes or any other figure -- it leaves the
"headroom" column unresolved and says so loudly, rather than guessing.

T4 DERATE FACTOR. No T4 is reachable from this environment, so this is a
published-spec estimate, not a measurement, and is reported as such:
NVIDIA T4's tensor-core FP16 throughput is a published 65 TFLOPS. This
project's dev card (RTX 5060 Ti, Blackwell) is a materially newer consumer
part with substantially higher published FP16 tensor throughput. Lacking a
verified apples-to-apples FP16-tensor TFLOPS figure for the 5060 Ti from a
source this script can cite with confidence, the derate factor used is a
conservative round number (3.0x SLOWER on T4) reflecting the well-established
generation/class gap between a 2018 datacenter inference card and a current
consumer card, explicitly flagged as an ESTIMATE requiring empirical
validation on real T4 hardware before being trusted for a submission-critical
decision.

    python scripts/36_throughput_bench.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

T4_DERATE_FACTOR = 3.0
T4_DERATE_BASIS = (
    "ESTIMATE, not measured: no T4 reachable from this environment. NVIDIA T4 "
    "publishes 65 TFLOPS FP16 tensor throughput; the dev card (RTX 5060 Ti) is "
    "a materially newer, higher-throughput consumer part. 3.0x is a "
    "conservative round-number reflecting that generation/class gap, not a "
    "cited TFLOPS ratio -- validate on real T4 hardware before relying on it "
    "for a submission-critical decision."
)
STACK_SIZE = 25000  # images, matching submission/README.md's own bench stack
# The real organiser limit is STILL not recorded anywhere in the repo (see
# module docstring for what was searched). 2026-08-05: rather than leave the
# whole table unresolved pending that number, report against three assumed
# limits so member-selection decisions have something to plan against now.
ASSUMED_LIMITS_MIN = (15.0, 30.0, 60.0)
HEADROOM_TARGET = 0.30

BACKBONES = [
    # Resolutions are each backbone's ACTUAL training resolution -- confirmed
    # from configs/sweep_a4.yaml (image_size: 384), configs/g1_rn50_swsl.yaml
    # / g3_rn50_gastronet.yaml (384, RN50 is fully convolutional and stays at
    # A4's resolution per src/backbones.py's own docstring), and
    # configs/g2_vitb_dinov2.yaml (378, confirmed separately). 2026-08-05:
    # the first bench run used 512 for ConvNeXt-Base/RN50 -- an earlier
    # instruction to benchmark at the raw per-slice submission-stack frame
    # size, since superseded; only G2's 378 was ever right. Fixed to
    # per-backbone training resolution for all three.
    {"key": "convnext_base", "label": "ConvNeXt-Base", "arch": "convnext_base.fb_in1k",
     "resolution": 384, "pretrained": False},
    {"key": "resnet50", "label": "RN50", "arch": "resnet50",
     "resolution": 384, "pretrained": False},
    {"key": "vitb14", "label": "ViT-B/14 (DINOv2, registers)",
     "arch": "vit_base_patch14_reg4_dinov2", "resolution": 378, "pretrained": False},
]


def assert_gpu_idle() -> None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    used_mib = [int(x.strip()) for x in out.stdout.strip().splitlines() if x.strip()]
    if any(m > 0 for m in used_mib):
        print(f"REFUSING TO RUN: GPU memory in use ({used_mib} MiB) -- a "
              f"throughput number measured under load is worthless. Wait for "
              f"the card to be idle and re-run.", file=sys.stderr)
        sys.exit(1)
    print(f"GPU idle confirmed: {used_mib} MiB used. Proceeding.")


def bench_backbone(spec: Dict[str, Any], n_warmup: int = 5, n_timed: int = 30,
                   batch_size: int = 16) -> Dict[str, Any]:
    import torch
    import timm

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # img_size matters here: vit_base_patch14_reg4_dinov2's default pretrained
    # config is 518px, and timm does NOT infer the benchmark's intended
    # resolution from the input tensor shape alone -- it asserts patch_embed's
    # configured img_size against the actual input and raises if they
    # disagree. Same try/except-TypeError pattern src/model.py's build_model()
    # already uses: convnets reject the img_size kwarg outright (fully
    # convolutional, no fixed input size), ViT-family archs require it for a
    # non-default resolution.
    try:
        model = timm.create_model(spec["arch"], pretrained=spec["pretrained"],
                                  img_size=spec["resolution"], num_classes=1)
    except TypeError:
        model = timm.create_model(spec["arch"], pretrained=spec["pretrained"],
                                  num_classes=1)
    model.eval().cuda()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    res = spec["resolution"]
    x = torch.randn(batch_size, 3, res, res, device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(n_warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_timed):
            model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    per_batch_s = elapsed / n_timed
    per_image_s = per_batch_s / batch_size
    images_per_s = 1.0 / per_image_s

    del model
    torch.cuda.empty_cache()

    return {"key": spec["key"], "label": spec["label"], "arch": spec["arch"],
           "resolution": res, "batch_size": batch_size,
           "load_seconds": load_s, "per_image_seconds_dev_gpu": per_image_s,
           "images_per_second_dev_gpu": images_per_s,
           "per_image_seconds_t4_estimate": per_image_s * T4_DERATE_FACTOR,
           "images_per_second_t4_estimate": images_per_s / T4_DERATE_FACTOR,
           "load_seconds_t4_estimate": load_s * T4_DERATE_FACTOR}


def ensemble_table(bench: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    configs = [
        (1, 1, 1, 1, "single backbone, single fold, single seed, no TTA"),
        (1, 5, 1, 1, "single backbone, k=5 folds, single seed"),
        (1, 5, 5, 1, "single backbone, k=5 folds x 5 seeds"),
        (1, 5, 5, 2, "single backbone, k=5x5, 2-way TTA (orig+hflip)"),
        (2, 5, 5, 1, "2 backbones, k=5x5 each"),
        (2, 5, 5, 2, "2 backbones, k=5x5 each, 2-way TTA"),
        (3, 5, 5, 1, "3 backbones (all of A4/G1/G3-class), k=5x5 each"),
        (3, 5, 5, 2, "3 backbones, k=5x5 each, 2-way TTA"),
    ]
    for n_backbones, n_folds, n_seeds, n_tta, desc in configs:
        # Cost dominated by whichever backbone(s) are actually in the config;
        # since this table doesn't pin WHICH backbones, use the mean per-image
        # T4-estimate cost across the 3 benchmarked backbones as a stand-in
        # for "a representative backbone", scaled by n_backbones. This is
        # explicit and stated, not hidden in the numbers.
        mean_per_image_t4 = sum(
            b["per_image_seconds_t4_estimate"] for b in bench.values()) / len(bench)
        mean_load_t4 = sum(
            b["load_seconds_t4_estimate"] for b in bench.values()) / len(bench)

        n_checkpoints = n_backbones * n_folds * n_seeds
        total_load_s = n_checkpoints * mean_load_t4
        total_inference_s = (STACK_SIZE * n_backbones * n_folds * n_seeds * n_tta
                             * mean_per_image_t4)
        total_s = total_load_s + total_inference_s
        total_min = total_s / 60.0

        per_limit = {}
        for limit in ASSUMED_LIMITS_MIN:
            headroom = (limit - total_min) / limit
            per_limit[limit] = {"headroom_frac": headroom,
                                "fits_30pct_headroom": headroom >= HEADROOM_TARGET}

        rows.append({
            "config": f"{n_backbones}x{n_folds}x{n_seeds}x{n_tta}",
            "description": desc,
            "n_checkpoints": n_checkpoints,
            "n_tta": n_tta,
            "est_load_minutes": total_load_s / 60.0,
            "est_inference_minutes": total_inference_s / 60.0,
            "est_total_minutes_t4": total_min,
            "per_limit": per_limit,
        })
    return rows


def main() -> int:
    assert_gpu_idle()

    bench = {}
    for spec in BACKBONES:
        print(f"benchmarking {spec['label']} @ {spec['resolution']}px, fp16...")
        r = bench_backbone(spec)
        bench[spec["key"]] = r
        print(f"  load={r['load_seconds']:.2f}s  "
             f"dev_gpu={r['images_per_second_dev_gpu']:.1f} img/s  "
             f"T4-estimate={r['images_per_second_t4_estimate']:.1f} img/s")

    table = ensemble_table(bench)

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stack_size": STACK_SIZE,
        "assumed_limits_minutes": list(ASSUMED_LIMITS_MIN),
        "runtime_limit_status": "NOT FOUND anywhere in the repo -- see this "
                                "script's docstring for what was searched; "
                                "table below reports against three assumed "
                                "limits instead of leaving it unresolved",
        "t4_derate_factor": T4_DERATE_FACTOR,
        "t4_derate_basis": T4_DERATE_BASIS,
        "per_backbone": bench,
        "ensemble_configs": table,
    }

    report_path = os.path.join(REPO_ROOT, "reports", "throughput_bench.json")
    with open(report_path, "w") as fh:
        json.dump(out, fh, indent=2)

    md = []
    A = md.append
    A("# JOB 5b/5c -- throughput bench and member-budget ceiling")
    A("")
    A(f"Generated by `scripts/36_throughput_bench.py`. GPU confirmed idle "
     f"before benchmarking. Stack size: {STACK_SIZE} images (matching "
     f"`submission/README.md`'s own `make bench` stack).")
    A("")
    A("**RUNTIME LIMIT: still NOT FOUND anywhere in this repository.** "
     "Searched `submission/`, `reports/`, `manifests/`, `scripts/`, and repo "
     "root (README, do_test_run.sh, Makefile, every *.md) for any recorded "
     "organiser time/runtime limit -- nothing. Rather than leave the whole "
     "table unresolved pending that number, it is reported against three "
     f"assumed limits ({', '.join(f'{m:.0f} min' for m in ASSUMED_LIMITS_MIN)}) "
     "so member-selection has something to plan against now. Replace with "
     "the real number the moment it's known.")
    A("")
    A(f"**T4 derate factor: {T4_DERATE_FACTOR}x slower than the dev GPU "
     f"(RTX 5060 Ti) -- ESTIMATE, not measured.** {T4_DERATE_BASIS}")
    A("")
    A("## Per-backbone benchmark (fp16, dev GPU measured, T4 estimated)")
    A("")
    A("| backbone | resolution | load (s) | dev img/s | T4-est img/s | "
     "T4-est ms/image |")
    A("|---|---|---|---|---|---|")
    for b in bench.values():
        A(f"| {b['label']} | {b['resolution']}px | {b['load_seconds']:.2f} | "
         f"{b['images_per_second_dev_gpu']:.1f} | "
         f"{b['images_per_second_t4_estimate']:.1f} | "
         f"{b['per_image_seconds_t4_estimate']*1000:.2f} |")
    A("")
    A("Resolutions are each backbone's ACTUAL training resolution: 384px "
     "(ConvNeXt-Base, RN50 -- confirmed from `configs/sweep_a4.yaml` / "
     "`configs/g1_rn50_swsl.yaml` / `configs/g3_rn50_gastronet.yaml`, all "
     "`image_size: 384`) and 378px (ViT-B/14, confirmed from "
     "`configs/g2_vitb_dinov2.yaml`). An earlier pass benchmarked "
     "ConvNeXt-Base/RN50 at 512px against a since-superseded instruction; "
     "fixed here.")
    A("")
    A("## Ensemble configuration table (T4 estimate)")
    A("")
    A("`config` = n_backbones x n_folds x n_seeds x n_TTA. Cost uses the MEAN "
     "T4-estimated per-image/load cost across the 3 benchmarked backbones as "
     "a stand-in for \"a representative backbone\" -- this table does not pin "
     "which specific backbones fill each slot.")
    A("")
    limit_headers = " | ".join(f"fits {m:.0f}min (30% headroom)?" for m in ASSUMED_LIMITS_MIN)
    A("| config | description | n checkpoints | est. load (min) | "
     f"est. inference (min) | est. TOTAL (min, T4) | {limit_headers} |")
    A("|---|---|---|---|---|---|" + "---|" * len(ASSUMED_LIMITS_MIN))
    for r in table:
        cells = []
        for m in ASSUMED_LIMITS_MIN:
            pl = r["per_limit"][m]
            cells.append(f"{pl['headroom_frac']*100:+.0f}% "
                        f"({'fits' if pl['fits_30pct_headroom'] else 'NO'})")
        A(f"| {r['config']} | {r['description']} | {r['n_checkpoints']} | "
         f"{r['est_load_minutes']:.1f} | {r['est_inference_minutes']:.1f} | "
         f"**{r['est_total_minutes_t4']:.1f}** | " + " | ".join(cells) + " |")
    A("")
    A("**Largest configuration fitting 30% headroom, per assumed limit:**")
    A("")
    for m in ASSUMED_LIMITS_MIN:
        fitting = [r for r in table if r["per_limit"][m]["fits_30pct_headroom"]]
        if fitting:
            best = max(fitting, key=lambda r: r["est_total_minutes_t4"])
            A(f"- **{m:.0f} min limit**: `{best['config']}` "
             f"({best['description']}) -- {best['est_total_minutes_t4']:.1f} "
             f"min, {best['per_limit'][m]['headroom_frac']*100:.0f}% headroom.")
        else:
            A(f"- **{m:.0f} min limit**: NOTHING in this table fits with 30% "
             f"headroom -- even the smallest configuration "
             f"({table[0]['config']}, {table[0]['est_total_minutes_t4']:.1f} "
             f"min) does not clear it.")
    A("")
    A("This is still an ESTIMATE against an UNCONFIRMED limit -- get the real "
     "number before finalizing member selection.")
    A("")

    md_path = os.path.join(REPO_ROOT, "reports", "throughput_bench.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(f"\nwritten: {report_path}")
    print(f"written: {md_path}")
    print(f"\nRUNTIME LIMIT still not found in repo -- reported against "
         f"assumed limits {ASSUMED_LIMITS_MIN} min instead. See "
         f"reports/throughput_bench.md for the full table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
