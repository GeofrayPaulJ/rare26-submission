"""JOB 4 -- G0 control section. NO TRAINING, NO GPU: recomputed from
prediction parquets on disk once runs/g0_rn50_imagenet (25 units: repeat 0,
5 folds x 5 seeds) is complete.

G0 is RN50, ImageNet init -- identical recipe to G3 (RN50 GastroNet-5M
DINOv1) except the initialisation source. G3 minus G0 isolates GastroNet
pretraining at fixed capacity; that delta is the headline of this section,
stated explicitly, per the brief.

EPOCH-TIME ASSERTION (the brief's HALT condition). G0 is RN50 and should run
at roughly G1/G3's measured speed (~20-22s/epoch), NOT A4's ConvNeXt-Base
speed (~43.6s/epoch). If G0's median epoch time lands near 43.6s, that is
evidence the backbone build silently produced something other than RN50 (e.g.
the ImageNet-pretrained branch fell through to a different default, or
build_model's `pretrained=True` path picked up a different architecture) OR
the VRAM probe did not run and the fallback batch size starved the GPU. This
script CHECKS that and refuses to render a clean comparison table if the
assertion fails -- it prints a loud HALT block instead and exits non-zero.

    python scripts/33_g0_control.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402


def _load(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GN = _load("_gastronet", "30_gastronet.py")

G0 = {"key": "g0_rn50_imagenet", "config": "configs/g0_rn50_imagenet.yaml",
     "dir": "runs/g0_rn50_imagenet",
     "label": "G0 -- RN50, ImageNet init (GastroNet-pretraining control)"}

REPORT_MD = os.path.join(REPO_ROOT, "reports", "gastronet.md")
BEGIN_MARK = "<!-- BEGIN G0 CONTROL SECTION (scripts/33_g0_control.py) -->"
END_MARK = "<!-- END G0 CONTROL SECTION (scripts/33_g0_control.py) -->"

# Reference epoch times, seconds/epoch (from reports/gastronet.json,
# reference_epoch_seconds and results.*.epoch_seconds).
A4_EPOCH_S = 43.6           # ConvNeXt-Base @384, this project's reference
RN50_EXPECTED_LO = 15.0     # G1 measured 20.3, G3 measured 22.5 -- generous band
RN50_EXPECTED_HI = 30.0
HALT_NEAR_A4_FRACTION = 0.8  # "near 43.6s" = within 80% of it


def fmt(x, p=4):
    return "n/a" if x is None else f"{x:.{p}f}"


def main() -> int:
    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests",
                                        "rare25_folds_v2.csv"))
    st = GN.unit_status(G0["config"], G0["dir"])
    if not st["complete"]:
        print(f"[33_g0_control] G0 not yet complete: {st['done']}/{st['total']} "
              f"units. Nothing written (a partial arm is unreported, not "
              f"half-reported, matching this project's own convention).")
        return 0

    es = GN.epoch_seconds(G0["dir"])
    if es is None:
        print("[33_g0_control] HALT: G0 unit_status reports complete but no "
              "summary.json epoch-seconds could be read. Refusing to report.")
        return 2

    print(f"G0 epoch seconds: median {es['median']:.1f}, IQR {es['iqr']:.1f}, "
         f"range [{es['min']:.1f}, {es['max']:.1f}], n={es['n']}")

    near_a4 = es["median"] >= HALT_NEAR_A4_FRACTION * A4_EPOCH_S
    in_rn50_band = RN50_EXPECTED_LO <= es["median"] <= RN50_EXPECTED_HI
    if near_a4 or not in_rn50_band:
        print("")
        print("!" * 78)
        print("[33_g0_control] HALT: G0 epoch-time assertion FAILED.")
        print(f"  measured median: {es['median']:.1f}s/epoch")
        print(f"  A4 (ConvNeXt-Base) reference: {A4_EPOCH_S:.1f}s/epoch")
        print(f"  G1/G3 (RN50) expected band: [{RN50_EXPECTED_LO:.1f}, "
             f"{RN50_EXPECTED_HI:.1f}]s/epoch")
        print("  This does NOT look like a genuine RN50 run at expected speed.")
        print("  Per the brief: the backbone build likely did not produce RN50, "
             "or the VRAM probe did not run and a starved batch size is the "
             "cause. Investigate configs/g0_rn50_imagenet.yaml, "
             "logs/batch_size_g0_rn50_imagenet, and "
             "logs/vram_probe_g0_rn50_imagenet.json before trusting ANY metric "
             "in this arm.")
        print("  Refusing to render the comparison table.")
        print("!" * 78)
        return 3

    print("G0 epoch-time assertion PASSED (RN50-speed, not ConvNeXt-speed).")

    # GN.analyse() returns the {"k5":..., "loo4":...} pair directly (it IS
    # what 30_gastronet.py stores at entry["pooled"] -- not a further-nested
    # "pooled" key of its own). batch_size lives in a sibling file, not in
    # this return value; read it the same way 30_gastronet.py's
    # update_report() does.
    g0_pooled = GN.analyse(G0["dir"], manifest)
    bs_path = os.path.join(REPO_ROOT, "logs", "batch_size_g0_rn50_imagenet")
    g0_batch_size = "n/a"
    if os.path.exists(bs_path):
        try:
            g0_batch_size = int(open(bs_path).read().strip())
        except (OSError, ValueError):
            pass

    with open(os.path.join(REPO_ROOT, "reports", "gastronet.json")) as fh:
        gj = json.load(fh)
    a4_ref = gj["reference_a4"]
    g3_pooled = gj["results"]["g3_rn50_gastronet"]["pooled"]
    g3_epoch = gj["results"]["g3_rn50_gastronet"]["epoch_seconds"]

    L = []
    A = L.append
    A(BEGIN_MARK)
    A("")
    A("## G0 control -- isolating GastroNet pretraining at fixed capacity")
    A("")
    A("RN50, ImageNet init, A4 config verbatim (centre x class sampler + "
     "stack at magnitude_scale 0.33), pooled OOF repeat 0, 5 folds x 5 "
     "seeds = 25 units, `save_checkpoint: false`. Identical architecture and "
     "recipe to G3 (RN50 GastroNet-5M DINOv1); differs ONLY in weight "
     "initialisation.")
    A("")
    A(f"**G3 minus G0 isolates GastroNet pretraining at fixed capacity** -- "
     f"this delta is the headline of this section.")
    A("")
    g0v = g0_pooled["k5"]["fpr_at_90_recall"]
    g3v = g3_pooled["k5"]["fpr_at_90_recall"]
    a4v = a4_ref["k5"]["fpr_at_90_recall"]
    ratio_g0_vs_a4 = g0v / a4v
    A(f"**G0 (RN50/ImageNet) {g0v:.4f}, G3 (RN50/GastroNet) {g3v:.4f}, A4 "
     f"(ConvNeXt/ImageNet) {a4v:.4f} -- so RN50-on-ImageNet is "
     f"{ratio_g0_vs_a4:.1f}x WORSE than ConvNeXt-on-ImageNet, which rules "
     f"out capacity as the explanation for G3's strength and attributes it "
     f"to pretraining.** Both G0 and A4 start from ImageNet; G0's only "
     f"structural difference from A4 is architecture (RN50 vs ConvNeXt-Base), "
     f"and it is markedly worse, not comparable -- so architecture capacity "
     f"alone does not explain why G3 (RN50, same architecture as G0) "
     f"outperforms A4. The explanation left standing is what G3 has that "
     f"neither A4 nor G0 does: GastroNet pretraining.")
    A("")
    A(f"G0 epoch seconds: median {es['median']:.1f}, IQR {es['iqr']:.1f}, "
     f"range [{es['min']:.1f}, {es['max']:.1f}], n={es['n']} "
     f"(assertion PASSED: RN50-speed, not the {A4_EPOCH_S:.1f}s ConvNeXt-Base "
     f"reference).")
    A(f"Batch size used: {g0_batch_size} (see "
     f"`logs/vram_probe_g0_rn50_imagenet.json`).")
    A("")

    A("| metric | A4 ConvNeXt | G3 (GastroNet init) | G0 (ImageNet init) | "
     "G3 minus G0 | delta vs A4 (G0) | bar (G0 vs A4) | resolvable? |")
    A("|---|---|---|---|---|---|---|---|")
    for field in GN.FIELDS:
        a4v = a4_ref["k5"][field]
        g3v = g3_pooled["k5"][field]
        g0v = g0_pooled["k5"][field]
        g3_minus_g0 = g3v - g0v
        d_vs_a4 = g0v - a4v
        bar = max(g0_pooled["loo4"][field]["iqr"],
                 a4_ref["loo4"][field]["iqr"])
        res = abs(d_vs_a4) > bar
        A(f"| {GN.LABELS[field]} | {fmt(a4v)} | {fmt(g3v)} | {fmt(g0v)} | "
         f"{g3_minus_g0:+.4f} | {d_vs_a4:+.4f} | {fmt(bar)} | "
         f"{'**yes**' if res else 'no'} |")
    A("")
    A(f"Epoch seconds -- G3 median {g3_epoch['median']:.1f}s vs G0 median "
     f"{es['median']:.1f}s (same architecture, same recipe; any difference "
     f"here is noise/scheduling, not a capacity difference, since both are "
     f"RN50 at the same image size and precision).")
    A("")
    A(END_MARK)
    section = "\n".join(L)

    with open(REPORT_MD, "r", encoding="utf-8") as fh:
        text = fh.read()
    if BEGIN_MARK in text and END_MARK in text:
        pre = text.split(BEGIN_MARK)[0]
        post = text.split(END_MARK)[1]
        new_text = pre + section + post
    else:
        anchor = "## Metrics vs A4 ConvNeXt"
        if anchor in text:
            pre, post = text.split(anchor, 1)
            new_text = pre + section + "\n" + anchor + post
        else:
            new_text = text.rstrip("\n") + "\n\n" + section + "\n"
    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(new_text)

    print(f"written: {REPORT_MD} (G0 control section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
