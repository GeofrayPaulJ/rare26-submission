"""Verify GastroNet backbone checkpoint identity BEFORE any GPU time is spent.

The brief is explicit: confirm each checkpoint is the architecture it claims to
be, report param counts and key shapes, and HALT rather than guess. This script
does exactly that and nothing else -- no training, no GPU, no model
construction that could mask a mismatch behind a lenient loader.

WHY IDENTITY IS CHECKED FROM THE STATE DICT, NOT FROM A CONSTRUCTED MODEL.
Building a model and calling load_state_dict(strict=False) would happily accept
a ResNet checkpoint into a ViT and report "loaded", with every tensor silently
dropped. The only trustworthy evidence of what a .pth file IS lives in its own
key names and tensor shapes, so that is what is inspected here.

    python scripts/31_verify_backbones.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")

# Filenames as they exist ON DISK. The SWSL checkpoint's name carries a
# URL-encoded '+' (%2B) from however it was downloaded; referring to it by the
# decoded name silently misses the file, so the literal on-disk name is used.
TARGETS = [
    {"key": "rn50_swsl_gastronet",
     "file": "RN50_Billion-Scale-SWSL%2BGastroNet-5M_DINOv1.pth",
     "expect": "resnet50", "label": "RN50 Billion-Scale-SWSL + GastroNet-5M (DINOv1)"},
    {"key": "dinov2_vitb",
     "file": "dinov2.pth",
     "expect": "vit_base", "label": "DINOv2 ViT-B"},
    {"key": "rn50_gastronet",
     "file": "RN50_GastroNet-5M_DINOv1.pth",
     "expect": "resnet50", "label": "RN50 GastroNet-5M (DINOv1), SWSL control"},
]

# What each architecture must look like. ViT-B/14 per the DINOv2 release:
# 768-wide, 12 blocks, 14x14 patches, ~86.5M params.
EXPECT = {
    "resnet50": {
        "params_m": (23.0, 26.5),
        "markers": ["conv1.weight", "layer1.0.conv1.weight", "layer4.2.conv3.weight"],
        "conv1_shape": [64, 3, 7, 7],
    },
    "vit_base": {
        "params_m": (85.0, 88.0),
        "width": 768,
        "blocks": 12,
        "patch": 14,
    },
}


def unwrap(obj: Any) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Return (state_dict, notes). Checkpoints in the wild nest the weights
    under any of several keys, or are the bare tensor dict."""
    notes: List[str] = []
    sd = obj
    for k in ("state_dict", "model", "teacher", "student", "network"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            notes.append(f"unwrapped from ['{k}']")
            sd = sd[k]
            break
    if not isinstance(sd, dict):
        return {}, notes + [f"not a dict: {type(sd).__name__}"]
    tensors = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    if not tensors:
        return {}, notes + ["no tensors found"]
    # Strip a wrapper prefix if ANY key carries it, keeping only those keys.
    # Requiring EVERY key to share the prefix is wrong and was a real bug here:
    # a DINO checkpoint stores {backbone.*, dino_head.*} side by side, so the
    # all() form stripped nothing, left the head's parameters in the count, and
    # reported a genuine ViT-B as "0 blocks, 109M params, HALT". The
    # self-supervised projection head is not part of the backbone and must be
    # excluded from both the identity check and the parameter count -- it is
    # discarded when the encoder is used downstream.
    for pref in ("module.", "backbone.", "encoder."):
        matching = {k: v for k, v in tensors.items() if k.startswith(pref)}
        if matching and len(matching) >= 0.5 * len(tensors):
            dropped = len(tensors) - len(matching)
            tensors = {k[len(pref):]: v for k, v in matching.items()}
            notes.append(f"stripped prefix '{pref}'"
                         + (f"; excluded {dropped} non-backbone tensors "
                            f"(e.g. SSL projection head)" if dropped else ""))
            break
    return tensors, notes


def classify(sd: Dict[str, torch.Tensor]) -> str:
    keys = set(sd)
    if any(k.startswith("layer4.") for k in keys) and "conv1.weight" in keys:
        return "resnet50"
    if any("blocks." in k for k in keys) and any("patch_embed" in k for k in keys):
        return "vit_base"
    return "unknown"


def describe_vit(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    pe = sd.get("patch_embed.proj.weight")
    if pe is not None:
        out["patch_embed.proj.weight"] = list(pe.shape)
        out["width"] = int(pe.shape[0])
        out["patch"] = int(pe.shape[-1])
    pos = sd.get("pos_embed")
    if pos is not None:
        out["pos_embed"] = list(pos.shape)
        out["pos_embed_width"] = int(pos.shape[-1])
    blocks = {k.split(".")[1] for k in sd if k.startswith("blocks.")}
    out["n_blocks"] = len(blocks)
    return out


def describe_rn50(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("conv1.weight", "layer1.0.conv1.weight", "layer4.2.conv3.weight",
              "fc.weight"):
        if k in sd:
            out[k] = list(sd[k].shape)
    stages = {}
    for st in ("layer1", "layer2", "layer3", "layer4"):
        blocks = {k.split(".")[1] for k in sd if k.startswith(st + ".")}
        if blocks:
            stages[st] = len(blocks)
    out["blocks_per_stage"] = stages
    return out


def verify_one(t: Dict[str, str]) -> Dict[str, Any]:
    path = os.path.join(WEIGHTS_DIR, t["file"])
    res: Dict[str, Any] = {"key": t["key"], "label": t["label"],
                           "file": t["file"], "expected_arch": t["expect"]}
    if not os.path.exists(path):
        return {**res, "verdict": "HALT", "reason": "file does not exist"}
    size = os.path.getsize(path)
    res["size_bytes"] = size
    if size == 0:
        return {**res, "verdict": "HALT",
                "reason": "file is 0 bytes (truncated/failed download)"}
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 -- any load failure is a HALT
        return {**res, "verdict": "HALT",
                "reason": f"torch.load failed: {type(e).__name__}: {e}"}

    sd, notes = unwrap(obj)
    res["notes"] = notes
    if not sd:
        return {**res, "verdict": "HALT", "reason": "no tensors in checkpoint"}

    n_params = sum(v.numel() for v in sd.values())
    res["n_tensors"] = len(sd)
    res["params_total"] = n_params
    res["params_millions"] = round(n_params / 1e6, 2)

    found = classify(sd)
    res["detected_arch"] = found
    if found != t["expect"]:
        return {**res, "verdict": "HALT",
                "reason": f"expected {t['expect']}, detected {found}"}

    spec = EXPECT[t["expect"]]
    lo, hi = spec["params_m"]
    in_range = lo <= n_params / 1e6 <= hi
    res["params_expected_range_m"] = [lo, hi]
    res["params_in_range"] = in_range

    if found == "vit_base":
        d = describe_vit(sd)
        res["key_shapes"] = d
        checks = {
            "width_768": d.get("width") == spec["width"]
                         or d.get("pos_embed_width") == spec["width"],
            "blocks_12": d.get("n_blocks") == spec["blocks"],
            "patch_14": d.get("patch") == spec["patch"],
            "params_in_range": in_range,
        }
    else:
        d = describe_rn50(sd)
        res["key_shapes"] = d
        checks = {
            "conv1_7x7_64": d.get("conv1.weight") == spec["conv1_shape"],
            "bottleneck_3_4_6_3": d.get("blocks_per_stage") == {
                "layer1": 3, "layer2": 4, "layer3": 6, "layer4": 3},
            "params_in_range": in_range,
        }
    res["checks"] = checks
    res["verdict"] = "OK" if all(checks.values()) else "HALT"
    if res["verdict"] == "HALT":
        res["reason"] = "failed: " + ", ".join(k for k, v in checks.items() if not v)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", default="reports/backbone_verification.json")
    args = ap.parse_args(argv)

    results = [verify_one(t) for t in TARGETS]

    print("=" * 78)
    print("BACKBONE CHECKPOINT VERIFICATION")
    print("=" * 78)
    for r in results:
        mark = "OK  " if r["verdict"] == "OK" else "HALT"
        print(f"\n[{mark}] {r['label']}")
        print(f"       file     : {r['file']}")
        print(f"       size     : {r.get('size_bytes', 0) / 1e6:.1f} MB")
        if "params_millions" in r:
            print(f"       params   : {r['params_millions']}M "
                  f"({r.get('n_tensors')} tensors), expected "
                  f"{r.get('params_expected_range_m')}")
            print(f"       detected : {r.get('detected_arch')} "
                  f"(expected {r['expected_arch']})")
        for k, v in (r.get("key_shapes") or {}).items():
            print(f"         {k}: {v}")
        for k, v in (r.get("checks") or {}).items():
            print(f"         check {k}: {'pass' if v else 'FAIL'}")
        if r.get("notes"):
            print(f"       notes    : {'; '.join(r['notes'])}")
        if r["verdict"] != "OK":
            print(f"       REASON   : {r.get('reason')}")

    out = os.path.join(REPO_ROOT, args.out_json)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from src.io import write_text_durable
    write_text_durable(out, json.dumps(results, indent=2, default=str))
    print(f"\nwritten: {args.out_json}")

    n_halt = sum(1 for r in results if r["verdict"] != "OK")
    print("=" * 78)
    print(f"{len(results) - n_halt}/{len(results)} verified OK"
          + (f", {n_halt} HALT" if n_halt else ""))
    return 1 if n_halt else 0


if __name__ == "__main__":
    raise SystemExit(main())
