"""D2 (reports/a4_loco_corrected_pre_registration.md) -- MANDATORY
re-adjudication of the G1 and G3 Rule 2 verdicts against CORRECTED-code A4
LOCO (`runs/a4_checkpointed_loco`), triggered iff corrected A4 LOCO differs
from old-formula A4 LOCO (`runs/sweep_a2_loco`) beyond its own conservative
bar in EITHER direction (see scripts/49_a4_loco_chain.py /
reports/a4_checkpointed_loco.json for that determination). NO TRAINING, NO
GPU: `runs/g1_rn50_swsl_loco` and `runs/g3_rn50_gastronet_loco` already
exist on disk, per the pre-registration's own point that this
re-adjudication costs nothing beyond what JOB 1 already trained.

Do NOT run this unless JOB 1's result actually triggers D2. If D1's null
(D3) holds -- corrected A4 LOCO within bar of old-formula in both
directions -- both standing Rule 2 verdicts (G3 REJECTED; G1 was never
formally gated by Rule 2 to begin with, see below) hold as-is and this
script is not needed.

WHAT THIS DOES NOT DO: overwrite the historical verdict computed against
old-formula A4. `reports/g3_loco_rule2.json` (the original, old-formula
verdict) is left untouched; this script writes SEPARATE
`*_corrected.json` files and a side-by-side section in
reports/gastronet.md, the same "old vs new, both shown" convention already
used by the G-vs-A4-corrected section (scripts/41_g_vs_a4_corrected.py) for
the pooled-OOF tables.

G1 note: G1's LOCO section (scripts/30_gastronet.py's render_g1_loco_section)
has never had a FORMAL Rule 2 verdict computed for it -- only a delta table
and fusion table. This script is therefore G1's FIRST formal Rule 2
application, not a re-adjudication of a prior G1 verdict; it uses corrected
A4 as control from the start, matching where the project's evidence now
stands.

    python scripts/50_readjudicate_rule2_corrected_a4.py
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
MS = _load("_magnitude_sweep", "26_magnitude_sweep.py")
G3M = _load("_g3loco", "32_g3_loco.py")

SEEDS = GN.SEEDS
CENTRES = GN.CENTRES
A0_LOCO_DIR = GN.A0_LOCO_DIR
CORRECTED_A4_LOCO_DIR = "runs/a4_checkpointed_loco"

G1_LOCO_DIR = "runs/g1_rn50_swsl_loco"
G1_CONFIG = "configs/g1_rn50_swsl.yaml"
G3_LOCO_DIR = "runs/g3_rn50_gastronet_loco"
G3_CONFIG = "configs/g3_rn50_gastronet.yaml"

REPORT_MD = os.path.join(REPO_ROOT, "reports", "gastronet.md")


def _pooled_from_gastronet_json(key: str):
    with open(os.path.join(REPO_ROOT, "reports", "gastronet.json")) as fh:
        gj = json.load(fh)
    a4_ref = gj["reference_a4"]
    pooled = gj["results"][key]["pooled"]
    return a4_ref, pooled


def _loco_bundle(manifest: pd.DataFrame, backbone_dir: str, backbone_config: str,
                 backbone_key_short: str) -> Dict[str, Any]:
    st = GN.unit_status_loco(backbone_config, backbone_dir)
    if not st["complete"]:
        return {"status": st}
    a0 = MS.analyse_loco(A0_LOCO_DIR, manifest, seeds=SEEDS)
    a4 = MS.analyse_loco(CORRECTED_A4_LOCO_DIR, manifest, seeds=SEEDS)
    bk = MS.analyse_loco(backbone_dir, manifest, seeds=SEEDS)
    return {"status": st, "a0": a0, "a4": a4, backbone_key_short: bk}


def apply_rule2(loco: Dict[str, Any], backbone_key_short: str,
                a4_pooled: Dict[str, Any], bk_pooled: Dict[str, Any]) -> Dict[str, Any]:
    """Same logic as scripts/32_g3_loco.py's apply_rule2_control_a4, made
    backbone-agnostic (bk instead of hardcoded g3) so it applies to G1 and
    G3 identically -- Rule 2 (reports/a4_pre_registration.md) with control =
    A4 (corrected)."""
    a4d = {c: loco["a4"][c] for c in CENTRES}
    bkd = {c: loco[backbone_key_short][c] for c in CENTRES}

    per_centre = {}
    both_improve = True
    any_beyond_bar = False
    for c in CENTRES:
        delta = a4d[c]["k5_fpr"] - bkd[c]["k5_fpr"]
        bar = max(a4d[c]["loo4_spread"]["iqr"], bkd[c]["loo4_spread"]["iqr"])
        beyond = delta > bar
        per_centre[c] = {"a4_k5_fpr": a4d[c]["k5_fpr"], "bk_k5_fpr": bkd[c]["k5_fpr"],
                         "delta": delta, "bar": bar, "beyond_bar": beyond}
        if delta <= 0:
            both_improve = False
        if beyond:
            any_beyond_bar = True

    a4_asym = a4_pooled["k5"]["fpr_asymmetry"]
    bk_asym = bk_pooled["k5"]["fpr_asymmetry"]
    asym_bar = max(a4_pooled["loo4"]["fpr_asymmetry"]["iqr"],
                   bk_pooled["loo4"]["fpr_asymmetry"]["iqr"])
    asym_reduced = (abs(a4_asym) - abs(bk_asym)) > asym_bar

    veto = False
    veto_detail = []
    for field in ("fpr_at_90_recall", "fpr_prior_equalised"):
        a4v = a4_pooled["k5"][field]
        bkv = bk_pooled["k5"][field]
        bar = max(a4_pooled["loo4"][field]["iqr"], bk_pooled["loo4"][field]["iqr"])
        regressed = (bkv - a4v) > bar
        veto_detail.append({"field": field, "a4": a4v, "bk": bkv, "bar": bar,
                            "regressed_beyond_bar": regressed})
        if regressed:
            veto = True

    accept = both_improve and any_beyond_bar and asym_reduced and not veto
    return {
        "control": "A4 (corrected, runs/a4_checkpointed_loco)",
        "per_centre": per_centre,
        "both_directions_improve": both_improve,
        "at_least_one_beyond_bar": any_beyond_bar,
        "asymmetry": {"a4": a4_asym, "bk": bk_asym, "bar": asym_bar,
                     "reduced_beyond_bar": asym_reduced},
        "veto": {"triggered": veto, "detail": veto_detail},
        "accept": accept,
    }


def render(label: str, rule2: Dict[str, Any]) -> str:
    L = []
    A = L.append
    A(f"### Rule 2 re-adjudication -- control = A4 CORRECTED ({label})")
    A("")
    A("Triggered by `reports/a4_loco_corrected_pre_registration.md` D2: "
     "corrected-code A4 LOCO differed from old-formula A4 LOCO beyond bar "
     "in at least one direction, so every Rule 2 verdict gated on the "
     "old-formula comparator is formally recomputed here against "
     "`runs/a4_checkpointed_loco` instead. The historical (old-formula) "
     "verdict is left on record elsewhere in this file, not overwritten.")
    A("")
    A(f"1. Both LOCO directions improve over corrected A4: "
     f"**{'yes' if rule2['both_directions_improve'] else 'no'}**.")
    A(f"2. At least one delta exceeds the conservative bar for its centre: "
     f"**{'yes' if rule2['at_least_one_beyond_bar'] else 'no'}**.")
    asym = rule2["asymmetry"]
    A(f"3. Centre asymmetry reduced vs corrected A4 beyond the conservative "
     f"bar: A4={asym['a4']:.4f}, backbone={asym['bk']:.4f}, "
     f"bar={asym['bar']:.4f} -- **{'yes' if asym['reduced_beyond_bar'] else 'no'}**.")
    veto = rule2["veto"]
    A(f"4. VETO (in-distribution pooled or prior-equalised FPR@90R "
     f"regressing beyond its own bar): "
     f"**{'TRIGGERED' if veto['triggered'] else 'not triggered'}**.")
    for d in veto["detail"]:
        A(f"   - {d['field']}: A4={d['a4']:.4f} backbone={d['bk']:.4f} "
         f"bar={d['bar']:.4f} regressed={d['regressed_beyond_bar']}")
    A("")
    A(f"**Rule 2 verdict (control = A4 corrected): "
     f"{'ACCEPTED' if rule2['accept'] else 'REJECTED'}**")
    A("")
    return "\n".join(L)


def main() -> int:
    if not os.path.isdir(os.path.join(REPO_ROOT, CORRECTED_A4_LOCO_DIR)):
        print(f"[50_readjudicate] HALT: {CORRECTED_A4_LOCO_DIR} not found -- "
              f"JOB 1 has not completed. Refusing to re-adjudicate against "
              f"data that does not exist.")
        return 2

    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))

    results = {}
    for backbone_key_short, backbone_dir, backbone_config, gj_key, label in (
        ("g1", G1_LOCO_DIR, G1_CONFIG, "g1_rn50_swsl", "G1"),
        ("g3", G3_LOCO_DIR, G3_CONFIG, "g3_rn50_gastronet", "G3"),
    ):
        loco = _loco_bundle(manifest, backbone_dir, backbone_config, backbone_key_short)
        if not loco["status"]["complete"]:
            print(f"[50_readjudicate] {label}: LOCO not complete "
                 f"({loco['status']}), skipping.")
            continue
        a4_pooled, bk_pooled = _pooled_from_gastronet_json(gj_key)
        rule2 = apply_rule2(loco, backbone_key_short, a4_pooled, bk_pooled)
        results[label] = rule2

        out_json = os.path.join(REPO_ROOT, "reports",
                                f"{backbone_key_short}_loco_rule2_corrected.json")
        with open(out_json, "w") as fh:
            json.dump(rule2, fh, indent=2)
        print(f"[50_readjudicate] {label} vs corrected A4: "
             f"{'ACCEPTED' if rule2['accept'] else 'REJECTED'} "
             f"(written {out_json})")

    if not results:
        print("[50_readjudicate] Nothing to write -- no backbone LOCO arm complete.")
        return 3

    with open(REPORT_MD, "r", encoding="utf-8") as fh:
        text = fh.read()

    BEGIN = "<!-- BEGIN RULE2 CORRECTED-A4 READJUDICATION (scripts/50_readjudicate_rule2_corrected_a4.py) -->"
    END = "<!-- END RULE2 CORRECTED-A4 READJUDICATION (scripts/50_readjudicate_rule2_corrected_a4.py) -->"
    section_lines = [BEGIN, "",
                     "## Rule 2 re-adjudication against corrected A4 (D2, reports/a4_loco_corrected_pre_registration.md)",
                     ""]
    for label, rule2 in results.items():
        section_lines.append(render(label, rule2))
    section_lines.append(END)
    section = "\n".join(section_lines)

    if BEGIN in text and END in text:
        pre, post = text.split(BEGIN)[0], text.split(END)[1]
        text = pre + section + post
    else:
        text = text.rstrip("\n") + "\n\n" + section + "\n"
    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"written: {REPORT_MD} (Rule 2 corrected-A4 re-adjudication section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
