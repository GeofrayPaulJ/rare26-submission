"""JOB 3 -- G3 LOCO section. NO TRAINING, NO GPU: recomputed from prediction
parquets on disk once runs/g3_rn50_gastronet_loco (10 units: 2 centres x 5
seeds) is complete.

Mirrors scripts/30_gastronet.py's G1-LOCO section (`analyse_g1_loco` /
`render_g1_loco_section`) exactly, with two differences:

  1. G3 instead of G1.
  2. Rule 2 (reports/a4_pre_registration.md) is applied FORMALLY here, with
     control = A4 instead of A0 -- i.e. every "improves over A0" clause in the
     original gate becomes "improves over A4". This is the brief's explicit
     instruction for JOB 3: G3 is being asked whether it earns an ensemble
     slot on top of the already-accepted A4, not whether A4-style treatment
     beats the untreated baseline (that question was already settled when A4
     was accepted).

INSERTION. Appended to reports/gastronet.md ABOVE the pooled-OOF tables (i.e.
directly after "## Status", before "## Metrics vs A4 ConvNeXt"), matching
where the G1 LOCO section already sits relative to G1's own pooled-OOF table
(that one is BELOW its pooled table; G3's placement is deliberately different
per this task's brief). Idempotent: re-running replaces the previously
inserted section (delimited by HTML comment markers) rather than duplicating
it.

RUN ONLY AFTER THE FULL GPU QUEUE HAS SETTLED (g3_loco AND g0_rn50_imagenet
both complete, or at minimum g3_loco alone if only this section is wanted).
scripts/23_sweep.py calls scripts/30_gastronet.py's update_gastronet() after
EVERY arm in RUN_ORDER_GASTRONET, which fully regenerates reports/gastronet.md
from scratch via render() -- and render() does not know about g3_loco, so it
does not clobber a section that was never there, but it also does not
preserve one that was inserted before g0_rn50_imagenet's own boundary update
fires. Running this script BEFORE the queue reaches gastronet_complete risks
exactly one throwaway re-insertion after g0 finishes; running it AFTER is the
one that lasts, so run it after the queue halts/completes.

    python scripts/32_g3_loco.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from src.evaluate import metric_block  # noqa: E402


def _load(name: str, filename: str):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GN = _load("_gastronet", "30_gastronet.py")
MS = _load("_magnitude_sweep", "26_magnitude_sweep.py")
RA = _load("_reanalysis", "17_reanalysis.py")

SEEDS = GN.SEEDS
CENTRES = GN.CENTRES
A0_LOCO_DIR = GN.A0_LOCO_DIR
A4_LOCO_DIR = GN.A4_LOCO_DIR
FUSION = GN.FUSION

G3_LOCO = {"key": "g3_loco", "config": "configs/g3_rn50_gastronet.yaml",
          "dir": "runs/g3_rn50_gastronet_loco",
          "label": "G3 -- RN50 GastroNet-5M (DINOv1), SWSL control"}

REPORT_MD = os.path.join(REPO_ROOT, "reports", "gastronet.md")

BEGIN_MARK = "<!-- BEGIN G3 LOCO SECTION (scripts/32_g3_loco.py) -->"
END_MARK = "<!-- END G3 LOCO SECTION (scripts/32_g3_loco.py) -->"


def analyse_g3_loco(manifest: pd.DataFrame) -> Dict[str, Any]:
    st = GN.unit_status_loco(G3_LOCO["config"], G3_LOCO["dir"])
    result: Dict[str, Any] = {"status": st}
    if not st["complete"]:
        return result

    a0 = MS.analyse_loco(A0_LOCO_DIR, manifest, seeds=SEEDS)
    a4 = MS.analyse_loco(A4_LOCO_DIR, manifest, seeds=SEEDS)
    g3 = MS.analyse_loco(G3_LOCO["dir"], manifest, seeds=SEEDS)

    fusion: Dict[int, Dict[str, float]] = {}
    for centre in CENTRES:
        fa = GN.loco_k5_frame(A4_LOCO_DIR, centre, manifest)
        fb = GN.loco_k5_frame(G3_LOCO["dir"], centre, manifest)
        fused = GN.fuse_loco_frames(fa, fb)
        y, s = fused["label_int"].to_numpy(), fused["logit"].to_numpy()
        m = metric_block(y, s)
        fusion[centre] = {"fpr_at_90_recall": m["fpr_at_90_recall"],
                          "roc_auc": m["roc_auc"]}

    result.update({"a0": a0, "a4": a4, "g3": g3, "fusion": fusion})
    return result


def apply_rule2_control_a4(loco: Dict[str, Any], a4_pooled: Dict[str, Any],
                           g3_pooled: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 2 from reports/a4_pre_registration.md, applied FORMALLY with
    control = A4 (every clause that read "over A0" now reads "over A4").
    a4_pooled / g3_pooled are each {"k5": {...}, "loo4": {...}} dicts in the
    same shape as reports/gastronet.json's "results"/"<key>"/"pooled"."""
    a4d = {c: loco["a4"][c] for c in CENTRES}
    g3d = {c: loco["g3"][c] for c in CENTRES}

    # Condition 1+2: both directions improve over A4, at least one beyond bar.
    per_centre = {}
    both_improve = True
    any_beyond_bar = False
    for c in CENTRES:
        delta = a4d[c]["k5_fpr"] - g3d[c]["k5_fpr"]  # positive = G3 better
        bar = max(a4d[c]["loo4_spread"]["iqr"], g3d[c]["loo4_spread"]["iqr"])
        beyond = delta > bar
        per_centre[c] = {"a4_k5_fpr": a4d[c]["k5_fpr"], "g3_k5_fpr": g3d[c]["k5_fpr"],
                         "delta": delta, "bar": bar, "beyond_bar": beyond}
        if delta <= 0:
            both_improve = False
        if beyond:
            any_beyond_bar = True

    # Condition 3: centre asymmetry (pooled k5) reduced vs A4, beyond bar.
    a4_asym = a4_pooled["k5"]["fpr_asymmetry"]
    g3_asym = g3_pooled["k5"]["fpr_asymmetry"]
    asym_bar = max(a4_pooled["loo4"]["fpr_asymmetry"]["iqr"],
                   g3_pooled["loo4"]["fpr_asymmetry"]["iqr"])
    asym_reduced = (abs(a4_asym) - abs(g3_asym)) > asym_bar

    # Condition 4 (VETO): in-distribution pooled FPR@90R or prior-equalised
    # regressing (increasing) beyond own bar.
    veto = False
    veto_detail = []
    for field in ("fpr_at_90_recall", "fpr_prior_equalised"):
        a4v = a4_pooled["k5"][field]
        g3v = g3_pooled["k5"][field]
        bar = max(a4_pooled["loo4"][field]["iqr"], g3_pooled["loo4"][field]["iqr"])
        regressed = (g3v - a4v) > bar
        veto_detail.append({"field": field, "a4": a4v, "g3": g3v, "bar": bar,
                            "regressed_beyond_bar": regressed})
        if regressed:
            veto = True

    accept = both_improve and any_beyond_bar and asym_reduced and not veto

    return {
        "per_centre": per_centre,
        "both_directions_improve": both_improve,
        "at_least_one_beyond_bar": any_beyond_bar,
        "asymmetry": {"a4": a4_asym, "g3": g3_asym, "bar": asym_bar,
                     "reduced_beyond_bar": asym_reduced},
        "veto": {"triggered": veto, "detail": veto_detail},
        "accept": accept,
    }


def render_section(loco: Dict[str, Any], rule2: Dict[str, Any] | None) -> str:
    L = []
    A = L.append
    A(BEGIN_MARK)
    A("")
    A("## G3 LOCO -- does the SWSL-ablated control earn a slot")
    A("")
    A("Both directions, k=5 seed ensemble FPR@90R with its own within-repeat "
     "LOO-4 IQR. `bar` is the conservative bar (max of A4's and G3's LOO-4 "
     "IQR); A0 is shown for context and is not part of the bar. Unlike the "
     "G1 LOCO section (below the pooled-OOF tables, because it was the first "
     "written), this section is placed ABOVE the pooled-OOF tables per this "
     "task's brief.")
    A("")
    if not loco["status"]["complete"]:
        st = loco["status"]
        A(f"**Not yet complete**: {st['done']}/{st['total']} units.")
        A("")
        A(END_MARK)
        return "\n".join(L)

    A("| direction | A0 | A4 | G3 | G3 vs A4 delta | bar | resolvable? |")
    A("|---|---|---|---|---|---|---|")
    for centre in CENTRES:
        a0d, a4d, g3d = loco["a0"][centre], loco["a4"][centre], loco["g3"][centre]
        bar = max(a4d["loo4_spread"]["iqr"], g3d["loo4_spread"]["iqr"])
        delta = g3d["k5_fpr"] - a4d["k5_fpr"]
        res = abs(delta) > bar
        A(f"| holdout_center_{centre} | {a0d['k5_fpr']:.4f} | {a4d['k5_fpr']:.4f} | "
         f"{g3d['k5_fpr']:.4f} | {delta:+.4f} | {bar:.4f} | "
         f"{'**yes**' if res else 'no'} |")
    A("")

    A("### A4+G3 fusion on LOCO predictions, per direction")
    A("")
    A("| direction | A4 alone | G3 alone | fused | fused vs A4 delta | bar | resolvable? |")
    A("|---|---|---|---|---|---|---|")
    for centre in CENTRES:
        a4d, g3d, fus = loco["a4"][centre], loco["g3"][centre], loco["fusion"][centre]
        bar = max(a4d["loo4_spread"]["iqr"], g3d["loo4_spread"]["iqr"])
        delta = fus["fpr_at_90_recall"] - a4d["k5_fpr"]
        res = abs(delta) > bar
        A(f"| holdout_center_{centre} | {a4d['k5_fpr']:.4f} | {g3d['k5_fpr']:.4f} | "
         f"{fus['fpr_at_90_recall']:.4f} | {delta:+.4f} | {bar:.4f} | "
         f"{'**yes**' if res else 'no'} |")
    A("")

    if rule2 is not None:
        A("### Rule 2 (reports/a4_pre_registration.md), applied FORMALLY -- control = A4")
        A("")
        A("Every clause of Rule 2 originally read \"improves over A0\"; here "
         "every occurrence of A0 is replaced by A4, since G3 is being asked "
         "whether it earns an ensemble slot on top of the already-accepted "
         "A4, not whether A4-style treatment beats the untreated baseline.")
        A("")
        A(f"1. Both LOCO directions improve over A4: "
         f"**{'yes' if rule2['both_directions_improve'] else 'no'}**.")
        A(f"2. At least one delta exceeds the conservative bar for its centre: "
         f"**{'yes' if rule2['at_least_one_beyond_bar'] else 'no'}**.")
        asym = rule2["asymmetry"]
        A(f"3. Centre asymmetry reduced vs A4 beyond the conservative bar: "
         f"A4={asym['a4']:.4f}, G3={asym['g3']:.4f}, bar={asym['bar']:.4f} -- "
         f"**{'yes' if asym['reduced_beyond_bar'] else 'no'}**.")
        veto = rule2["veto"]
        A(f"4. VETO (in-distribution pooled or prior-equalised FPR@90R "
         f"regressing beyond its own bar): "
         f"**{'TRIGGERED' if veto['triggered'] else 'not triggered'}**.")
        for d in veto["detail"]:
            A(f"   - {d['field']}: A4={d['a4']:.4f} G3={d['g3']:.4f} "
             f"bar={d['bar']:.4f} regressed={d['regressed_beyond_bar']}")
        A("")
        A(f"**Rule 2 verdict (control = A4): "
         f"{'ACCEPTED' if rule2['accept'] else 'REJECTED'}**")
        A("")

    A(END_MARK)
    return "\n".join(L)


def insert_into_report(section_text: str) -> None:
    with open(REPORT_MD, "r", encoding="utf-8") as fh:
        text = fh.read()

    if BEGIN_MARK in text and END_MARK in text:
        pre = text.split(BEGIN_MARK)[0]
        post = text.split(END_MARK)[1]
        new_text = pre + section_text + post
    else:
        anchor = "## Metrics vs A4 ConvNeXt"
        if anchor not in text:
            # complete/incomplete-backbone variant of the report has no
            # pooled-OOF section at all yet; append at the end instead.
            new_text = text.rstrip("\n") + "\n\n" + section_text + "\n"
        else:
            pre, post = text.split(anchor, 1)
            new_text = pre + section_text + "\n" + anchor + post

    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(new_text)


def main() -> int:
    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests",
                                        "rare25_folds_v2.csv"))
    loco = analyse_g3_loco(manifest)

    rule2 = None
    if loco["status"]["complete"]:
        import json
        with open(os.path.join(REPO_ROOT, "reports", "gastronet.json")) as fh:
            gj = json.load(fh)
        a4_ref = gj.get("reference_a4")
        g3_pooled = gj.get("results", {}).get("g3_rn50_gastronet", {}).get("pooled")
        if a4_ref is None or g3_pooled is None:
            print("[32_g3_loco] WARNING: could not find A4 reference or G3 "
                  "pooled figures in reports/gastronet.json; skipping Rule 2.")
        else:
            rule2 = apply_rule2_control_a4(loco, a4_ref, g3_pooled)

    section = render_section(loco, rule2)
    insert_into_report(section)
    print(f"written: {REPORT_MD} (G3 LOCO section "
          f"{'complete' if loco['status']['complete'] else 'incomplete: ' + str(loco['status'])})")

    verdict_path = os.path.join(REPO_ROOT, "reports", "g3_loco_rule2.json")
    if rule2 is not None:
        with open(verdict_path, "w") as fh:
            json.dump(rule2, fh, indent=2)
        print("")
        print("=" * 78)
        print(f"G3 LOCO -- RULE 2 VERDICT (control = A4): "
              f"{'ACCEPTED' if rule2['accept'] else 'REJECTED'}")
        print("=" * 78)
        print(f"  1. both directions improve over A4: {rule2['both_directions_improve']}")
        print(f"  2. at least one beyond conservative bar: {rule2['at_least_one_beyond_bar']}")
        print(f"  3. centre asymmetry reduced beyond bar: {rule2['asymmetry']['reduced_beyond_bar']}")
        print(f"  4. veto triggered: {rule2['veto']['triggered']}")
        print(f"written: {verdict_path}")
        print("=" * 78)
    else:
        with open(verdict_path, "w") as fh:
            json.dump({"accept": None, "reason": "loco incomplete or A4/G3 "
                      "pooled figures unavailable"}, fh, indent=2)
        print(f"G3 LOCO -- RULE 2 VERDICT: not computed (see above). "
              f"written: {verdict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
