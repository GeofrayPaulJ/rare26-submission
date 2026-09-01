"""Item 3 (2026-08-05 remediation) -- every G-vs-A4 pooled table in
gastronet.md, recomputed with runs/a4_checkpointed (current code) as the A4
column, OLD-baseline (runs/sweep_a4) figures retained alongside for direct
comparison. NO TRAINING, NO GPU.

Does NOT re-apply Rule 2 -- that needs new-code A4 LOCO (item 5/step 4b of
the remediation plan), not run yet. This is the pooled-OOF picture only.

    python scripts/41_g_vs_a4_corrected.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

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

REPORT_MD = os.path.join(REPO_ROOT, "reports", "gastronet.md")
BEGIN_MARK = "<!-- BEGIN G-VS-A4-CORRECTED SECTION (scripts/41_g_vs_a4_corrected.py) -->"
END_MARK = "<!-- END G-VS-A4-CORRECTED SECTION (scripts/41_g_vs_a4_corrected.py) -->"


def fmt(x, p=4):
    return "n/a" if x is None else f"{x:.{p}f}"


def main() -> int:
    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))

    old_a4 = GN.analyse(GN.REFERENCE["dir"], manifest)         # runs/sweep_a4
    new_a4 = GN.analyse("runs/a4_checkpointed", manifest)      # current code

    with open(os.path.join(REPO_ROOT, "reports", "gastronet.json")) as fh:
        gj = json.load(fh)

    L = []
    A = L.append
    A(BEGIN_MARK)
    A("")
    A("## G-vs-A4, recomputed against the CORRECTED A4 baseline (item 3, 2026-08-05)")
    A("")
    A("**Do not read this section in isolation.** `runs/sweep_a4` (\"A4 "
     "old\" below) trained under code from 2026-07-30, before the "
     "`downsample_ceiling_m` fix (2026-07-31) and a further set of core "
     "training-file changes on 2026-08-03 (`src/backbones.py`, "
     "`src/config.py`, `src/io.py`, `src/model.py`, `src/train.py`). "
     "`runs/a4_checkpointed` (\"A4 new\") is a same-config retrain under "
     "TODAY's code -- see `reports/a4_retrain_finding.md` for the full "
     "account. G1, G2, and G3 all trained 2026-08-03/04, i.e. already under "
     "current code -- so the OLD table was comparing them against a stale "
     "reference. Rule 2 is NOT re-applied here; that needs new-code A4 LOCO, "
     "which has not been run.")
    A("")
    A(f"A4 pooled-OOF FPR@90R -- old: **{fmt(old_a4['k5']['fpr_at_90_recall'])}**, "
     f"new: **{fmt(new_a4['k5']['fpr_at_90_recall'])}**.")
    A("")

    for key, label in (("g1_rn50_swsl", "G1"), ("g2_vitb_dinov2", "G2"),
                       ("g3_rn50_gastronet", "G3")):
        g = gj["results"][key]["pooled"]
        A(f"### {label} vs A4 (old vs new baseline)")
        A("")
        A("| metric | A4 old | A4 new | " + label + " | delta (old A4 - "
         + label + ") | delta (new A4 - " + label + ") | bar (new) | "
         "resolvable (new)? |")
        A("|---|---|---|---|---|---|---|---|")
        for f in GN.FIELDS:
            av_old = old_a4["k5"][f]
            av_new = new_a4["k5"][f]
            gv = g["k5"][f]
            d_old = av_old - gv
            d_new = av_new - gv
            bar_new = max(new_a4["loo4"][f]["iqr"], g["loo4"][f]["iqr"])
            res_new = abs(d_new) > bar_new
            A(f"| {GN.LABELS[f]} | {fmt(av_old)} | {fmt(av_new)} | {fmt(gv)} "
             f"| {d_old:+.4f} | {d_new:+.4f} | {fmt(bar_new)} | "
             f"{'**yes**' if res_new else 'no'} |")
        A("")

    A("### Headline: G3's pooled advantage over A4")
    A("")
    g3 = gj["results"]["g3_rn50_gastronet"]["pooled"]["k5"]["fpr_at_90_recall"]
    adv_old = old_a4["k5"]["fpr_at_90_recall"] - g3
    adv_new = new_a4["k5"]["fpr_at_90_recall"] - g3
    bar_new = max(new_a4["loo4"]["fpr_at_90_recall"]["iqr"],
                 gj["results"]["g3_rn50_gastronet"]["pooled"]["loo4"]["fpr_at_90_recall"]["iqr"])
    A(f"G3's pooled FPR@90R advantage over A4 falls from **{adv_old:.4f}** "
     f"(old baseline) to **{adv_new:.4f}** (corrected baseline) -- against "
     f"a conservative bar of {bar_new:.4f}, this is **"
     f"{'still' if abs(adv_new) > bar_new else 'no longer'} resolvable**.")
    A("")
    A(END_MARK)
    section = "\n".join(L)

    with open(REPORT_MD, "r", encoding="utf-8") as fh:
        text = fh.read()
    if BEGIN_MARK in text and END_MARK in text:
        pre, post = text.split(BEGIN_MARK)[0], text.split(END_MARK)[1]
        new_text = pre + section + post
    else:
        new_text = text.rstrip("\n") + "\n\n" + section + "\n"
    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(new_text)

    print(f"written: {REPORT_MD}")
    print(f"A4 old FPR@90R: {old_a4['k5']['fpr_at_90_recall']:.4f}  "
         f"A4 new FPR@90R: {new_a4['k5']['fpr_at_90_recall']:.4f}")
    print(f"G3 advantage: old {adv_old:.4f} -> new {adv_new:.4f} "
         f"(bar {bar_new:.4f}, resolvable={abs(adv_new) > bar_new})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
