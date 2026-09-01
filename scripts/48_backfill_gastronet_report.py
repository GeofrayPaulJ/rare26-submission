"""Backfill reports/gastronet.md with the two sections that exist only in
the DERIVED artefact paper/03_gastronet.md: the G0 LOCO section (computed by
scripts/44_pinned_arm_chain.py's render_g0_loco_report(), which writes only
reports/g0_loco.{md,json}, never gastronet.md) and the A4 pinned-085 verdict
(computed by the same script's render_and_write(), which writes only
reports/a4_pinned085_verdict.{md,json}). reports/ is this project's source of
truth; paper/ is generated from it (scripts/46_generate_paper.py reads
reports/*.json directly). A number computed once and shown only in the
derived file, never in the source-of-truth file, is a documentation gap, not
a new result -- this script transcribes nothing by hand, it reads the same
JSON files paper/03_gastronet.md already reads.

DOES NOT touch reference_a4 in reports/gastronet.json. The comparison
baseline for the A4-vs-G-backbone tables stays runs/a4_checkpointed (0.0280,
current code, unpinned) -- G0/G1/G2/G3 all trained under that same corrected
code, so pinning A4 to the pinned-085 arm would make A4 the only backbone
still compared on the old formula. The pinned-085 verdict answers a
different question (which parameter caused the pre-fix vs post-fix gap) and
is reported here as diagnostic history, not as a baseline change.

Idempotent: re-running replaces both sections between their own markers
rather than appending a second copy.

    python scripts/48_backfill_gastronet_report.py
"""
from __future__ import annotations

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
REPORT_MD = os.path.join(REPORT_DIR, "gastronet.md")

PINNED_BEGIN = "<!-- BEGIN A4 PINNED085 VERDICT SECTION (scripts/48_backfill_gastronet_report.py) -->"
PINNED_END = "<!-- END A4 PINNED085 VERDICT SECTION (scripts/48_backfill_gastronet_report.py) -->"
G0LOCO_BEGIN = "<!-- BEGIN G0 LOCO SECTION (scripts/48_backfill_gastronet_report.py) -->"
G0LOCO_END = "<!-- END G0 LOCO SECTION (scripts/48_backfill_gastronet_report.py) -->"


def _load(name: str) -> dict:
    with open(os.path.join(REPORT_DIR, name)) as fh:
        return json.load(fh)


def _splice(text: str, begin: str, end: str, section: str, anchor: str,
           anchor_before: bool) -> str:
    if begin in text and end in text:
        pre = text.split(begin)[0]
        post = text.split(end)[1]
        return pre + section + post
    if anchor in text:
        pre, post = text.split(anchor, 1)
        if anchor_before:
            return pre + section + "\n" + anchor + post
        return pre + anchor + "\n" + section + "\n" + post
    return text.rstrip("\n") + "\n\n" + section + "\n"


def render_pinned_section(v: dict) -> str:
    L = []
    A = L.append
    A(PINNED_BEGIN)
    A("")
    A("## A4 baseline status -- pinned-085 verdict (remediation item 5)")
    A("")
    A("From `reports/a4_pinned085_verdict.json`. `runs/a4_pinned085`: current "
     "code, `downsample_ceiling_m` pinned to exactly 0.85 (verified before "
     "launch), everything else identical to `configs/sweep_a4_checkpointed.yaml`. "
     "This isolates WHICH parameter caused the pre-fix (old code, 0.0529) vs "
     "post-fix (current code, unpinned, 0.0280) gap -- it does not change "
     "which arm this report treats as the A4 reference (still "
     "`runs/a4_checkpointed`, 0.0280, current code unpinned; see this file's "
     "`reference_a4` in `reports/gastronet.json`). G0/G1/G2/G3 all trained "
     "under the corrected code, so the reference stays on the corrected, "
     "unpinned arm for a like-for-like comparison across every backbone.")
    A("")
    A("| | FPR@90R | LOO-4 IQR |")
    A("|---|---|---|")
    A(f"| Old code (`runs/sweep_a4`) | {v['old_fpr']:.4f} | {v['old_iqr']:.4f} |")
    A(f"| Current code, unpinned (`runs/a4_checkpointed`) -- **this report's A4 reference** | "
     f"{v['unpinned_fpr']:.4f} | {v['unpinned_iqr']:.4f} |")
    A(f"| Current code, PINNED to 0.85 (`runs/a4_pinned085`) | "
     f"{v['pinned_fpr_at_90_recall']:.4f} | {v['pinned_loo4_iqr']:.4f} |")
    A("")
    A(f"Distance to old: {v['distance_to_old']:.4f}. Distance to unpinned: "
     f"{v['distance_to_unpinned']:.4f}.")
    A("")
    A(f"**{v['conclusion']}**")
    A("")
    A(f"IQR also restored toward old rather than staying near unpinned: "
     f"**{v['iqr_also_restored']}**. If the point estimate confirms the "
     f"parameter but the IQR does not follow, the variance-collapse question "
     f"remains open even after this result.")
    A("")
    A(PINNED_END)
    return "\n".join(L)


def render_g0_loco_section(g: dict) -> str:
    L = []
    A = L.append
    A(G0LOCO_BEGIN)
    A("")
    A("## G0 LOCO -- does GastroNet pretraining's advantage survive an unseen centre")
    A("")
    A(f"**{g['headline_question']}**")
    A("")
    A(f"Pooled (current code), from `reports/gastronet.json` / fresh "
     f"`runs/g0_rn50_imagenet` analysis: G3 minus G0 = "
     f"**{g['pooled_g3_minus_g0']:+.4f}**.")
    A("")
    A("| direction | A0 | A4 (old-formula, context only) | G3 | G0 | G3 minus G0 | bar | resolvable? |")
    A("|---|---|---|---|---|---|---|---|")
    for c in ("1", "2"):
        pc = g["per_centre"][c] if c in g["per_centre"] else g["per_centre"][int(c)]
        A(f"| holdout_center_{c} | {pc['a0_k5_fpr']:.4f} | {pc['a4_k5_fpr']:.4f} | "
         f"{pc['g3_k5_fpr']:.4f} | {pc['g0_k5_fpr']:.4f} | "
         f"{pc['g3_minus_g0']:+.4f} | {pc['bar']:.4f} | "
         f"{'**yes**' if pc['resolvable'] else 'no'} |")
    A("")
    A(f"_{g['a4_loco_caveat']}_")
    A("")
    A(f"**{g['conclusion']}**")
    A("")
    A(G0LOCO_END)
    return "\n".join(L)


def main() -> int:
    with open(REPORT_MD, "r", encoding="utf-8") as fh:
        text = fh.read()

    pinned = _load("a4_pinned085_verdict.json")
    g0_loco = _load("g0_loco.json")

    pinned_section = render_pinned_section(pinned)
    text = _splice(text, PINNED_BEGIN, PINNED_END, pinned_section,
                   anchor="## Status", anchor_before=True)

    g0loco_section = render_g0_loco_section(g0_loco)
    g0_control_end = "<!-- END G0 CONTROL SECTION (scripts/33_g0_control.py) -->"
    text = _splice(text, G0LOCO_BEGIN, G0LOCO_END, g0loco_section,
                   anchor=g0_control_end, anchor_before=False)

    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write(text)

    print(f"written: {REPORT_MD} (A4 pinned-085 verdict + G0 LOCO sections backfilled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
