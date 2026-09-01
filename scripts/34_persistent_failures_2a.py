"""JOB 2a -- re-derive the persistent-failure positive set. NO TRAINING, NO GPU.

The original "seven" (reports/noise_floor.md section 6.1) came from A0,
repeat 0, seeds 0-4 only -- five (repeat, seed) groups, each a single
(non-ensembled) model's pooled-OOF predictions. A positive is a
"threshold setter" in a group if it ranks in the bottom 16 of that group's
158 positives by raw logit (the 16 given up to reach 90% recall: 0.9*158 =
142.2). "Persistent failure" = threshold-setter in EVERY group.

Repeats 1 and 2 now exist for A0 (runs/sweep_a0, repeats 1-2; repeat 0 lives
separately in runs/noise_floor_a). This script re-derives the set two ways:

  PRIMARY   -- same single-arm methodology as the original, extended along
              the axis that is actually new: A0's repeat x seed groups,
              5 -> 15.
  SECONDARY -- a robustness check pooling groups across all four arms that
              now carry 3 repeats (A0, A1, A2, A4), since the brief names
              all four in the same breath as "repeats 1 and 2 now exist".
              This is a strictly harder bar (persistent-failure-across-
              MODELS, not just across-seeds-of-one-model) and is reported
              alongside the primary reading rather than in place of it, so
              the ambiguity in the brief is resolved by evidence, not by
              silent choice.

    python scripts/34_persistent_failures_2a.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402

from src.evaluate import pool_oof  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
N_POSITIVES = 158
BOTTOM_N = 16  # 0.9 * 158 = 142.2 -> 16 given up at 90% recall

# arm key -> (out_dir for repeat 0, out_dir for repeats 1-2)
ARM_DIRS = {
    "A0": {0: "runs/noise_floor_a", 1: "runs/sweep_a0", 2: "runs/sweep_a0"},
    "A1": {0: "runs/sweep_a1", 1: "runs/sweep_a1", 2: "runs/sweep_a1"},
    "A2": {0: "runs/sweep_a2", 1: "runs/sweep_a2", 2: "runs/sweep_a2"},
    "A4": {0: "runs/sweep_a4", 1: "runs/sweep_a4", 2: "runs/sweep_a4"},
}


def bottom16_for_group(out_dir: str, repeat: int, seed: int,
                       manifest: pd.DataFrame) -> Set[str]:
    pool = pool_oof(out_dir, repeat, seed, folds=FOLDS, manifest=manifest)
    pos = pool[pool["label_int"] == 1].copy()
    if len(pos) != N_POSITIVES:
        raise AssertionError(
            f"{out_dir} r{repeat} s{seed}: {len(pos)} positives, expected "
            f"{N_POSITIVES}"
        )
    pos = pos.sort_values("logit", ascending=True, kind="mergesort")
    return set(pos.iloc[:BOTTOM_N]["filepath"])


def persistent_set(groups: Sequence[Set[str]]) -> Set[str]:
    if not groups:
        return set()
    out = set(groups[0])
    for g in groups[1:]:
        out &= g
    return out


def main() -> int:
    manifest = pd.read_csv(os.path.join(REPO_ROOT, "manifests",
                                        "rare25_folds_v2.csv"))

    # ---- PRIMARY: A0 alone, repeat 0 only vs 3 repeats ----
    a0_groups_by_repeat: Dict[int, Dict[int, Set[str]]] = {0: {}, 1: {}, 2: {}}
    for repeat in (0, 1, 2):
        out_dir = os.path.join(REPO_ROOT, ARM_DIRS["A0"][repeat])
        for seed in SEEDS:
            a0_groups_by_repeat[repeat][seed] = bottom16_for_group(
                out_dir, repeat, seed, manifest)

    a0_r0_groups = list(a0_groups_by_repeat[0].values())
    a0_all_groups = ([a0_groups_by_repeat[r][s] for r in (0, 1, 2) for s in SEEDS])

    set_r0 = persistent_set(a0_r0_groups)
    set_3rep = persistent_set(a0_all_groups)

    # ---- SECONDARY: A0+A1+A2+A4 pooled, repeat 0 only vs 3 repeats ----
    cross_groups_by_repeat: Dict[int, List[Set[str]]] = {0: [], 1: [], 2: []}
    per_arm_r0: Dict[str, Set[str]] = {}
    for arm in ("A0", "A1", "A2", "A4"):
        for repeat in (0, 1, 2):
            out_dir = os.path.join(REPO_ROOT, ARM_DIRS[arm][repeat])
            for seed in SEEDS:
                g = bottom16_for_group(out_dir, repeat, seed, manifest)
                cross_groups_by_repeat[repeat].append(g)
                if repeat == 0:
                    per_arm_r0.setdefault(arm, set())
        # per-arm r0 set for reference
    for arm in ("A0", "A1", "A2", "A4"):
        out_dir = os.path.join(REPO_ROOT, ARM_DIRS[arm][0])
        per_arm_r0[arm] = persistent_set(
            [bottom16_for_group(out_dir, 0, s, manifest) for s in SEEDS])

    cross_set_r0 = persistent_set(cross_groups_by_repeat[0])
    cross_set_3rep = persistent_set(
        cross_groups_by_repeat[0] + cross_groups_by_repeat[1] + cross_groups_by_repeat[2])

    result = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "primary_A0_only": {
            "repeat0_only_n_groups": len(a0_r0_groups),
            "repeat0_only_set": sorted(set_r0),
            "repeat0_only_size": len(set_r0),
            "three_repeats_n_groups": len(a0_all_groups),
            "three_repeats_set": sorted(set_3rep),
            "three_repeats_size": len(set_3rep),
            "grew_shrank_or_changed": (
                "grew" if len(set_3rep) > len(set_r0) else
                "shrank" if len(set_3rep) < len(set_r0) else
                ("same membership" if set_3rep == set_r0 else "same size, different membership")
            ),
            "added": sorted(set_3rep - set_r0),
            "removed": sorted(set_r0 - set_3rep),
        },
        "secondary_cross_arm_A0_A1_A2_A4": {
            "repeat0_only_n_groups": len(cross_groups_by_repeat[0]),
            "repeat0_only_set": sorted(cross_set_r0),
            "repeat0_only_size": len(cross_set_r0),
            "three_repeats_n_groups": sum(len(v) for v in cross_groups_by_repeat.values()),
            "three_repeats_set": sorted(cross_set_3rep),
            "three_repeats_size": len(cross_set_3rep),
            "per_arm_repeat0_only_size": {a: len(s) for a, s in per_arm_r0.items()},
        },
    }

    os.makedirs(os.path.join(REPO_ROOT, "reports"), exist_ok=True)
    with open(os.path.join(REPO_ROOT, "reports", "persistent_failures_2a.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    lines = []
    lines.append("# JOB 2a -- persistent-failure set, re-derived\n")
    lines.append(f"Generated by `scripts/34_persistent_failures_2a.py`. "
                 f"No training, no GPU: recomputed from prediction parquets on disk.\n")
    lines.append("## PRIMARY reading: A0 alone (same methodology as the original "
                 "\"seven\", extended along the newly-available repeat axis)\n")
    lines.append(f"- repeat-0-only (5 groups): **{len(set_r0)}** persistent-failure "
                 f"positives: {sorted(set_r0)}")
    lines.append(f"- 3 repeats (15 groups): **{len(set_3rep)}** persistent-failure "
                 f"positives: {sorted(set_3rep)}")
    lines.append(f"- verdict: the set **{result['primary_A0_only']['grew_shrank_or_changed']}**.")
    if set_3rep - set_r0:
        lines.append(f"  - added at 3 repeats: {sorted(set_3rep - set_r0)}")
    if set_r0 - set_3rep:
        lines.append(f"  - dropped at 3 repeats: {sorted(set_r0 - set_3rep)}")
    lines.append("")
    lines.append("## SECONDARY reading (robustness check): pooled across A0, A1, A2, A4 "
                 "(all four now have 3 repeats)\n")
    lines.append(f"- repeat-0-only, pooled across 4 arms x 5 seeds = 20 groups: "
                 f"**{len(cross_set_r0)}** positives: {sorted(cross_set_r0)}")
    lines.append(f"- 3 repeats, pooled across 4 arms x 3 repeats x 5 seeds = 60 groups: "
                 f"**{len(cross_set_3rep)}** positives: {sorted(cross_set_3rep)}")
    lines.append(f"- per-arm (repeat-0-only) persistent-failure set sizes: "
                 f"{result['secondary_cross_arm_A0_A1_A2_A4']['per_arm_repeat0_only_size']}")
    lines.append("")
    lines.append("_Two readings are reported because the brief names all four "
                 "repeat-gated arms in the same sentence as the repeat axis; the "
                 "PRIMARY reading is the literal continuation of the original "
                 "single-arm derivation and is the one JOB 2b-2e's dossier is built "
                 "from unless stated otherwise._\n")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(REPO_ROOT, "reports", "persistent_failures_2a.md"), "w") as fh:
        fh.write(text)

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
