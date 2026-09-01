"""L3 -- LOCO AUROC commensurability.

CHECK FIRST (per instruction): no report in reports/ presents LOCO AUROC
as a finding -- reports/a4_checkpointed_loco.md reports FPR@90R (IQR)
only. The per-epoch training history in each run's summary.json DOES
carry a `roc_auc` field at the canonical (last) epoch, but that has never
been surfaced/aggregated in any report. Proceeding to compute, per
instruction, using the top-level val_loco_c{1,2}_s{seed}.parquet files
(the canonical last-epoch predictions -- verified below to match
summary.json's own last-epoch roc_auc, not re-derived from the per-epoch
dumps under preds/).

JOB A: per-direction AUROC, n=1 (seeds 0-4) and k=5 logit-averaged
ensemble, for the SHIPPING checkpoint family (a4_checkpointed_loco).
Bootstrap 95% CI (2000 resamples, image-level, with replacement).
Also: the two LOCO directions pooled together (same checkpoints,
covers the whole dataset once, each image scored by a model that never
saw its centre) as the "pooled OOF, same checkpoints" control.

JOB B: full-data k=5 checkpoints (deploy_a4_full_s{0-4}) -- checked for
any held-out LOCO scoring. None exists (S1V: full-data checkpoints train
on 100% of data with no holdout by construction). Not trained here to
create one, per instruction.

JOB C printed separately in reports/l3_loco_auroc.md, referencing this
script's numbers plus the standard (non-LOCO) 5-fold pooled OOF AUROC
already on record for the same checkpoint family (runs/a4_checkpointed,
single seed 0, folds 0-4) as the headline local comparator.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/106_l3_loco_auroc.py'
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20260819)
N_BOOT = 2000

LOCO_TEMPLATE = "runs/a4_checkpointed_loco/loco_c{c}_s{s}/val_loco_c{c}_s{s}.parquet"
FOLD_TEMPLATE = "runs/a4_checkpointed/r0_f{f}_s0/val_r0_f{f}_s0.parquet"


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOT):
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        aucs.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), len(aucs)


def load_direction(centre: int) -> dict:
    """seed -> dataframe (filepath, label_int, logit), all 5 seeds, same held-out set."""
    per_seed = {}
    for s in range(5):
        df = pd.read_parquet(LOCO_TEMPLATE.format(c=centre, s=s))
        per_seed[s] = df.set_index("filepath")
    return per_seed


def direction_report(centre: int) -> dict:
    per_seed = load_direction(centre)
    ref = per_seed[0]
    fps = ref.index.tolist()
    labels = ref["label_int"].to_numpy()
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())

    n1 = {}
    logits_matrix = np.zeros((len(fps), 5))
    for s in range(5):
        df = per_seed[s].loc[fps]
        assert (df["label_int"].to_numpy() == labels).all(), f"label mismatch seed {s}"
        logit = df["logit"].to_numpy()
        logits_matrix[:, s] = logit
        prob = 1 / (1 + np.exp(-logit))
        auc = roc_auc_score(labels, prob)
        lo, hi, n_eff = bootstrap_ci(labels, prob)
        n1[s] = {"auroc": round(float(auc), 4), "ci95": [round(lo, 4), round(hi, 4)]}

    ens_logit = logits_matrix.mean(axis=1)
    ens_prob = 1 / (1 + np.exp(-ens_logit))
    ens_auc = roc_auc_score(labels, ens_prob)
    ens_lo, ens_hi, _ = bootstrap_ci(labels, ens_prob)

    return {
        "centre_held_out": centre,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n1_per_seed": n1,
        "k5_ensemble": {"auroc": round(float(ens_auc), 4), "ci95": [round(ens_lo, 4), round(ens_hi, 4)]},
        "_labels": labels,
        "_ens_prob": ens_prob,
        "_fps": fps,
        "_per_seed_probs": {s: 1 / (1 + np.exp(-logits_matrix[:, s])) for s in range(5)},
    }


def pooled_both_directions(d1: dict, d2: dict) -> dict:
    """Both LOCO directions concatenated -- covers the whole dataset once,
    each image scored by a model that never saw its own centre. Same
    checkpoints, "pooled OOF" control JOB A asks for."""
    labels = np.concatenate([d1["_labels"], d2["_labels"]])

    n1_pooled = {}
    for s in range(5):
        probs = np.concatenate([d1["_per_seed_probs"][s], d2["_per_seed_probs"][s]])
        auc = roc_auc_score(labels, probs)
        lo, hi, _ = bootstrap_ci(labels, probs)
        n1_pooled[s] = {"auroc": round(float(auc), 4), "ci95": [round(lo, 4), round(hi, 4)]}

    ens_probs = np.concatenate([d1["_ens_prob"], d2["_ens_prob"]])
    ens_auc = roc_auc_score(labels, ens_probs)
    ens_lo, ens_hi, _ = bootstrap_ci(labels, ens_probs)

    return {
        "n_pos": int(labels.sum()), "n_neg": int((labels == 0).sum()),
        "n1_per_seed": n1_pooled,
        "k5_ensemble": {"auroc": round(float(ens_auc), 4), "ci95": [round(ens_lo, 4), round(ens_hi, 4)]},
    }


def standard_pooled_oof() -> dict:
    """The STANDARD (non-LOCO, grouped-random-fold) 5-fold pooled OOF AUROC
    for the same checkpoint family (runs/a4_checkpointed, seed 0, folds 0-4)
    -- the project's existing headline local comparator, recomputed here for
    a verified, single-seed, apples-to-apples figure (not the 5-seed x
    5-fold 0.9706 pooled figure quoted elsewhere, which mixes 5 seeds)."""
    frames = []
    for f in range(5):
        df = pd.read_parquet(FOLD_TEMPLATE.format(f=f))
        frames.append(df[["filepath", "label_int", "logit"]])
    pooled = pd.concat(frames, ignore_index=True)
    labels = pooled["label_int"].to_numpy()
    probs = 1 / (1 + np.exp(-pooled["logit"].to_numpy()))
    auc = roc_auc_score(labels, probs)
    lo, hi, _ = bootstrap_ci(labels, probs)
    return {
        "n_pos": int(labels.sum()), "n_neg": int((labels == 0).sum()),
        "auroc": round(float(auc), 4), "ci95": [round(lo, 4), round(hi, 4)],
    }


def verify_against_summary_json() -> dict:
    """Sanity check: does the top-level val_loco_c1_s0.parquet's AUROC match
    summary.json's own last-epoch (canonical) roc_auc field?"""
    with open("runs/a4_checkpointed_loco/loco_c1_s0/summary.json") as fh:
        summ = json.load(fh)
    last_epoch = summ["history"][-1]
    assert last_epoch["epoch"] == 30
    recorded = last_epoch["roc_auc"]

    df = pd.read_parquet("runs/a4_checkpointed_loco/loco_c1_s0/val_loco_c1_s0.parquet")
    prob = 1 / (1 + np.exp(-df["logit"].to_numpy()))
    recomputed = roc_auc_score(df["label_int"].to_numpy(), prob)
    return {"summary_json_last_epoch_roc_auc": recorded, "recomputed_from_parquet": float(recomputed),
            "match": abs(recorded - recomputed) < 1e-6}


def main() -> int:
    check = verify_against_summary_json()
    print("[verify] top-level parquet vs summary.json last-epoch roc_auc:")
    print(json.dumps(check, indent=2))
    assert check["match"], "top-level parquet does not match canonical last epoch -- STOP"

    d1 = direction_report(1)
    d2 = direction_report(2)
    pooled_loco = pooled_both_directions(d1, d2)
    std_pooled = standard_pooled_oof()

    out = {
        "holdout_centre_1": {k: v for k, v in d1.items() if not k.startswith("_")},
        "holdout_centre_2": {k: v for k, v in d2.items() if not k.startswith("_")},
        "pooled_both_loco_directions_same_checkpoints": pooled_loco,
        "standard_5fold_pooled_oof_seed0": std_pooled,
        "full_data_k5_checkpoints_held_out_scoring": "NONE EXISTS -- deploy_a4_full_s{0-4} train on "
            "100% of data with no holdout by construction (S1V, reports/s1v_full_data_verification.md). "
            "Not trained here to create one, per instruction.",
    }

    with open("reports/l3_loco_auroc_raw.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n=== holdout_centre_1 ===")
    print(json.dumps(out["holdout_centre_1"], indent=2))
    print("\n=== holdout_centre_2 ===")
    print(json.dumps(out["holdout_centre_2"], indent=2))
    print("\n=== pooled both LOCO directions (same checkpoints) ===")
    print(json.dumps(pooled_loco, indent=2))
    print("\n=== standard 5-fold pooled OOF, seed 0 (same checkpoint family) ===")
    print(json.dumps(std_pooled, indent=2))
    print("\nwrote reports/l3_loco_auroc_raw.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
