"""EVC paste-detector control. CPU ONLY, NO GPU, sklearn not torch.

If a trivial classifier can tell composited (pasted) images apart from
genuine positive-class images using only generic low-level pixel
statistics, the paste augmentation is adding a detectable artifact
signature rather than naturalistic positive signal -- a model trained on
it could learn to key off that signature instead of the lesion itself, and
the augmentation would not be expected to transfer to real held-out data.
HIGH AUC VOIDS THE ARM, per instruction: this script's verdict is read
before any G3+EVC training unit runs.

Positive class = 40 composited pastes (runs/evc_paste_preview/paste_*.png).
Negative-of-this-classifier class = 158 genuine `class_label == "neoplasia"`
RARE25 images from 00_source (i.e. "real positives" in the disease sense,
but the NEGATIVE class for THIS classifier, which asks "is this image a
paste or not").

Features are generic whole-image statistics, not paste-specific (no mask,
no boundary ring) -- deliberately, so a detector that succeeds is doing so
on the same footing a downstream classifier would have, not on privileged
information about where the paste is.

    docker exec Prometheus bash -lc 'cd /workspace/RARE26 && python scripts/54_evc_paste_detector.py'
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREVIEW_DIR = os.path.join(REPO_ROOT, "runs", "evc_paste_preview")
RARE_ROOT = os.path.join(REPO_ROOT, "00_source")
OUT_MD = os.path.join(REPO_ROOT, "reports", "evc_paste_detector.md")
OUT_JSON = os.path.join(REPO_ROOT, "reports", "evc_paste_detector.json")
SEED = 20260808
N_FOLDS = 5


def features(img_bgr: np.ndarray) -> dict:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 100, 200)

    # high-frequency energy ratio via 2D FFT: fraction of spectral energy
    # outside the central (low-frequency) 1/4-radius disc
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = min(cy, cx)
    hf_mask = r > 0.25 * r_max
    hf_ratio = float(mag[hf_mask].sum() / (mag.sum() + 1e-9))

    return {
        "L_mean": float(lab[..., 0].mean()), "L_std": float(lab[..., 0].std()),
        "a_mean": float(lab[..., 1].mean()), "a_std": float(lab[..., 1].std()),
        "b_mean": float(lab[..., 2].mean()), "b_std": float(lab[..., 2].std()),
        "H_std": float(hsv[..., 0].std()),
        "S_mean": float(hsv[..., 1].mean()), "S_std": float(hsv[..., 1].std()),
        "V_std": float(hsv[..., 2].std()),
        "laplacian_var": float(lap.var()),
        "edge_density": float((edges > 0).mean()),
        "high_freq_energy_ratio": hf_ratio,
    }


def main() -> int:
    paste_paths = sorted(glob.glob(os.path.join(PREVIEW_DIR, "paste_*.png")))
    assert len(paste_paths) == 40, f"expected 40 preview pastes, found {len(paste_paths)}"

    rare = pd.read_csv(os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv"))
    pos = rare[rare["class_label"] == "neoplasia"].reset_index(drop=True)

    rows = []
    for p in paste_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        feat = features(img)
        feat.update({"source": os.path.basename(p), "is_paste": 1})
        rows.append(feat)

    for _, r in pos.iterrows():
        img = cv2.imread(os.path.join(RARE_ROOT, r["filepath"]))
        if img is None:
            continue
        feat = features(img)
        feat.update({"source": r["filepath"], "is_paste": 0})
        rows.append(feat)

    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c not in ("source", "is_paste")]
    X = df[feat_cols].values
    y = df["is_paste"].values
    n_paste, n_real = int(y.sum()), int((1 - y).sum())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=SEED, class_weight="balanced"),
    }

    results = {}
    for name, model in models.items():
        oof = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, oof)
        results[name] = {"pooled_oof_auc": round(float(auc), 4)}

    # feature importance from the RF, fit on all data, for interpretation only
    rf_full = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=SEED, class_weight="balanced")
    rf_full.fit(X, y)
    importances = sorted(zip(feat_cols, rf_full.feature_importances_), key=lambda kv: -kv[1])

    verdict_auc = max(r["pooled_oof_auc"] for r in results.values())
    HIGH_AUC_THRESHOLD = 0.90
    voided = verdict_auc >= HIGH_AUC_THRESHOLD

    out = {"n_paste": n_paste, "n_real_positive": n_real, "n_folds": N_FOLDS,
           "seed": SEED, "features": feat_cols, "results": results,
           "feature_importance_rf": [(k, round(float(v), 4)) for k, v in importances],
           "high_auc_threshold": HIGH_AUC_THRESHOLD,
           "verdict_auc": round(float(verdict_auc), 4), "arm_voided": bool(voided)}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)

    lines = ["# EVC paste-detector control",
             "",
             f"`scripts/54_evc_paste_detector.py`, CPU only (sklearn, no torch/GPU). "
             f"{n_paste} pasted composites (`runs/evc_paste_preview/paste_*.png`) vs "
             f"{n_real} genuine `class_label == \"neoplasia\"` RARE25 images, "
             f"{N_FOLDS}-fold stratified CV, pooled out-of-fold ROC-AUC discriminating "
             f"paste (1) vs real positive (0). Features are generic whole-image "
             f"Lab/HSV/edge/frequency statistics -- no mask or boundary information, "
             f"so the classifier has no more information than a downstream training "
             f"pipeline would.",
             "",
             f"**Verdict threshold: AUC >= {HIGH_AUC_THRESHOLD} voids the arm.**",
             "",
             "| model | pooled OOF AUC |",
             "|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['pooled_oof_auc']} |")
    lines += ["",
              f"**Verdict AUC (max across models): {out['verdict_auc']}** -- "
              + ("**ARM VOIDED: pasted composites are trivially detectable from "
                 "generic pixel statistics alone; do not use this augmentation for "
                 "training.**" if voided else
                 "arm NOT voided on this control; pasted composites are not trivially "
                 "separable from real positives on generic pixel statistics."),
              "",
              "## RF feature importances (interpretation only, not part of the verdict)",
              "",
              "| feature | importance |", "|---|---|"]
    for k, v in importances:
        lines.append(f"| {k} | {v} |")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[paste-detector] n_paste={n_paste} n_real={n_real} verdict_auc={out['verdict_auc']} voided={voided}")
    print(f"[paste-detector] wrote {OUT_MD} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
