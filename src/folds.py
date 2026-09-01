"""Split resolution from the fold manifest.

Two independent split families live in one manifest:

  * 10 repeats x 5 folds of grouped cross-validation, in columns
    fold_r0 .. fold_r9. A value of -1 means "never use this row" (it failed
    QC / is not kept for training). Any other value is the fold index.

  * two leave-one-centre-out splits, in columns holdout_center_1 and
    holdout_center_2, each valued "train"/"test".

Everything here returns lists of `filepath` strings; the dataset turns those
into pixels. Nothing here decodes an image.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

FOLD_SENTINEL = -1  # rows tagged -1 are excluded from every split


def _load(manifest: str) -> pd.DataFrame:
    return pd.read_csv(manifest)


def _fold_col(repeat: int) -> str:
    if not 0 <= repeat <= 9:
        raise ValueError(f"repeat must be 0..9, got {repeat}")
    return f"fold_r{repeat}"


def get_split(
    repeat: int, fold: int, manifest: str
) -> Tuple[List[str], List[str]]:
    """Return (train_filepaths, val_filepaths) for one (repeat, fold).

    * Validation  = rows whose fold_r{repeat} == fold.
    * Train       = rows whose fold_r{repeat} is a real fold != fold.
    * fold == -1 rows appear in neither. keep_for_training is honoured (rows
      that are not kept carry -1 in every fold column, and we assert that).
    """
    df = _load(manifest)
    col = _fold_col(repeat)

    if "keep_for_training" in df.columns:
        # rows not kept must already be sentinel-tagged; guard the invariant.
        bad = df[(df["keep_for_training"] == False) & (df[col] != FOLD_SENTINEL)]
        if len(bad):
            raise AssertionError(
                f"{len(bad)} rows are keep_for_training=False but not {FOLD_SENTINEL} "
                f"in {col}; the manifest is inconsistent"
            )

    usable = df[df[col] != FOLD_SENTINEL]
    val = usable[usable[col] == fold]["filepath"].tolist()
    train = usable[usable[col] != fold]["filepath"].tolist()
    return train, val


def get_full_data_split(manifest: str) -> List[str]:
    """All usable filepaths (repeat 0's fold_r0 != -1, i.e. keep_for_training),
    no split at all. Additive only -- no existing call site touches this.
    Used solely by scripts/59_full_data_ema.py for a deploy-scale checkpoint
    trained on 100% of labelled data; there is deliberately no held-out set
    to return alongside it."""
    df = _load(manifest)
    usable = df[df[_fold_col(0)] != FOLD_SENTINEL]
    return usable["filepath"].tolist()


def get_holdout_split(
    centre: int, manifest: str
) -> Tuple[List[str], List[str]]:
    """Return (train_filepaths, test_filepaths) for a leave-one-centre-out split.

    centre == 1 uses holdout_center_1 (center_1 is the held-out test centre),
    centre == 2 uses holdout_center_2. Rows with -1 in fold_r0 (i.e. not kept
    for training) are excluded from the train side, matching the CV splits.
    """
    if centre not in (1, 2):
        raise ValueError(f"centre must be 1 or 2, got {centre}")
    df = _load(manifest)
    col = f"holdout_center_{centre}"

    train_mask = df[col] == "train"
    if "fold_r0" in df.columns:
        # keep the train side consistent with the CV usable set
        train_mask &= df["fold_r0"] != FOLD_SENTINEL

    train = df[train_mask]["filepath"].tolist()
    test = df[df[col] == "test"]["filepath"].tolist()
    return train, test
