"""Prediction dump schema.

One parquet per (repeat, fold, seed). The `logit` column is the raw pre-sigmoid
output stored at full float64 -- never a probability, never clipped, never
rounded. At test time there are 23,176 negatives; pushed through a sigmoid they
saturate and pile up as ties at the operating threshold, and ties inflate the
false-positive count directly. Keeping the raw logit at float64 preserves the
ordering the metric depends on.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd

# exact column order of the dump
SCHEMA = [
    "filepath",
    "centre",
    "class_label",
    "label_int",
    "visibility",
    "group_id_v2",
    "repeat",
    "fold",
    "seed",
    "logit",
]


def build_frame(
    filepath: Sequence[str],
    centre: Sequence[str],
    class_label: Sequence[str],
    label_int: Sequence[int],
    visibility: Sequence,
    group_id_v2: Sequence,
    repeat: int,
    fold: int,
    seed: int,
    logit: Sequence[float],
) -> pd.DataFrame:
    """Assemble the prediction frame with the exact schema and dtypes."""
    logit_arr = np.asarray(logit, dtype=np.float64)
    df = pd.DataFrame(
        {
            "filepath": np.asarray(filepath, dtype=object),
            "centre": np.asarray(centre, dtype=object),
            "class_label": np.asarray(class_label, dtype=object),
            "label_int": np.asarray(label_int, dtype=np.int64),
            "visibility": np.asarray(visibility, dtype=object),
            "group_id_v2": np.asarray(group_id_v2, dtype=object),
            "repeat": np.int64(repeat),
            "fold": np.int64(fold),
            "seed": np.int64(seed),
            "logit": logit_arr,
        },
        columns=SCHEMA,
    )
    return df


def fsync_file(path: str) -> None:
    """Force this file's CONTENT to durable storage.

    Opening read-only is enough: fsync acts on the underlying inode, not on the
    handle's access mode, and this way the caller does not have to hold the
    write handle open to reach it (torch.save and DataFrame.to_parquet both
    close their own handle before returning).
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: str) -> None:
    """Force a DIRECTORY ENTRY to durable storage.

    Distinct from fsync_file and not redundant with it. fsync on a file makes
    its bytes durable; it says nothing about whether the directory entry
    pointing at those bytes survived. After os.replace the new name lives only
    in the parent directory's metadata, so a host reset between the replace and
    the directory's own writeback can leave the OLD name (or no name) even
    though the new file's content is safely on disk.

    No-op on platforms that refuse to open a directory (Windows). That is not a
    silent gap: every unattended run executes inside the Linux container, where
    it works -- the Windows path is only ever used for interactive analysis.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def durable_replace(tmp: str, final: str) -> None:
    """fsync(content) -> os.replace -> fsync(parent dir).

    The full sequence needed for "either the old file or the new file, never a
    torn one, even across an abrupt host reset". os.replace alone gives
    atomicity against a killed PROCESS (the page cache outlives it) but not
    against a killed MACHINE -- that distinction cost this project a halted
    weekend run when a checkpoint was renamed into place with its bytes still
    unflushed. Every durable write in this repo goes through here.
    """
    fsync_file(tmp)
    os.replace(tmp, final)
    fsync_dir(os.path.dirname(os.path.abspath(final)))


def write_text_durable(path: str, text: str, encoding: str = "utf-8") -> None:
    """Write a whole text file (a report, a sentinel) durably.

    Used for anything a human or a later stage READS BACK as a result. A
    report truncated by a host reset is worse than a missing one: a missing
    file is an obvious failure, a half-written one gets believed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    durable_replace(tmp, path)


def write_predictions(df: pd.DataFrame, path: str) -> None:
    """Validate and write one prediction parquet.

    Asserts the schema, that `logit` is float64, and that no logit is NaN/inf --
    a NaN slipping into the dump silently corrupts the bootstrap metric.
    """
    if list(df.columns) != SCHEMA:
        raise ValueError(f"columns {list(df.columns)} != schema {SCHEMA}")
    if df["logit"].dtype != np.float64:
        raise TypeError(f"logit must be float64, got {df['logit'].dtype}")

    logit = df["logit"].to_numpy()
    if not np.isfinite(logit).all():
        n_bad = int((~np.isfinite(logit)).sum())
        raise AssertionError(f"{n_bad} non-finite logit values (NaN/inf); refusing to write")

    # Atomic write: a kill mid-write (SIGKILL, container death, host reset) must
    # never leave a partial/zero-byte parquet at `path` that a resume check
    # could mistake for a completed dump. Temp name in the same directory, so
    # os.replace is a same-filesystem rename rather than a copy.
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    durable_replace(tmp, path)


def read_predictions(path: str) -> pd.DataFrame:
    """Read a prediction parquet back, re-asserting the schema."""
    df = pd.read_parquet(path)
    if list(df.columns) != SCHEMA:
        raise ValueError(f"columns {list(df.columns)} != schema {SCHEMA}")
    return df
