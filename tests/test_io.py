"""Test 7: io round-trip preserves logit at full float64 precision."""
import os

import numpy as np
import pytest

from src.io import build_frame, read_predictions, write_predictions


def _frame(logits):
    n = len(logits)
    return build_frame(
        filepath=[f"center_1/x/{i}.png" for i in range(n)],
        centre=["center_1"] * n,
        class_label=["non-dysplastic"] * n,
        label_int=[0] * n,
        visibility=[None] * n,
        group_id_v2=list(range(n)),
        repeat=3,
        fold=2,
        seed=0,
        logit=logits,
    )


def test_logit_roundtrip_exact(tmp_path):
    # values chosen to expose any float32 truncation / rounding
    logits = np.array(
        [
            -37.123456789012345,
            0.0,
            1e-12,
            123456.7890123456,
            -1e-8,
            np.nextafter(1.0, 2.0),
            np.pi,
        ],
        dtype=np.float64,
    )
    df = _frame(logits)
    assert df["logit"].dtype == np.float64

    path = os.path.join(tmp_path, "r3_f2_s0.parquet")
    write_predictions(df, path)
    back = read_predictions(path)

    assert back["logit"].dtype == np.float64
    # bitwise equality, not approximate
    assert (back["logit"].to_numpy() == logits).all()


def test_write_rejects_non_finite(tmp_path):
    for bad in (np.nan, np.inf, -np.inf):
        df = _frame(np.array([0.1, bad, 0.2], dtype=np.float64))
        with pytest.raises(AssertionError):
            write_predictions(df, os.path.join(tmp_path, "bad.parquet"))
