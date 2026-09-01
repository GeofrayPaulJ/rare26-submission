"""Tests 4-5: split hygiene -- no leakage by filepath or group."""
import pandas as pd
import pytest

from src.folds import get_split
from tests.conftest import MANIFEST


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(MANIFEST)


def test_no_filepath_in_train_and_val(df):
    """Test 4: no filepath appears in both train and val, any (repeat, fold)."""
    for repeat in range(10):
        for fold in range(5):
            train, val = get_split(repeat, fold, MANIFEST)
            overlap = set(train) & set(val)
            assert not overlap, (
                f"repeat {repeat} fold {fold}: {len(overlap)} filepaths in both sides"
            )
            # sanity: -1 rows are excluded entirely
            usable = (df[f"fold_r{repeat}"] != -1).sum()
            assert len(train) + len(val) == usable


def test_no_group_spans_train_and_val(df):
    """Test 5: no group_id_v2 spans train and val, all 10 repeats x 5 folds."""
    g = df.set_index("filepath")["group_id_v2"]
    for repeat in range(10):
        for fold in range(5):
            train, val = get_split(repeat, fold, MANIFEST)
            train_groups = set(g.loc[train])
            val_groups = set(g.loc[val])
            shared = train_groups & val_groups
            assert not shared, (
                f"repeat {repeat} fold {fold}: {len(shared)} group_id_v2 span both sides"
            )
