"""Shared fixtures. Tests run from the repo root inside the Prometheus container,
so paths are relative to /workspace/RARE26."""
import os
import sys

import pandas as pd
import pytest

# make `import src...` work when pytest is invoked from the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import Config  # noqa: E402

MANIFEST = os.path.join(REPO_ROOT, "manifests", "rare25_folds_v2.csv")
REDACTION = os.path.join(REPO_ROOT, "manifests", "redaction_check.csv")
IMAGE_ROOT = os.path.join(REPO_ROOT, "00_source")


@pytest.fixture(scope="session")
def manifest_path():
    return MANIFEST


@pytest.fixture(scope="session")
def manifest_df():
    return pd.read_csv(MANIFEST)


@pytest.fixture(scope="session")
def sample_filepaths(manifest_df):
    """A small, deterministic subset of usable images to keep tests fast."""
    usable = manifest_df[manifest_df["fold_r0"] != -1]
    # take a spread of both classes so labels are exercised
    pos = usable[usable["class_label"] == "neoplasia"].head(15)
    neg = usable[usable["class_label"] == "non-dysplastic"].head(45)
    return pd.concat([pos, neg])["filepath"].tolist()


def make_config(**overrides):
    """Config for tests: small images, no workers, tuned per test via overrides."""
    base = dict(
        seed=0,
        image_size=128,
        cache_size=160,
        num_workers=0,
        batch_size=8,
        manifest=MANIFEST,
        image_root=IMAGE_ROOT,
        redaction_stats=REDACTION,
    )
    base.update(overrides)
    return Config(**base)
