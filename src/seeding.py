"""Deterministic seeding from a single integer.

`seed_everything` seeds python, numpy, torch (CPU + CUDA) and configures the
determinism knobs that do not cost throughput, then returns a dict describing
everything it set so a run can log its exact reproducibility state.

`seed_worker` is the DataLoader ``worker_init_fn``: it re-derives each worker's
seeds from torch's per-worker base seed so worker RNGs are distinct yet
reproducible.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """Seed every RNG from `seed`. Returns the settings applied, for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    settings: Dict[str, Any] = {
        "seed": seed,
        "PYTHONHASHSEED": str(seed),
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": True,
    }

    if deterministic:
        # cuBLAS needs this set for deterministic GEMMs under determinism mode.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: keep training even if an op has no deterministic kernel,
        # rather than crippling throughput or crashing the run.
        torch.use_deterministic_algorithms(True, warn_only=True)
        settings.update(
            torch_deterministic_algorithms=True,
            cudnn_deterministic=True,
            cudnn_benchmark=False,
            CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        )

    return settings


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn. torch gives each worker a distinct base seed;
    derive numpy/random from it so every worker RNG is distinct and reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
