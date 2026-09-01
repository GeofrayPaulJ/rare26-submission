"""Realised per-stratum draw counts for one epoch, under each sampler.

Confirms the centre x class weighting does what it was asked to do, by drawing
a real epoch and counting it rather than by inspecting the weight vector and
believing the arithmetic.

Prints three split families side by side, because the contrast is the point:

  pooled CV (r0 f0)   both centres in training -- four strata, and the
                      difference between the two samplers is visible
  holdout_center_1    training side is center_2 only
  holdout_center_2    training side is center_1 only

On the two LOCO splits the sampler has only two strata to work with and is
exactly the class-balanced baseline. That is why the factorial screen has two
identifiable arms on LOCO and not four -- see reports/dg_screen.md section 1.

    python scripts/21_sampler_demo.py
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import Config  # noqa: E402
from src.data import BarrettDataset, make_sampler, stratum_draw_counts  # noqa: E402
from src.folds import get_holdout_split, get_split  # noqa: E402

SAMPLERS = ("weighted", "balanced_centre_class")


def report(title: str, filepaths, manifest: pd.DataFrame, cfg: Config,
           seed: int) -> None:
    # No image decoding: the samplers read labels and centres only, so building
    # a RAM cache of several thousand frames to count draws would be waste.
    ds = BarrettDataset(filepaths, manifest, cfg, train=True, cache=None,
                        build_cache=False)

    print(f"\n{title}")
    print(f"  {len(ds)} training rows")

    composition = {}
    for name in SAMPLERS:
        gen = torch.Generator()
        gen.manual_seed(seed)
        sampler = make_sampler(ds, Config(**{**cfg.to_dict(), "sampler": name,
                                             "aug": cfg.aug}), gen)
        composition[name] = stratum_draw_counts(sampler, ds, gen)

    strata = sorted(set().union(*(c.keys() for c in composition.values())))
    width = max(len(s) for s in strata) + 2
    header = f"  {'stratum':{width}s}" + "".join(f"{n:>28s}" for n in SAMPLERS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in strata:
        row = f"  {s:{width}s}"
        for name in SAMPLERS:
            n = composition[name].get(s, 0)
            total = sum(composition[name].values())
            row += f"{n:>20d} ({100 * n / total:4.1f}%)"
        print(row)

    identical = composition[SAMPLERS[0]] == composition[SAMPLERS[1]]
    n_centres = len(set(k.split("|")[0] for k in strata))
    if identical:
        print(f"  -> IDENTICAL draws ({n_centres} centre in training): the "
              f"centre x class sampler IS the class-balanced sampler here")
    else:
        print(f"  -> the two samplers differ ({n_centres} centres in training)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs",
                                                     "dg_c0.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    manifest_path = os.path.join(REPO_ROOT, cfg.manifest)
    manifest = pd.read_csv(manifest_path)

    train, _ = get_split(0, 0, manifest_path)
    report("POOLED CV (repeat 0, fold 0) -- both centres in training",
           train, manifest, cfg, args.seed)

    for centre in (1, 2):
        train, _ = get_holdout_split(centre, manifest_path)
        report(f"LOCO holdout_center_{centre} -- single-centre training side",
               train, manifest, cfg, args.seed)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
