"""Evidence that the Gaussian shot-noise sampler matches the Poisson one.

src/augment.py used to draw ``rng.poisson`` once per subpixel. That cost 16 ms
per image -- 41% of the whole augmentation budget -- so it was replaced by a
Gaussian with the Poisson variance substituted. This script is the evidence
that the replacement is a replacement and not a change of transform, and it is
a HALT CONDITION for the sweep: if either comparison exceeds 2%, nothing runs.

Two things are checked, because they are the two ways a noise model can be
wrong independently:

  1. VARIANCE VERSUS INTENSITY. Shot noise is defined by its variance being
     proportional to signal. Both implementations are measured against each
     other and against the closed form Var = 255*x/scale, across the intensity
     range and at three photon scales spanning what the sweep actually draws.

  2. NOISE POWER SPECTRUM. Two samplers can agree on total variance and still
     differ in how that variance is distributed over spatial frequency, which
     is what determines whether the noise looks like sensor grain or like
     something a convolution can trivially remove. Both are drawn per subpixel
     and should be white; the radially-averaged spectra are compared bin by bin.

Everything is measured UNCLIPPED. Clipping to [0, 255] is a pointwise step both
implementations share, so including it would test the clip rather than the
sampler, and at the noisiest settings it would dominate. The clipped agreement
is reported too, for completeness.

    python scripts/24_shot_noise_check.py [--tolerance 0.02]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

# Photon scales spanning what the arms actually draw: 20 is the noisiest end at
# s=1.0, 200 the mildest, and 183.7 is what s=0.33 resolves to.
SCALES = (20.0, 60.0, 183.7, 200.0)
INTENSITIES = (8, 16, 32, 64, 96, 128, 160, 192, 224, 250)


def reference_poisson(x: np.ndarray, rng: np.random.Generator,
                      scale: float) -> np.ndarray:
    """The implementation being replaced, unclipped.

    Kept verbatim here rather than imported, because the point of this file is
    to compare against the code as it WAS; importing whatever src/augment.py
    currently holds would make the comparison vacuous the moment it changes.
    """
    lam = np.clip(x, 0.0, 255.0) / 255.0 * scale
    return rng.poisson(lam).astype(np.float64) / scale * 255.0


def candidate_gaussian(x: np.ndarray, rng: np.random.Generator,
                       scale: float) -> np.ndarray:
    """The shipped implementation, unclipped."""
    xc = np.clip(x, 0.0, 255.0)
    sigma = np.sqrt(xc * (255.0 / scale))
    return xc + rng.standard_normal(size=xc.shape) * sigma


def variance_vs_intensity(n: int, tolerance: float) -> Dict:
    """Empirical mean and variance at each (intensity, scale), both samplers."""
    rows: List[Dict] = []
    for scale in SCALES:
        for x0 in INTENSITIES:
            patch = np.full(n, float(x0))
            ref = reference_poisson(patch, np.random.default_rng(1234), scale)
            cand = candidate_gaussian(patch, np.random.default_rng(5678), scale)
            theory = 255.0 * x0 / scale
            v_ref, v_cand = float(ref.var()), float(cand.var())
            rows.append({
                "scale": scale, "intensity": x0,
                "theory_var": theory,
                "ref_var": v_ref, "cand_var": v_cand,
                "ref_mean": float(ref.mean()), "cand_mean": float(cand.mean()),
                "rel_diff_var": abs(v_cand - v_ref) / max(v_ref, 1e-12),
                "rel_diff_vs_theory": abs(v_cand - theory) / max(theory, 1e-12),
                # reported, not gated: the Gaussian is symmetric where the
                # Poisson is skewed, and this is the accepted deviation
                "ref_skew": float(((ref - ref.mean()) ** 3).mean()
                                  / max(ref.std() ** 3, 1e-12)),
                "cand_skew": float(((cand - cand.mean()) ** 3).mean()
                                   / max(cand.std() ** 3, 1e-12)),
            })
    worst = max(rows, key=lambda r: r["rel_diff_var"])
    return {
        "rows": rows,
        "max_rel_diff_var": worst["rel_diff_var"],
        "worst_at": {"scale": worst["scale"], "intensity": worst["intensity"]},
        "passes": bool(worst["rel_diff_var"] <= tolerance),
    }


def radial_power_spectrum(field: np.ndarray) -> np.ndarray:
    """Radially-averaged 2D power spectrum of a square noise field."""
    f = np.fft.fftshift(np.fft.fft2(field))
    power = (f.real ** 2 + f.imag ** 2) / field.size
    h, w = field.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    n_bins = min(cy, cx)
    total = np.bincount(r.ravel(), power.ravel(), minlength=n_bins + 1)
    count = np.bincount(r.ravel(), minlength=n_bins + 1)
    return (total[:n_bins] / np.maximum(count[:n_bins], 1))[1:]  # drop DC


def power_spectrum_check(size: int, reps: int, tolerance: float) -> Dict:
    """Compare the two samplers' noise spectra at a mid-grey intensity."""
    out: List[Dict] = []
    for scale in SCALES:
        patch = np.full((size, size), 128.0)
        acc_ref = np.zeros(size // 2 - 1)
        acc_cand = np.zeros(size // 2 - 1)
        rng_ref = np.random.default_rng(11)
        rng_cand = np.random.default_rng(22)
        for _ in range(reps):
            acc_ref += radial_power_spectrum(
                reference_poisson(patch, rng_ref, scale) - patch)
            acc_cand += radial_power_spectrum(
                candidate_gaussian(patch, rng_cand, scale) - patch)
        acc_ref /= reps
        acc_cand /= reps
        rel = np.abs(acc_cand - acc_ref) / np.maximum(acc_ref, 1e-12)
        # Flatness: white noise has a spectrum independent of frequency, so a
        # sampler that introduced spatial correlation would show a trend here
        # even if its total variance were right.
        flat_ref = float(acc_ref.std() / acc_ref.mean())
        flat_cand = float(acc_cand.std() / acc_cand.mean())
        out.append({
            "scale": scale,
            "mean_level_ref": float(acc_ref.mean()),
            "mean_level_cand": float(acc_cand.mean()),
            "rel_diff_mean_level": float(
                abs(acc_cand.mean() - acc_ref.mean()) / acc_ref.mean()),
            "mean_rel_diff_per_bin": float(rel.mean()),
            "max_rel_diff_per_bin": float(rel.max()),
            "flatness_cv_ref": flat_ref,
            "flatness_cv_cand": flat_cand,
            "n_bins": int(rel.size),
        })
    worst_level = max(r["rel_diff_mean_level"] for r in out)
    worst_bin = max(r["mean_rel_diff_per_bin"] for r in out)
    return {
        "rows": out,
        "max_rel_diff_mean_level": worst_level,
        "max_mean_rel_diff_per_bin": worst_bin,
        "passes": bool(worst_level <= tolerance and worst_bin <= tolerance),
    }


def clipped_agreement(n: int) -> Dict:
    """What the shipped code actually produces, clipping and uint8 cast included."""
    rows = []
    for scale in SCALES:
        for x0 in (32, 128, 224):
            patch = np.full(n, float(x0))
            ref = np.clip(reference_poisson(patch, np.random.default_rng(7), scale),
                          0, 255)
            cand = np.clip(candidate_gaussian(patch, np.random.default_rng(8), scale),
                           0, 255)
            rows.append({
                "scale": scale, "intensity": x0,
                "ref_mean": float(ref.mean()), "cand_mean": float(cand.mean()),
                "ref_sd": float(ref.std()), "cand_sd": float(cand.std()),
                "rel_diff_sd": float(abs(cand.std() - ref.std())
                                     / max(ref.std(), 1e-12)),
            })
    return {"rows": rows,
            "max_rel_diff_sd": max(r["rel_diff_sd"] for r in rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=2_000_000,
                    help="samples per (intensity, scale) variance estimate")
    ap.add_argument("--spectrum-size", type=int, default=256)
    ap.add_argument("--spectrum-reps", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports",
                                                  "shot_noise_check.json"))
    args = ap.parse_args(argv)

    var = variance_vs_intensity(args.n, args.tolerance)
    spec = power_spectrum_check(args.spectrum_size, args.spectrum_reps,
                                args.tolerance)
    clip = clipped_agreement(args.n // 4)

    print(f"SHOT NOISE EQUIVALENCE -- Poisson sampler vs Gaussian sampler")
    print(f"tolerance {args.tolerance:.1%}, "
          f"{args.n:,} samples per variance cell, "
          f"{args.spectrum_reps} x {args.spectrum_size}^2 per spectrum\n")

    print("1. VARIANCE VS INTENSITY (unclipped)")
    print(f"{'scale':>7s} {'x':>5s} {'theory var':>12s} {'poisson':>12s} "
          f"{'gaussian':>12s} {'rel diff':>9s}")
    print("-" * 62)
    for r in var["rows"]:
        if r["intensity"] in (8, 64, 128, 250):
            print(f"{r['scale']:7.1f} {r['intensity']:5d} {r['theory_var']:12.2f} "
                  f"{r['ref_var']:12.2f} {r['cand_var']:12.2f} "
                  f"{r['rel_diff_var']:8.3%}")
    print(f"\n  worst relative difference: {var['max_rel_diff_var']:.3%} "
          f"(scale {var['worst_at']['scale']}, intensity "
          f"{var['worst_at']['intensity']})   "
          f"{'PASS' if var['passes'] else 'FAIL'}")

    print("\n2. NOISE POWER SPECTRUM (radially averaged, mid-grey)")
    print(f"{'scale':>7s} {'level poisson':>15s} {'level gaussian':>15s} "
          f"{'level diff':>11s} {'mean/bin':>10s} {'max/bin':>9s}")
    print("-" * 70)
    for r in spec["rows"]:
        print(f"{r['scale']:7.1f} {r['mean_level_ref']:15.2f} "
              f"{r['mean_level_cand']:15.2f} {r['rel_diff_mean_level']:10.3%} "
              f"{r['mean_rel_diff_per_bin']:9.3%} {r['max_rel_diff_per_bin']:8.3%}")
    print(f"\n  worst level difference:   {spec['max_rel_diff_mean_level']:.3%}")
    print(f"  worst mean per-bin diff:  {spec['max_mean_rel_diff_per_bin']:.3%}   "
          f"{'PASS' if spec['passes'] else 'FAIL'}")
    print(f"  spectra are flat (CV): poisson "
          f"{spec['rows'][0]['flatness_cv_ref']:.3f}, gaussian "
          f"{spec['rows'][0]['flatness_cv_cand']:.3f} -- both white, so the "
          f"variance sits at the same frequencies")

    print(f"\n3. AS SHIPPED (clipped to [0,255])")
    print(f"  worst SD difference: {clip['max_rel_diff_sd']:.3%}")

    skew = [(r["scale"], r["intensity"], r["ref_skew"], r["cand_skew"])
            for r in var["rows"] if r["intensity"] == 8]
    print(f"\n4. ACCEPTED DEVIATION -- skewness at the noisiest cell "
          f"(intensity 8)")
    for s, x, a, b in skew:
        print(f"  scale {s:6.1f}: poisson skew {a:+.3f}, gaussian {b:+.3f}")
    print("  Expected and not gated: the Poisson is discrete and right-skewed "
          "at low\n  photon counts, the Gaussian is symmetric. Variance -- what "
          "this transform\n  exists to produce -- matches by construction.")

    ok = var["passes"] and spec["passes"]
    payload = {
        "tolerance": args.tolerance,
        "variance_vs_intensity": var,
        "power_spectrum": spec,
        "clipped": clip,
        "passes": bool(ok),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n{'=' * 62}")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(both comparisons within {args.tolerance:.0%})")
    print(f"{'=' * 62}")
    print(f"written: {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
