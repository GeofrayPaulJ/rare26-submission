"""
12_compression_check.py -- Measure whether de-identification compression artifacts
correlate with class label or hospital in the RARE25 training images.

BACKGROUND: the RARE25 paper states de-identification introduced compression-related
anonymisation artifacts into a SUBSET of training images, absent from test data. If
those artifacts correlate with class, a model can learn "does this image show
compression damage" as a proxy for the label and find nothing to grip on at test time.

LESSON APPLIED FROM 09_imaging_mode.py: that script assumed a two-population split
existed and let Otsu's method manufacture one out of what turned out to be a single
continuous distribution (later confirmed: R/G ratio never dropped below 1.0 anywhere
in the dataset -- there was no second population). This script tests for bimodality
FIRST, with two independent tests that must BOTH agree (Hartigan's dip test AND a
1-vs-2-component Gaussian-mixture BIC comparison with Ashman's D > 2 for separation),
before ever calling an indicator "bimodal". A unimodal result is reported as exactly
that -- not forced into a binary split.

WHAT IT MEASURES (inside the FOV only, sampling the largest square inscribed in a
circle of radius 0.8 * fov_radius -- reusing inner_square() from 05_fov_crop.py so
the sampling geometry is identical to the rest of this pipeline, and centred/sized
to stay clear of the vignette edge):

  1. blockiness_ratio    -- mean |pixel difference| across 8x8 grid boundaries,
                            aligned to the RAW image's absolute pixel grid (not the
                            crop's local grid -- compression happens before any
                            crop we do), divided by the same at non-boundary
                            positions. Meaningfully > 1 indicates JPEG block edges.
  2. unique_colour_frac  -- distinct RGB values in the sampled region, as a
                            fraction of sampled pixels. Lossy requantization
                            reduces this (posterization); also reported as a raw
                            unique_colour_count.
  3. high_freq_energy_ratio -- FFT power spectrum (Hann-windowed) of the sampled
                            patch; fraction of power at radius > 0.5x Nyquist.
  4. dct_comb_score      -- the image is split into its own native 8x8 blocks
                            (grid-aligned to the raw image, as in #1); the AC
                            coefficient at position (0,1) is collected across all
                            blocks and histogrammed. Quantisation produces comb-like
                            periodic gaps in that histogram; this score is the
                            largest non-DC peak in the histogram's own FFT,
                            normalised by total non-DC magnitude. Higher = more
                            comb-like = more evidence of block quantisation.

Correlations between all four are reported plainly -- if they disagree, that means
they are not measuring one common "artifact score" and none is treated as such.

Part 3 answers the question that actually matters regardless of Part 2's outcome:
do these four continuous indicators, fed into a logistic regression evaluated on
the EXISTING grouped fold_r0 splits (no new splits made), predict class label or
hospital above chance (AUC > 0.5)? That is the leak that would hurt at test time.

Read-only: 00_source and manifests/rare25_folds.csv are never modified. Output is a
new file, manifests/compression_check.csv, plus two visual montages. Deterministic
(fixed random_state everywhere) and idempotent.

USAGE:
    python 12_compression_check.py
    python 12_compression_check.py --workers 8 --hf-cutoff 0.5
"""

from __future__ import annotations

import argparse
import importlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
MANIFESTS = ROOT / "manifests"
SRC = ROOT / "00_source"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

INDICATORS = ["blockiness_ratio", "unique_colour_frac", "high_freq_energy_ratio", "dct_comb_score"]

# 8x8 orthonormal DCT-II basis matrix, built via scipy (not hand-derived) so the
# per-block transform below is guaranteed consistent with scipy's own convention.
_DCT8 = dctn(np.eye(8), axes=0, norm="ortho")


def get_roi(arr: np.ndarray, cx: float, cy: float, radius: float, frac: float = 0.8):
    """Largest axis-aligned square inside a circle of radius frac*radius, reusing
    05_fov_crop.inner_square. Returns (gray, rgb, left, top) or None if too small."""
    fov05 = importlib.import_module("05_fov_crop")
    h, w = arr.shape[:2]
    left, top, right, bottom = fov05.inner_square(cx, cy, radius * frac, w, h)
    if right - left < 32 or bottom - top < 32:
        return None
    rgb = arr[top:bottom, left:right]
    gray = rgb.mean(axis=2).astype(np.float64)
    return gray, rgb, left, top


def blockiness_ratio(gray: np.ndarray, left: int, top: int) -> float:
    h, w = gray.shape
    if h < 16 or w < 16:
        return float("nan")

    hdiff = np.abs(np.diff(gray, axis=1))  # (h, w-1)
    vdiff = np.abs(np.diff(gray, axis=0))  # (h-1, w)

    col_abs = np.arange(w - 1) + left
    row_abs = np.arange(h - 1) + top
    h_boundary = (col_abs + 1) % 8 == 0
    v_boundary = (row_abs + 1) % 8 == 0

    boundary_vals = np.concatenate([hdiff[:, h_boundary].ravel(), vdiff[v_boundary, :].ravel()])
    nonboundary_vals = np.concatenate([hdiff[:, ~h_boundary].ravel(), vdiff[~v_boundary, :].ravel()])

    nb_mean = nonboundary_vals.mean()
    if nb_mean <= 1e-9:
        return float("nan")
    return float(boundary_vals.mean() / nb_mean)


def unique_colour_stats(rgb: np.ndarray) -> tuple[int, float]:
    flat = rgb.reshape(-1, 3).astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    n_unique = int(np.unique(packed).size)
    return n_unique, n_unique / packed.size


def high_freq_energy_ratio(gray: np.ndarray, cutoff: float = 0.5) -> float:
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    g = (gray - gray.mean()) * window
    spectrum = np.fft.fftshift(np.fft.fft2(g))
    power = np.abs(spectrum) ** 2

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt(((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2)

    total = power.sum()
    if total <= 0:
        return float("nan")
    return float(power[radius > cutoff].sum() / total)


def dct_comb_score(gray: np.ndarray, left: int, top: int, coeff=(0, 1)) -> float:
    h, w = gray.shape
    # Align blocks to the RAW image's absolute 8-pixel grid, not the crop's local grid.
    x_offset = (-left) % 8
    y_offset = (-top) % 8
    g = gray[y_offset:, x_offset:]
    bh, bw = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    if bh < 8 or bw < 8:
        return float("nan")
    g = g[:bh, :bw]

    blocks = g.reshape(bh // 8, 8, bw // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    if blocks.shape[0] < 32:
        return float("nan")

    d = _DCT8 @ blocks @ _DCT8.T
    coeffs = d[:, coeff[0], coeff[1]]

    hist, _ = np.histogram(coeffs, bins=64)
    hist = hist.astype(np.float64) - hist.mean()
    spec = np.abs(np.fft.rfft(hist))[1:]  # drop DC bin
    total = spec.sum()
    if total <= 1e-9:
        return 0.0
    return float(spec.max() / total)


def process_one(args: tuple) -> dict:
    filepath, cx, cy, radius, hf_cutoff = args
    row = {"filepath": filepath}
    row.update({k: float("nan") for k in INDICATORS})
    row["unique_colour_count"] = None
    row["roi_error"] = ""
    try:
        with Image.open(SRC / filepath) as img:
            arr = np.asarray(img.convert("RGB"))
        roi = get_roi(arr, cx, cy, radius, frac=0.8)
        if roi is None:
            row["roi_error"] = "ROI too small"
            return row
        gray, rgb, left, top = roi

        row["blockiness_ratio"] = blockiness_ratio(gray, left, top)
        n_unique, frac_unique = unique_colour_stats(rgb)
        row["unique_colour_count"] = n_unique
        row["unique_colour_frac"] = frac_unique
        row["high_freq_energy_ratio"] = high_freq_energy_ratio(gray, cutoff=hf_cutoff)
        row["dct_comb_score"] = dct_comb_score(gray, left, top)
    except Exception as exc:  # noqa: BLE001
        row["roi_error"] = str(exc)
    return row


def ascii_histogram(values: np.ndarray, bins: int = 30, width: int = 40) -> str:
    values = values[~np.isnan(values)]
    hist, edges = np.histogram(values, bins=bins)
    peak = hist.max() if hist.max() > 0 else 1
    lines = []
    for count, lo in zip(hist, edges[:-1]):
        bar = "#" * int(width * count / peak)
        lines.append(f"  {lo:12.5f} {bar:<{width}s} {count:5d}")
    return "\n".join(lines)


def ashmans_d(means: np.ndarray, stds: np.ndarray) -> float:
    return float(abs(means[0] - means[1]) * np.sqrt(2) / np.sqrt(stds[0] ** 2 + stds[1] ** 2))


def bimodality_test(values: np.ndarray, name: str) -> dict:
    values = values[~np.isnan(values)]
    out = {"indicator": name, "n": len(values)}

    dip_bimodal = None
    try:
        import diptest as _dt
        dip, pval = _dt.diptest(values)
        out["dip_statistic"], out["dip_pvalue"] = float(dip), float(pval)
        dip_bimodal = pval < 0.05
        out["dip_method"] = "hartigan_dip"
    except ImportError:
        out["dip_method"] = "unavailable"

    X = values.reshape(-1, 1)
    gm1 = GaussianMixture(n_components=1, random_state=0).fit(X)
    gm2 = GaussianMixture(n_components=2, random_state=0).fit(X)
    bic1, bic2 = gm1.bic(X), gm2.bic(X)
    out["bic_1_component"], out["bic_2_component"] = float(bic1), float(bic2)
    out["bic_delta_favoring_2"] = float(bic1 - bic2)

    means = gm2.means_.ravel()
    stds = np.sqrt(gm2.covariances_.ravel())
    d = ashmans_d(means, stds)
    out["ashmans_d"] = d
    out["gmm2_means"] = means.tolist()
    out["gmm2_stds"] = stds.tolist()

    gmm_bimodal = (bic1 - bic2 > 10) and (d > 2)
    out["gmm_bimodal"] = bool(gmm_bimodal)

    if dip_bimodal is None:
        final = gmm_bimodal
    else:
        final = bool(dip_bimodal and gmm_bimodal)  # both tests must agree -- conservative
    out["bimodal"] = final
    out["_gmm2"] = gm2
    return out


def cv_auc(df: pd.DataFrame, feature_cols: list[str], y: np.ndarray, fold_col: str = "fold_r0") -> np.ndarray:
    aucs = []
    for k in sorted(df[fold_col].unique()):
        train_mask = (df[fold_col] != k).to_numpy()
        test_mask = (df[fold_col] == k).to_numpy()
        y_train, y_test = y[train_mask], y[test_mask]
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue
        scaler = StandardScaler().fit(df.loc[train_mask, feature_cols].values)
        X_train = scaler.transform(df.loc[train_mask, feature_cols].values)
        X_test = scaler.transform(df.loc[test_mask, feature_cols].values)
        clf = LogisticRegression(max_iter=1000, random_state=0).fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))
    return np.array(aucs)


def build_montage(df: pd.DataFrame, score_col: str, ascending: bool, out_path: Path,
                   patch_size: int = 160, title: str = "") -> None:
    subset = df.sort_values(score_col, ascending=ascending).head(16).reset_index(drop=True)
    cols, rows = 4, 4
    label_h, pad, header_h = 16, 4, 22
    cell_w, cell_h = patch_size + pad, patch_size + label_h + pad
    canvas = Image.new("RGB", (cols * cell_w + pad, header_h + rows * cell_h + pad), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 4), title, fill=(255, 255, 255), font=font)

    for idx, row in subset.iterrows():
        r, c = divmod(idx, cols)
        with Image.open(SRC / row["filepath"]) as img:
            img = img.convert("RGB")
            w, h = img.size
            left = int(np.clip(row["fov_centre_x"] - patch_size / 2, 0, max(0, w - patch_size)))
            top = int(np.clip(row["fov_centre_y"] - patch_size / 2, 0, max(0, h - patch_size)))
            patch = img.crop((left, top, left + patch_size, top + patch_size))
        x0 = pad + c * cell_w
        y0 = header_h + pad + r * cell_h
        canvas.paste(patch, (x0, y0))
        draw.text((x0, y0 + patch_size + 1), f"{row[score_col]:.4f}", fill=(255, 255, 0), font=font)

    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hf-cutoff", type=float, default=0.5,
                     help="High-frequency-energy cutoff as a fraction of Nyquist radius")
    args = ap.parse_args()

    src_path = MANIFESTS / "rare25_folds.csv"
    if not src_path.exists():
        raise SystemExit(f"Missing {src_path}. Run the RARE25 pipeline (00-07) first.")
    manifest = pd.read_csv(src_path)  # read-only, never written back to

    jobs = [
        (row.filepath, row.fov_centre_x, row.fov_centre_y, row.fov_radius, args.hf_cutoff)
        for row in manifest.itertuples()
    ]
    print(f"Measuring compression indicators for {len(jobs)} images "
          f"(ROI = square inscribed in a circle at 0.8x fitted FOV radius)...")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in tqdm(pool.map(process_one, jobs, chunksize=32), total=len(jobs)):
            results.append(res)

    ind_df = pd.DataFrame(results)
    df = manifest.merge(ind_df, on="filepath", how="left")

    n_errors = int((df["roi_error"].fillna("") != "").sum())
    print(f"\nDone. ROI/measurement failures: {n_errors}")
    if n_errors:
        for _, row in df[df["roi_error"].fillna("") != ""].head(20).iterrows():
            print(f"  {row['filepath']}: {row['roi_error']}")

    clean = df.dropna(subset=INDICATORS).reset_index(drop=True)
    print(f"Usable rows for analysis: {len(clean)} / {len(df)}")

    # ================= PART 1: correlation between indicators =================
    print("\n" + "=" * 70)
    print("PART 1 -- correlation between the four indicators")
    print("=" * 70)
    corr = clean[INDICATORS].corr()
    print(corr.round(3).to_string())
    off_diag = corr.where(~np.eye(len(INDICATORS), dtype=bool))
    max_abs = off_diag.abs().max().max()
    if max_abs < 0.3:
        print(f"\nAll pairwise correlations are weak (max |r|={max_abs:.2f}). These four "
              f"indicators are NOT measuring one common underlying thing -- no single "
              f"'artifact score' is justified. Each is reported and tested independently.")
    else:
        print(f"\nStrongest pairwise correlation: |r|={max_abs:.2f}.")

    # ================= PART 2: bimodality =================
    print("\n" + "=" * 70)
    print("PART 2 -- is there a genuinely distinct artifacted subset?")
    print("=" * 70)
    bimodal_flags = {}
    for name in INDICATORS:
        vals = clean[name].to_numpy()
        print(f"\n--- {name} ---")
        print(ascii_histogram(vals))
        bt = bimodality_test(vals, name)
        bimodal_flags[name] = bt
        dip_str = (f"dip={bt.get('dip_statistic', float('nan')):.4f} "
                   f"p={bt.get('dip_pvalue', float('nan')):.4f}" if "dip_statistic" in bt else "dip test unavailable")
        print(f"  {dip_str}")
        print(f"  GMM: BIC(1-comp)={bt['bic_1_component']:.1f}  BIC(2-comp)={bt['bic_2_component']:.1f}  "
              f"delta(favoring 2)={bt['bic_delta_favoring_2']:.1f}  Ashman's D={bt['ashmans_d']:.2f}")
        verdict = "BIMODAL" if bt["bimodal"] else "UNIMODAL"
        print(f"  VERDICT: {verdict}"
              + ("" if bt["bimodal"] else " -- no distinct artifacted subset detectable on this indicator."))

    any_bimodal = any(bt["bimodal"] for bt in bimodal_flags.values())
    for name, bt in bimodal_flags.items():
        if bt["bimodal"]:
            gm2 = bt["_gmm2"]
            labels = gm2.predict(clean[[name]].values)
            # Orient labels so 0 = lower-mean component, 1 = higher-mean component.
            order = np.argsort(gm2.means_.ravel())
            remap = {order[0]: 0, order[1]: 1}
            clean[f"{name}_group"] = [remap[l] for l in labels]
    if not any_bimodal:
        print("\nNo indicator is genuinely bimodal by both tests. Per the brief, this is a "
              "valid finding: there is no detectable distinct 'artifacted' subset in these "
              "training images by any of the four measures. Part 3 proceeds using the "
              "continuous indicator values (no binary artifact label is manufactured).")

    # ================= PART 3: does compression predict class / hospital? =================
    print("\n" + "=" * 70)
    print("PART 3 -- can the four indicators alone predict class label or hospital?")
    print("(Logistic regression, features = the 4 continuous indicators only, "
          "evaluated on the existing fold_r0 group-respecting 5-fold split.)")
    print("=" * 70)

    y_class = (clean["class_label"] == "neoplasia").astype(int).to_numpy()
    aucs_class = cv_auc(clean, INDICATORS, y_class, fold_col="fold_r0")
    print(f"\nTarget = class label (neoplasia vs non-dysplastic):")
    print(f"  fold AUCs: {np.round(aucs_class, 4).tolist()}")
    print(f"  mean AUC = {aucs_class.mean():.4f}  (std={aucs_class.std():.4f}, "
          f"range=[{aucs_class.min():.4f}, {aucs_class.max():.4f}])")
    if aucs_class.mean() > 0.6:
        print("  -> MEANINGFULLY ABOVE 0.5: compression artifacts carry class information "
              "that will not exist at test time. This is a shortcut.")
    else:
        print("  -> near 0.5: no material class leak via these indicators.")

    y_centre = (clean["centre"] == "center_2").astype(int).to_numpy()
    aucs_centre = cv_auc(clean, INDICATORS, y_centre, fold_col="fold_r0")
    print(f"\nTarget = hospital (center_1 vs center_2):")
    print(f"  fold AUCs: {np.round(aucs_centre, 4).tolist()}")
    print(f"  mean AUC = {aucs_centre.mean():.4f}  (std={aucs_centre.std():.4f}, "
          f"range=[{aucs_centre.min():.4f}, {aucs_centre.max():.4f}])")
    if aucs_centre.mean() > 0.6:
        print("  -> MEANINGFULLY ABOVE 0.5: compression indicators predict hospital of origin.")
    else:
        print("  -> near 0.5: no material hospital leak via these indicators.")

    # ================= Indicator breakdown by class / hospital =================
    print("\n" + "=" * 70)
    print("Indicator breakdown by class and hospital")
    print("=" * 70)
    for group_col in ["class_label", "centre"]:
        print(f"\n--- grouped by {group_col} ---")
        stats = clean.groupby(group_col)[INDICATORS].agg(["mean", "median"])
        print(stats.round(5).to_string())
        groups = clean[group_col].unique()
        if len(groups) == 2:
            g1, g2 = groups
            for name in INDICATORS:
                v1 = clean.loc[clean[group_col] == g1, name]
                v2 = clean.loc[clean[group_col] == g2, name]
                lo, hi = max(v1.min(), v2.min()), min(v1.max(), v2.max())
                overlap = max(0.0, hi - lo)
                print(f"  {name:24s} {g1} range=[{v1.min():.4f},{v1.max():.4f}]  "
                      f"{g2} range=[{v2.min():.4f},{v2.max():.4f}]  "
                      f"{'OVERLAP' if overlap > 0 else 'DISJOINT'}")

    # ================= PART 4: montages =================
    print("\n" + "=" * 70)
    print("PART 4 -- visual montages of highest/lowest blockiness")
    print("=" * 70)
    top_path = MANIFESTS / "compression_check_blockiness_top16.png"
    bot_path = MANIFESTS / "compression_check_blockiness_bottom16.png"
    build_montage(clean, "blockiness_ratio", ascending=False, out_path=top_path,
                  title="Highest blockiness_ratio (16)")
    build_montage(clean, "blockiness_ratio", ascending=True, out_path=bot_path,
                  title="Lowest blockiness_ratio (16)")
    print(f"Saved: {top_path}")
    print(f"Saved: {bot_path}")

    # ================= Save output =================
    out_csv = MANIFESTS / "compression_check.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWritten: {out_csv}")

    # ================= Plain-language summary =================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Distinct artifacted subset detectable: {'YES' if any_bimodal else 'NO'}"
          + (f" (bimodal on: {[n for n, b in bimodal_flags.items() if b['bimodal']]})" if any_bimodal else ""))
    print(f"Compression indicators predict class label: "
          f"{'YES' if aucs_class.mean() > 0.6 else 'NO'} (mean AUC={aucs_class.mean():.3f})")
    print(f"Compression indicators predict hospital: "
          f"{'YES' if aucs_centre.mean() > 0.6 else 'NO'} (mean AUC={aucs_centre.mean():.3f})")


if __name__ == "__main__":
    main()
