"""Shared helpers for the RARE25/RARE26 Barrett's Esophagus data inventory pipeline.

Design note: the dataset layout is not assumed to be fixed. locate_dataset_root()
discovers class-labeled folders (neo/ndbe or similarly-named) anywhere under a set
of candidate search locations, then infers the dataset root as the common ancestor
of those folders. infer_class_label() matches on keyword substrings rather than
exact folder names so minor naming variants (e.g. "non_dysplastic", "NDBE",
"Neoplasia") still resolve correctly.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

# Expected dataset totals, per the RARE25/RARE26 task brief.
EXPECTED_TOTAL = 3095
EXPECTED_POSITIVE = 158
EXPECTED_NEGATIVE = 2937

# Order matters: negative keywords are checked first since some negative labels
# ("non-neoplastic") can contain a positive substring ("neo").
NEGATIVE_KEYWORDS = [
    "ndbe", "non-dys", "non_dys", "nondys", "non-neo", "non_neo", "nonneo",
    "negative", "benign", "normal", "control",
]
POSITIVE_KEYWORDS = [
    "neo", "dysplasia", "dysplastic", "positive", "hgd", "malignant", "abnormal", "cancer",
]

SKIP_DIR_NAMES = {"manifests", "scripts", "__pycache__", ".git", ".qodo"}


def infer_class_label(folder_name: str) -> str:
    d = folder_name.lower().strip().replace(" ", "")
    dn = d.replace("_", "-")
    for kw in NEGATIVE_KEYWORDS:
        if kw.replace("_", "-") in dn:
            return "non-dysplastic"
    for kw in POSITIVE_KEYWORDS:
        if kw in d:
            return "neoplasia"
    return "unknown"


def _candidate_roots() -> list[Path]:
    cands = [Path.cwd(), Path.cwd() / "00_source", Path("/workspace"), Path("/workspace/00_source")]
    env_root = os.environ.get("DATASET_ROOT")
    if env_root:
        cands.insert(0, Path(env_root))
    seen, out = set(), []
    for c in cands:
        try:
            if c.exists() and c.is_dir() and c.resolve() not in seen:
                seen.add(c.resolve())
                out.append(c)
        except OSError:
            continue
    return out


def locate_dataset_root(explicit: str | None = None, max_depth: int = 6) -> Path:
    """Discover the dataset root by finding class-labeled folders containing images.

    Returns the common ancestor directory of every discovered class folder
    (e.g. the parent of center_1/, center_2/, ... when a centre level exists,
    or the direct parent of neo/ndbe when it doesn't).
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Specified dataset root does not exist: {explicit}")
        return p

    class_dirs: list[tuple[Path, int]] = []
    for base in _candidate_roots():
        base_depth = len(base.resolve().parts)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIR_NAMES]
            depth = len(Path(dirpath).resolve().parts) - base_depth
            if depth > max_depth:
                dirnames[:] = []
                continue
            label = infer_class_label(Path(dirpath).name)
            if label == "unknown":
                continue
            n_images = sum(1 for f in filenames if Path(f).suffix.lower() in IMAGE_EXTENSIONS)
            if n_images > 0:
                class_dirs.append((Path(dirpath).resolve(), n_images))
        if class_dirs:
            break  # first candidate base that yields matches wins

    if not class_dirs:
        searched = ", ".join(str(b) for b in _candidate_roots())
        raise FileNotFoundError(
            f"Could not locate dataset root: no class-labeled image folders found "
            f"under candidate paths [{searched}]. Set DATASET_ROOT env var explicitly."
        )

    # If every class dir shares one grandparent (centre-based layout: root/centre/class/*),
    # and there's more than one such grandparent grouping, root = that shared grandparent.
    grandparents = Counter(p.parent.parent for p, _ in class_dirs)
    parents = Counter(p.parent for p, _ in class_dirs)

    if len(grandparents) == 1:
        return next(iter(grandparents))
    if len(parents) == 1:
        return next(iter(parents))
    # Fallback: shallowest common ancestor across all discovered class dirs.
    return min(parents, key=lambda p: len(p.parts))


def relative_centre_and_class(filepath: Path, dataset_root: Path) -> tuple[str, str]:
    """Derive (centre, class_label) from a file's path relative to dataset_root."""
    rel_parts = filepath.resolve().relative_to(dataset_root.resolve()).parts
    if len(rel_parts) < 2:
        return "unknown", "unknown"
    class_folder = rel_parts[-2]
    class_label = infer_class_label(class_folder)
    centre = rel_parts[-3] if len(rel_parts) >= 3 else "unknown"
    return centre, class_label


def iter_image_files(dataset_root: Path):
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIR_NAMES]
        for f in filenames:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                yield Path(dirpath) / f
