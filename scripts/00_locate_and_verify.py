"""Phase 0: locate the dataset root, print its directory tree, and sanity-check
the total image / per-class counts against the expected 3,095 / 158 / 2,937 figures.

Usage: python 00_locate_and_verify.py [--root PATH] [--max-depth N]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import (
    EXPECTED_NEGATIVE,
    EXPECTED_POSITIVE,
    EXPECTED_TOTAL,
    IMAGE_EXTENSIONS,
    SKIP_DIR_NAMES,
    iter_image_files,
    locate_dataset_root,
    relative_centre_and_class,
)


def print_tree(root: Path, max_entries_per_dir: int = 12, max_depth: int = 4) -> None:
    print(f"\n{root}")

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as e:
            print(f"{prefix}[unreadable: {e}]")
            return
        entries = [e for e in entries if not (e.is_dir() and e.name in SKIP_DIR_NAMES)]
        shown = entries[:max_entries_per_dir]
        for i, entry in enumerate(shown):
            is_last = i == len(shown) - 1 and len(entries) <= max_entries_per_dir
            connector = "`-- " if is_last else "|-- "
            if entry.is_dir():
                sub_files = [f for f in entry.iterdir() if f.is_file()] if depth + 1 <= max_depth else []
                img_count = sum(1 for f in sub_files if f.suffix.lower() in IMAGE_EXTENSIONS)
                suffix = f"  ({img_count} images)" if img_count else ""
                print(f"{prefix}{connector}{entry.name}/{suffix}")
                walk(entry, prefix + ("    " if is_last else "|   "), depth + 1)
            else:
                print(f"{prefix}{connector}{entry.name}")
        if len(entries) > max_entries_per_dir:
            print(f"{prefix}`-- ... ({len(entries) - max_entries_per_dir} more entries)")

    walk(root, "", 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="Explicit dataset root (skips auto-discovery)")
    ap.add_argument("--max-depth", type=int, default=4, help="Max depth for tree printing")
    args = ap.parse_args()

    root = locate_dataset_root(args.root)
    print(f"Dataset root located: {root}")

    print("\n=== Directory tree ===")
    print_tree(root, max_depth=args.max_depth)

    print("\n=== Scanning all image files ===")
    class_counts: Counter[str] = Counter()
    centre_counts: Counter[str] = Counter()
    centre_class_counts: Counter[tuple[str, str]] = Counter()
    total = 0
    non_image_files = []

    for img_path in iter_image_files(root):
        total += 1
        centre, class_label = relative_centre_and_class(img_path, root)
        class_counts[class_label] += 1
        centre_counts[centre] += 1
        centre_class_counts[(centre, class_label)] += 1

    # Also report non-image files sitting in the tree (e.g. leftover archives) for visibility.
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS:
            if not any(part in SKIP_DIR_NAMES for part in p.relative_to(root).parts):
                non_image_files.append(p.relative_to(root))

    print(f"\nTotal image files found: {total}")
    print("\nBy class label:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label:20s} {count}")
    print("\nBy centre:")
    for centre, count in sorted(centre_counts.items()):
        print(f"  {centre:20s} {count}")
    print("\nBy centre x class:")
    for (centre, label), count in sorted(centre_class_counts.items()):
        print(f"  {centre:15s} / {label:16s} {count}")

    if non_image_files:
        print(f"\nNon-image files present in tree (excluded from manifest, left untouched):")
        for f in non_image_files:
            print(f"  {f}")

    print("\n=== Expectation check ===")
    pos = class_counts.get("neoplasia", 0)
    neg = class_counts.get("non-dysplastic", 0)
    unknown = class_counts.get("unknown", 0)

    mismatches = []
    if total != EXPECTED_TOTAL:
        mismatches.append(f"TOTAL: expected {EXPECTED_TOTAL}, found {total} (diff {total - EXPECTED_TOTAL:+d})")
    if pos != EXPECTED_POSITIVE:
        mismatches.append(f"POSITIVE/neo: expected {EXPECTED_POSITIVE}, found {pos} (diff {pos - EXPECTED_POSITIVE:+d})")
    if neg != EXPECTED_NEGATIVE:
        mismatches.append(f"NEGATIVE/ndbe: expected {EXPECTED_NEGATIVE}, found {neg} (diff {neg - EXPECTED_NEGATIVE:+d})")
    if unknown:
        mismatches.append(f"UNKNOWN class label: {unknown} images could not be classified from folder structure")

    if mismatches:
        print("MISMATCH DETECTED vs expected counts:")
        for m in mismatches:
            print(f"  - {m}")
    else:
        print(f"OK: counts match expectations exactly (total={total}, positive={pos}, negative={neg}).")


if __name__ == "__main__":
    main()
