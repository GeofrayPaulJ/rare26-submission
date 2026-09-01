"""STEP 0 (2026-08-07) -- provenance WITHOUT git.

This machine's policy keeps run provenance out of any git history (the
checkout has zero commits, so every provenance.json records git_sha=None and
scripts/run_cv.py's 12-file source_digest is the only attribution that
exists). That digest can DETECT a source change but cannot RECONSTRUCT the
code that produced an old run. This module closes that gap locally:

  1. snapshot(): copy src/, scripts/, configs/ into runs/_src/<tree_digest>/
     (skipped if that digest already has a snapshot). The digest is a sha256
     over EVERY file in those trees (sorted relative paths, contents), not
     run_cv.py's 12-file subset -- the subset provably under-covers (the
     2026-08-03 core-file changes that moved A4's baseline included files
     outside it). The legacy 12-file digest is recorded alongside for
     cross-referencing existing runs.
  2. provenance.json inside each snapshot: per-file sha256, tree digest,
     legacy digest, timestamp, file count.
  3. registry(): reports/source_registry.md -- digest -> timestamp -> first
     run that used it -> file count. Backfilled from every
     runs/*/provenance.json on disk; digests whose source trees no longer
     exist are marked UNKNOWN explicitly rather than silently dropped.

Wired into scripts/run_cv.py's main() (best-effort: a snapshot failure must
never block a run) so every future run start self-archives its exact tree.

    python scripts/51_source_snapshot.py           # snapshot now + rebuild registry
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:                                    # host python may lack it
    def tqdm(it, **kw):                                # type: ignore
        return it

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_ROOT = os.path.join(REPO_ROOT, "runs", "_src")
REGISTRY_MD = os.path.join(REPO_ROOT, "reports", "source_registry.md")
TREES = ("src", "scripts", "configs")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_SUFFIXES = (".pyc", ".pyo")

# Same 12 files scripts/run_cv.py hashes -- kept in sync manually; a drift
# here only weakens the legacy cross-reference, not the snapshot itself.
LEGACY_SOURCES = (
    "src/config.py", "src/data.py", "src/folds.py", "src/io.py",
    "src/metrics.py", "src/model.py", "src/augment.py", "src/seeding.py",
    "src/train.py", "src/evaluate.py", "scripts/run_cv.py", "scripts/08_score.py",
)


def _iter_tree_files() -> List[str]:
    rels: List[str] = []
    for tree in TREES:
        base = os.path.join(REPO_ROOT, tree)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in sorted(filenames):
                if fn.endswith(SKIP_SUFFIXES):
                    continue
                rels.append(os.path.relpath(os.path.join(dirpath, fn), REPO_ROOT)
                            .replace("\\", "/"))
    return sorted(rels)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_digests() -> Tuple[str, Dict[str, str], str]:
    """(tree_digest, per_file, legacy_digest) for the CURRENT working tree."""
    per_file: Dict[str, str] = {}
    combined = hashlib.sha256()
    for rel in _iter_tree_files():
        d = _sha256_file(os.path.join(REPO_ROOT, rel))
        per_file[rel] = d
        combined.update(f"{rel}:{d}\n".encode())
    legacy = hashlib.sha256()
    for rel in LEGACY_SOURCES:
        p = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(p):
            legacy.update(f"{rel}:ABSENT\n".encode())
            continue
        legacy.update(f"{rel}:{_sha256_file(p)}\n".encode())
    return combined.hexdigest(), per_file, legacy.hexdigest()


def snapshot(quiet: bool = False) -> str:
    """Archive the current tree under runs/_src/<tree_digest>/. Idempotent:
    an existing complete snapshot (provenance.json present) is not rewritten."""
    tree_digest, per_file, legacy_digest = compute_digests()
    dest = os.path.join(SNAP_ROOT, tree_digest)
    marker = os.path.join(dest, "provenance.json")
    if os.path.exists(marker):
        if not quiet:
            print(f"[snapshot] {tree_digest[:16]} already archived; skipping copy.")
        return tree_digest

    tmp = dest + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    files = list(per_file)
    it = files if quiet else tqdm(files, desc=f"snapshot {tree_digest[:12]}",
                                  unit="file", file=sys.stderr)
    for rel in it:
        srcp = os.path.join(REPO_ROOT, rel)
        dstp = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dstp), exist_ok=True)
        shutil.copy2(srcp, dstp)

    prov = {
        "tree_sha256": tree_digest,
        "legacy_run_cv_sha256": legacy_digest,
        "archived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": len(per_file),
        "files": per_file,
    }
    with open(os.path.join(tmp, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    if os.path.exists(dest):        # torn earlier attempt (dir but no marker)
        shutil.rmtree(dest)
    os.rename(tmp, dest)
    if not quiet:
        print(f"[snapshot] archived {len(per_file)} files as {tree_digest[:16]}")
    return tree_digest


def _scan_run_digests() -> List[Dict[str, Optional[str]]]:
    """provenance.json lives at UNIT level (runs/<run>/<unit>/provenance.json,
    written by src/train.py per unit) -- scan both that depth and the run
    level for safety. One row per (digest, run): the earliest unit's
    written_utc represents the run."""
    rows: List[Dict[str, Optional[str]]] = []
    runs_root = os.path.join(REPO_ROOT, "runs")
    if not os.path.isdir(runs_root):
        return rows
    for name in sorted(os.listdir(runs_root)):
        if name == "_src" or not os.path.isdir(os.path.join(runs_root, name)):
            continue
        candidates = [os.path.join(runs_root, name, "provenance.json")]
        for unit in sorted(os.listdir(os.path.join(runs_root, name))):
            candidates.append(os.path.join(runs_root, name, unit, "provenance.json"))
        best: Dict[str, Dict[str, Optional[str]]] = {}   # digest -> earliest row
        for p in candidates:
            if not os.path.exists(p):
                continue
            try:
                with open(p) as fh:
                    prov = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            digest = (prov.get("source_sha256") or {}).get("combined_sha256")
            ts = (prov.get("written_utc") or prov.get("created_utc")
                  or prov.get("updated_utc")
                  or time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                   time.gmtime(os.path.getmtime(p))))
            key = digest or "ABSENT"
            if key not in best or (ts or "") < (best[key]["utc"] or ""):
                best[key] = {"legacy_digest": digest, "run": name, "utc": ts}
        rows.extend(best.values())
    return rows


def rebuild_registry() -> None:
    """reports/source_registry.md from run provenance + archived snapshots."""
    rows = _scan_run_digests()
    snaps: Dict[str, Dict] = {}          # legacy digest -> snapshot provenance
    if os.path.isdir(SNAP_ROOT):
        for d in os.listdir(SNAP_ROOT):
            p = os.path.join(SNAP_ROOT, d, "provenance.json")
            if os.path.exists(p):
                with open(p) as fh:
                    sp = json.load(fh)
                snaps[sp["legacy_run_cv_sha256"]] = sp

    # first run per legacy digest, chronological
    first: Dict[str, Dict] = {}
    for r in sorted(rows, key=lambda r: r["utc"] or ""):
        d = r["legacy_digest"] or "ABSENT"
        if d not in first:
            first[d] = r

    L = []
    A = L.append
    A("# Source registry -- code provenance without git")
    A("")
    A(f"Rebuilt {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
     f"`scripts/51_source_snapshot.py`. This machine keeps no git history "
     f"(zero commits; every run's `git_sha` is None), so run attribution "
     f"rests on content digests. Two digests appear below: the LEGACY digest "
     f"is `scripts/run_cv.py`'s 12-file `source_digest()` (what existing "
     f"`runs/*/provenance.json` files actually recorded); the TREE digest is "
     f"the full-tree sha256 (src/ + scripts/ + configs/) used to key the "
     f"snapshots in `runs/_src/`. Digests whose source trees predate this "
     f"mechanism cannot be reconstructed and are marked **UNKNOWN** rather "
     f"than silently omitted -- for those runs, attribution is the digest "
     f"value itself plus whatever reports describe about the code of that "
     f"era, nothing more.")
    A("")
    A("| legacy digest (12-file) | first seen (UTC) | first run using it | snapshot |")
    A("|---|---|---|---|")
    for d, r in sorted(first.items(), key=lambda kv: kv[1]["utc"] or ""):
        if d in snaps:
            sp = snaps[d]
            snap_cell = (f"`runs/_src/{sp['tree_sha256'][:16]}...` "
                         f"({sp['file_count']} files, archived {sp['archived_utc']})")
        else:
            snap_cell = "**UNKNOWN** -- tree predates snapshotting, not reconstructible"
        A(f"| `{(d or 'ABSENT')[:16]}...` | {r['utc']} | `runs/{r['run']}` | {snap_cell} |")
    A("")
    A(f"Runs scanned: {len(rows)}; distinct digests: {len(first)}; "
     f"archived snapshots: {len(snaps)}.")
    A("")

    os.makedirs(os.path.dirname(REGISTRY_MD), exist_ok=True)
    with open(REGISTRY_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"[registry] written: {REGISTRY_MD} "
         f"({len(first)} digests, {len(snaps)} with snapshots)")


def snapshot_at_run_start() -> None:
    """Called from scripts/run_cv.py main(). Best-effort by contract: a
    provenance failure must never cost a night of GPU time."""
    try:
        snapshot(quiet=True)
        rebuild_registry()
    except Exception as exc:                            # noqa: BLE001
        print(f"[snapshot] non-fatal: {exc}", file=sys.stderr)


def main() -> int:
    snapshot()
    rebuild_registry()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
