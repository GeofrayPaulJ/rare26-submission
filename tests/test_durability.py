"""Durable-write ordering guards.

WHY THIS FILE EXISTS. On 2026-08-03 an unattended run halted because
save_checkpoint() did tmp-write + os.replace() with no fsync. os.replace is
atomic against a killed PROCESS -- the page cache outlives it -- but not
against a killed MACHINE: the rename can reach disk while the bytes it points
at have not, leaving a correctly-named, unreadable file. That produced
"PytorchStreamReader failed reading zip archive: failed finding central
directory" on the next resume and stopped the sweep at 297/330 units.

A SIGKILL test does not catch this (page cache survives), and neither does
`docker kill` (the Docker VM's cache survives the container). Rather than rely
on a crash test that cannot actually evict dirty pages, these tests assert the
SYSCALL ORDERING that makes the guarantee, which is the thing that was
actually missing:

    fsync(file content)  ->  os.replace  ->  fsync(parent directory)

Each half matters and they are not interchangeable. fsync on the file makes
the BYTES durable; fsync on the parent directory makes the NAME that points at
them durable. Skipping the second one leaves a window where the new content is
safe on disk under a name that did not survive.
"""
from __future__ import annotations

import os
import stat

import numpy as np
import pandas as pd
import pytest

from src.io import (
    SCHEMA, build_frame, durable_replace, write_predictions, write_text_durable,
)


class SyscallRecorder:
    """Record the order of os.fsync / os.replace, tagging each fsync by
    whether its fd refers to a directory or a regular file."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, str]] = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            try:
                kind = ("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
            except OSError:
                kind = "unknown"
            self.calls.append(("fsync", kind))
            return real_fsync(fd)

        def replace(src, dst, **kw):
            self.calls.append(("replace", os.path.basename(str(dst))))
            return real_replace(src, dst, **kw)

        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(os, "replace", replace)

    @property
    def kinds(self) -> list[str]:
        return [c[0] for c in self.calls]

    def assert_durable_sequence(self) -> None:
        """fsync(file) strictly before replace, fsync(dir) strictly after."""
        assert ("replace" in self.kinds), f"no os.replace happened: {self.calls}"
        i = self.kinds.index("replace")
        before = self.calls[:i]
        after = self.calls[i + 1:]
        assert any(c == ("fsync", "file") for c in before), (
            f"content was NOT fsynced before the rename -- this is the exact "
            f"2026-08-03 bug. calls={self.calls}")
        assert any(c == ("fsync", "dir") for c in after), (
            f"parent directory was NOT fsynced after the rename, so the new "
            f"name may not survive a host reset. calls={self.calls}")


def _frame(n=8):
    return build_frame(
        filepath=[f"img_{i}.png" for i in range(n)],
        centre=["center_1"] * n,
        class_label=["neoplasia"] * n,
        label_int=[0, 1] * (n // 2),
        visibility=[None] * n,
        group_id_v2=["g"] * n,
        repeat=0, fold=0, seed=0,
        logit=np.linspace(-2.0, 2.0, n).astype(np.float64),
    )


def test_durable_replace_orders_its_syscalls(tmp_path, monkeypatch):
    rec = SyscallRecorder(monkeypatch)
    final = os.path.join(tmp_path, "thing.bin")
    tmp = final + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(b"payload")
    durable_replace(tmp, final)
    rec.assert_durable_sequence()
    assert open(final, "rb").read() == b"payload"


def test_write_predictions_is_durable(tmp_path, monkeypatch):
    rec = SyscallRecorder(monkeypatch)
    path = os.path.join(tmp_path, "val_r0_f0_s0.parquet")
    write_predictions(_frame(), path)
    rec.assert_durable_sequence()
    assert list(pd.read_parquet(path).columns) == SCHEMA


def test_write_text_durable_is_durable(tmp_path, monkeypatch):
    rec = SyscallRecorder(monkeypatch)
    path = os.path.join(tmp_path, "reports", "x.md")
    write_text_durable(path, "# hello\n")
    rec.assert_durable_sequence()
    assert open(path).read() == "# hello\n"


def test_save_checkpoint_is_durable(tmp_path, monkeypatch):
    """The exact call site whose missing fsync halted the 2026-08-03 run."""
    torch = pytest.importorskip("torch")
    from src.train import save_checkpoint

    rec = SyscallRecorder(monkeypatch)
    path = os.path.join(tmp_path, "last.pt")
    save_checkpoint(path, {"epoch": 3, "w": torch.zeros(4)})
    rec.assert_durable_sequence()
    assert torch.load(path, map_location="cpu", weights_only=False)["epoch"] == 3


def test_write_json_atomic_is_durable(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from run_cv import write_json_atomic

    rec = SyscallRecorder(monkeypatch)
    path = os.path.join(tmp_path, "run_index.json")
    write_json_atomic(path, {"a": 1})
    rec.assert_durable_sequence()


def test_no_torn_file_is_ever_visible_at_the_final_path(tmp_path):
    """The property all of the above exist to provide: the final path only
    ever holds a complete file, because content is only ever built under the
    .tmp name and moved by a single atomic rename."""
    final = os.path.join(tmp_path, "val.parquet")
    write_predictions(_frame(), final)
    first = pd.read_parquet(final)
    # a second write of different content must never expose an intermediate
    df2 = _frame(16)
    write_predictions(df2, final)
    assert len(pd.read_parquet(final)) == 16
    assert len(first) == 8
    assert not os.path.exists(final + ".tmp"), "tmp file leaked into the run dir"
