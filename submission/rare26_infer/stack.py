"""Slice-wise readers for the input image stack.

THIS FILE EXISTS BECAUSE THE TEMPLATE'S READER CANNOT RUN HERE. The reference
implementation does:

    SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path))

which materialises the entire stack as one array. At the scale this challenge
ships -- 25,000 frames of 512x637x3 uint8 -- that single call is 24.5 GB before
PyTorch has allocated anything, against a 32 GB host cap. It does not fail
gracefully; it gets OOM-killed partway through with no useful message.

So nothing here ever holds more than one slice at a time. Both backends expose
the same tiny contract -- ``len()`` and ``read(i) -> HWC uint8 RGB`` -- and both
open their own file handle per worker process, because neither a TiffFile nor a
SimpleITK reader survives being forked into a DataLoader worker.

Two formats are supported because the organisers ship .tiff in the template's
test fixture but .mha is equally valid on the platform, and which one arrives is
not something we get told in advance.
"""
from __future__ import annotations

import logging
import os
from glob import glob
from typing import List, Optional

import numpy as np

from .preprocess import to_rgb_uint8

logger = logging.getLogger("rare26")

TIFF_EXTS = (".tif", ".tiff")
ITK_EXTS = (".mha", ".mhd", ".nii", ".nii.gz", ".nrrd")


def find_stack_file(directory: str) -> str:
    """The one image file in the input directory.

    Grand Challenge mounts exactly one file per image socket, but glob order is
    filesystem order, so it is sorted before picking -- an arbitrary choice
    would be a silent source of run-to-run variation if that ever changed.

    2026-08-10 (D6): the one-file assumption above is exactly that -- an
    assumption, stated in the platform's favour but never verified against
    a multi-case job. D6's experiment showed that if more than one stack
    file ever lands here, everything but the sorted-first file is silently
    discarded. The behaviour is deliberately UNCHANGED (patching in a guess
    about the platform's multi-case contract would be a second unverified
    assumption); instead every candidate is now logged loudly, so a
    production log can prove -- rather than leave unknowable -- whether
    only one file was mounted.
    """
    candidates: List[str] = []
    for ext in TIFF_EXTS + ITK_EXTS:
        candidates.extend(glob(os.path.join(directory, f"*{ext}")))
    if not candidates:
        listing = sorted(os.listdir(directory)) if os.path.isdir(directory) else "<missing>"
        raise FileNotFoundError(
            f"no image stack found in {directory}; directory contains: {listing}"
        )
    candidates = sorted(candidates)
    if len(candidates) > 1:
        logger.error(
            "MULTIPLE stack files found in %s: %s -- processing ONLY %r and "
            "DISCARDING the other %d file(s). If this is a multi-case job, "
            "those cases will receive no predictions. This code assumes one "
            "file per socket; that assumption is now violated.",
            directory, [os.path.basename(c) for c in candidates],
            os.path.basename(candidates[0]), len(candidates) - 1,
        )
    else:
        logger.info("stack file: %s (sole candidate in %s)",
                    os.path.basename(candidates[0]), directory)
    return candidates[0]


class TiffStack:
    """Multi-page TIFF, read one page at a time.

    tifffile parses the page directory up front (cheap -- offsets only, no pixel
    data) and then decodes individual pages on demand, which is exactly the
    access pattern needed. BigTIFF is handled transparently, and it has to be:
    a 25,000-frame uncompressed stack is ~24 GB, well past classic TIFF's 4 GB
    offset ceiling, so any real full-size stack is necessarily BigTIFF.
    """

    def __init__(self, path: str) -> None:
        import tifffile

        self.path = path
        self._tifffile = tifffile
        self._tif = None
        # Count pages once in the parent, then close: the handle itself is not
        # fork-safe and must be reopened lazily inside each worker.
        with tifffile.TiffFile(path) as tif:
            self._n = len(tif.pages)
        self._owner_pid: Optional[int] = None

    def __len__(self) -> int:
        return self._n

    def _handle(self):
        pid = os.getpid()
        if self._tif is None or self._owner_pid != pid:
            self._tif = self._tifffile.TiffFile(self.path)
            self._owner_pid = pid
        return self._tif

    def read(self, i: int) -> np.ndarray:
        page = self._handle().pages[i]
        return to_rgb_uint8(np.asarray(page.asarray()))


class ItkStack:
    """MetaImage/NIfTI/NRRD stack read through SimpleITK's streaming interface.

    ``SetExtractIndex``/``SetExtractSize`` on an ``ImageFileReader`` pull a
    single z-slice off disk without decoding the rest of the volume, which is
    what makes this bounded in memory. ReadImageInformation() parses the header
    only.
    """

    def __init__(self, path: str) -> None:
        import SimpleITK as sitk

        self.path = path
        self._sitk = sitk
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        self._size = list(reader.GetSize())        # (X, Y[, Z])
        self._ncomp = reader.GetNumberOfComponents()
        self._dim = len(self._size)
        self._n = self._size[2] if self._dim >= 3 else 1
        self._reader = None
        self._owner_pid: Optional[int] = None

    def __len__(self) -> int:
        return self._n

    def _handle(self):
        pid = os.getpid()
        if self._reader is None or self._owner_pid != pid:
            r = self._sitk.ImageFileReader()
            r.SetFileName(self.path)
            r.ReadImageInformation()
            self._reader = r
            self._owner_pid = pid
        return self._reader

    def read(self, i: int) -> np.ndarray:
        r = self._handle()
        if self._dim >= 3:
            r.SetExtractIndex([0, 0, int(i)])
            r.SetExtractSize([self._size[0], self._size[1], 1])
        arr = self._sitk.GetArrayFromImage(r.Execute())
        arr = np.squeeze(arr)
        # A 3-channel slice can come back as (Y, X, 3) already, or as (3, Y, X)
        # when the writer stored components on the leading axis.
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        return to_rgb_uint8(arr)


def open_stack(path: str):
    """Pick the reader that matches the file's extension."""
    low = path.lower()
    if low.endswith(TIFF_EXTS):
        return TiffStack(path)
    if low.endswith(ITK_EXTS):
        return ItkStack(path)
    raise ValueError(f"unrecognised stack format: {path}")
