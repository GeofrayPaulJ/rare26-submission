"""Dataset, crop modes and the RAM cache.

The dataset yields ``(tensor, label_int, filepath)``.

RAM CACHE -- load-bearing, not an optimisation.
The images sit on an NTFS volume bind-mounted into a Linux container. Random
small-file reads across that boundary starve the GPU. So on first construction
we decode every image exactly once, apply the crop, resize to a fixed
``cache_size`` square, and hold the result in a *process-shared* uint8 tensor
(N x cache_size x cache_size x 3). Every epoch after the first reads from RAM;
DataLoader workers share the one tensor (``share_memory_``) rather than copying
it. The per-item work left at read time is: final resize -> augment -> to tensor.

Per-item augmentation is seeded from (config.seed, current epoch, item index)
via ``BarrettDataset.set_epoch`` / ``_rng``, so it is reproducible for a given
epoch and resume-safe, but not frozen across epochs the way (seed, index)
alone would be.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .augment import RedactionDist, apply as apply_aug, load_redaction_dist
from .config import Config
from .seeding import seed_worker

logger = logging.getLogger(__name__)

# cv2 spins up its own thread pool; inside a DataLoader worker that oversubscribes
# the CPU. One thread per worker is faster and keeps resize deterministic.
cv2.setNumThreads(0)

LABEL_MAP = {"non-dysplastic": 0, "neoplasia": 1}
_DEGENERATE_PX = 64  # inner boxes smaller than this are logged as suspicious

# geometry columns pulled from the manifest, indexed by filepath
_GEOM_COLS = [
    "centre", "class_label", "group_id_v2", "visibility",
    "width", "height",
    "inner_left", "inner_top", "inner_right", "inner_bottom",
    "fov_centre_x", "fov_centre_y", "fov_radius",
]


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to (size, size). INTER_AREA for downscale (the common case here),
    INTER_LINEAR for the rare upscale. Deterministic either way."""
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (size <= h and size <= w) else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def _crop_inscribed_square(img: np.ndarray, row: dict) -> np.ndarray:
    """Crop the square inscribed in the FOV circle (inner_* box). Resizing this
    to a fixed side normalises the per-hospital radius difference automatically."""
    h, w = img.shape[:2]
    il = int(max(0, min(w, row["inner_left"])))
    it = int(max(0, min(h, row["inner_top"])))
    ir = int(max(0, min(w, row["inner_right"])))
    ib = int(max(0, min(h, row["inner_bottom"])))
    if ir - il < _DEGENERATE_PX or ib - it < _DEGENERATE_PX:
        logger.warning(
            "degenerate inner box (%dx%d) for %s; using it as-is",
            ir - il, ib - it, row.get("filepath", "?"),
        )
    if ir <= il or ib <= it:
        # fully collapsed box: fall back to the whole frame rather than crash
        return img
    return img[it:ib, il:ir, :]


def _crop_fov_bbox(img: np.ndarray, row: dict) -> np.ndarray:
    """Crop the circle's bounding box (clipped to the image), then pad to square
    with black. Keeps peripheral tissue and the scope border."""
    h, w = img.shape[:2]
    cx, cy, r = row["fov_centre_x"], row["fov_centre_y"], row["fov_radius"]
    x0 = int(max(0, np.floor(cx - r)))
    y0 = int(max(0, np.floor(cy - r)))
    x1 = int(min(w, np.ceil(cx + r)))
    y1 = int(min(h, np.ceil(cy + r)))
    if x1 <= x0 or y1 <= y0:
        return img
    crop = img[y0:y1, x0:x1, :]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    canvas = np.zeros((side, side, 3), dtype=crop.dtype)  # black pad
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw, :] = crop
    return canvas


_CROP_FUNCS = {
    "inscribed_square": _crop_inscribed_square,
    "fov_bbox": _crop_fov_bbox,
}


def _decode(path: str) -> np.ndarray:
    """Decode an image to HWC uint8 RGB. cv2.imread on a bytes buffer avoids
    cv2's own path handling and returns BGR, which we flip to RGB."""
    with open(path, "rb") as fh:
        buf = np.frombuffer(fh.read(), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"failed to decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class BarrettDataset(Dataset):
    """One split's worth of images. Set ``build_cache=True`` to decode into the
    shared RAM tensor up front; pass an existing ``cache`` tensor to reuse one.
    With ``cache=None`` and ``build_cache=False`` the dataset decodes on the fly
    (used to prove the cache is byte-identical to fresh reads)."""

    def __init__(
        self,
        filepaths: List[str],
        manifest: pd.DataFrame,
        config: Config,
        train: bool,
        image_root: Optional[str] = None,
        redaction_dist: Optional[RedactionDist] = None,
        cache: Optional[torch.Tensor] = None,
        build_cache: bool = False,
    ) -> None:
        self.filepaths = list(filepaths)
        self.cfg = config
        self.train = train
        self.image_root = image_root if image_root is not None else config.image_root
        self.crop_fn = _CROP_FUNCS[config.crop_mode]

        # per-filepath geometry / labels
        idx = manifest.set_index("filepath")
        missing = [f for f in self.filepaths if f not in idx.index]
        if missing:
            raise KeyError(f"{len(missing)} filepaths absent from manifest, e.g. {missing[0]}")
        self.rows = [
            {**idx.loc[fp, _GEOM_COLS].to_dict(), "filepath": fp}
            for fp in self.filepaths
        ]

        # redaction distribution: only needed when black boxes are on and training
        if redaction_dist is None and train and config.aug.random_black_boxes:
            redaction_dist = load_redaction_dist(config.redaction_stats)
        self.redaction_dist = redaction_dist

        self.cache = cache
        self.cache_build_seconds: Optional[float] = None
        if build_cache and cache is None:
            self.cache = self._build_cache()

        # Epoch counter for the per-item RNG, held in OS-level shared memory
        # (see set_epoch below for why a plain attribute does not work here).
        self._epoch = torch.multiprocessing.Value("l", 0)

    # -- epoch ----------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Update the epoch the per-item RNG derives from.

        With ``persistent_workers=True`` the DataLoader forks/spawns its worker
        processes once and keeps them alive across every subsequent epoch. Each
        worker holds its own COPY of this dataset object from that one-time
        handoff, so a plain ``self.epoch = epoch`` set in the main process after
        construction is invisible to the workers -- they keep reading whatever
        value they were copied with. ``torch.multiprocessing.Value`` instead
        allocates the counter in real OS-level shared memory (mmap), so every
        process holding a reference to it -- main and workers alike -- reads
        and writes the same backing memory. That is what makes this call
        actually reach workers that were already alive before it was made.
        """
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    # -- cache --------------------------------------------------------------
    def _cache_crop(self, i: int) -> np.ndarray:
        """The canonical cached value for item i: decoded, cropped, resized to
        cache_size, HWC uint8. Both the RAM path and the on-the-fly path go
        through here, which is what makes them byte-identical."""
        row = self.rows[i]
        img = _decode(os.path.join(self.image_root, row["filepath"]))
        img = self.crop_fn(img, row)
        return _resize_square(img, self.cfg.cache_size)

    def _build_cache(self) -> torch.Tensor:
        n, s = len(self.filepaths), self.cfg.cache_size
        cache = torch.empty((n, s, s, 3), dtype=torch.uint8)
        t0 = time.time()
        # decoding every image off the bind mount is the longest silent stretch
        # of a run; show it rather than look hung
        for i in tqdm(
            range(n),
            desc=f"cache {'train' if self.train else 'val'}",
            unit="img", leave=False, dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        ):
            cache[i] = torch.from_numpy(self._cache_crop(i))
        self.cache_build_seconds = time.time() - t0
        cache.share_memory_()  # place in shared memory so workers do not copy it
        logger.info(
            "cache built: %d imgs, %.1f s, %.2f GB resident",
            n, self.cache_build_seconds, cache.numel() / 1e9,
        )
        return cache

    def cache_bytes(self) -> int:
        return 0 if self.cache is None else self.cache.numel() * self.cache.element_size()

    # -- item ---------------------------------------------------------------
    def _rng(self, i: int) -> np.random.Generator:
        """Per-item RNG seeded from (base seed, epoch, index). Independent of
        worker id. A fixed (seed, epoch, index) always gives byte-identical
        augmentation -- which is what makes a resumed run retrace an
        uninterrupted one -- but the epoch component means two DIFFERENT
        epochs draw different augmentations for the same image, instead of the
        dataset being effectively fixed after the first epoch. The epoch comes
        from shared memory (see set_epoch) so this reads the current value
        even inside a persistent worker process."""
        epoch = self._epoch.value
        return np.random.default_rng([self.cfg.seed, epoch, i])

    def __len__(self) -> int:
        return len(self.filepaths)

    def labels(self) -> np.ndarray:
        """Integer label per item, in dataset order. Used to build the weighted
        sampler and to line the prediction dump up with the manifest."""
        return np.asarray(
            [LABEL_MAP[r["class_label"]] for r in self.rows], dtype=np.int64
        )

    def centres(self) -> np.ndarray:
        """Centre string per item, in dataset order. Used to build the
        centre x class sampler. Kept as strings rather than codes: the manifest
        is the authority on how many centres exist, and an integer encoding here
        would silently invent an ordering."""
        return np.asarray([str(r["centre"]) for r in self.rows], dtype=object)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        if self.cache is not None:
            cache_uint8 = self.cache[i].numpy()
        else:
            cache_uint8 = self._cache_crop(i)

        img = _resize_square(cache_uint8, self.cfg.image_size)
        arr = apply_aug(
            img, self.cfg.aug, self._rng(i), self.train, self.redaction_dist
        )
        tensor = torch.from_numpy(arr)  # CHW float32
        label = LABEL_MAP[self.rows[i]["class_label"]]
        return tensor, label, self.rows[i]["filepath"]


def make_weighted_sampler(
    dataset: BarrettDataset,
    generator: torch.Generator,
) -> torch.utils.data.WeightedRandomSampler:
    """Inverse-class-frequency sampler, with replacement, one epoch = len(dataset).

    Weight of an item is 1/count(its class), so the two classes are drawn in
    equal expectation despite the ~5% positive rate. Drawing WITH replacement
    over an epoch of the original length means a positive is seen many times per
    epoch and most negatives are not seen at all -- that is the intended trade,
    and it is why the loss below carries no additional class weighting. Doing
    both would double-count the imbalance correction.
    """
    labels = dataset.labels()
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
    if (counts == 0).any():
        present = np.flatnonzero(counts)
        raise ValueError(
            f"weighted sampler needs both classes present; found only {present.tolist()}"
        )
    weights = (1.0 / counts)[labels]
    logger.info(
        "weighted sampler: counts=%s, weight ratio pos/neg=%.1f",
        counts.astype(int).tolist(), counts[0] / counts[1],
    )
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def make_balanced_centre_class_sampler(
    dataset: BarrettDataset,
    generator: torch.Generator,
) -> torch.utils.data.WeightedRandomSampler:
    """Sampler balanced on the (centre, class) product, not on class alone.

    WHY THIS EXISTS. In the pooled training set center_2 carries a 12.5%
    neoplasia prior against center_1's 2.6%. Balancing on class alone leaves
    that association intact: a model can lower its training loss by learning
    "this looks like center_2" and it will be rewarded for doing so. Hospital
    identity is therefore a legitimately predictive TRAINING feature and a
    worthless TEST feature, since the twelve evaluation centres are none of
    these. Giving each of the four (centre, class) strata equal total weight
    makes centre and class independent in the sampled stream, so the shortcut
    stops paying.

    WEIGHTING. An item in stratum s gets weight 1/count(s), so every non-empty
    stratum sums to exactly 1.0 -- "equal total weight each". Draws are with
    replacement over an epoch of len(dataset), matching make_weighted_sampler,
    so this changes the composition of an epoch and not its length.

    DEGENERATE CASE -- READ THIS BEFORE INTERPRETING ANY RESULT. When the
    training rows hold only one centre, two of the four strata are empty and
    the weight vector this returns is *element-wise identical* to
    make_weighted_sampler's, because 1/count(centre, class) and 1/count(class)
    are then the same number for every item. The sampler is not approximately
    equivalent in that case, it is exactly equivalent, and training is
    bit-identical. That is precisely the situation on both leave-one-centre-out
    splits, where the training side is single-centre by construction. The
    warning below fires so that fact cannot be discovered only after the GPU
    time has been spent. tests/test_sampler.py pins the identity.
    """
    labels = dataset.labels()
    centres = dataset.centres()
    strata = np.array(
        [f"{c}|{'neoplasia' if y else 'non-dysplastic'}"
         for c, y in zip(centres, labels)],
        dtype=object,
    )
    uniq, inverse = np.unique(strata.astype(str), return_inverse=True)
    counts = np.bincount(inverse, minlength=len(uniq)).astype(np.float64)

    n_centres = len(np.unique(centres.astype(str)))
    n_classes = len(np.unique(labels))
    if n_classes < 2:
        raise ValueError(
            f"balanced_centre_class needs both classes present; found strata "
            f"{uniq.tolist()}"
        )

    weights = (1.0 / counts)[inverse]
    logger.info(
        "balanced_centre_class sampler: %d populated strata of %d possible -- %s",
        len(uniq), n_centres * n_classes,
        ", ".join(f"{u}={int(c)}" for u, c in zip(uniq, counts)),
    )
    if n_centres < 2:
        logger.warning(
            "balanced_centre_class was requested but the training rows hold "
            "ONE centre (%s). Only %d strata are populated, so this sampler is "
            "EXACTLY equivalent to sampler='weighted' and the run will be "
            "bit-identical to the class-balanced baseline. It is not a weaker "
            "version of the intended intervention -- it is the same run.",
            uniq[0].split("|")[0], len(uniq),
        )
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def stratum_draw_counts(
    sampler: torch.utils.data.Sampler,
    dataset: BarrettDataset,
    generator: Optional[torch.Generator] = None,
) -> "OrderedDict[str, int]":
    """Realised (centre, class) composition of ONE epoch drawn from ``sampler``.

    This is the confirmation that the weighting did what it was asked to do,
    rather than what it was believed to do. It draws a real epoch and counts it.

    Pass the sampler's own `generator` and its state is saved and restored
    around the draw, so calling this never perturbs the stream that training
    subsequently consumes -- without that, asking the question would change the
    answer to every later one.
    """
    state = generator.get_state() if generator is not None else None
    try:
        indices = list(sampler)
    finally:
        if state is not None:
            generator.set_state(state)

    labels = dataset.labels()
    centres = dataset.centres()
    counts: "OrderedDict[str, int]" = OrderedDict()
    for i in indices:
        key = f"{centres[i]}|{'neoplasia' if labels[i] else 'non-dysplastic'}"
        counts[key] = counts.get(key, 0) + 1
    return OrderedDict(sorted(counts.items()))


def make_sampler(
    dataset: BarrettDataset,
    config: Config,
    generator: torch.Generator,
) -> torch.utils.data.Sampler:
    """The sampler named by ``config.sampler``, drawing from ``generator``.

    Always returns an explicit sampler object rather than leaning on
    DataLoader's ``shuffle=True``. That matters more than it looks: with
    shuffle=True the implicit RandomSampler draws from the DataLoader's OWN
    generator, and the DataLoader also consumes a value from that generator to
    seed its worker processes -- but only when it actually constructs workers.
    With persistent_workers that happens once, on the first epoch. A resumed run
    constructs its workers on whatever epoch it restarts at, consuming the draw
    at a different point and shifting every subsequent batch order.
    Owning the sampler's generator separately makes the batch order a function
    of (seed, epoch) alone, which is the property tests/test_resume.py pins.
    """
    if config.sampler == "weighted":
        return make_weighted_sampler(dataset, generator)
    if config.sampler == "balanced_centre_class":
        return make_balanced_centre_class_sampler(dataset, generator)
    if config.sampler in ("random", "none"):
        return torch.utils.data.RandomSampler(dataset, generator=generator)
    if config.sampler == "sequential":
        return torch.utils.data.SequentialSampler(dataset)
    raise ValueError(f"unknown sampler {config.sampler!r}")


def make_loader(
    dataset: BarrettDataset,
    config: Config,
    shuffle: bool,
    sampler: Optional[torch.utils.data.Sampler] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader whose ordering is a deterministic function of
    ``config.seed``. Same seed -> same order; different seed -> different order.

    Pass ``generator`` to own the ordering RNG from outside -- the training loop
    re-seeds it per epoch so batch order is a pure function of (seed, epoch),
    which is what makes a resumed run retrace an uninterrupted one. When a
    ``sampler`` is supplied, ``shuffle`` must be False (DataLoader forbids both).
    """
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(config.seed)
    if sampler is not None and shuffle:
        raise ValueError("pass either sampler or shuffle=True, not both")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )
