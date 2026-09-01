"""Partial-AUC surrogate loss, restricted to the low-FPR region the challenge
actually scores.

REVISED 2026-08-07 (step 6 of that night's brief) -- seven changes from the
first draft, each with its own rationale:

  1. EXPLICIT fp32. This project has a bf16 defect on record that collapsed
     617 logits to 53 distinct values (see src/model.py's fp32
     classifier-head docstring). Every computation here now promotes half
     precision inputs to fp32 and runs under ``autocast(enabled=False)``, so
     an enclosing bf16/fp16 autocast cannot quantise pairwise margins.
     fp64 inputs are left at fp64 (gradcheck needs that).

  2. SQUARED HINGE, not sigmoid, as the training surrogate. Sigmoid
     saturates on the largest margins -- which are precisely the hardest,
     most-misranked pairs -- so its gradient vanishes exactly where the FPR
     is being set. The squared hinge ``relu(margin - (s_pos - s_neg))^2``
     has gradient GROWING with margin violation instead. The sigmoid
     relaxation survives only inside ``partial_auc_surrogate`` (an
     ESTIMATOR of pAUC, still useful for monitoring and for the exact
     equivalence tests against src/metrics.py); the LOSS no longer uses it.

  3. WARMUP SCHEDULE (``pauc_beta_schedule``). At initialisation the ranking
     is random, so "the top beta of negatives" is a random subset and most
     of the batch receives no gradient. The schedule returns None (=plain
     BCE) for the first ``warmup_epochs`` epochs, then anneals beta linearly
     from 1.0 (all pairs -- full-AUC hinge) down to the target across
     ``anneal_epochs`` epochs, and holds the target thereafter.

  4. TWO-SIDED RESTRICTION. The metric's threshold is set by the BOTTOM
     decile of positives, not by all of them -- but a strict decile spends
     ~19% of its gradient on the 3-of-bottom-16 positives that every arm on
     record fails to rescue. The loss therefore restricts to the bottom
     QUARTILE of positives (configurable, ``pos_fraction`` default 0.25) x
     the top ``beta`` of negatives. ``partial_auc_surrogate`` keeps
     ``pos_fraction=1.0`` as its default so the exact-equivalence identity
     with src/metrics.partial_auc still holds where the tests pin it.

  5. ceil(), NOT round(), on both selection counts, minimum 1 -- a small
     batch can never select zero pairs and silently return zero loss. All
     knobs (beta, pos_fraction, temperature, margin, lambda, schedule) are
     src/config.py fields now, not literals.

  6. COMBINED LOSS: ``lambda * BCE + (1 - lambda) * pAUC``, lambda default
     0.3, wired in src/train.py behind ``pauc_enabled`` (default False --
     nothing changes for any existing config). Pure pAUC (lambda=0) leaves
     the logit scale arbitrary; the fusion audit's member SDs run 2.70-5.66,
     so a lambda=0 member fused by raw-logit averaging would be dominated or
     drowned -- if lambda=0 is ever swept, z-score fusion for that member is
     MANDATORY, decided now, before any result exists to bias the call.

  7. TESTS (tests/test_losses.py): monotonicity (raise one positive's
     score, the loss must fall) and torch.autograd.gradcheck on an fp64
     batch -- the earlier suite checked gradient FINITENESS only, which a
     sign error would pass.

SELECTION IS HARD, VALUES ARE SOFT -- unchanged from the first draft:
``torch.topk`` picks WHICH pairs enter the sum (non-differentiable, same
trade-off as hard-negative mining everywhere else in ML); gradient flows
through every selected score's VALUE.

DEGENERATE BATCHES: a batch with zero positives or zero negatives carries no
ranking information; both functions return an exact 0.0 that stays connected
to the autograd graph, so ``.backward()`` on a combined loss never sees NaN
and contributes exactly zero gradient from this term.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

DEFAULT_MAX_FPR = 0.15        # matches src.metrics.PAUC_MAX_FPR
DEFAULT_POS_FRACTION = 0.25   # bottom quartile of positives (see point 4)
DEFAULT_TEMPERATURE = 1.0     # surrogate (estimator) only -- not the loss
DEFAULT_MARGIN = 1.0
DEFAULT_LAMBDA = 0.3          # weight on BCE in the combined loss
DEFAULT_WARMUP_EPOCHS = 5     # plain BCE for epochs [0, warmup)
DEFAULT_ANNEAL_EPOCHS = 5     # beta 1.0 -> target across these epochs

_LOW_PRECISION = (torch.float16, torch.bfloat16)


def _at_least_fp32(t: torch.Tensor) -> torch.Tensor:
    """fp16/bf16 -> fp32; fp32 and fp64 pass through unchanged (gradcheck
    runs at fp64 and must stay there)."""
    return t.float() if t.dtype in _LOW_PRECISION else t


def _zero_connected(*tensors: torch.Tensor) -> torch.Tensor:
    """An exact 0.0 that still participates in autograd for every tensor
    passed in, so calling .backward() through it is always safe -- never
    "does not require grad", never NaN, regardless of which inputs are
    empty."""
    total = None
    for t in tensors:
        contribution = 0.0 * t.sum()
        total = contribution if total is None else total + contribution
    return total


def _hard_indicator(diff: torch.Tensor) -> torch.Tensor:
    """Exact step function with the standard rank-statistic tie convention:
    1 if diff > 0, 0 if diff < 0, 0.5 if diff == 0 exactly. No gradient."""
    return torch.where(
        diff > 0, torch.ones_like(diff),
        torch.where(diff < 0, torch.zeros_like(diff),
                   torch.full_like(diff, 0.5)),
    )


def select_restricted(
    logits_pos: torch.Tensor,
    logits_neg: torch.Tensor,
    max_fpr: float,
    pos_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The two-sided restricted sets: bottom-``pos_fraction`` of positives x
    top-``max_fpr`` of negatives. Counts use ceil() with a floor of 1 --
    round() could select zero negatives on a small batch and return a zero
    loss with zero gradient, silently."""
    n_neg = logits_neg.numel()
    n_pos = logits_pos.numel()
    m_neg = min(n_neg, max(1, math.ceil(max_fpr * n_neg)))
    k_pos = min(n_pos, max(1, math.ceil(pos_fraction * n_pos)))
    top_neg, _ = torch.topk(logits_neg, k=m_neg, largest=True, sorted=False)
    bottom_pos, _ = torch.topk(logits_pos, k=k_pos, largest=False, sorted=False)
    return bottom_pos, top_neg


def partial_auc_surrogate(
    logits_pos: torch.Tensor,
    logits_neg: torch.Tensor,
    max_fpr: float = DEFAULT_MAX_FPR,
    temperature: float = DEFAULT_TEMPERATURE,
    hard: bool = False,
    pos_fraction: float = 1.0,
) -> torch.Tensor:
    """Restricted-area ESTIMATE of pAUC[0, max_fpr] -- an estimator for
    monitoring and testing, no longer the training loss (see module
    docstring, point 2).

    With the default ``pos_fraction=1.0`` and ``hard=True`` this is EXACTLY
    ``src.metrics.partial_auc(...)["raw"]`` whenever max_fpr * n_neg is an
    integer and no tie straddles the top-m boundary (tests pin this).
    ``hard=False`` uses the sigmoid relaxation -- smooth, 0.5 at ties, and
    converging to the step function as temperature -> 0.
    """
    if logits_pos.numel() == 0 or logits_neg.numel() == 0:
        return _zero_connected(logits_pos, logits_neg)

    device_type = logits_pos.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        pos = _at_least_fp32(logits_pos)
        neg = _at_least_fp32(logits_neg)
        sel_pos, sel_neg = select_restricted(pos, neg, max_fpr, pos_fraction)
        diff = sel_pos.unsqueeze(-1) - sel_neg.unsqueeze(0)  # (k_pos, m_neg)
        if hard:
            indicator = _hard_indicator(diff)
        else:
            temp = max(temperature, 1e-8)  # guard against a caller passing 0
            indicator = torch.sigmoid(diff / temp)
        return indicator.mean()


def partial_auc_loss(
    logits_pos: torch.Tensor,
    logits_neg: torch.Tensor,
    max_fpr: float = DEFAULT_MAX_FPR,
    pos_fraction: float = DEFAULT_POS_FRACTION,
    margin: float = DEFAULT_MARGIN,
    temperature: float = DEFAULT_TEMPERATURE,  # accepted for BC; unused by hinge
) -> torch.Tensor:
    """Squared-hinge ranking loss over the two-sided restricted pair grid --
    something to MINIMISE; 0.0 iff every selected positive outranks every
    selected negative by at least ``margin``.

    Per selected pair: ``relu(margin - (s_pos - s_neg)) ** 2``. Gradient
    GROWS linearly with margin violation, so the worst-ranked pairs -- the
    ones actually setting FPR@90R -- get the largest updates, the exact
    opposite of the saturating sigmoid this replaces.

    REDUCTION: mean over the selected pair grid. The grid is rectangular
    (the same top-m negatives are compared against every selected positive),
    so mean-over-pairs and mean-over-positives-of-per-positive-means are
    identical here; there is no hidden factor-of-k rescaling of the
    learning rate either way.

    ``temperature`` is accepted and ignored: the hinge has no temperature.
    It stays in the signature so a config sweeping the surrogate's
    temperature does not crash against the loss.

    A degenerate batch (no positives, or no negatives) contributes exactly
    ``0.0`` -- there is no ranking information in a batch that lacks one
    class, and "contributes nothing, exactly zero gradient" is the one
    behaviour that cannot bias training either way (same convention as
    src.metrics._degenerate).
    """
    del temperature
    if logits_pos.numel() == 0 or logits_neg.numel() == 0:
        return _zero_connected(logits_pos, logits_neg)

    device_type = logits_pos.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        pos = _at_least_fp32(logits_pos)
        neg = _at_least_fp32(logits_neg)
        sel_pos, sel_neg = select_restricted(pos, neg, max_fpr, pos_fraction)
        diff = sel_pos.unsqueeze(-1) - sel_neg.unsqueeze(0)  # (k_pos, m_neg)
        violation = torch.relu(margin - diff)
        return (violation ** 2).mean()


def pauc_beta_schedule(
    epoch: int,
    target_beta: float = DEFAULT_MAX_FPR,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
    anneal_epochs: int = DEFAULT_ANNEAL_EPOCHS,
) -> Optional[float]:
    """Cold-start schedule (module docstring, point 3).

    Returns None for epochs [0, warmup_epochs) -- the caller must train on
    plain BCE there, because a random initial ranking makes the top-beta
    negative selection a random subset and starves most of the batch of
    gradient. From ``warmup_epochs`` onward, beta anneals LINEARLY from 1.0
    (first annealing epoch) to ``target_beta`` (last annealing epoch,
    ``warmup_epochs + anneal_epochs - 1``), then holds the target. With the
    defaults (5 and 5): BCE for epochs 0-4, beta 1.0 at epoch 5, the target
    0.15 at epoch 9 and thereafter.
    """
    if epoch < warmup_epochs:
        return None
    if anneal_epochs <= 1:
        return target_beta
    t = min(1.0, (epoch - warmup_epochs) / (anneal_epochs - 1))
    return 1.0 + t * (target_beta - 1.0)
