"""Calibrated accept gate: first-crossing pricing and the fixed-sequence
scan of Section 3.4 (Eqs. 6-11).

Calibration episodes run gate-free and log the per-step running scores and
working hypotheses; the deployed episode under any threshold lambda is that
log truncated at its first step with q_t <= lambda. One gate-free pass
therefore prices every candidate threshold at once:

    Q_i        = min_t q_{i,t}                    (record score)
    t_i(lam)   = first step with q_t <= lam       (crossing step)
    L_i(lam)   = clipped CER of the hypothesis released at t_i(lam)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..metrics.edit import cer
from ..types import Trajectory
from .bounds import BOUNDS


@dataclass(frozen=True)
class PricedTrajectory:
    """Per-region quantities on a fixed threshold grid."""

    region_id: str
    record_score: float          # Q_i
    losses: np.ndarray           # L_i(lambda) on the grid; NaN where Q_i > lambda
    crossing_cost: np.ndarray    # cumulative cost at the crossing step; NaN likewise


def crossing_records(trajectory: Trajectory) -> List[Tuple[float, str, float]]:
    """Strict prefix minima of the score sequence, in episode order.

    Each element is (q, working_text, cumulative_cost) at a step where the
    running score reached a new record low; the first crossing for any
    threshold lambda is the earliest record with q <= lambda.
    """
    records: List[Tuple[float, str, float]] = []
    best = math.inf
    for step in trajectory.steps:
        if step.q < best:
            best = step.q
            records.append((step.q, step.working_text, step.cumulative_cost))
    return records


def price_trajectory(trajectory: Trajectory, grid: np.ndarray) -> PricedTrajectory:
    """Evaluate L_i(lambda) for every grid point from one gate-free log."""
    if trajectory.reference is None:
        raise ValueError("pricing requires a reference transcription")
    records = crossing_records(trajectory)
    losses = np.full(len(grid), np.nan)
    costs = np.full(len(grid), np.nan)
    # records are in decreasing q; the earliest record with q <= lambda wins.
    for q, text, cost in records:
        mask = np.isnan(losses) & (grid >= q)
        if mask.any():
            losses[mask] = cer(text, trajectory.reference)
            costs[mask] = cost
    return PricedTrajectory(
        region_id=trajectory.region_id,
        record_score=records[-1][0] if records else math.inf,
        losses=losses,
        crossing_cost=costs,
    )


@dataclass(frozen=True)
class GateResult:
    """Outcome of the fixed-sequence calibration scan (Eq. 11)."""

    lambda_hat: Optional[float]  # None: no grid point certifies -> abstain everywhere
    grid: np.ndarray
    n_accept: np.ndarray         # n(lambda)
    risk_hat: np.ndarray         # empirical selective risk R^(lambda)
    risk_ucb: np.ndarray         # upper confidence bound R^+_delta(lambda)


def calibrate(
    priced: Sequence[PricedTrajectory],
    alpha: float,
    delta: float,
    grid: Sequence[float],
    bound: str = "bentkus",
    min_accept: Optional[int] = None,
) -> GateResult:
    """Fixed-sequence scan from the conservative end (Eqs. 9-11).

    The scan certifies a prefix of the grid; lambda_hat is the largest grid
    point whose entire scanned prefix carries UCB <= alpha, and None when
    the first scanned point already fails. No multiplicity correction is
    required (Appendix A, first-unsafe-index argument).

    ``min_accept`` implements the preregistered scan-start floor discussed
    in Section 3.4: the scan begins at the smallest grid point whose
    acceptance count clears the floor. The rule is measurable in the
    scores alone (never the losses), so the fixed-sequence argument is
    untouched; with the default None, the strict Eq. (11) convention
    applies and an empty bottom grid cell vetoes all certification.
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must lie in (0, 1)")
    ucb_fn = BOUNDS[bound]
    grid_arr = np.asarray(sorted(grid), dtype=float)
    loss_matrix = np.vstack([p.losses for p in priced])  # regions x grid
    accepted = ~np.isnan(loss_matrix)
    n_accept = accepted.sum(axis=0)
    sums = np.nansum(np.where(accepted, loss_matrix, 0.0), axis=0)
    risk_hat = np.where(n_accept > 0, sums / np.maximum(n_accept, 1), np.nan)
    risk_ucb = np.array(
        [ucb_fn(float(risk_hat[j]), int(n_accept[j]), delta) if n_accept[j] > 0 else math.inf
         for j in range(len(grid_arr))]
    )
    start = 0
    if min_accept is not None:
        eligible = np.nonzero(n_accept >= max(min_accept, 1))[0]
        if eligible.size == 0:
            return GateResult(lambda_hat=None, grid=grid_arr, n_accept=n_accept,
                              risk_hat=risk_hat, risk_ucb=risk_ucb)
        start = int(eligible[0])
    lambda_hat: Optional[float] = None
    for j in range(start, len(grid_arr)):
        if risk_ucb[j] <= alpha:
            lambda_hat = float(grid_arr[j])
        else:
            break  # fixed-sequence: the first failure ends the certified prefix
    return GateResult(lambda_hat=lambda_hat, grid=grid_arr, n_accept=n_accept,
                      risk_hat=risk_hat, risk_ucb=risk_ucb)
