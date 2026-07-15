"""Selective metrics of Section 4.1: test-side risk-coverage sweep
(Eq. test-metrics), empirical validity V-hat (Eq. validity), mean cost.

Test episodes are logged gate-free exactly as calibration episodes are,
so one pass prices the whole sweep.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..calibration.gate import PricedTrajectory, calibrate


def risk_coverage_curves(priced: Sequence[PricedTrajectory],
                         grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """R-hat_test(lambda) with the max(n, 1) guard, and coverage phi-hat."""
    loss_matrix = np.vstack([p.losses for p in priced])
    accepted = ~np.isnan(loss_matrix)
    n_accept = accepted.sum(axis=0)
    sums = np.nansum(np.where(accepted, loss_matrix, 0.0), axis=0)
    risk = sums / np.maximum(n_accept, 1)
    coverage = n_accept / loss_matrix.shape[0]
    return risk, coverage


def realized_at(priced: Sequence[PricedTrajectory], lam: Optional[float],
                grid: np.ndarray) -> Tuple[float, float]:
    """Realized (risk, coverage) at a calibrated threshold; abstain-everywhere
    (lam None) realizes zero risk vacuously at zero coverage."""
    if lam is None:
        return 0.0, 0.0
    j = int(np.searchsorted(grid, lam, side="right") - 1)
    risk, coverage = risk_coverage_curves(priced, grid)
    return float(risk[j]), float(coverage[j])


def validity_hat(priced: Sequence[PricedTrajectory], alpha: float, delta: float,
                 grid: np.ndarray, n_resamples: int = 100,
                 calibration_fraction: float = 0.5,
                 seed: int = 0, bound: str = "bentkus") -> float:
    """V-hat: fraction of calibration/test resplits whose realized selective
    risk stays below alpha (checks Proposition 1 in frequency)."""
    rng = np.random.default_rng(seed)
    pool: List[PricedTrajectory] = list(priced)
    n_calibration = int(len(pool) * calibration_fraction)
    hits = 0
    for _ in range(n_resamples):
        order = rng.permutation(len(pool))
        calibration = [pool[i] for i in order[:n_calibration]]
        test = [pool[i] for i in order[n_calibration:]]
        gate = calibrate(calibration, alpha, delta, grid, bound=bound)
        risk, _ = realized_at(test, gate.lambda_hat, grid)
        hits += int(risk <= alpha)
    return hits / n_resamples


def mean_cost(priced: Sequence[PricedTrajectory], lam: Optional[float],
              grid: np.ndarray, full_costs: Sequence[float]) -> float:
    """Mean spent cost per region at threshold lambda: the crossing cost for
    accepted regions, the full episode cost for deferred ones."""
    if lam is None:
        return float(np.mean(full_costs))
    j = int(np.searchsorted(grid, lam, side="right") - 1)
    costs = []
    for p, full in zip(priced, full_costs):
        crossing = p.crossing_cost[j]
        costs.append(full if np.isnan(crossing) else float(crossing))
    return float(np.mean(costs))
