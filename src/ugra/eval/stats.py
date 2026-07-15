"""Statistical testing protocol of Section 4.1.

Paired bootstrap over test regions of the per-region mean across planner
seeds (no seed pooling), Holm correction across the tested family, and
Wilson binomial intervals for coverage comparisons.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def paired_bootstrap_pvalue(losses_a: Sequence[float], losses_b: Sequence[float],
                            n_boot: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for mean(losses_a) != mean(losses_b), paired.

    Inputs are per-region losses already averaged across seeds.
    """
    a = np.asarray(losses_a, dtype=float)
    b = np.asarray(losses_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must align")
    differences = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_boot, len(differences)))
    boot_means = differences[indices].mean(axis=1)
    p_low = float(np.mean(boot_means >= 0.0))
    p_high = float(np.mean(boot_means <= 0.0))
    return max(min(1.0, 2.0 * min(p_low, p_high)), 1.0 / n_boot)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def wilson_interval(successes: int, total: int,
                    confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (coverage CIs)."""
    if total == 0:
        return 0.0, 1.0
    from scipy.stats import norm

    z = norm.ppf(0.5 + confidence / 2.0)
    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)
