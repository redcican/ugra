"""Upper confidence bounds on the selective risk (Eq. 8 and Lemma 2).

Both bounds take the empirical selective risk over the n(lambda) accepted
calibration regions and return an upper confidence bound at level delta,
valid for i.i.d. losses in [0, 1]. The convention of Appendix A applies:
an empty acceptance set never certifies (UCB = +inf).
"""
from __future__ import annotations

import math

from scipy.stats import binom


def hoeffding_ucb(r_hat: float, n: int, delta: float) -> float:
    """Hoeffding form of Eq. (8): r_hat + sqrt(log(1/delta) / (2 n))."""
    if n <= 0:
        return math.inf
    return r_hat + math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def bentkus_ucb(r_hat: float, n: int, delta: float, tol: float = 1e-9) -> float:
    """Bentkus form (Lemma 2, following Bates et al. 2021).

    inf { r in [0, 1] : BinomCDF(ceil(n * r_hat); n, r) <= delta / e },
    computed by bisection; the CDF is non-increasing in r. Where no r in
    [0, 1] satisfies the condition (for example k = n), the infimum over
    the empty set is +inf; since the loss is bounded by one, we return 1.0,
    a bound that certifies only the vacuous case alpha >= 1.
    """
    if n <= 0:
        return math.inf
    k = math.ceil(n * r_hat)
    threshold = delta / math.e
    if binom.cdf(k, n, 1.0) > threshold:
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if binom.cdf(k, n, mid) <= threshold:
            hi = mid
        else:
            lo = mid
    return hi


BOUNDS = {"hoeffding": hoeffding_ucb, "bentkus": bentkus_ucb}
