"""Unit tests for the concentration bounds (Eq. 8, Lemma 2)."""
import math

import pytest
from scipy.stats import binom

from ugra.calibration.bounds import bentkus_ucb, hoeffding_ucb


@pytest.mark.unit
class TestHoeffding:
    def test_closed_form(self):
        assert hoeffding_ucb(0.02, 200, 0.1) == pytest.approx(
            0.02 + math.sqrt(math.log(10) / 400))

    def test_empty_acceptance_never_certifies(self):
        assert hoeffding_ucb(0.0, 0, 0.1) == math.inf

    def test_monotone_in_n(self):
        assert hoeffding_ucb(0.02, 400, 0.1) < hoeffding_ucb(0.02, 100, 0.1)

    def test_monotone_in_delta(self):
        assert hoeffding_ucb(0.02, 100, 0.2) < hoeffding_ucb(0.02, 100, 0.05)


@pytest.mark.unit
class TestBentkus:
    def test_zero_loss_closed_form(self):
        # k = 0: BinomCDF(0; n, r) = (1-r)^n <= delta/e  =>  r = 1 - (delta/e)^(1/n)
        n, delta = 150, 0.1
        expected = 1.0 - (delta / math.e) ** (1.0 / n)
        assert bentkus_ucb(0.0, n, delta) == pytest.approx(expected, abs=1e-6)

    def test_upper_bounds_the_empirical_risk(self):
        assert bentkus_ucb(0.05, 300, 0.1) > 0.05

    def test_infimum_satisfies_the_condition(self):
        n, delta, r_hat = 250, 0.1, 0.03
        r = bentkus_ucb(r_hat, n, delta)
        assert binom.cdf(math.ceil(n * r_hat), n, r + 1e-6) <= delta / math.e

    def test_degenerate_k_equals_n(self):
        # all accepted losses at 1: no r < 1 can satisfy the condition
        assert bentkus_ucb(1.0, 50, 0.1) == 1.0

    def test_empty_acceptance_never_certifies(self):
        assert bentkus_ucb(0.0, 0, 0.1) == math.inf

    def test_tighter_than_hoeffding_at_small_risk(self):
        # the regime the paper operates in: small empirical risk, moderate n
        assert bentkus_ucb(0.01, 200, 0.1) < hoeffding_ucb(0.01, 200, 0.1)
