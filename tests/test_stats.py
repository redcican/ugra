"""Unit tests for the testing protocol (paired bootstrap, Holm, Wilson)."""
import numpy as np
import pytest

from ugra.eval.stats import holm, paired_bootstrap_pvalue, wilson_interval


@pytest.mark.unit
class TestPairedBootstrap:
    def test_detects_a_real_shift(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(0, 1, 500)
        better = np.clip(base - 0.08 + rng.normal(0, 0.02, 500), 0, 1)
        assert paired_bootstrap_pvalue(better, base, n_boot=2000, seed=1) < 0.01

    def test_no_shift_is_not_rejected(self):
        rng = np.random.default_rng(3)
        a = rng.uniform(0, 1, 500)
        b = a + rng.normal(0, 0.05, 500)  # paired noise, zero mean shift
        assert paired_bootstrap_pvalue(a, b, n_boot=2000, seed=1) > 0.05

    def test_deterministic_under_seed(self):
        rng = np.random.default_rng(5)
        a, b = rng.uniform(0, 1, 100), rng.uniform(0, 1, 100)
        assert (paired_bootstrap_pvalue(a, b, seed=7)
                == paired_bootstrap_pvalue(a, b, seed=7))

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(ValueError):
            paired_bootstrap_pvalue([0.1], [0.1, 0.2])


@pytest.mark.unit
class TestHolm:
    def test_known_example(self):
        adjusted = holm([0.01, 0.04, 0.03])
        assert adjusted[0] == pytest.approx(0.03)   # 3 * 0.01
        assert adjusted[2] == pytest.approx(0.06)   # max(2 * 0.03, previous)
        assert adjusted[1] == pytest.approx(0.06)   # monotone step-down

    def test_capped_at_one(self):
        assert holm([0.9, 0.95]).max() <= 1.0

    def test_monotone_in_input_order_of_significance(self):
        adjusted = holm([0.001, 0.5])
        assert adjusted[0] < adjusted[1]


@pytest.mark.unit
class TestWilson:
    def test_contains_point_estimate(self):
        low, high = wilson_interval(83, 100)
        assert low < 0.83 < high

    def test_degenerate_total(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_narrower_with_more_data(self):
        low_s, high_s = wilson_interval(83, 100)
        low_l, high_l = wilson_interval(830, 1000)
        assert (high_l - low_l) < (high_s - low_s)
