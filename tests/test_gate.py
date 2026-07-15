"""Tests for first-crossing pricing and the fixed-sequence scan
(Section 3.4, Eqs. 6-11), including a simulation check of Proposition 1.
"""
import math

import numpy as np
import pytest

from ugra.calibration.gate import calibrate, crossing_records, price_trajectory
from ugra.metrics.edit import cer
from ugra.types import StepRecord, Trajectory

GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)


def _trajectory(qs, texts, reference="target text"):
    steps = tuple(
        StepRecord(step=i + 1, q=float(q), working_text=t, cumulative_cost=4.0 + i)
        for i, (q, t) in enumerate(zip(qs, texts))
    )
    return Trajectory(region_id="r", reference=reference, steps=steps,
                      decisions=(), end_reason="no_action", total_cost=4.0 + len(qs))


@pytest.mark.unit
class TestPricing:
    def test_records_are_strict_prefix_minima(self):
        traj = _trajectory([0.5, 0.7, 0.3, 0.3, 0.1], list("abcde"))
        records = crossing_records(traj)
        assert [r[0] for r in records] == [0.5, 0.3, 0.1]

    def test_first_crossing_loss_brute_force(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            n_steps = int(rng.integers(1, 8))
            qs = rng.uniform(0, 1, n_steps)
            texts = ["target text" if rng.random() < 0.5 else "tgt txt!"
                     for _ in range(n_steps)]
            traj = _trajectory(qs, texts)
            priced = price_trajectory(traj, GRID)
            for j, lam in enumerate(GRID):
                crossing = next((i for i, q in enumerate(qs) if q <= lam), None)
                if crossing is None:
                    assert np.isnan(priced.losses[j])
                else:
                    assert priced.losses[j] == pytest.approx(
                        cer(texts[crossing], "target text"))

    def test_record_score_is_min(self):
        traj = _trajectory([0.9, 0.4, 0.6], list("abc"))
        assert price_trajectory(traj, GRID).record_score == pytest.approx(0.4)

    def test_reference_required(self):
        traj = _trajectory([0.5], ["a"], reference=None)
        with pytest.raises(ValueError):
            price_trajectory(traj, GRID)


@pytest.mark.unit
class TestScan:
    def test_prefix_property(self):
        # good regions cross at low q with zero loss; bad ones at high q.
        good = [_trajectory([0.02], ["target text"]) for _ in range(400)]
        bad = [_trajectory([0.8], ["wrong words entirely"]) for _ in range(100)]
        priced = [price_trajectory(t, GRID) for t in good + bad]
        # strict Eq. (11): no mass at grid[0] -> the empty bottom cell vetoes
        strict = calibrate(priced, alpha=0.05, delta=0.1, grid=GRID)
        assert strict.lambda_hat is None
        # preregistered scan-start floor (Section 3.4): certification resumes
        gate = calibrate(priced, alpha=0.05, delta=0.1, grid=GRID, min_accept=1)
        assert gate.lambda_hat is not None
        assert 0.02 <= gate.lambda_hat < 0.8
        certified = (GRID <= gate.lambda_hat) & (gate.n_accept > 0)
        assert (gate.risk_ucb[certified] <= 0.05).all()

    def test_no_certification_returns_none(self):
        bad = [_trajectory([0.01], ["wrong words entirely"]) for _ in range(50)]
        priced = [price_trajectory(t, GRID) for t in bad]
        gate = calibrate(priced, alpha=0.05, delta=0.1, grid=GRID)
        assert gate.lambda_hat is None

    def test_empty_grid_cells_do_not_certify(self):
        # nothing crosses below 0.5 -> UCB is +inf there; the scan stops at
        # the first grid point, so lambda_hat is None even though larger
        # thresholds would have been safe. Honest fixed-sequence behavior.
        late = [_trajectory([0.5], ["target text"]) for _ in range(300)]
        priced = [price_trajectory(t, GRID) for t in late]
        gate = calibrate(priced, alpha=0.05, delta=0.1, grid=GRID)
        assert math.isinf(gate.risk_ucb[0])
        assert gate.lambda_hat is None

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            calibrate([], alpha=0.0, delta=0.1, grid=GRID)


@pytest.mark.slow
class TestProposition1:
    def test_pac_guarantee_frequency(self):
        """Simulate Proposition 1: over many calibration draws, the realized
        selective risk at lambda-hat exceeds alpha in at most ~delta of
        draws (plus Monte-Carlo slack)."""
        alpha, delta = 0.1, 0.1
        rng = np.random.default_rng(0)
        grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)

        def sample_population(n, generator):
            # single-step trajectories: Q ~ U(0,1); loss noisy-monotone in Q
            trajectories = []
            for k in range(n):
                q = float(generator.uniform())
                wrong = generator.random() < 0.6 * q  # P(error | Q=q)
                text = "target text" if not wrong else "wrong words"
                trajectories.append(_trajectory([q], [text]))
            return trajectories

        violations = 0
        n_reps = 200
        for _ in range(n_reps):
            calibration = [price_trajectory(t, grid)
                           for t in sample_population(400, rng)]
            gate = calibrate(calibration, alpha, delta, grid)
            if gate.lambda_hat is None:
                continue  # abstain everywhere: vacuously safe
            # true selective risk at lambda-hat, by construction of the channel:
            # E[loss | Q <= lam] with loss = cer("wrong words","target text") * P(wrong)
            lam = gate.lambda_hat
            unit_loss = cer("wrong words", "target text")
            true_risk = unit_loss * 0.6 * lam / 2.0  # E[0.6 Q | Q<=lam] = 0.3 lam
            violations += int(true_risk > alpha)
        assert violations / n_reps <= delta + 0.05  # Monte-Carlo slack
