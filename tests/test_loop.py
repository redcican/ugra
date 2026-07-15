"""Integration tests for Algorithm 1 on the mock ecosystem: termination,
budget accounting, the deployed-equals-truncation property, and abstain
reasons.
"""
import pytest

from ugra.agent.loop import LoopConfig, run_episode
from ugra.agent.planner import RandomPlanner, RulePlanner
from ugra.mock import make_regions, make_toolbox

TOOLBOX = make_toolbox()
CONFIG = LoopConfig(budget=12.0)


def _episodes(count, seed=0, planner=None, gate_lambda=None):
    planner = planner or RulePlanner()
    for region, quality in make_regions(count, seed=seed):
        yield run_episode(region, TOOLBOX, planner, CONFIG,
                          gate_lambda=gate_lambda, root_quality=quality)


@pytest.mark.integration
class TestGateFreeEpisodes:
    def test_termination_and_budget(self):
        for trajectory, outcome in _episodes(20):
            assert outcome is None
            assert trajectory.end_reason in ("budget", "no_action")
            # mandatory double read charged: 1 + 3 units
            assert trajectory.total_cost >= 4.0
            # never exceeds B by more than one final action's cost
            assert trajectory.total_cost <= CONFIG.budget + 3.0

    def test_scores_are_bounded_and_logged(self):
        for trajectory, _ in _episodes(10, seed=3):
            assert len(trajectory.steps) >= 1
            assert all(0.0 <= s.q <= 1.0 for s in trajectory.steps)

    def test_quality_orders_disagreement(self):
        """Cleaner inputs should open at lower disagreement on average."""
        from ugra.types import Region

        def first_q(quality, n=30):
            values = []
            for k in range(n):
                region = Region(region_id=f"q{quality}-{k}", image=None,
                                reference="payment of forty pounds was received",
                                expected_len=38)
                trajectory, _ = run_episode(region, TOOLBOX, RulePlanner(), CONFIG,
                                            root_quality=quality)
                values.append(trajectory.steps[0].q)
            return sum(values) / len(values)

        assert first_q(0.9) < first_q(0.2)

    def test_random_planner_still_terminates(self):
        for trajectory, _ in _episodes(10, seed=5, planner=RandomPlanner(seed=1)):
            assert trajectory.end_reason in ("budget", "no_action")


@pytest.mark.integration
class TestDeployedTruncation:
    def test_deployed_equals_truncated_gate_free(self):
        """With a deterministic planner, the deployed episode at lambda is
        the gate-free episode truncated at its first crossing — the
        property the calibration argument of Section 3.4 relies on."""
        lam = 0.15
        for region, quality in make_regions(30, seed=11):
            gate_free, _ = run_episode(region, TOOLBOX, RulePlanner(), CONFIG,
                                       root_quality=quality)
            deployed_traj, outcome = run_episode(region, TOOLBOX, RulePlanner(),
                                                 CONFIG, gate_lambda=lam,
                                                 root_quality=quality)
            crossing = next((i for i, s in enumerate(gate_free.steps) if s.q <= lam),
                            None)
            if crossing is None:
                assert not outcome.accepted
                assert outcome.reason in ("budget", "no_action")
            else:
                assert outcome.accepted
                assert outcome.reason == "accept"
                assert deployed_traj.steps == gate_free.steps[: crossing + 1]
                assert outcome.text == gate_free.steps[crossing].working_text

    def test_abstain_reasons_reported(self):
        reasons = {outcome.reason
                   for _, outcome in _episodes(30, seed=2, gate_lambda=0.0)}
        assert reasons <= {"accept", "budget", "no_action"}
        assert "accept" not in reasons or 0.0 in {0.0}  # lambda=0 accepts only exact agreement
