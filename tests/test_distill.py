"""Unit tests for distillation-pair extraction (Section 3.5)."""
import json

import pytest

from ugra.distill.dataset import accepted_prefix_pairs, export_jsonl
from ugra.types import DecisionRecord, StepRecord, Trajectory


def _trajectory(qs, n_decisions=None):
    steps = tuple(StepRecord(step=i + 1, q=q, working_text=f"t{i}",
                             cumulative_cost=4.0 + i)
                  for i, q in enumerate(qs))
    n_decisions = len(qs) - 1 if n_decisions is None else n_decisions
    decisions = tuple(DecisionRecord(step=i + 1, summary=f"s{i}",
                                     action_label=f"a{i}")
                      for i in range(n_decisions))
    return Trajectory(region_id="r", reference="x", steps=steps,
                      decisions=decisions, end_reason="no_action",
                      total_cost=4.0 + len(qs))


@pytest.mark.unit
class TestPrefixPairs:
    def test_pairs_stop_before_the_crossing(self):
        trajectory = _trajectory([0.6, 0.4, 0.1, 0.05])
        pairs = accepted_prefix_pairs(trajectory, lambda_hat=0.15)
        # crossing at step 3 -> decisions from steps 1 and 2 only
        assert [a for _, a in pairs] == ["a0", "a1"]

    def test_never_crossing_yields_nothing(self):
        trajectory = _trajectory([0.6, 0.5, 0.4])
        assert accepted_prefix_pairs(trajectory, lambda_hat=0.1) == []

    def test_immediate_crossing_yields_nothing(self):
        # accepted at the mandatory double read: no planner decision to learn
        trajectory = _trajectory([0.02, 0.01])
        assert accepted_prefix_pairs(trajectory, lambda_hat=0.15) == []

    def test_export_jsonl(self, tmp_path):
        trajectories = [_trajectory([0.6, 0.1]), _trajectory([0.9, 0.8])]
        out = tmp_path / "pairs.jsonl"
        count = export_jsonl(trajectories, lambda_hat=0.15, path=out)
        lines = out.read_text().splitlines()
        assert count == len(lines) == 1
        record = json.loads(lines[0])
        assert set(record) == {"summary", "action"}
