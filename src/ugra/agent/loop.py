"""Algorithm 1: uncertainty-guided reading of one region.

Two execution modes share this single implementation:
  - gate-free (``gate_lambda=None``): the calibration/test logging pass of
    Section 3.4 — the episode runs until no legal action remains or the
    budget is exhausted, and every step's (q_t, working hypothesis) is
    recorded;
  - deployed (``gate_lambda`` set): the accept test q <= lambda-hat runs
    after every step, and the episode stops at the first crossing.

Because the planner summary never contains the threshold, the deployed
episode under lambda is the gate-free episode truncated at its first
crossing — the property the calibration argument relies on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..metrics.edit import disagreement
from ..types import Action, DecisionRecord, DeployedOutcome, Region, StepRecord, Trajectory
from .planner import Planner
from .score import running_score
from .serialization import serialize_state
from .state import EpisodeState
from ..tools.base import ReadContext, ToolBox


@dataclass(frozen=True)
class LoopConfig:
    budget: float = 12.0
    floors: Optional[dict] = None  # verifier name -> floor; default 0.25 each

    def floor_map(self, toolbox: ToolBox) -> dict:
        if self.floors is not None:
            return dict(self.floors)
        return {v.name: 0.25 for v in toolbox.verifiers}


def _legal_actions(state: EpisodeState, toolbox: ToolBox, spent: float,
                   budget: float) -> List[Action]:
    remaining = budget - spent
    legal: List[Action] = []
    for restorer in toolbox.restorers:
        if restorer.cost > remaining:
            continue
        for view in state.views:
            if (restorer.name, view.view_id) not in state.tried:
                legal.append(Action(kind="restore", tool=restorer.name, target=view.view_id))
    for reader in toolbox.readers:
        if reader.cost > remaining:
            continue
        for view in state.views:
            if (reader.name, view.view_id) not in state.tried:
                legal.append(Action(kind="read", tool=reader.name, target=view.view_id))
    working = state.working_hypothesis()
    if working is not None:
        for verifier in toolbox.verifiers:
            if verifier.cost > remaining:
                continue
            if (verifier.name, working.hyp_id) not in state.tried:
                legal.append(Action(kind="verify", tool=verifier.name, target=working.hyp_id))
    return legal


def _execute(action: Action, state: EpisodeState, toolbox: ToolBox,
             region: Region, step: int) -> float:
    if action.kind == "restore":
        restorer = next(t for t in toolbox.restorers if t.name == action.tool)
        source = state.views[action.target]
        image, quality = restorer.apply(source.image, source.quality)
        state.add_view(image, quality, parent_id=source.view_id, tool=restorer.name)
        state.tried.add((restorer.name, source.view_id))
        return restorer.cost
    if action.kind == "read":
        reader = next(r for r in toolbox.readers if r.name == action.tool)
        view = state.views[action.target]
        text, confidence = reader.read(
            view.image,
            ReadContext(region_id=region.region_id, view_id=view.view_id,
                        quality=view.quality, reference=region.reference),
        )
        state.add_hypothesis(text, confidence, reader.name, view.view_id, step)
        state.tried.add((reader.name, view.view_id))
        return reader.cost
    if action.kind == "verify":
        verifier = next(v for v in toolbox.verifiers if v.name == action.tool)
        hypothesis = state.hypotheses[action.target]
        value = verifier.score(hypothesis.text, image=None,
                               expected_len=region.expected_len)
        state.add_score(hypothesis.hyp_id, verifier.name, value)
        state.tried.add((verifier.name, hypothesis.hyp_id))
        return verifier.cost
    raise ValueError(f"unknown action kind: {action.kind}")


def run_episode(
    region: Region,
    toolbox: ToolBox,
    planner: Planner,
    config: LoopConfig = LoopConfig(),
    gate_lambda: Optional[float] = None,
    root_quality: float = 0.5,
) -> Tuple[Trajectory, Optional[DeployedOutcome]]:
    """Run one episode; returns the (possibly truncated) trajectory and,
    in deployed mode, the terminal outcome.
    """
    floors = config.floor_map(toolbox)
    state = EpisodeState(region.image, root_quality, toolbox.specialist.name)

    # Mandatory double read (Section 3.2): charged against the budget.
    spent = 0.0
    for reader in toolbox.readers:
        spent += _execute(Action(kind="read", tool=reader.name, target=0),
                          state, toolbox, region, step=0)

    steps: List[StepRecord] = []
    decisions: List[DecisionRecord] = []
    u_history: List[float] = []
    action_history: List[str] = []
    end_reason = "no_action"
    step = 0
    while True:
        step += 1
        hyp_s, hyp_g = state.latest_per_reader(
            (toolbox.specialist.name, toolbox.generalist.name))
        u_t = disagreement(hyp_s.text if hyp_s else "", hyp_g.text if hyp_g else "")
        working = state.working_hypothesis()
        q_t = running_score(u_t, state.scores.get(working.hyp_id, {}) if working else {},
                            floors)
        u_history.append(u_t)
        steps.append(StepRecord(step=step, q=q_t,
                                working_text=working.text if working else "",
                                cumulative_cost=spent))

        if gate_lambda is not None and q_t <= gate_lambda:
            outcome = DeployedOutcome(region_id=region.region_id, accepted=True,
                                      text=working.text if working else "",
                                      reason="accept", cost=spent, steps_used=step)
            return _pack(region, steps, decisions, "accept", spent), outcome

        legal = _legal_actions(state, toolbox, spent, config.budget)
        if not legal or spent >= config.budget:
            end_reason = "budget" if spent >= config.budget else "no_action"
            break

        summary = serialize_state(state, hyp_s, hyp_g, u_history, action_history,
                                  config.budget - spent, legal)
        action = planner.choose(summary, legal)
        decisions.append(DecisionRecord(step=step, summary=summary,
                                        action_label=action.label()))
        spent += _execute(action, state, toolbox, region, step)
        action_history.append(action.label())

    trajectory = _pack(region, steps, decisions, end_reason, spent)
    outcome = None
    if gate_lambda is not None:
        outcome = DeployedOutcome(region_id=region.region_id, accepted=False,
                                  text=None, reason=end_reason, cost=spent,
                                  steps_used=step)
    return trajectory, outcome


def _pack(region: Region, steps: List[StepRecord],
          decisions: List[DecisionRecord], end_reason: str,
          total_cost: float) -> Trajectory:
    return Trajectory(region_id=region.region_id, reference=region.reference,
                      steps=tuple(steps), decisions=tuple(decisions),
                      end_reason=end_reason, total_cost=total_cost)
