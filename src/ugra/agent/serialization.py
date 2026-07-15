"""State serialization for the text-only planner (Section 3.3).

The planner never sees pixels; the summary carries hypotheses with their
disagreeing spans, verifier scores, reader confidences, the tool history
with the change in u each action produced, and the remaining budget. The
threshold lambda-hat is deliberately absent, which is what licenses the
gate-free calibration pass of Section 3.4.
"""
from __future__ import annotations

import difflib
from typing import List, Optional, Sequence

from ..types import Action, Hypothesis
from .state import EpisodeState


def disagreeing_spans(text_a: str, text_b: str, limit: int = 4) -> str:
    matcher = difflib.SequenceMatcher(a=text_a, b=text_b, autojunk=False)
    spans = [
        f"[{op[1]}:{op[2]}] '{text_a[op[1]:op[2]]}' vs '{text_b[op[3]:op[4]]}'"
        for op in matcher.get_opcodes() if op[0] != "equal"
    ]
    return "; ".join(spans[:limit]) if spans else "none"


def serialize_state(
    state: EpisodeState,
    hyp_specialist: Optional[Hypothesis],
    hyp_generalist: Optional[Hypothesis],
    u_history: Sequence[float],
    action_history: Sequence[str],
    budget_left: float,
    legal: Sequence[Action],
) -> str:
    lines: List[str] = ["## state"]
    if hyp_specialist is not None:
        lines.append(f'specialist: "{hyp_specialist.text}" (conf {hyp_specialist.confidence:.2f}, view v{hyp_specialist.view_id})')
    if hyp_generalist is not None:
        lines.append(f'generalist: "{hyp_generalist.text}" (conf {hyp_generalist.confidence:.2f}, view v{hyp_generalist.view_id})')
    if hyp_specialist is not None and hyp_generalist is not None:
        lines.append(f"disagreeing spans: {disagreeing_spans(hyp_specialist.text, hyp_generalist.text)}")
    working = state.working_hypothesis()
    if working is not None:
        scores = state.scores.get(working.hyp_id, {})
        rendered = ", ".join(f"{k}={v:.2f}" for k, v in scores.items()) or "none yet"
        lines.append(f"verifier scores of working hypothesis: {rendered}")
    lines.append("u history: " + ", ".join(f"{u:.3f}" for u in u_history))
    if action_history:
        lines.append("actions so far: " + " -> ".join(action_history))
    lines.append(f"remaining budget: {budget_left:.2f}")
    lines.append("## legal actions")
    for index, action in enumerate(legal):
        target_kind = "hypothesis" if action.kind == "verify" else "view"
        lines.append(f"A{index}: {action.tool} on {target_kind} {action.target}")
    lines.append("Reply with exactly one action id, e.g. A0.")
    return "\n".join(lines)
