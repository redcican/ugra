"""Core immutable value types shared across the package.

Notation follows Section 3.1 of the paper: an episode evolves a state
s_t = (v_t, H_t, e_t) over views, hypotheses, and an evidence log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Region:
    """One document region x with (train/calibration-time) reference y*.

    ``image`` is a PIL image for real corpora and may be ``None`` in the
    mock ecosystem, where readers synthesize text from ``reference`` and
    the view quality channel instead.
    """

    region_id: str
    image: Any = None
    reference: Optional[str] = None
    expected_len: Optional[int] = None  # layout-verifier prior, if known
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class View:
    """A view v of a region: the crop after the restorations applied so far.

    ``quality`` is an abstract legibility channel in [0, 1] used by the
    mock ecosystem and by quality-aware restorers; real restorers may
    carry it through unchanged.
    """

    view_id: int
    image: Any
    quality: float
    parent_id: Optional[int]
    tool: str  # producing tool name; "root" for the original crop


@dataclass(frozen=True)
class Hypothesis:
    """An annotated transcription hypothesis (Section 3.1, Eq. 3 context)."""

    hyp_id: int
    text: str
    confidence: float
    reader: str
    view_id: int
    step: int


@dataclass(frozen=True)
class Action:
    """One element of the action space (Eq. 2).

    ``kind`` is one of "restore" | "read" | "verify" — the apply actions of
    Eq. (2); the two terminal actions are produced by the gates, never by
    the planner (Section 3.1). ``target`` is a view id for restorers and
    readers, a hypothesis id for verifiers.
    """

    kind: str
    tool: str
    target: int

    def label(self) -> str:
        return f"{self.tool}@{self.target}"


@dataclass(frozen=True)
class StepRecord:
    """Per-step scalar record needed by the calibration of Section 3.4."""

    step: int
    q: float
    working_text: str
    cumulative_cost: float


@dataclass(frozen=True)
class DecisionRecord:
    """Planner decision, logged for distillation (Section 3.5)."""

    step: int
    summary: str
    action_label: str


@dataclass(frozen=True)
class Trajectory:
    """A gate-free episode log (Section 3.4).

    The deployed episode under any threshold lambda is the truncation of
    this record at its first step with q_t <= lambda.
    """

    region_id: str
    reference: Optional[str]
    steps: Tuple[StepRecord, ...]
    decisions: Tuple[DecisionRecord, ...]
    end_reason: str  # "budget" | "no_action"
    total_cost: float

    @property
    def record_score(self) -> float:
        """Q = min_t q_t, the smallest running score along the episode."""
        return min(s.q for s in self.steps)


@dataclass(frozen=True)
class DeployedOutcome:
    """Result of running the loop with the calibrated gate active."""

    region_id: str
    accepted: bool
    text: Optional[str]
    reason: str  # "accept" | "score" | "budget" | "no_action"
    cost: float
    steps_used: int
