"""Trajectory distillation data (Section 3.5).

Every episode is logged as (state summary, action) pairs; pairs are kept
from episodes that would have been ACCEPTED under the calibrated gate,
that is, from the prefix up to the first crossing q_t <= lambda-hat. The
export is JSONL ready for supervised fine-tuning of the student planner.

Ordering constraint (Section 3.5): the student is trained first, and the
gate is recalibrated afterwards on episodes produced by the distilled
loop; this module only prepares the training pairs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from ..types import Trajectory


def accepted_prefix_pairs(trajectory: Trajectory,
                          lambda_hat: float) -> List[Tuple[str, str]]:
    """(summary, action label) pairs before the first crossing, or [] if the
    episode never crosses (it would have been deferred)."""
    crossing_step = None
    best = math.inf
    for record in trajectory.steps:
        if record.q < best:
            best = record.q
        if record.q <= lambda_hat:
            crossing_step = record.step
            break
    if crossing_step is None:
        return []
    return [(d.summary, d.action_label) for d in trajectory.decisions
            if d.step < crossing_step]


def export_jsonl(trajectories: Iterable[Trajectory], lambda_hat: float,
                 path: str | Path) -> int:
    """Write all accepted-prefix pairs to JSONL; returns the pair count."""
    path = Path(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            for summary, action in accepted_prefix_pairs(trajectory, lambda_hat):
                handle.write(json.dumps({"summary": summary, "action": action},
                                        ensure_ascii=False) + "\n")
                count += 1
    return count
