"""Running score q_t (Section 3.4).

Default: q_t equals the current disagreement u_t, overridden to 1 while
any verifier score of the working hypothesis sits below its floor. The
score is reference-free by construction.
"""
from __future__ import annotations

from typing import Mapping


def running_score(u_t: float, working_scores: Mapping[str, float],
                  floors: Mapping[str, float]) -> float:
    for verifier, floor in floors.items():
        if verifier in working_scores and working_scores[verifier] < floor:
            return 1.0
    return min(max(u_t, 0.0), 1.0)
