"""Episode state s_t = (v_t, H_t, e_t) and the working hypothesis (Eq. 3).

The state is the single mutable object of an episode; everything it stores
is append-only, and all selection rules follow Section 3.1:
  - working hypothesis = argmax of the mean verifier score z_bar,
  - z_bar = 0 for hypotheses without verifier evidence,
  - ties break toward the most recent view, then toward the specialist.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..types import Hypothesis, View


class EpisodeState:
    def __init__(self, root_image: Any, root_quality: float, specialist_name: str) -> None:
        self.views: List[View] = [View(view_id=0, image=root_image, quality=root_quality,
                                       parent_id=None, tool="root")]
        self.hypotheses: List[Hypothesis] = []
        self.scores: Dict[int, Dict[str, float]] = {}  # hyp_id -> verifier -> z
        self.log: List[str] = []
        self.tried: set[Tuple[str, int]] = set()  # (tool name, target id)
        self.specialist_name = specialist_name

    # ---- append-only updates -------------------------------------------
    def add_view(self, image: Any, quality: float, parent_id: int, tool: str) -> View:
        view = View(view_id=len(self.views), image=image, quality=quality,
                    parent_id=parent_id, tool=tool)
        self.views.append(view)
        return view

    def add_hypothesis(self, text: str, confidence: float, reader: str,
                       view_id: int, step: int) -> Hypothesis:
        hyp = Hypothesis(hyp_id=len(self.hypotheses), text=text, confidence=confidence,
                         reader=reader, view_id=view_id, step=step)
        self.hypotheses.append(hyp)
        self.scores[hyp.hyp_id] = {}
        return hyp

    def add_score(self, hyp_id: int, verifier: str, value: float) -> None:
        self.scores[hyp_id][verifier] = value

    # ---- selection rules (Eq. 3) ---------------------------------------
    def z_bar(self, hyp_id: int) -> float:
        values = self.scores.get(hyp_id, {})
        return sum(values.values()) / len(values) if values else 0.0

    def working_hypothesis(self) -> Optional[Hypothesis]:
        if not self.hypotheses:
            return None
        return max(
            self.hypotheses,
            key=lambda h: (
                self.z_bar(h.hyp_id),
                h.view_id,                                   # most recent view
                1 if h.reader == self.specialist_name else 0,  # then specialist
                h.hyp_id,                                    # then latest read
            ),
        )

    def latest_per_reader(self, reader_names: Tuple[str, str]) -> Tuple[Optional[Hypothesis], Optional[Hypothesis]]:
        """Each reader's hypothesis from its most recent read (Section 3.3)."""
        latest: Dict[str, Hypothesis] = {}
        for hyp in self.hypotheses:
            latest[hyp.reader] = hyp  # append order = read order
        return latest.get(reader_names[0]), latest.get(reader_names[1])
