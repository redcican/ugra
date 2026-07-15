"""Planners (Section 3.3): the served LLM planner, plus the deterministic
rule planner and the random planner used by the ablations of Section 4.4.

Planners only ever choose among the legal apply actions; the terminal
decisions belong to the calibrated gates. Any malformed LLM output falls
back to the harness default (the first legal action), so free text can
never leak into the control flow.
"""
from __future__ import annotations

import abc
import logging
import re
from typing import Optional, Sequence

import numpy as np

from ..types import Action

logger = logging.getLogger(__name__)

_FAMILY_PRIORITY = {"restore": 0, "read": 1, "verify": 2}


class Planner(abc.ABC):
    name: str

    @abc.abstractmethod
    def choose(self, summary: str, legal: Sequence[Action]) -> Action:
        ...


class RulePlanner(Planner):
    """Deterministic heuristic that interleaves restoration and re-reading:
    read the newest view first if any reader has not seen it, otherwise
    restore the newest view, otherwise verify the working hypothesis.
    Serves as the no-LLM default and makes episodes reproducible in tests.
    """

    name = "rule"

    def __init__(self, costs: Optional[dict] = None) -> None:
        self.costs = costs or {}

    def choose(self, summary: str, legal: Sequence[Action]) -> Action:
        newest_view = max((a.target for a in legal if a.kind in ("read", "restore")),
                          default=-1)
        reads = [a for a in legal if a.kind == "read" and a.target == newest_view]
        if reads:
            return min(reads, key=lambda a: (self.costs.get(a.tool, 0.0), a.tool))
        restores = [a for a in legal if a.kind == "restore" and a.target == newest_view]
        if restores:
            return min(restores, key=lambda a: (self.costs.get(a.tool, 0.0), a.tool))
        return min(legal, key=lambda a: (_FAMILY_PRIORITY.get(a.kind, 3),
                                         self.costs.get(a.tool, 0.0), -a.target, a.tool))


class RandomPlanner(Planner):
    """Uniform choice among legal actions; the validity-vs-coverage
    ablation of Section 4.4 (a poor planner loses coverage, never
    validity).
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def choose(self, summary: str, legal: Sequence[Action]) -> Action:
        return legal[int(self._rng.integers(len(legal)))]


class LLMPlanner(Planner):
    """Served instruction-tuned LLM (Qwen3-235B-A22B or the distilled
    Qwen3-8B student) behind an OpenAI-compatible endpoint.

    Output is constrained to the action vocabulary: the model must answer
    with one action id (A<k>); anything else falls back to the harness
    default with a logged warning.
    """

    name = "llm"

    SYSTEM_PROMPT = (
        "You control the re-perception policy of a document-reading agent. "
        "Given the state summary, pick the single legal action most likely "
        "to improve the evidence: bleed-through calls for binarization, "
        "small script for magnification, disputed proper nouns for the "
        "lexicon, unusual layouts for the generalist reader. Reply with "
        "the action id only."
    )

    def __init__(self, endpoint: str, model: str, temperature: float = 0.3,
                 top_p: float = 0.9, max_tokens: int = 512,
                 timeout: float = 60.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

    def choose(self, summary: str, legal: Sequence[Action]) -> Action:
        import requests  # lazy

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": summary},
            ],
        }
        try:
            response = requests.post(f"{self.endpoint}/chat/completions",
                                     json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except Exception as error:  # noqa: BLE001 — network failure -> harness default
            logger.warning("planner call failed (%s); harness default applied", error)
            return legal[0]
        match = re.search(r"A(\d+)", content)
        if match and int(match.group(1)) < len(legal):
            return legal[int(match.group(1))]
        logger.warning("planner returned no legal action id (%r); harness default", content)
        return legal[0]
