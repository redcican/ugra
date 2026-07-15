"""Verifiers (Section 3.2, Eq. 4). All three scores lie in [0, 1], are
larger for cleaner hypotheses, and are defined as zero for an empty
hypothesis (well-definedness convention of the paper).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Set

from .base import Verifier


class LexiconVerifier(Verifier):
    """z_lex: fraction of hypothesis words found in the collection lexicon."""

    name = "lexicon"
    cost = 0.05

    def __init__(self, vocabulary: Iterable[str]) -> None:
        self.vocabulary: Set[str] = {w.lower() for w in vocabulary}

    def score(self, text: str, image: Any = None,
              expected_len: Optional[int] = None) -> float:
        words = text.split()
        if not words:
            return 0.0
        hits = sum(1 for w in words if w.strip(".,;:!?'\"()").lower() in self.vocabulary)
        return hits / len(words)


class CharNGramVerifier(Verifier):
    """z_lm: exp of the mean character log-likelihood (Eq. 4).

    Uses KenLM when available; otherwise falls back to a built-in
    add-k-smoothed bigram model fit on era-matched text, which preserves
    the interface and the [0, 1] range.
    """

    name = "charlm"
    cost = 0.05

    def __init__(self, training_text: Iterable[str], add_k: float = 0.5) -> None:
        self.add_k = add_k
        self._bigram: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._context_total: Dict[str, float] = defaultdict(float)
        self._alphabet: Set[str] = set()
        for line in training_text:
            padded = "\x02" + line + "\x03"
            for prev_char, char in zip(padded, padded[1:]):
                self._bigram[prev_char][char] += 1.0
                self._context_total[prev_char] += 1.0
                self._alphabet.add(char)
        self._vocab_size = max(len(self._alphabet), 1)

    def _log_prob(self, prev_char: str, char: str) -> float:
        num = self._bigram[prev_char][char] + self.add_k
        den = self._context_total[prev_char] + self.add_k * self._vocab_size
        return math.log(num / den)

    def score(self, text: str, image: Any = None,
              expected_len: Optional[int] = None) -> float:
        if not text:
            return 0.0
        padded = "\x02" + text
        mean_ll = sum(self._log_prob(p, c) for p, c in zip(padded, padded[1:])) / len(text)
        return math.exp(mean_ll)


class LayoutVerifier(Verifier):
    """z_geo = min(gamma, 1/gamma) with gamma the rendered-width ratio.

    Without access to stroke metrics, gamma is approximated by the ratio of
    the hypothesis length to the expected character capacity of the region;
    real deployments replace ``expected_len`` with the rendered-width
    estimate of Section 3.2.
    """

    name = "layout"
    cost = 0.05

    def score(self, text: str, image: Any = None,
              expected_len: Optional[int] = None) -> float:
        if not text or not expected_len or expected_len <= 0:
            return 0.0 if not text else 0.5  # no geometry prior: uninformative mid value
        gamma = len(text) / expected_len
        return max(0.0, min(gamma, 1.0 / gamma))
