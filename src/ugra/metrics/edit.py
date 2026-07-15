"""Edit-distance metrics: Eq. (1) CER, Eq. (2) WER, Eq. (5) disagreement.

The calibrated loss (Eq. 1) is clipped at one so that it is bounded, which
the concentration bounds of the calibration require; aggregate error rates
reported in tables pool raw edit distances and are not clipped
(Section 4.1 of the paper).
"""
from __future__ import annotations

from typing import List, Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences (chars or word lists)."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    previous: List[int] = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cer(hypothesis: str, reference: str) -> float:
    """Clipped character error rate, Eq. (1): min(1, lev / max(|y*|, 1))."""
    return min(1.0, levenshtein(hypothesis, reference) / max(len(reference), 1))


def raw_char_edits(hypothesis: str, reference: str) -> int:
    """Unclipped character edit count, for pooled (micro) aggregation."""
    return levenshtein(hypothesis, reference)


def wer(hypothesis: str, reference: str) -> float:
    """Word error rate, Eq. (2): word-level Levenshtein over reference words."""
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    return levenshtein(hyp_words, ref_words) / max(len(ref_words), 1)


def disagreement(hyp_a: str, hyp_b: str) -> float:
    """Symmetrically normalized edit distance u, Eq. (5).

    Neither string is a reference, so the denominator treats them
    symmetrically; the constant 1 guards the empty-empty case.
    """
    return levenshtein(hyp_a, hyp_b) / max(len(hyp_a), len(hyp_b), 1)


def pooled_cer(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Pooled (micro) CER: summed edit distance over summed reference length.

    This is the aggregation used for corpus-level and pooled rows in the
    paper's tables; it carries no clip.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must align")
    total_edits = sum(levenshtein(h, r) for h, r in zip(hypotheses, references))
    total_len = sum(len(r) for r in references)
    return total_edits / max(total_len, 1)
