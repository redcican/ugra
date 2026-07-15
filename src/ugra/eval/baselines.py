"""External baseline runners (Section 4.1): single-pass readers, test-time
augmentation at fixed cost, and restore-then-read pipelines. Ablation
variants of the agent itself live in the experiment scripts, not here.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

from ..tools.base import ReadContext, Reader, Restorer
from ..types import Region


def single_pass(reader: Reader, region: Region,
                root_quality: float = 0.5) -> Tuple[str, float]:
    """One read of the root view; the loss is fixed at capture time."""
    return reader.read(region.image, ReadContext(region_id=region.region_id,
                                                 view_id=0, quality=root_quality,
                                                 reference=region.reference))


def _majority_vote(candidates: Sequence[str]) -> str:
    """Character-level majority vote after right-padding to a common length.

    A simplification of alignment-based voting; positions where no symbol
    reaches a plurality fall back to the first candidate.
    """
    if not candidates:
        return ""
    width = max(len(c) for c in candidates)
    padded = [c.ljust(width, "\x00") for c in candidates]
    voted: List[str] = []
    for position in range(width):
        counts = Counter(p[position] for p in padded)
        symbol, _ = counts.most_common(1)[0]
        voted.append(symbol if symbol != "\x00" else "")
    return "".join(voted).rstrip()


def tta(reader: Reader, region: Region, n_views: int = 8,
        root_quality: float = 0.5, jitter: float = 0.08) -> str:
    """Test-time augmentation: n augmented views, majority voting; iteration
    without reasoning, at fixed cost n * reader.cost.
    """
    candidates = []
    for k in range(n_views):
        quality = max(0.0, min(1.0, root_quality + ((-1) ** k) * jitter * (k // 2 + 1) / n_views))
        text, _ = reader.read(region.image, ReadContext(
            region_id=region.region_id, view_id=1000 + k, quality=quality,
            reference=region.reference))
        candidates.append(text)
    return _majority_vote(candidates)


def restore_then_read(restorer: Restorer, reader: Reader, region: Region,
                      root_quality: float = 0.5) -> Tuple[str, float]:
    """Unconditional restoration followed by one read (published-pipeline
    pattern: DE-GAN / DocRes + TrOCR)."""
    image, quality = restorer.apply(region.image, root_quality)
    return reader.read(image, ReadContext(region_id=region.region_id, view_id=1,
                                          quality=quality, reference=region.reference))
