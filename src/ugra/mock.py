"""Mock ecosystem: a synthetic reading channel that lets the full pipeline
run end to end without model weights (tests, demos, protocol dry runs).

The channel is deliberately faithful to the paper's mechanism: two readers
with DECORRELATED errors read a reference string through a view-quality
channel; restorers raise quality; higher quality means fewer corruption
events, so disagreement u falls as the agent re-perceives — exactly the
signal the gate calibrates.
"""
from __future__ import annotations

import hashlib
import string
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .tools.base import ReadContext, Reader, Restorer, ToolBox
from .tools.verifiers import CharNGramVerifier, LayoutVerifier, LexiconVerifier
from .types import Region

_ALPHABET = string.ascii_lowercase + " "


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little")


class MockReader(Reader):
    """Noisy-channel reader: corrupts the reference at a rate driven by
    (1 - view quality) times a per-reader base error. Deterministic per
    (reader, region, view), so repeated reads of one view agree with
    themselves and the (reader, view) legality rule stays meaningful.
    """

    def __init__(self, name: str, cost: float, base_error: float,
                 persona_seed: int) -> None:
        self.name = name
        self.cost = cost
        self.base_error = base_error
        self.persona_seed = persona_seed

    def read(self, image: Any, context: ReadContext) -> Tuple[str, float]:
        reference = context.reference or ""
        rng = np.random.default_rng(
            _stable_seed(self.name, self.persona_seed, context.region_id, context.view_id))
        error_rate = self.base_error * max(0.0, 1.0 - context.quality)
        characters: List[str] = []
        for char in reference:
            roll = rng.random()
            if roll < error_rate * 0.7:      # substitution
                characters.append(_ALPHABET[int(rng.integers(len(_ALPHABET)))])
            elif roll < error_rate * 0.85:   # deletion
                continue
            elif roll < error_rate:          # insertion
                characters.append(char)
                characters.append(_ALPHABET[int(rng.integers(len(_ALPHABET)))])
            else:
                characters.append(char)
        text = "".join(characters)
        confidence = float(np.clip(0.95 - 0.5 * error_rate + rng.normal(0, 0.03), 0, 1))
        return text, confidence


class MockRestorer(Restorer):
    """Raises the view quality by a fixed gain (a clean binarizer stand-in)."""

    def __init__(self, name: str, cost: float, gain: float) -> None:
        self.name = name
        self.cost = cost
        self.gain = gain

    def apply(self, image: Any, quality: float) -> Tuple[Any, float]:
        return image, min(1.0, quality + self.gain)


_SENTENCES = (
    "the committee resolved to defer the question until march",
    "payment of forty pounds was received by the parish clerk",
    "witnesses appeared before the council on the third day",
    "the letters were carried down the river to the printing house",
    "an inventory of the estate was taken in the presence of the mayor",
    "the surveyor recorded the boundary stones along the north field",
    "rations of bread and salt were issued to the garrison at dawn",
    "the registry lists three apprentices bound to the cooper",
)


def make_regions(count: int, seed: int = 0,
                 qualities: Sequence[float] = (0.75, 0.45, 0.2)) -> List[Tuple[Region, float]]:
    """Synthetic regions across severity tiers; returns (region, root_quality)."""
    rng = np.random.default_rng(seed)
    out: List[Tuple[Region, float]] = []
    for index in range(count):
        reference = _SENTENCES[int(rng.integers(len(_SENTENCES)))]
        quality = float(qualities[int(rng.integers(len(qualities)))])
        region = Region(region_id=f"mock-{seed}-{index}", image=None,
                        reference=reference, expected_len=len(reference))
        out.append((region, quality))
    return out


def make_toolbox(lexicon_extra: Optional[Sequence[str]] = None) -> ToolBox:
    """Default mock toolbox mirroring Table 2's families and cost tiers."""
    vocabulary = set(w for s in _SENTENCES for w in s.split())
    if lexicon_extra:
        vocabulary.update(lexicon_extra)
    return ToolBox(
        specialist=MockReader("specialist", cost=1.0, base_error=0.35, persona_seed=11),
        generalist=MockReader("generalist", cost=3.0, base_error=0.30, persona_seed=97),
        restorers=(
            MockRestorer("binarize", cost=0.05, gain=0.25),
            MockRestorer("contrast", cost=0.05, gain=0.10),
            MockRestorer("restore_learned", cost=1.0, gain=0.45),
        ),
        verifiers=(
            LexiconVerifier(vocabulary),
            CharNGramVerifier(_SENTENCES),
            LayoutVerifier(),
        ),
    )
