"""Tool interfaces of the perception layer (Section 3.2, Table 2).

Three families with fixed signatures:
    restorers  tau: v -> v'
    readers    rho: v -> (y_hat, c)
    verifiers  nu:  (v, y_hat) -> z in [0, 1]
Every tool carries a cost in normalized units (one specialist read = 1).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReadContext:
    """Everything a reader may condition on besides the image itself.

    The mock ecosystem reads ``reference`` through the quality channel;
    real readers use only ``image`` and ignore the rest.
    """

    region_id: str
    view_id: int
    quality: float
    reference: Optional[str] = None


class Restorer(abc.ABC):
    """v -> v'. Returns the transformed image and the new quality."""

    name: str
    cost: float

    @abc.abstractmethod
    def apply(self, image: Any, quality: float) -> Tuple[Any, float]:
        ...


class Reader(abc.ABC):
    """v -> (y_hat, c). Confidence c is logged as evidence, never a gate."""

    name: str
    cost: float

    @abc.abstractmethod
    def read(self, image: Any, context: ReadContext) -> Tuple[str, float]:
        ...


class Verifier(abc.ABC):
    """(v, y_hat) -> z in [0, 1]; larger is cleaner. Zero for empty text."""

    name: str
    cost: float

    @abc.abstractmethod
    def score(self, text: str, image: Any = None,
              expected_len: Optional[int] = None) -> float:
        ...


@dataclass(frozen=True)
class ToolBox:
    """The toolset T of an agent instance.

    ``specialist`` and ``generalist`` are the two mandatory readers of
    Section 3.2; their family diversity is what makes disagreement an
    informative signal.
    """

    specialist: Reader
    generalist: Reader
    restorers: Sequence[Restorer]
    verifiers: Sequence[Verifier]

    @property
    def readers(self) -> Tuple[Reader, Reader]:
        return (self.specialist, self.generalist)

    def min_cost(self) -> float:
        costs = [t.cost for t in (*self.readers, *self.restorers, *self.verifiers)]
        return min(costs)
