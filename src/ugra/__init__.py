"""UGRA: Uncertainty-Guided Reading Agent.

Reference implementation of "An Uncertainty-Guided Perception, Reasoning,
and Action Agent for Text Recognition in Degraded Documents": a budgeted
perception-reasoning-action loop over restorers, readers, and verifiers,
closed by accept/abstain gates calibrated with distribution-free risk
control (PAC (alpha, delta) bound on the mean CER of accepted output).
"""

__version__ = "0.1.0"

from .types import (  # noqa: F401
    Action,
    DecisionRecord,
    DeployedOutcome,
    Hypothesis,
    Region,
    StepRecord,
    Trajectory,
    View,
)
