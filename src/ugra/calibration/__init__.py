"""Calibrated gates (Section 3.4): pricing, bounds, fixed-sequence scan."""

from .bounds import bentkus_ucb, hoeffding_ucb  # noqa: F401
from .gate import (  # noqa: F401
    GateResult,
    PricedTrajectory,
    calibrate,
    crossing_records,
    price_trajectory,
)
