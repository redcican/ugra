"""Metrics: edit-distance losses (Eqs. 1, 2, 5).

Selective metrics (risk-coverage sweep, validity, mean cost) live in
``ugra.metrics.selective`` and are imported as a submodule — they depend
on the calibration pricing types, which in turn depend on the edit
metrics, so eager re-export here would be circular.
"""

from .edit import cer, disagreement, levenshtein, pooled_cer, wer  # noqa: F401
