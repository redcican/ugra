"""Unit tests for the synthetic degradation protocol (Section 4.1)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from ugra.degradation.operators import OPERATORS, apply_operator


def _page() -> Image.Image:
    image = Image.new("L", (400, 80), 245)
    ImageDraw.Draw(image).text((10, 25), "payment of forty pounds", fill=20)
    return image


def _distortion(original: Image.Image, degraded: Image.Image) -> float:
    a = np.asarray(original.convert("L"), dtype=float)
    b = np.asarray(degraded.convert("L"), dtype=float)
    return float(np.abs(a - b).mean())


@pytest.mark.unit
class TestOperators:
    @pytest.mark.parametrize("operator", sorted(OPERATORS))
    def test_deterministic_under_seed(self, operator):
        page = _page()
        first = apply_operator(page, operator, level=3, seed=42)
        second = apply_operator(page, operator, level=3, seed=42)
        assert np.array_equal(np.asarray(first), np.asarray(second))

    @pytest.mark.parametrize("operator", sorted(OPERATORS))
    def test_severity_increases_distortion(self, operator):
        page = _page()
        mild = _distortion(page, apply_operator(page, operator, level=1, seed=0))
        harsh = _distortion(page, apply_operator(page, operator, level=5, seed=0))
        assert harsh > mild

    def test_unknown_operator_rejected(self):
        with pytest.raises(ValueError):
            apply_operator(_page(), "sepia", level=1)

    def test_level_range_enforced(self):
        with pytest.raises(ValueError):
            apply_operator(_page(), "blur", level=6)
