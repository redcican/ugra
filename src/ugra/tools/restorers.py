"""Classical restorers (Table 2). Learned restorers (DocRes, Real-ESRGAN)
are thin wrappers around external checkpoints and load lazily; the
classical tier is dependency-free beyond Pillow/NumPy.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .base import Restorer


class MagnifyCrop(Restorer):
    """Center magnification; cheap re-perception for small script."""

    name = "magnify"
    cost = 0.05

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def apply(self, image: Any, quality: float) -> Tuple[Any, float]:
        if image is None:  # mock channel
            return None, quality
        w, h = image.size
        cw, ch = int(w / self.factor), int(h / self.factor)
        left, top = (w - cw) // 2, (h - ch) // 2
        crop = image.crop((left, top, left + cw, top + ch))
        return crop.resize((w, h), Image.LANCZOS), quality


class ContrastNormalize(Restorer):
    name = "contrast"
    cost = 0.05

    def apply(self, image: Any, quality: float) -> Tuple[Any, float]:
        if image is None:
            return None, quality
        return ImageOps.autocontrast(image.convert("L")), quality


class SauvolaBinarize(Restorer):
    """Adaptive binarization (Sauvola and Pietikainen, 2000)."""

    name = "sauvola"
    cost = 0.05

    def __init__(self, window: int = 25, k: float = 0.2, r: float = 128.0) -> None:
        self.window, self.k, self.r = window, k, r

    def apply(self, image: Any, quality: float) -> Tuple[Any, float]:
        if image is None:
            return None, quality
        gray = np.asarray(image.convert("L"), dtype=float)
        mean = np.asarray(
            Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.BoxBlur(self.window // 2)),
            dtype=float,
        )
        sq = np.asarray(
            Image.fromarray(np.clip(gray ** 2 / 255.0, 0, 255).astype(np.uint8)).filter(
                ImageFilter.BoxBlur(self.window // 2)
            ),
            dtype=float,
        ) * 255.0
        std = np.sqrt(np.clip(sq - mean ** 2, 0.0, None))
        threshold = mean * (1.0 + self.k * (std / self.r - 1.0))
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(binary), quality


class LearnedRestorer(Restorer):
    """Wrapper for a learned restoration checkpoint (DocRes / Real-ESRGAN).

    The heavy import happens at first use; construction is free so the
    toolbox can be assembled without the model dependencies installed.
    """

    cost = 1.0

    def __init__(self, name: str, loader: str) -> None:
        self.name = name
        self._loader = loader
        self._model = None

    def _load(self):
        raise NotImplementedError(
            f"learned restorer '{self.name}' requires the optional model stack; "
            f"install extras [models] and provide the checkpoint ({self._loader})"
        )

    def apply(self, image: Any, quality: float) -> Tuple[Any, float]:
        if self._model is None:
            self._load()
        return self._model(image), quality  # pragma: no cover


def classical_restorers() -> Tuple[Restorer, ...]:
    return (MagnifyCrop(), ContrastNormalize(), SauvolaBinarize())
