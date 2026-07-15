"""Synthetic degradation protocol (Section 4.1): five operators at five
severity levels, seeded and reproducible to the pixel. Level parameters
are placeholders pending the anchoring run that equalizes the specialist's
relative accuracy steps (the anchoring caveat of the paper applies).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from PIL import Image, ImageFilter

# level -> parameter, level 1 mildest, 5 harshest (PROVISIONAL anchoring).
PARAMS: Dict[str, Dict[int, float]] = {
    "blur":  {1: 0.6, 2: 1.2, 3: 2.0, 4: 3.0, 5: 4.5},        # Gaussian radius
    "noise": {1: 6.0, 2: 12.0, 3: 20.0, 4: 32.0, 5: 48.0},     # additive sigma
    "bleed": {1: 0.12, 2: 0.22, 3: 0.35, 4: 0.50, 5: 0.68},    # ghost opacity
    "fade":  {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.40, 5: 0.25},    # ink retention
    "warp":  {1: 1.5, 2: 3.0, 3: 5.0, 4: 8.0, 5: 12.0},        # sine amplitude px
}


def _blur(image: Image.Image, level: int, rng: np.random.Generator) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(PARAMS["blur"][level]))


def _noise(image: Image.Image, level: int, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=float)
    array += rng.normal(0.0, PARAMS["noise"][level], array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _bleed(image: Image.Image, level: int, rng: np.random.Generator) -> Image.Image:
    """Bleed-through composited from the mirrored page (donor fallback)."""
    gray = np.asarray(image.convert("L"), dtype=float)
    ghost = np.asarray(
        image.convert("L").transpose(Image.FLIP_LEFT_RIGHT).filter(ImageFilter.GaussianBlur(1.5)),
        dtype=float,
    )
    opacity = PARAMS["bleed"][level]
    ghost_ink = (255.0 - ghost) * opacity
    return Image.fromarray(np.clip(gray - ghost_ink, 0, 255).astype(np.uint8))


def _fade(image: Image.Image, level: int, rng: np.random.Generator) -> Image.Image:
    keep = PARAMS["fade"][level]
    array = np.asarray(image.convert("L"), dtype=float)
    array = 255.0 - (255.0 - array) * keep
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _warp(image: Image.Image, level: int, rng: np.random.Generator) -> Image.Image:
    amplitude = PARAMS["warp"][level]
    period = 140.0
    phase = rng.uniform(0.0, 2.0 * np.pi)
    array = np.asarray(image.convert("L"))
    output = np.full_like(array, 255)
    height = array.shape[0]
    for x in range(array.shape[1]):
        dy = int(round(amplitude * np.sin(2.0 * np.pi * x / period + phase)))
        if dy >= 0:
            output[dy:height, x] = array[0:height - dy, x]
        else:
            output[0:height + dy, x] = array[-dy:height, x]
    return Image.fromarray(output)


OPERATORS: Dict[str, Callable[[Image.Image, int, np.random.Generator], Image.Image]] = {
    "blur": _blur, "noise": _noise, "bleed": _bleed, "fade": _fade, "warp": _warp,
}


def apply_operator(image: Image.Image, operator: str, level: int,
                   seed: int = 0) -> Image.Image:
    """Apply one operator at one severity level, deterministically per seed."""
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator: {operator}")
    if level not in range(1, 6):
        raise ValueError("severity level must be in 1..5")
    rng = np.random.default_rng(seed)
    return OPERATORS[operator](image, level, rng)
