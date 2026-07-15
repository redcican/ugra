"""Readers (Section 3.2): the TrOCR specialist and the served Qwen3-VL
generalist. Both are thin wrappers with lazy imports so that the package
core (calibration, metrics, mock ecosystem) stays dependency-free.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Optional, Tuple

from .base import ReadContext, Reader


class TrOCRReader(Reader):
    """Specialist: TrOCR-large, fine-tuned per corpus (cost tier 1)."""

    name = "specialist"
    cost = 1.0

    def __init__(self, checkpoint: str = "microsoft/trocr-large-handwritten",
                 device: str = "cuda") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # lazy

        self._processor = TrOCRProcessor.from_pretrained(self.checkpoint)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.checkpoint).to(self.device)
        self._model.eval()

    def read(self, image: Any, context: ReadContext) -> Tuple[str, float]:
        if self._model is None:
            self._load()
        import torch  # lazy

        pixel_values = self._processor(image.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            out = self._model.generate(
                pixel_values.to(self.device),
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = self._processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        scores = torch.stack(out.scores, dim=1).softmax(-1)
        token_conf = scores.max(-1).values.mean().item() if scores.numel() else 0.0
        return text, float(token_conf)


class ServedVLMReader(Reader):
    """Generalist: an OpenAI-compatible served VLM (Qwen3-VL-30B-A3B-Instruct).

    Greedy decoding; the self-reported confidence is requested in-band and
    logged as evidence only (never used as a stopping criterion).
    """

    name = "generalist"
    cost = 3.0

    PROMPT = (
        "Transcribe the text in this document image exactly, preserving "
        "spelling. Reply as JSON: {\"text\": ..., \"confidence\": 0-1}."
    )

    def __init__(self, endpoint: str, model: str, timeout: float = 120.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _encode(image: Any) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def read(self, image: Any, context: ReadContext) -> Tuple[str, float]:
        import json

        import requests  # lazy

        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{self._encode(image)}"}},
                    {"type": "text", "text": self.PROMPT},
                ],
            }],
        }
        response = requests.post(f"{self.endpoint}/chat/completions",
                                 json=payload, timeout=self.timeout)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            return str(parsed.get("text", "")), float(parsed.get("confidence", 0.0))
        except (ValueError, TypeError):
            return content.strip(), 0.0


def build_reader(kind: str, **kwargs: Any) -> Reader:
    if kind == "trocr":
        return TrOCRReader(**kwargs)
    if kind == "served_vlm":
        return ServedVLMReader(**kwargs)
    raise ValueError(f"unknown reader kind: {kind}")
