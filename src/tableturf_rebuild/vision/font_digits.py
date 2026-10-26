from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True, slots=True)
class DigitRecognition:
    digit: str | None
    confidence: float
    candidates: tuple[tuple[str, float], ...]


class SplatoonDigitRecognizer:
    DEFAULT_FONT = Path(r"E:\Splatoon\字体\SPLATOON2.OTF")
    GLYPH_SIZE = 64

    def __init__(self, font_path: Path | None = None):
        self.font_path = font_path or self.DEFAULT_FONT
        self.templates = {
            digit: self._templates(digit)
            for digit in "0123456789"
        }

    def recognize(self, glyph: np.ndarray) -> DigitRecognition:
        normalized = _normalize_glyph(glyph)
        scores = []
        for digit, templates in self.templates.items():
            best = 0.0
            for template in templates:
                intersection = np.count_nonzero(normalized & template)
                union = np.count_nonzero(normalized | template)
                best = max(best, intersection / union if union else 0.0)
            scores.append((digit, float(best)))
        scores.sort(key=lambda item: item[1], reverse=True)
        best_digit, best_score = scores[0]
        return DigitRecognition(
            digit=best_digit if best_score >= 0.72 else None,
            confidence=best_score,
            candidates=tuple(scores[:5]),
        )

    def _templates(self, digit: str) -> tuple[np.ndarray, ...]:
        result = []
        for size in range(20, 101, 2):
            image = Image.new("L", (160, 160), 0)
            ImageDraw.Draw(image).text(
                (10, 10),
                digit,
                font=ImageFont.truetype(str(self.font_path), size),
                fill=255,
            )
            result.append(_normalize_glyph(np.asarray(image) > 0))
        return tuple(result)


def _normalize_glyph(glyph: np.ndarray) -> np.ndarray:
    values = glyph.astype(np.uint8) * 255
    rows, columns = np.where(values > 0)
    if rows.size == 0:
        return np.zeros(
            (SplatoonDigitRecognizer.GLYPH_SIZE,) * 2,
            dtype=bool,
        )
    cropped = values[
        rows.min() : rows.max() + 1,
        columns.min() : columns.max() + 1,
    ]
    return (
        cv2.resize(
            cropped,
            (
                SplatoonDigitRecognizer.GLYPH_SIZE,
                SplatoonDigitRecognizer.GLYPH_SIZE,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
        > 127
    )
