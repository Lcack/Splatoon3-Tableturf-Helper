from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.vision.calibration import (
    VisionProfile,
    VisionRegion,
    warp_region,
)


@dataclass(frozen=True, slots=True)
class ScoreObservation:
    self_value: int | None
    opponent_value: int | None
    self_text: str
    opponent_text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class _DigitTemplate:
    digit: str
    glyph: np.ndarray


class ScoreTemplateReader:
    GLYPH_SIZE = (32, 40)

    def __init__(
        self,
        profile_dir: Path,
        template_path: Path | None = None,
    ):
        self.profile_dir = profile_dir
        config_path = template_path or (profile_dir / "score_templates.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.templates: list[_DigitTemplate] = []
        profile_paths = set()
        for reference in config.get("references", ()):
            profile_path = profile_dir / str(reference["profile"])
            profile_paths.add(profile_path)
            profile = VisionProfile.load(profile_path)
            source = Path(profile.notes.removeprefix("source=").strip())
            frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            for region in profile.regions:
                value = str(reference.get("values", {}).get(region.name, ""))
                if not value:
                    continue
                glyphs = _segment_digits(warp_region(frame, region, 192, 128))
                if len(glyphs) != len(value):
                    continue
                self.templates.extend(
                    _DigitTemplate(digit, glyph)
                    for digit, glyph in zip(value, glyphs, strict=True)
                )
        if not self.templates:
            raise ValueError("no usable score digit templates")
        self.profile_paths = tuple(sorted(profile_paths))

    def read(self, frame: np.ndarray) -> ScoreObservation:
        values = {}
        confidences = []
        for profile_path in self.profile_paths:
            profile = VisionProfile.load(profile_path)
            for region in profile.regions:
                text, confidence = self._read_region(frame, region)
                values[region.name] = text
                confidences.append(confidence)
        self_text = values.get("我方分数", "??")
        opponent_text = values.get("对方分数", "??")
        return ScoreObservation(
            self_value=int(self_text) if self_text.isdigit() else None,
            opponent_value=(
                int(opponent_text) if opponent_text.isdigit() else None
            ),
            self_text=self_text,
            opponent_text=opponent_text,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
        )

    def _read_region(
        self,
        frame: np.ndarray,
        region: VisionRegion,
    ) -> tuple[str, float]:
        glyphs = _segment_digits(warp_region(frame, region, 192, 128))
        if not glyphs:
            return "??", 0.0
        text = []
        confidences = []
        for glyph in glyphs:
            digit, confidence = self._match(glyph)
            text.append(digit)
            confidences.append(confidence)
        return "".join(text), float(np.mean(confidences))

    def _match(self, glyph: np.ndarray) -> tuple[str, float]:
        scores = {}
        for template in self.templates:
            intersection = np.count_nonzero(glyph & template.glyph)
            union = np.count_nonzero(glyph | template.glyph)
            score = intersection / union if union else 0.0
            scores[template.digit] = max(scores.get(template.digit, 0.0), score)
        digit, score = max(scores.items(), key=lambda item: item[1])
        return (digit, float(score)) if score >= 0.72 else ("?", float(score))


def _segment_digits(patch: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )
    projection = np.count_nonzero(binary, axis=0)
    minimum = max(3, binary.shape[0] // 30)
    active = np.flatnonzero(projection >= minimum)
    if active.size == 0:
        return []
    groups = []
    start = previous = int(active[0])
    for value in active[1:]:
        value = int(value)
        if value > previous + 1:
            groups.append((start, previous))
            start = value
        previous = value
    groups.append((start, previous))
    glyphs = []
    for x1, x2 in groups:
        if x2 - x1 + 1 < 8:
            continue
        columns = binary[:, x1 : x2 + 1]
        rows = np.flatnonzero(np.count_nonzero(columns, axis=1))
        if rows.size == 0 or rows[-1] - rows[0] + 1 < 12:
            continue
        glyph = columns[rows[0] : rows[-1] + 1, :]
        glyph = cv2.resize(
            glyph,
            ScoreTemplateReader.GLYPH_SIZE,
            interpolation=cv2.INTER_NEAREST,
        )
        glyphs.append(glyph > 0)
    return glyphs
