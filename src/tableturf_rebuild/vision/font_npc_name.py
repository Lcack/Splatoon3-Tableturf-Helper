from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.calibration import VisionRegion, warp_region


@dataclass(frozen=True, slots=True)
class NpcNameCandidate:
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class NpcNameRecognition:
    name: str | None
    confidence: float
    candidates: tuple[NpcNameCandidate, ...]


class FontNpcNameRecognizer:
    DEFAULT_FONTS = (
        Path(r"E:\Splatoon\字体\华康新综艺体W7-P.ttf"),
        Path(r"E:\Splatoon\字体\华康新综艺体W9-P.ttf"),
    )
    GLYPH_SIZE = 64

    def __init__(
        self,
        names: tuple[str, ...],
        font_path: Path | None = None,
        font_paths: tuple[Path, ...] | None = None,
    ):
        self.names = tuple(name for name in names if name)
        if font_paths is not None:
            self.font_paths = font_paths
        elif font_path is not None:
            self.font_paths = (font_path,)
        else:
            self.font_paths = self.DEFAULT_FONTS
        self.font_path = self.font_paths[0]
        self.characters = tuple(sorted(set("".join(self.names))))
        self.templates = {
            character: self._character_templates(character)
            for character in self.characters
        }

    @classmethod
    def default(cls) -> "FontNpcNameRecognizer":
        catalog = GameCatalog.from_package()
        return cls(
            tuple(
                npc.name_zh
                for npc in catalog.npcs.values()
                if npc.name_zh
            )
        )

    def recognize(
        self,
        frame: np.ndarray,
        region: VisionRegion,
        names: tuple[str, ...] | None = None,
    ) -> NpcNameRecognition:
        patch = warp_region(frame, region, 420, 120)
        allowed_names = self.names if names is None else names
        glyphs_by_length = {
            length: _character_bands(patch, length)
            for length in sorted({len(name) for name in allowed_names})
        }
        candidates = []
        for name in allowed_names:
            glyphs = glyphs_by_length.get(len(name), [])
            if len(name) != len(glyphs):
                continue
            scores = [
                self._match_character(glyph, character)
                for glyph, character in zip(glyphs, name, strict=True)
            ]
            candidates.append(
                NpcNameCandidate(
                    name=name,
                    score=float(
                        0.8 * np.mean(scores)
                        + 0.2 * min(scores, default=0.0)
                    ),
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        if not candidates:
            return NpcNameRecognition(None, 0.0, ())
        best = candidates[0]
        second = candidates[1].score if len(candidates) > 1 else 0.0
        margin = best.score - second
        same_length_count = sum(len(name) == len(best.name) for name in allowed_names)
        unique_restricted = names is not None and same_length_count == 1
        accepted = (
            best.score >= 0.58
            if unique_restricted
            else best.score >= 0.72 and margin >= 0.02
        )
        return NpcNameRecognition(
            name=best.name if accepted else None,
            confidence=min(1.0, best.score) if accepted else best.score,
            candidates=tuple(candidates[:5]),
        )

    def _character_templates(
        self,
        character: str,
    ) -> np.ndarray:
        return _build_character_templates(character, self.font_paths)

    def _match_character(
        self,
        glyph: np.ndarray,
        character: str,
    ) -> float:
        templates = self.templates[character]
        intersection = np.count_nonzero(
            templates & glyph[np.newaxis, :, :],
            axis=(1, 2),
        )
        union = np.count_nonzero(
            templates | glyph[np.newaxis, :, :],
            axis=(1, 2),
        )
        scores = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float64),
            where=union > 0,
        )
        return float(scores.max(initial=0.0))


def _character_bands(
    patch: np.ndarray,
    expected_count: int,
) -> list[np.ndarray]:
    if expected_count <= 0:
        return []
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    projection = np.count_nonzero(binary, axis=0)
    active = np.flatnonzero(projection >= 2)
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

    clusters = []
    start, end = groups[0]
    for group_start, group_end in groups[1:]:
        if group_start > end + 12:
            clusters.append((start, end))
            start = group_start
        end = group_end
    clusters.append((start, end))
    clusters = [
        (x1, x2)
        for x1, x2 in clusters
        if x2 - x1 + 1 >= max(10, expected_count * 5)
    ]
    if not clusters:
        return []

    x1, x2 = max(clusters, key=lambda item: item[1] - item[0])
    cluster_groups = [
        (left, right)
        for left, right in groups
        if left >= x1 and right <= x2
    ]
    valid_groups = []
    for left, right in cluster_groups:
        rows = np.flatnonzero(
            np.count_nonzero(binary[:, left : right + 1], axis=1)
        )
        if (
            right - left + 1 >= 5
            and rows.size
            and rows[-1] - rows[0] + 1 >= 8
        ):
            valid_groups.append((left, right + 1))

    if len(valid_groups) == expected_count:
        boundaries = [segment[0] for segment in valid_groups] + [
            valid_groups[-1][1]
        ]
    else:
        boundaries = [x1]
        target_width = (x2 - x1 + 1) / expected_count
        search_radius = max(3, round(target_width * 0.18))
        for index in range(1, expected_count):
            target = round(x1 + target_width * index)
            lower = max(boundaries[-1] + 5, target - search_radius)
            upper = min(
                x2 - 5 * (expected_count - index),
                target + search_radius,
            )
            if lower >= upper:
                boundary = target
            else:
                scores = []
                for boundary in range(lower, upper + 1):
                    valley = sum(
                        int(projection[position])
                        for position in range(
                            max(x1, boundary - 1),
                            min(x2 + 1, boundary + 2),
                        )
                    )
                    scores.append(
                        (valley + abs(boundary - target) * 3, boundary)
                    )
                boundary = min(scores)[1]
            boundaries.append(boundary)
        boundaries.append(x2 + 1)

    glyphs = []
    for left, right in zip(
        boundaries[:-1],
        boundaries[1:],
        strict=True,
    ):
        column_slice = binary[:, left:right]
        rows = np.flatnonzero(np.count_nonzero(column_slice, axis=1))
        if rows.size == 0 or rows[-1] - rows[0] + 1 < 4:
            return []
        glyphs.append(
            _normalize_glyph(
                column_slice[rows[0] : rows[-1] + 1, :] > 0
            )
        )
    return glyphs


@lru_cache(maxsize=None)
def _build_character_templates(
    character: str,
    font_paths: tuple[Path, ...],
) -> np.ndarray:
    variants: dict[bytes, np.ndarray] = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for font_path in font_paths:
        for size in range(24, 85, 4):
            font = ImageFont.truetype(str(font_path), size)
            image = Image.new("L", (170, 170), 0)
            ImageDraw.Draw(image).text(
                (10, 10),
                character,
                font=font,
                fill=255,
            )
            normalized = _normalize_glyph(np.asarray(image) > 0)
            for variant in (
                normalized,
                cv2.dilate(
                    normalized.astype(np.uint8),
                    kernel,
                )
                > 0,
                cv2.erode(
                    normalized.astype(np.uint8),
                    kernel,
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0,
            ):
                variants[variant.tobytes()] = variant
    return np.stack(tuple(variants.values()))


def _normalize_glyph(glyph: np.ndarray) -> np.ndarray:
    values = glyph.astype(np.uint8) * 255
    rows, columns = np.where(values > 0)
    if rows.size == 0:
        return np.zeros(
            (FontNpcNameRecognizer.GLYPH_SIZE,) * 2,
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
                FontNpcNameRecognizer.GLYPH_SIZE,
                FontNpcNameRecognizer.GLYPH_SIZE,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
        > 127
    )


FontClosedSetRecognizer = FontNpcNameRecognizer
