from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.vision.calibration import warp_region
from tableturf_rebuild.vision.profile_matcher import VisionProfileMatcher
from tableturf_rebuild.vision.profile_semantics import (
    ProfileSemantics,
    RegionRole,
)


@dataclass(frozen=True, slots=True)
class SelectionProfileMatch:
    variant: str
    focus_slot: tuple[int, int] | None
    score: float
    profile_path: Path
    profile_score: float = 0.0
    brightness_margin: float = 0.0
    slot_brightness: tuple[tuple[int, int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class HandVisibilityObservation:
    selection_visible: bool | None
    ink_value: float
    bright_fraction: float
    cursor_slot: tuple[int, int] | None = None


def hand_ink_visibility(hsv: np.ndarray) -> tuple[bool | None, float, float]:
    """Measure card ink, excluding the black grid and gray illustration."""
    mask = (
        (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 55)
        & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 60)
    )
    values = hsv[:, :, 2][mask]
    if len(values) < 16:
        return None, 0.0, 0.0
    ink_value = float(np.percentile(values, 90))
    bright_fraction = float(np.mean(values >= 190))
    visible = (
        True if ink_value >= 200 and bright_fraction >= 0.25
        else False if ink_value <= 170 and bright_fraction <= 0.05
        else None
    )
    return visible, ink_value, bright_fraction


class SelectionProfileResolver:
    def __init__(self, profile_dir: Path):
        self.profile_dir = profile_dir
        index = json.loads(
            (profile_dir / "index.json").read_text(encoding="utf-8")
        )
        values = index.get("selection_geometry", {})
        self.paths = {
            str(variant): profile_dir / str(filename)
            for variant, filename in values.items()
        }
        self.semantics = {
            variant: ProfileSemantics.load(path)
            for variant, path in self.paths.items()
            if path.exists()
        }
        self.matcher = VisionProfileMatcher(
            {
                variant: (path,)
                for variant, path in self.paths.items()
            }
        )

    @classmethod
    def default(cls) -> "SelectionProfileResolver":
        profile_dir = Path(__file__).resolve().parents[3] / "profiles" / "vision"
        return cls(profile_dir)

    def match(self, frame: np.ndarray) -> tuple[SelectionProfileMatch, ...]:
        scores = self.matcher.score(frame, keys=set(self.paths))
        cursor = self.hand_cursor_slot(frame) or self.cursor_slot(frame)
        if cursor is not None and cursor[0] == 3:
            # The bottom buttons remove the white card arrow. The original
            # six-slot yellow cursor lets a resumed task navigate back safely.
            return (SelectionProfileMatch("idle", cursor, 0.99, self.paths["idle"]),)
        result = []
        for item in scores:
            brightness = self._slot_brightness(frame, item.key)
            margin = self._brightness_margin(item.key, brightness)
            bonus = max(-0.15, min(0.35, margin / 100.0))
            score = item.score + bonus
            if cursor is not None:
                # A disabled selected card is dimmer than its neighbors. The
                # white down-arrow is direct focus evidence; brightness is not.
                score = (0.95 + 0.05 * min(1.0, item.score)
                         if _focus_slot(item.key) == cursor
                         else 0.5 * min(1.0, item.score))
            result.append(
                SelectionProfileMatch(
                    variant=item.key,
                    focus_slot=_focus_slot(item.key),
                    score=score,
                    profile_path=self.paths[item.key],
                    profile_score=item.score,
                    brightness_margin=margin,
                    slot_brightness=brightness,
                )
            )
        result.sort(key=lambda match: match.score, reverse=True)
        return tuple(result)

    @staticmethod
    def button_cursor_slot(frame: np.ndarray) -> tuple[int, int] | None:
        cursor = SelectionProfileResolver.hand_cursor_slot(frame)
        return cursor if cursor is not None and cursor[0] == 3 else None

    @staticmethod
    def hand_cursor_slot(frame: np.ndarray) -> tuple[int, int] | None:
        """Animated yellow squid at all six hand/button slots (original GPL-3.0)."""
        height, width = frame.shape[:2]
        hits = []
        slots = ((1, 1, 470, 14), (1, 2, 480, 266),
                 (2, 1, 793, 14), (2, 2, 797, 266),
                 (3, 1, 909, 14), (3, 2, 898, 266))
        for slot_row, col, row, left in slots:
            # Cover the squid's animation excursion instead of requiring a
            # fixed percentage of one 30 px crop at a particular phase.
            patch = frame[max(0, round((row - 12) * height / 1080)):round((row + 42) * height / 1080),
                          max(0, round((left - 12) * width / 1920)):round((left + 42) * width / 1920)]
            if not patch.size:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (28, 100, 150), (38, 255, 255))
            _, _, components, _ = cv2.connectedComponentsWithStats(mask)
            largest = max(components[1:, cv2.CC_STAT_AREA], default=0)
            if largest >= 150 * (height / 1080) ** 2:
                hits.append((slot_row, col))
        return hits[0] if len(hits) == 1 else None

    @staticmethod
    def cursor_slot(frame: np.ndarray) -> tuple[int, int] | None:
        """Find the animated white down-arrow above the selected hand card."""
        height, width = frame.shape[:2]
        scale = height / 1080.0
        hsv = cv2.cvtColor(frame[:int(height * 0.56), :int(width * 0.28)],
                           cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 180)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        slots = set()
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if not (150 * scale ** 2 <= area <= 1600 * scale ** 2
                    and 22 * scale <= w <= 70 * scale
                    and 14 * scale <= h <= 50 * scale):
                continue
            row = (1 if 0.13 * height <= y <= 0.23 * height
                   else 2 if 0.43 * height <= y <= 0.53 * height else None)
            if row is None or not 0.04 * width < x + w / 2 < 0.27 * width:
                continue
            triangle = cv2.approxPolyDP(contour, 0.045 * cv2.arcLength(contour, True), True)
            if len(triangle) != 3:
                continue
            vertices = sorted(triangle[:, 0].tolist(), key=lambda point: point[1])
            top_a, top_b, apex = vertices
            if (abs(top_a[1] - top_b[1]) > h * 0.25
                    or apex[1] - max(top_a[1], top_b[1]) < h * 0.65
                    or abs(apex[0] - (top_a[0] + top_b[0]) / 2) > w * 0.25):
                continue
            slots.add((row, 1 if x + w / 2 < 0.14 * width else 2))
        return next(iter(slots)) if len(slots) == 1 else None

    def hand_visibility(self, frame: np.ndarray) -> HandVisibilityObservation:
        """Measure the yellow hand ink independently of card identity/focus.

        In the captured UI, selection ink is bright and placement ink is dim.
        Leave intermediate values unknown so transitions are not forced by them.
        """
        semantics = self.semantics.get("idle")
        cursor = self.hand_cursor_slot(frame) or self.cursor_slot(frame)
        values = []
        if semantics is not None:
            for item in semantics.by_role(RegionRole.HAND_GRID):
                hsv = cv2.cvtColor(
                    warp_region(frame, item.region, 64, 64), cv2.COLOR_BGR2HSV
                )
                mask = (
                    (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 55)
                    & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 60)
                )
                values.extend(hsv[:, :, 2][mask].tolist())
        if len(values) < 64:
            return HandVisibilityObservation(True if cursor else None, 0.0, 0.0, cursor)
        ink_value = float(np.percentile(values, 90))
        bright_fraction = float(np.mean(np.asarray(values) >= 190))
        visible = (
            True if ink_value >= 200 and bright_fraction >= 0.25
            else False if ink_value <= 170 and bright_fraction <= 0.05
            else None
        )
        return HandVisibilityObservation(True if cursor else visible, ink_value, bright_fraction, cursor)

    def _slot_brightness(
        self,
        frame: np.ndarray,
        variant: str,
    ) -> tuple[tuple[int, int, float], ...]:
        semantics = self.semantics.get(variant)
        if semantics is None:
            return ()
        values = []
        for item in semantics.by_role(RegionRole.HAND_CARD):
            if item.slot_row is None or item.slot_col is None:
                continue
            patch = warp_region(frame, item.region, 64, 96)
            value = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
            mask = np.ones(value.shape, dtype=bool)
            mask[20:76, 14:50] = False
            values.append(
                (
                    item.slot_row,
                    item.slot_col,
                    float(value[mask].mean()),
                )
            )
        return tuple(sorted(values))

    @staticmethod
    def _brightness_margin(
        variant: str,
        brightness: tuple[tuple[int, int, float], ...],
    ) -> float:
        target = _focus_slot(variant)
        if target is None or not brightness:
            return 0.0
        values = {
            (row, col): value for row, col, value in brightness
        }
        if target not in values:
            return 0.0
        others = [
            value for slot, value in values.items() if slot != target
        ]
        if not others:
            return 0.0
        return values[target] - float(np.median(others))

    def resolve(
        self,
        frame: np.ndarray,
        minimum_score: float = 0.75,
        minimum_margin: float = 0.01,
    ) -> SelectionProfileMatch | None:
        scores = self.match(frame)
        if not scores or scores[0].score < minimum_score:
            return None
        if len(scores) > 1 and scores[0].score - scores[1].score < minimum_margin:
            return None
        return scores[0]


def _focus_slot(variant: str) -> tuple[int, int] | None:
    prefix = "focus_"
    if not variant.startswith(prefix):
        return None
    parts = variant.removeprefix(prefix).split("_")
    if len(parts) != 2:
        return None
    return int(parts[0]), int(parts[1])
