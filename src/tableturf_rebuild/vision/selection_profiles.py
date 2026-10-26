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
        result = []
        for item in scores:
            brightness = self._slot_brightness(frame, item.key)
            margin = self._brightness_margin(item.key, brightness)
            bonus = max(-0.15, min(0.35, margin / 100.0))
            result.append(
                SelectionProfileMatch(
                    variant=item.key,
                    focus_slot=_focus_slot(item.key),
                    score=item.score + bonus,
                    profile_path=self.paths[item.key],
                    profile_score=item.score,
                    brightness_margin=margin,
                    slot_brightness=brightness,
                )
            )
        result.sort(key=lambda match: match.score, reverse=True)
        return tuple(result)

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
