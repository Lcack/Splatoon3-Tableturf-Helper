from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.vision.calibration import (
    VisionProfile,
    VisionRegion,
    warp_region,
)
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.font_digits import SplatoonDigitRecognizer
from tableturf_rebuild.vision.font_npc_name import FontClosedSetRecognizer


@dataclass(frozen=True, slots=True)
class NpcMenuObservation:
    npc_id: str | None
    npc_name: str | None
    level_wins: tuple[int | None, int | None, int | None]
    difficulty: int | None
    stage_id: str | None
    stage_name: str | None
    binding_consistent: bool | None
    name_confidence: float
    wins_confidence: float
    difficulty_confidence: float
    stage_confidence: float


class NpcMenuReader:
    GLYPH_SIZE = (32, 40)

    def __init__(self, profile_dir: Path):
        self.catalog = GameCatalog.from_package()
        self.profiles = {
            "NPC_ROSTER": VisionProfile.load(profile_dir / "npc选择.json"),
            "MATCH_SETUP": VisionProfile.load(profile_dir / "难度选择.json"),
        }
        self.name_recognizer = FontClosedSetRecognizer.default()
        self.stage_recognizer = FontClosedSetRecognizer(
            tuple(
                stage.name_zh
                for stage in self.catalog.stages.values()
            )
        )
        self.npc_id_by_name = {
            npc.name_zh: npc.npc_id
            for npc in self.catalog.npcs.values()
        }
        self.stage_id_by_name = {
            stage.name_zh: stage.stage_id
            for stage in self.catalog.stages.values()
        }
        self.digit_recognizer = SplatoonDigitRecognizer()

    def read(self, frame: np.ndarray, state: str) -> NpcMenuObservation:
        profile = self.profiles[state]
        name, name_confidence = self._read_name(frame, profile)
        stage_name, stage_confidence = self._read_stage(frame, profile)
        wins = tuple(
            self._read_win(frame, profile.region_map()[f"等级{level}胜利数"])
            for level in (1, 2, 3)
        )
        difficulty = None
        difficulty_confidence = 0.0
        if state == "MATCH_SETUP":
            difficulty, difficulty_confidence = self._read_difficulty(
                frame,
                profile.region_map()["难度选择"],
            )
        win_values = tuple(value for value, _ in wins)
        npc_id = self.npc_id_by_name.get(name or "")
        stage_id = self.stage_id_by_name.get(stage_name or "")
        if stage_id is not None and stage_confidence >= 0.55:
            allowed_names = tuple(
                npc.name_zh
                for npc in self.catalog.npcs.values()
                if npc.name_zh and stage_id in npc.map_ids
            )
            if allowed_names and len(allowed_names) < len(self.catalog.npcs):
                restricted_name, restricted_confidence = self._read_name(
                    frame,
                    profile,
                    names=allowed_names,
                )
                if restricted_name is not None and (
                    name is None
                    or restricted_confidence >= name_confidence - 0.01
                ):
                    name = restricted_name
                    name_confidence = restricted_confidence
                    npc_id = self.npc_id_by_name.get(name)
        binding_consistent = None
        if npc_id is not None and stage_id is not None:
            npc = self.catalog.npcs[npc_id]
            if difficulty is not None and 1 <= difficulty <= len(npc.map_ids):
                binding_consistent = stage_id == npc.map_ids[difficulty - 1]
            else:
                binding_consistent = stage_id in npc.map_ids
            if binding_consistent:
                stage_confidence = min(1.0, stage_confidence + 0.15)
            else:
                stage_id = None
                stage_name = None
                stage_confidence *= 0.5
        return NpcMenuObservation(
            npc_id=npc_id,
            npc_name=name,
            level_wins=win_values,
            difficulty=difficulty,
            stage_id=stage_id,
            stage_name=stage_name,
            binding_consistent=binding_consistent,
            name_confidence=name_confidence,
            wins_confidence=(
                float(np.mean([confidence for _, confidence in wins]))
                if wins
                else 0.0
            ),
            difficulty_confidence=difficulty_confidence,
            stage_confidence=stage_confidence,
        )

    def _read_name(
        self,
        frame: np.ndarray,
        profile: VisionProfile,
        names: tuple[str, ...] | None = None,
    ) -> tuple[str | None, float]:
        region = _expand_name_region(profile.region_map()["npc名字"])
        recognition = self.name_recognizer.recognize(
            frame,
            region,
            names=names,
        )
        return recognition.name, recognition.confidence

    def _read_stage(
        self,
        frame: np.ndarray,
        profile: VisionProfile,
    ) -> tuple[str | None, float]:
        region = profile.region_map()["对战场地名"]
        recognition = self.stage_recognizer.recognize(frame, region)
        return recognition.name, recognition.confidence

    def _read_win(
        self,
        frame: np.ndarray,
        region: VisionRegion,
    ) -> tuple[int | None, float]:
        glyphs = _segment_win_digits(warp_region(frame, region, 360, 72))
        if not glyphs:
            return None, 0.0
        text = []
        confidences = []
        for glyph in glyphs:
            digit, confidence = self._match_digit(glyph)
            text.append(digit)
            confidences.append(confidence)
        value_text = "".join(text)
        return (
            int(value_text) if value_text.isdigit() else None,
            float(np.mean(confidences)),
        )

    def _match_digit(self, glyph: np.ndarray) -> tuple[str, float]:
        recognition = self.digit_recognizer.recognize(glyph)
        return recognition.digit or "?", recognition.confidence

    def _read_difficulty(
        self,
        frame: np.ndarray,
        region: VisionRegion,
    ) -> tuple[int | None, float]:
        patch = warp_region(frame, region, 350, 67)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        candidates = []
        for index in range(1, count):
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            if (
                area >= 80
                and 5 <= width <= 25
                and 20 <= height <= 35
                and y >= 10
            ):
                candidates.append((x, y, width, height, index))
        if not candidates:
            return None, 0.0

        # The selector is a sentence followed by the current level digit.
        # The digit is the right-most tall component.
        x, y, width, height, index = max(candidates)
        glyph = labels[y : y + height, x : x + width] == index
        recognition = self.digit_recognizer.recognize(glyph)
        if (
            recognition.digit not in {"1", "2", "3"}
            or recognition.confidence < 0.72
        ):
            return None, recognition.confidence
        return int(recognition.digit), recognition.confidence


def _segment_win_digits(patch: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if x >= 190 and height >= 30 and area >= 50:
            candidates.append((x, y, width, height, index))
    if not candidates:
        return []
    glyphs = []
    for x, y, width, height, index in sorted(candidates):
        glyph = (labels[y : y + height, x : x + width] == index).astype(
            np.uint8
        ) * 255
        glyphs.append(
            cv2.resize(
                glyph,
                NpcMenuReader.GLYPH_SIZE,
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
    return glyphs


def _expand_name_region(region: VisionRegion) -> VisionRegion:
    if region.kind != "quad":
        return region
    top_left, top_right, bottom_right, bottom_left = region.points
    left_expand = 0.04
    top_slope = (
        (top_right[1] - top_left[1])
        / max(1e-9, top_right[0] - top_left[0])
    )
    bottom_slope = (
        (bottom_right[1] - bottom_left[1])
        / max(1e-9, bottom_right[0] - bottom_left[0])
    )
    return VisionRegion(
        name=region.name,
        kind="quad",
        points=(
            (
                top_left[0] - left_expand,
                top_left[1] - top_slope * left_expand,
            ),
            top_right,
            bottom_right,
            (
                bottom_left[0] - left_expand,
                bottom_left[1] - bottom_slope * left_expand,
            ),
        ),
        grid_rows=region.grid_rows,
        grid_cols=region.grid_cols,
    )
