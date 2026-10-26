from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.vision.calibration import (
    VisionProfile,
    VisionRegion,
    warp_region,
)


class MatchOutcome(StrEnum):
    WIN = "WIN"
    LOSS = "LOSS"
    DRAW = "DRAW"
    UNKNOWN = "UNKNOWN"


class PlayerResult(StrEnum):
    WIN = "WIN"
    LOSE = "LOSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class MatchResultObservation:
    outcome: MatchOutcome
    opponent_result: PlayerResult
    self_result: PlayerResult
    confidence: float


class MatchResultReader:
    def __init__(
        self,
        profile_path: Path,
    ):
        self.profile_path = profile_path
        self.profile = VisionProfile.load(profile_path)
        source = Path(self.profile.notes.removeprefix("source=").strip())
        reference = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if reference is None:
            raise ValueError(f"cannot read match result reference: {source}")
        self.opponent_region = self._region("对方结果")
        self.self_region = self._region("我方结果")
        self.win_descriptor = _descriptor(
            warp_region(reference, self.opponent_region, 128, 48)
        )
        self.lose_descriptor = _descriptor(
            warp_region(reference, self.self_region, 128, 48)
        )

    def read(self, frame: np.ndarray) -> MatchResultObservation:
        opponent_result, opponent_conf = self._classify(
            frame,
            self.opponent_region,
        )
        self_result, self_conf = self._classify(frame, self.self_region)
        if (
            opponent_result is PlayerResult.WIN
            and self_result is PlayerResult.LOSE
        ):
            outcome = MatchOutcome.LOSS
        elif (
            opponent_result is PlayerResult.LOSE
            and self_result is PlayerResult.WIN
        ):
            outcome = MatchOutcome.WIN
        elif (
            opponent_result is PlayerResult.UNKNOWN
            or self_result is PlayerResult.UNKNOWN
        ):
            outcome = MatchOutcome.UNKNOWN
        else:
            outcome = MatchOutcome.UNKNOWN
        return MatchResultObservation(
            outcome=outcome,
            opponent_result=opponent_result,
            self_result=self_result,
            confidence=min(opponent_conf, self_conf),
        )

    def _classify(
        self,
        frame: np.ndarray,
        region: VisionRegion,
    ) -> tuple[PlayerResult, float]:
        descriptor = _descriptor(warp_region(frame, region, 128, 48))
        win_score = _similarity(descriptor, self.win_descriptor)
        lose_score = _similarity(descriptor, self.lose_descriptor)
        margin = win_score - lose_score
        if margin >= 0.08:
            return PlayerResult.WIN, min(1.0, margin * 2.0)
        if margin <= -0.08:
            return PlayerResult.LOSE, min(1.0, -margin * 2.0)
        return PlayerResult.UNKNOWN, max(0.0, 1.0 - abs(margin) / 0.08)

    def _region(self, name: str) -> VisionRegion:
        region = self.profile.region_map().get(name)
        if region is None:
            raise ValueError(
                f"{self.profile_path.name}: missing result region {name}"
            )
        return region


def _descriptor(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    edges = cv2.Canny(gray.astype(np.uint8), 40, 100).astype(np.float32)
    gray_value = _normalize(gray)
    edge_value = _normalize(edges)
    return np.concatenate((gray_value, edge_value * 0.6))


def _normalize(value: np.ndarray) -> np.ndarray:
    flat = value.reshape(-1).astype(np.float64)
    flat -= flat.mean()
    length = float(np.linalg.norm(flat))
    return flat / length if length > 1e-9 else np.zeros_like(flat)


def _similarity(first: np.ndarray, second: np.ndarray) -> float:
    return (float(np.dot(first, second)) + 1.0) / 2.0
