from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
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
    """Result labels with similarity-based quality, not a calibrated probability.

    ``confidence`` is zero unless both visible result labels form a recognized
    WIN/LOSS pair. A nonzero value has already passed the reader's absolute
    template-quality and class-separation checks.
    """

    outcome: MatchOutcome
    opponent_result: PlayerResult
    self_result: PlayerResult
    confidence: float


class MatchResultReader:
    # A relative preference alone is not evidence of a result label. Unrelated
    # menus can have a sizeable WIN/LOSE margin while matching neither template.
    MIN_TEMPLATE_SIMILARITY = 0.85
    MIN_CLASS_MARGIN = 0.08
    MIN_PATCH_BRIGHTNESS = 180.0
    MIN_PATCH_CONTRAST = 12.0

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
        # Winning and losing use different typography and decorations at each
        # side of the result page. Swapping crops from a single LOSS capture
        # does not represent the actual bottom WIN/top LOSE layout.
        templates = profile_path.resolve().parents[2] / "assets" / "match_result"
        self._side_templates = self._load_side_templates(templates)

    def read(self, frame: np.ndarray) -> MatchResultObservation:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) < 2
        ):
            return MatchResultObservation(
                MatchOutcome.UNKNOWN, PlayerResult.UNKNOWN, PlayerResult.UNKNOWN, 0.0
            )
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
            confidence=(
                min(opponent_conf, self_conf)
                if outcome is not MatchOutcome.UNKNOWN
                else 0.0
            ),
        )

    def _classify(
        self,
        frame: np.ndarray,
        region: VisionRegion,
    ) -> tuple[PlayerResult, float]:
        patch = warp_region(frame, region, 128, 48)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Descriptor normalization removes exposure information. Preserve a
        # separate visibility check so a dim/fading result is not promoted to a
        # confident match merely because its remaining outline is similar.
        brightness = float(np.percentile(np.max(patch, axis=2), 95))
        if (
            brightness < self.MIN_PATCH_BRIGHTNESS
            or float(np.std(gray)) < self.MIN_PATCH_CONTRAST
        ):
            return PlayerResult.UNKNOWN, 0.0
        descriptor = _descriptor(patch)
        win_score = _similarity(descriptor, self.win_descriptor)
        lose_score = _similarity(descriptor, self.lose_descriptor)
        side_template = self._side_templates.get(region.name)
        if side_template is not None:
            text_region, text_descriptor = side_template
            # These alternate crops contain only the result word. Decorative
            # ink, confetti and the player's name card must not become a
            # requirement for recognizing the WIN layout.
            text_patch = warp_region(frame, text_region, 128, 48)
            score = 0.0
            if (np.percentile(text_patch.max(axis=2), 95) >= self.MIN_PATCH_BRIGHTNESS
                and float(np.std(cv2.cvtColor(text_patch, cv2.COLOR_BGR2GRAY))) >= self.MIN_PATCH_CONTRAST):
                score = _similarity(_descriptor(text_patch), text_descriptor)
            if region.name == self.self_region.name:
                win_score = max(win_score, score)
            else:
                lose_score = max(lose_score, score)
        margin = win_score - lose_score
        best_score = max(win_score, lose_score)
        if (
            best_score < self.MIN_TEMPLATE_SIMILARITY
            or abs(margin) < self.MIN_CLASS_MARGIN
        ):
            return PlayerResult.UNKNOWN, 0.0
        if margin > 0:
            return PlayerResult.WIN, win_score
        return PlayerResult.LOSE, lose_score

    def _load_side_templates(self, directory: Path) -> dict:
        manifest = directory / "templates.json"
        if not manifest.is_file():
            return {}
        items = json.loads(manifest.read_text(encoding="utf-8"))
        result = {}
        for side, region in (("opponent", self.opponent_region), ("self", self.self_region)):
            item = items[side]
            path = directory / item["file"]
            template = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if template is None:
                raise ValueError(f"cannot read match result template: {path}")
            text_region = VisionRegion(region.name, "rect", tuple(map(tuple, item["points"])))
            result[region.name] = (text_region, _descriptor(template))
        return result

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
    descriptor = np.concatenate((gray_value, edge_value * 0.6))
    length = float(np.linalg.norm(descriptor))
    return descriptor / length if length > 1e-9 else np.zeros_like(descriptor)


def _normalize(value: np.ndarray) -> np.ndarray:
    flat = value.reshape(-1).astype(np.float64)
    flat -= flat.mean()
    length = float(np.linalg.norm(flat))
    return flat / length if length > 1e-9 else np.zeros_like(flat)


def _similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not np.isfinite(denominator) or denominator <= 1e-9:
        return 0.0
    cosine = float(np.dot(first, second)) / denominator
    return (float(np.clip(cosine, -1.0, 1.0)) + 1.0) / 2.0
