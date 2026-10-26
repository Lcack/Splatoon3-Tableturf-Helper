from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.vision.calibration import VisionRegion
from tableturf_rebuild.vision.match_result_reader import (
    MatchOutcome,
    MatchResultReader,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "vision" / "比赛结果.json"
SOURCE = (
    ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-221749"
    / "session-20261021-222704_annotations"
    / "011_video_state.png"
)


def _bounds(region: VisionRegion, width: int, height: int) -> tuple[int, int, int, int]:
    points = region.pixel_points(width, height)
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def test_reference_result_is_loss() -> None:
    reader = MatchResultReader(PROFILE)
    frame = cv2.imread(str(SOURCE))
    assert frame is not None

    observation = reader.read(frame)

    assert observation.outcome is MatchOutcome.LOSS
    assert observation.opponent_result.value == "WIN"
    assert observation.self_result.value == "LOSE"


def test_swapped_result_text_is_win() -> None:
    reader = MatchResultReader(PROFILE)
    frame = cv2.imread(str(SOURCE))
    assert frame is not None
    height, width = frame.shape[:2]
    opponent_box = _bounds(reader.opponent_region, width, height)
    self_box = _bounds(reader.self_region, width, height)
    ox1, oy1, ox2, oy2 = opponent_box
    sx1, sy1, sx2, sy2 = self_box
    opponent_patch = frame[oy1:oy2, ox1:ox2].copy()
    self_patch = frame[sy1:sy2, sx1:sx2].copy()
    swapped = frame.copy()
    swapped[sy1:sy2, sx1:sx2] = cv2.resize(
        opponent_patch,
        (sx2 - sx1, sy2 - sy1),
        interpolation=cv2.INTER_AREA,
    )
    swapped[oy1:oy2, ox1:ox2] = cv2.resize(
        self_patch,
        (ox2 - ox1, oy2 - oy1),
        interpolation=cv2.INTER_AREA,
    )

    observation = reader.read(swapped)

    assert observation.outcome is MatchOutcome.WIN
    assert observation.opponent_result.value == "LOSE"
    assert observation.self_result.value == "WIN"
