from pathlib import Path

import cv2

from tableturf_rebuild.vision.score_reader import ScoreTemplateReader


ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = ROOT / "profiles" / "vision"
FRAME = (
    ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-221749"
    / "session-20261021-222704_annotations"
    / "045_video_state.png"
)


def test_score_templates_read_known_reference() -> None:
    reader = ScoreTemplateReader(VISION_DIR)
    frame = cv2.imread(str(FRAME))
    assert frame is not None

    observation = reader.read(frame)

    assert observation.self_value == 52
    assert observation.opponent_value == 48
    assert observation.self_text == "52"
    assert observation.opponent_text == "48"
    assert observation.confidence >= 0.99
