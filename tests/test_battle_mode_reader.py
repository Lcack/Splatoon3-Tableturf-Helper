from pathlib import Path

import cv2
import numpy as np
import pytest

from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.vision.battle_mode_reader import BattleModeReader


ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "artifacts/state_capture/run-20261021-221749/session-20261021-222704_annotations"


@pytest.mark.parametrize("sample, expected", [
    ("008", False),  # ordinary selection, SP=0, disabled gray button
    ("045", False),  # ordinary selection, SP sufficient for a card, blue button
    ("017", False),  # ordinary placement, blue button
    ("010", True), ("024", True),  # real annotated ON selection frames
    ("027", True),  # real annotated ON placement, dim gray button
    ("036", True),  # both cards revealed; toggle still has its ON label
])
@pytest.mark.parametrize("height", [1080, 720, 540])
def test_button_label_disambiguates_special_mode_from_availability(sample, expected, height):
    frame = cv2.imread(str(FRAMES / f"{sample}_video_state.png"))
    assert frame is not None
    frame = cv2.resize(frame, (round(height * 16 / 9), height))
    result = BattleModeReader().read(frame)
    assert result.special_mode is expected
    assert result.confidence >= 0.88
    if sample == "008" and height == 1080:
        # This ordinary disabled button passes the old dark-HSV ON test.
        assert result.dark_fraction > 0.4


@pytest.mark.parametrize("sample", ["001", "002", "011", "012", "029", "030", "031", "039"])
def test_nonbattle_screens_are_unknown(sample):
    frame = cv2.imread(str(FRAMES / f"{sample}_video_state.png"))
    assert BattleModeReader().read(frame).special_mode is None


def test_dark_capture_and_missing_text_are_unknown():
    frame = cv2.imread(str(FRAMES / "024_video_state.png"))
    reader = BattleModeReader()
    assert reader.read((frame.astype(float) * .4).astype(np.uint8)).special_mode is None
    no_label = frame.copy()
    no_label[847:897, 330:410] = (0, 230, 230)
    assert reader.read(no_label).special_mode is None


def test_mode_does_not_use_the_sp_balance():
    frame = cv2.imread(str(FRAMES / "024_video_state.png"))
    frame[965:1015, 45:550] = 0
    assert BattleModeReader().read(frame).special_mode is True


def test_current_live_ordinary_button_is_off():
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/special-mode-off-available/analysis.png"))
    assert frame is not None
    assert BattleModeReader().read(frame).special_mode is False


@pytest.mark.parametrize("height", [1080, 720, 540])
def test_focused_live_button_label_enlargement_remains_off(height):
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/special-button-stall/analysis.png"))
    assert frame is not None
    frame = cv2.resize(frame, (round(height * 16 / 9), height))
    reader = BattleModeReader()
    observation = reader.read(frame)
    assert observation.special_mode is False
    assert observation.off_score >= reader.TEXT_MINIMUM
    assert observation.off_score - observation.on_score >= reader.TEXT_MARGIN
    assert observation.ordinary_blue_fraction >= .5
    # This is glyph-scale coverage, not a weaker appearance threshold.
    assert reader.TEXT_MINIMUM == .88 and reader.TEXT_MARGIN == .15


def test_runtime_publishes_independent_special_mode_evidence():
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("Rectangle")
    runtime.state_machine.force_state(GameState.SPECIAL_SELECT)
    frame = cv2.imread(str(FRAMES / "024_video_state.png"))
    observation = runtime.update(frame, include_selection=True)
    assert observation.special_mode is True
    assert observation.mode_evidence.special_mode is True
    assert observation.timings_ms["battle_mode"] > 0
