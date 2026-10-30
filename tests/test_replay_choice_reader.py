"""A replay prompt must survive background changes and retain all three labels."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from tableturf_rebuild.vision.replay_choice_reader import ReplayChoiceReader


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/diagnostics/win-replay-stall-140226/capture.jpg"
OLD = ROOT / "artifacts/state_capture/run-20261021-221749/session-20261021-222704_annotations"


@pytest.fixture(scope="module")
def reader():
    return ReplayChoiceReader()


def frame(path):
    image = cv2.imread(str(path))
    assert image is not None, path
    return image


@pytest.mark.parametrize("path", (
    CURRENT,
    CURRENT.parent / "analysis.png",
    ROOT / "artifacts/diagnostics/loss-result-stall/replay-capture.jpg",
    OLD / "032_video_state.png",
    OLD / "039_video_state.png",
))
def test_actual_win_loss_and_different_npc_prompts_match(reader, path):
    reading = reader.read(frame(path))
    assert reading.visible and reading.confidence >= .95, reading


@pytest.mark.parametrize("height", (360, 540, 720, 1080, 1440, 2160))
def test_normalized_capture_resolution(reader, height):
    image = cv2.resize(frame(CURRENT), (height * 16 // 9, height), interpolation=cv2.INTER_AREA)
    reading = reader.read(image)
    assert reading.visible, reading


def test_button_text_polarity_can_change_without_inventing_focus(reader):
    image = frame(CURRENT)
    for key in ("cancel", "continue"):
        left, top, right, bottom = reader.TEXT_SEARCH_ROIS[key]
        image[top:bottom, left:right] = 255 - image[top:bottom, left:right]
    assert reader.read(image).visible


@pytest.mark.parametrize("missing", ("title", "cancel", "continue"))
def test_a_partially_visible_dialog_cannot_be_acknowledged(reader, missing):
    image = frame(CURRENT)
    left, top, right, bottom = reader.TEXT_SEARCH_ROIS[missing]
    image[top:bottom, left:right] = 75
    assert not reader.read(image).visible


def test_dark_transition_does_not_match_through_brightness_invariance(reader):
    dark = (frame(CURRENT).astype(np.float32) * .65).astype(np.uint8)
    reading = reader.read(dark)
    assert not reading.visible and reading.reason == "FRAME_TOO_DARK"


@pytest.mark.parametrize("name", (
    "015_video_state.png", "027_video_state.png", "030_video_state.png",
    "031_video_state.png", "038_video_state.png", "040_video_state.png",
))
def test_result_reward_placement_and_menu_frames_do_not_match(reader, name):
    assert not reader.read(frame(OLD / name)).visible


@pytest.mark.parametrize("image", (
    None, np.zeros((1080, 1920), dtype=np.uint8),
    np.zeros((200, 355, 3), dtype=np.uint8),
    np.zeros((480, 640, 3), dtype=np.uint8),
    np.ones((1080, 1920, 3), dtype=np.float32),
))
def test_invalid_or_unsupported_capture_is_unknown(reader, image):
    assert not reader.read(image).visible


def test_missing_template_is_unknown(tmp_path):
    reader = ReplayChoiceReader(tmp_path / "missing.npz")
    assert reader.read(frame(CURRENT)).reason == "MISSING_DIALOG_TEXT_TEMPLATES"
