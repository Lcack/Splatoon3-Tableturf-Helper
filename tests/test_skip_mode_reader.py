from pathlib import Path

import cv2
import numpy as np
import pytest

from tableturf_rebuild.vision.skip_mode_reader import SkipModeReader


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "artifacts/diagnostics"
FRAMES = ROOT / "artifacts/state_capture/run-20261021-221749/session-20261021-222704_annotations"


def read_frame(path):
    frame = cv2.imread(str(path))
    assert frame is not None
    return frame


@pytest.mark.parametrize("sample, expected", [
    ("skip-button", False),  # actual focus on the normal skip button
    ("skip-discard", True),  # actual dialog with the full return label
    ("special-button-stall", False),  # focus on the other bottom button
    ("all-gray-skip", False),  # unavailable cards are not a discard dialog
])
@pytest.mark.parametrize("height", [1080, 720, 540])
def test_live_button_text_acknowledges_discard_instead_of_focus(sample, expected, height):
    frame = read_frame(DIAGNOSTICS / sample / "analysis.png")
    frame = cv2.resize(frame, (round(height * 16 / 9), height))
    result = SkipModeReader().read(frame)
    assert result.discard_mode is expected
    assert result.confidence >= .88
    assert result.reason is None
    assert abs(result.discard_score - result.normal_score) >= .15


@pytest.mark.parametrize("sample, expected", [
    ("008", False), ("017", False),  # regular hand and placement
    ("024", False), ("027", False),  # disabled skip during special mode
    ("009", True), ("023", True), ("025", True), ("026", True),
])
@pytest.mark.parametrize("height", [1080, 720, 540])
def test_independent_annotated_button_labels(sample, expected, height):
    frame = read_frame(FRAMES / f"{sample}_video_state.png")
    frame = cv2.resize(frame, (round(height * 16 / 9), height))
    assert SkipModeReader().read(frame).discard_mode is expected


@pytest.mark.parametrize("sample", ["001", "002", "011", "012", "029", "030", "031", "039"])
def test_nonbattle_screens_do_not_acknowledge_a_discard_dialog(sample):
    frame = read_frame(FRAMES / f"{sample}_video_state.png")
    assert SkipModeReader().read(frame).discard_mode is None


@pytest.mark.parametrize("sample", ["skip-button", "skip-discard"])
def test_dark_and_missing_label_are_unknown_even_with_button_background(sample):
    frame = read_frame(DIAGNOSTICS / sample / "analysis.png")
    reader = SkipModeReader()
    dark = (frame.astype(float) * .4).astype(np.uint8)
    assert reader.read(dark).discard_mode is None
    assert reader.read(dark).reason == "FRAME_TOO_DARK"
    no_text = frame.copy()
    no_text[840:912, 108:202] = (0, 230, 230)
    assert reader.read(no_text).discard_mode is None
    partial_text = frame.copy()
    partial_text[855:901, 125:153] = (0, 230, 230)
    assert reader.read(partial_text).discard_mode is None


def test_enlarged_return_glyph_is_supported_without_lowering_thresholds():
    frame = read_frame(DIAGNOSTICS / "skip-discard/analysis.png")
    patch = frame[860:896, 130:175].copy()
    enlarged = cv2.resize(patch, None, fx=1.1, fy=1.1)
    frame[856:902, 125:182] = frame[870, 110]
    frame[858:858 + enlarged.shape[0], 128:128 + enlarged.shape[1]] = enlarged
    reader = SkipModeReader()
    assert reader.read(frame).discard_mode is True
    assert reader.TEXT_MINIMUM == .88
    assert reader.TEXT_MARGIN == .15


def test_card_brightness_and_sp_are_not_used_as_discard_confirmation():
    frame = read_frame(DIAGNOSTICS / "skip-discard/analysis.png")
    frame[0:820, 0:550] = 0
    frame[960:1050, 0:550] = 0
    assert SkipModeReader().read(frame).discard_mode is True


@pytest.mark.parametrize("frame", [None, np.zeros((2, 2), dtype=np.uint8),
                                   np.zeros((540, 960, 3), dtype=float),
                                   np.zeros((480, 640, 3), dtype=np.uint8)])
def test_invalid_frame_does_not_invent_a_mode(frame):
    assert SkipModeReader().read(frame).discard_mode is None


def test_missing_templates_leave_mode_unknown(tmp_path):
    frame = read_frame(DIAGNOSTICS / "skip-discard/analysis.png")
    result = SkipModeReader(tmp_path / "missing.npz").read(frame)
    assert result.discard_mode is None
    assert result.reason == "MISSING_BUTTON_TEXT_TEMPLATES"
