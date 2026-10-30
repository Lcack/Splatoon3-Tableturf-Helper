from pathlib import Path

import cv2
import numpy as np
import pytest

from tableturf_rebuild.vision.sp_reader import SpReader


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("sample, expected", [
    ("long-card-stall", (0, 2)),
    ("gray-hand", (0, 1)),
    ("two-cell-preview", (0, 1)),
])
def test_sp_reads_actual_hud_balance_independently_of_board_specials(sample, expected):
    frame = cv2.imread(str(ROOT / f"artifacts/diagnostics/{sample}/analysis.png"))
    assert frame is not None
    observation = SpReader().read(frame)
    assert (observation.self_value, observation.opponent_value) == expected
    assert observation.self_confidence >= 0.8
    assert observation.opponent_confidence >= 0.8
    assert len(observation.self_slot_ratios) == len(observation.opponent_slot_ratios) == 15


@pytest.mark.parametrize("size", [(1280, 720), (960, 540)])
def test_sp_slot_geometry_scales_with_capture_resolution(size):
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/long-card-stall/analysis.png"))
    observation = SpReader().read(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    assert (observation.self_value, observation.opponent_value) == (0, 2)


def test_idle_dimming_cannot_become_a_zero_sp_balance():
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/long-card-stall/analysis.png"))
    dimmed = (frame.astype(np.float32) * 0.3).astype(np.uint8)
    observation = SpReader().read(dimmed)
    assert observation.self_value is observation.opponent_value is None
    assert observation.self_reason == observation.opponent_reason == "FRAME_TOO_DARK"


def test_warning_banner_can_hide_opponent_balance_without_hiding_own_balance():
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/blocked-card/analysis.png"))
    observation = SpReader().read(frame)
    assert observation.self_value == 0
    assert observation.opponent_value is None
    assert observation.opponent_reason == "HUD_NOT_VISIBLE"


def test_orange_sp_fill_is_counted_on_actual_skip_discard_screen():
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/skip-discard/analysis.png"))
    observation = SpReader().read(frame)
    assert observation.self_value == 1
    assert observation.self_confidence >= 0.8


def test_result_screen_cannot_be_misread_as_zero_sp():
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/placement-recovered-result/analysis.png"))
    observation = SpReader().read(frame)
    assert observation.self_value is observation.opponent_value is None
    assert observation.self_reason == observation.opponent_reason == "HUD_NOT_VISIBLE"


def test_missing_hud_rejects_blank_and_invalid_frames():
    reader = SpReader()
    for frame in (None, np.zeros((1080, 1920), np.uint8),
                  np.zeros((1080, 1920, 3), np.uint8),
                  np.full((1080, 1920, 3), 255, np.uint8),
                  np.zeros((400, 400, 3), np.uint8)):
        observation = reader.read(frame)
        assert observation.self_value is observation.opponent_value is None


def test_noncontiguous_or_half_colored_slots_are_unknown():
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/long-card-stall/analysis.png"))
    # Full orange slot after an unlit first slot violates the HUD fill order.
    frame[987:1002, 93:108] = (0, 160, 240)
    observation = SpReader().read(frame)
    assert observation.self_value is None
    assert observation.self_reason == "NONCONTIGUOUS_SP_SLOTS"
    frame[987:1002, 93:108] = 0
    frame[987:1002, 61:67] = (0, 160, 240)
    observation = SpReader().read(frame)
    assert observation.self_value is None
    assert observation.self_reason == "AMBIGUOUS_SLOT_COLOR"
