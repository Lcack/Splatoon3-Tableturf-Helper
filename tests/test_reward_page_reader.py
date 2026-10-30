"""The real level popup needs its title, points badge, bar and number fields."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from tableturf_rebuild.vision.reward_page_reader import RewardPageReader


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/live_task/run-20261030T013150Z-11628-f6eda8/evidence/061610-726000.png"
OLD = ROOT / "artifacts/state_capture/run-20261021-221749/session-20261021-222704_annotations"
EVIDENCE = CURRENT.parent
NOTICES = (EVIDENCE / "023746-528005.png", EVIDENCE / "034750-792978.png")


@pytest.fixture(scope="module")
def reader():
    return RewardPageReader()


def frame(path):
    image = cv2.imread(str(path))
    assert image is not None, path
    return image


@pytest.mark.parametrize("path", (
    CURRENT, OLD / "031_video_state.png",
    ROOT / "artifacts/diagnostics/loss-result-stall/after-A-capture.jpg",
))
def test_actual_experience_pages_with_different_levels_and_fill_match(reader, path):
    reading = reader.read(frame(path))
    assert reading.visible and reading.page_kind == "LEVEL_XP", reading
    assert reading.confidence >= .95


@pytest.mark.parametrize("height", (360, 480, 540, 600, 720, 900, 1080))
def test_supported_resolution_uses_high_title_threshold(reader, height):
    image = cv2.resize(frame(CURRENT), (height * 16 // 9, height), interpolation=cv2.INTER_AREA)
    reading = reader.read(image)
    assert reading.visible and reading.title_score >= reader.TITLE_MINIMUM, reading


@pytest.mark.parametrize("region", ("BADGE_ROI", "BAR_ROI", "LEVEL_ROI", "PROGRESS_ROI"))
def test_title_does_not_acknowledge_an_incomplete_popup(reader, region):
    image = frame(CURRENT)
    left, top, right, bottom = getattr(reader, region)
    image[top:bottom, left:right] = 75
    assert not reader.read(image).visible


def test_title_patch_alone_over_another_page_cannot_acknowledge_reward(reader):
    image = frame(OLD / "030_video_state.png")
    source = frame(CURRENT)
    left, top, right, bottom = reader.TITLE_SEARCH_ROI
    image[top:bottom, left:right] = source[top:bottom, left:right]
    reading = reader.read(image)
    assert reading.title_score >= reader.TITLE_MINIMUM and not reading.visible


def test_stale_analysis_before_the_experience_popup_is_not_a_reward_page(reader):
    path = ROOT / "artifacts/diagnostics/loss-result-stall/after-A-analysis.png"
    assert not reader.read(frame(path)).visible


@pytest.mark.parametrize("path", NOTICES)
@pytest.mark.parametrize("height", (360, 540, 720, 1080))
def test_two_real_single_confirm_notices_share_the_full_shell(reader, path, height):
    image = cv2.resize(frame(path), (height * 16 // 9, height), interpolation=cv2.INTER_AREA)
    reading = reader.read(image)
    assert reading.visible and reading.page_kind == "SINGLE_CONFIRM_NOTICE", reading
    assert reading.confidence >= .94


@pytest.mark.parametrize("region", ("NOTICE_TITLE_ROI", "NOTICE_BODY_ROI", "NOTICE_CONFIRM_ROI"))
def test_a_partial_notice_is_not_confirmed(reader, region):
    image = frame(NOTICES[0])
    left, top, right, bottom = getattr(reader, region)
    image[top:bottom, left:right] = 30
    assert not reader.read(image).visible


def test_notice_button_pasted_over_a_result_page_cannot_confirm_it(reader):
    image, source = frame(OLD / "030_video_state.png"), frame(NOTICES[0])
    left, top, right, bottom = reader.NOTICE_CONFIRM_ROI
    image[top:bottom, left:right] = source[top:bottom, left:right]
    assert not reader.read(image).visible


def test_solid_white_block_cannot_replace_notice_title_text(reader):
    image = frame(NOTICES[0])
    image[283:335, 830:1096] = 255
    assert not reader.read(image).visible


def test_dark_transition_is_unknown_even_when_correlation_remains_high(reader):
    image = (frame(CURRENT).astype(np.float32) * .65).astype(np.uint8)
    reading = reader.read(image)
    assert not reading.visible and reading.reason == "FRAME_TOO_DARK"


@pytest.mark.parametrize("name", (
    "015_video_state.png", "027_video_state.png", "030_video_state.png",
    "032_video_state.png", "038_video_state.png", "039_video_state.png", "040_video_state.png",
))
def test_result_replay_placement_and_menu_frames_do_not_match(reader, name):
    assert not reader.read(frame(OLD / name)).visible


@pytest.mark.parametrize("image", (
    None, np.zeros((1080, 1920), dtype=np.uint8),
    np.zeros((200, 355, 3), dtype=np.uint8),
    np.zeros((480, 640, 3), dtype=np.uint8),
    np.ones((1080, 1920, 3), dtype=np.float32),
))
def test_invalid_or_unsupported_capture_is_unknown(reader, image):
    assert not reader.read(image).visible


def test_missing_title_asset_is_unknown(tmp_path):
    reader = RewardPageReader(tmp_path / "missing.npz")
    assert reader.read(frame(CURRENT)).reason == "MISSING_LEVEL_TITLE_TEMPLATES"
