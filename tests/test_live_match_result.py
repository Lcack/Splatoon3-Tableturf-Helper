"""Real WIN/LOSS result regressions, including animated confetti phases."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from tableturf_rebuild.application.task_runner import TaskRunner
from tableturf_rebuild.vision.calibration import warp_region
from tableturf_rebuild.vision.match_result_reader import (
    MatchOutcome,
    MatchResultReader,
    PlayerResult,
    _descriptor,
    _similarity,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "vision" / "比赛结果.json"
STATES = (
    ROOT / "artifacts" / "state_capture" / "run-20261021-221749"
    / "session-20261021-222704_annotations"
)
REAL_LOSS = (
    STATES / "011_video_state.png",
    ROOT / "artifacts" / "diagnostics" / "placement-recovered-result" / "capture.jpg",
)
WIN_SAMPLES = ROOT / "artifacts" / "diagnostics" / "live-win-calibration"
REAL_WIN = (WIN_SAMPLES / "capture.jpg", *(WIN_SAMPLES / f"phase-{index:02d}.jpg" for index in range(8)))


@pytest.fixture(scope="module")
def reader():
    return MatchResultReader(PROFILE)


def load(path):
    frame = cv2.imread(str(path))
    assert frame is not None, path
    return frame


def assert_unknown(reading):
    assert reading.outcome is MatchOutcome.UNKNOWN
    assert reading.confidence == 0.0


@pytest.mark.parametrize("path", REAL_LOSS, ids=("reference-loss", "live-loss"))
def test_real_loss_has_bounded_quality_accepted_by_runner(reader, path):
    reading = reader.read(load(path))

    assert reading.outcome is MatchOutcome.LOSS
    assert reading.opponent_result is PlayerResult.WIN
    assert reading.self_result is PlayerResult.LOSE
    assert max(reader.MIN_TEMPLATE_SIMILARITY, TaskRunner.RESULT_MIN_CONFIDENCE) <= reading.confidence <= 1.0


@pytest.mark.parametrize("size", [(1280, 720), (960, 540)])
def test_real_loss_survives_capture_scaling(reader, size):
    frame = cv2.resize(load(REAL_LOSS[1]), size, interpolation=cv2.INTER_AREA)
    reading = reader.read(frame)
    assert reading.outcome is MatchOutcome.LOSS
    assert TaskRunner.RESULT_MIN_CONFIDENCE <= reading.confidence <= 1.0


@pytest.mark.parametrize("path", REAL_WIN, ids=("initial-live", *(f"confetti-{i}" for i in range(8))))
def test_real_win_and_held_out_confetti_phases_are_accepted(reader, path):
    reading = reader.read(load(path))
    assert reading.outcome is MatchOutcome.WIN
    assert reading.opponent_result is PlayerResult.LOSE
    assert reading.self_result is PlayerResult.WIN
    assert TaskRunner.RESULT_MIN_CONFIDENCE <= reading.confidence <= 1.0


@pytest.mark.parametrize("size", [(1280, 720), (960, 540)])
def test_real_win_survives_capture_scaling(reader, size):
    reading = reader.read(cv2.resize(load(REAL_WIN[-1]), size, interpolation=cv2.INTER_AREA))
    assert reading.outcome is MatchOutcome.WIN
    assert TaskRunner.RESULT_MIN_CONFIDENCE <= reading.confidence <= 1.0


@pytest.mark.parametrize("side", ["opponent", "self"])
def test_one_win_layout_label_cannot_confirm_a_victory(reader, side):
    frame = load(REAL_WIN[-1])
    region = reader.opponent_region if side == "opponent" else reader.self_region
    points = region.pixel_points(frame.shape[1], frame.shape[0])
    left, top = np.floor(points.min(axis=0)).astype(int)
    right, bottom = np.ceil(points.max(axis=0)).astype(int)
    frame[top:bottom + 1, left:right + 1] = 0
    assert_unknown(reader.read(frame))


def test_win_templates_do_not_depend_on_player_name_or_nameplate(reader):
    frame = load(REAL_WIN[-1])
    # Replace both name-card panels while leaving the result words intact.
    # The second crop includes a few pixels of the original profile's broad
    # ROI, so this also checks the tighter WIN-layout calibration is used.
    frame[166:327, 98:659] = (30, 160, 50)
    frame[689:850, 98:659] = (130, 30, 140)
    reading = reader.read(frame)
    assert reading.outcome is MatchOutcome.WIN
    assert reading.confidence >= TaskRunner.RESULT_MIN_CONFIDENCE


def test_descriptor_similarity_is_true_bounded_cosine(reader):
    assert np.linalg.norm(reader.win_descriptor) == pytest.approx(1.0)
    assert np.linalg.norm(reader.lose_descriptor) == pytest.approx(1.0)
    assert _similarity(reader.win_descriptor, reader.win_descriptor) == pytest.approx(1.0)
    assert _similarity(reader.win_descriptor * 2.0, reader.win_descriptor * 3.0) == pytest.approx(1.0)
    assert _similarity(reader.win_descriptor, -reader.win_descriptor) == pytest.approx(0.0)
    blank = _descriptor(np.zeros((48, 128, 3), dtype=np.uint8))
    assert not np.any(blank)
    assert _similarity(blank, reader.win_descriptor) == 0.0


@pytest.mark.parametrize("level", [0, 32, 127, 255])
def test_empty_or_flat_capture_is_unknown(reader, level):
    assert_unknown(reader.read(np.full((1080, 1920, 3), level, dtype=np.uint8)))


@pytest.mark.parametrize("factor", [0.0, 0.3, 0.6])
def test_dark_result_does_not_keep_normalized_high_confidence(reader, factor):
    frame = (load(REAL_LOSS[0]).astype(np.float64) * factor).astype(np.uint8)
    assert_unknown(reader.read(frame))


@pytest.mark.parametrize("frame", [
    None,
    np.empty((0, 0, 3), dtype=np.uint8),
    np.zeros((20, 20), dtype=np.uint8),
    np.zeros((20, 20, 4), dtype=np.uint8),
    np.zeros((20, 20, 3), dtype=np.float32),
])
def test_invalid_capture_is_unknown(reader, frame):
    assert_unknown(reader.read(frame))


@pytest.mark.parametrize("index", [1, 10, 12, 19, 24, 29, 31, 32, 39, 45])
def test_other_game_pages_do_not_report_a_winner(reader, index):
    assert_unknown(reader.read(load(STATES / f"{index:03d}_video_state.png")))


@pytest.mark.parametrize("seed", [1, 13, 43])
def test_unrelated_bright_texture_is_unknown(reader, seed):
    frame = np.random.default_rng(seed).integers(0, 256, (540, 960, 3), dtype=np.uint8)
    assert_unknown(reader.read(frame))


def test_bright_reward_page_needs_absolute_match_not_just_class_margin(reader):
    # This real reward page, synthetically brightened, clears visibility and
    # prefers the expected two labels by >.08 but matches neither label well.
    frame = np.clip(load(STATES / "031_video_state.png").astype(np.float64) * 2, 0, 255).astype(np.uint8)
    for region in (reader.opponent_region, reader.self_region):
        patch = warp_region(frame, region, 128, 48)
        assert np.percentile(patch.max(axis=2), 95) >= reader.MIN_PATCH_BRIGHTNESS
        descriptor = _descriptor(patch)
        scores = [_similarity(descriptor, template) for template in (reader.win_descriptor, reader.lose_descriptor)]
        assert abs(scores[0] - scores[1]) >= reader.MIN_CLASS_MARGIN
        assert max(scores) < reader.MIN_TEMPLATE_SIMILARITY
    assert_unknown(reader.read(frame))


def test_a_single_recognized_label_cannot_confirm_the_match(reader):
    frame = load(REAL_LOSS[0])
    points = reader.self_region.pixel_points(frame.shape[1], frame.shape[0])
    left, top = np.floor(points.min(axis=0)).astype(int)
    right, bottom = np.ceil(points.max(axis=0)).astype(int)
    frame[top:bottom + 1, left:right + 1] = 0

    reading = reader.read(frame)
    assert reading.opponent_result is PlayerResult.WIN
    assert reading.self_result is PlayerResult.UNKNOWN
    assert_unknown(reading)
