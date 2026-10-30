"""Real Cross sword cursor phases, including a dot over an unlit spawn point."""
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.card import CardShape
from tableturf_rebuild.domain.grid import Cell, PLAYABLE_CELLS
from tableturf_rebuild.vision.board_reader import BoardCellReading, BoardObservation
from tableturf_rebuild.vision.calibration import VisionRegion
from tableturf_rebuild.vision.legacy_preview import context_dot_score
from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader
from tableturf_rebuild.vision.preview_features import (
    contextual_preview_dot_score, preview_diagonal_score, preview_dot_score,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "artifacts/diagnostics/cross-saber-r180-stall"
PHASES = ("latest", "013712-418355", "013732-013973", "013748-397407", "013758-716069")
CARD = "MiniGame_SaberLight00"
TARGET_SAMPLES = ROOT / "artifacts/diagnostics/cross-saber-target-animation"
TARGET_PHASES = ("020838-694235", "020850-716015")


def load_sample(name, directory):
    snapshot = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    frame = cv2.imread(str(directory / ("analysis.png" if name == "latest" else f"{name}.png")))
    assert frame is not None
    saved = snapshot["observation"]["board"]
    readings = []
    for source in saved["readings"]:
        item = source.copy()
        for key in ("stage_cell", "inferred_cell"):
            item[key] = Cell(item[key])
        for key in ("mean_bgr", "median_hsv"):
            item[key] = tuple(item[key])
        readings.append(BoardCellReading(**item))
    cells = tuple(tuple(Cell(x) for x in row) for row in saved["cells"])
    board = BoardObservation(saved["stage_id"], saved["rows"], saved["cols"], cells,
        tuple(readings), saved["confidence"], 0.0, saved["preview_present"],
        sum(x in PLAYABLE_CELLS for row in cells for x in row))
    # The cursor can hide activation evidence. A false flag must not prevent
    # a positively observed dot texture from identifying the special square.
    background = replace(board, cells=tuple(tuple(Cell(x) for x in row)
        for row in snapshot["executor"]["selection_board"]),
        readings=tuple(replace(x, special_active=False) for x in board.readings))
    points = snapshot["executor"]["preview_attempts"][0]["region"]
    region = VisionRegion("native", "rect", tuple(map(tuple, points)), board.rows, board.cols)
    catalog = GameCatalog.from_package()
    card = replace(catalog.cards[CARD], shape=CardShape.from_matrix(snapshot["executor"]["visual_shapes"][CARD]))
    catalog = replace(catalog, cards={**catalog.cards, CARD: card})
    return catalog, frame, board, background, region


@pytest.fixture(scope="module", params=PHASES)
def sample(request):
    return load_sample(request.param, SAMPLES)


@pytest.fixture(scope="module", params=TARGET_PHASES)
def target_sample(request):
    return load_sample(request.param, TARGET_SAMPLES)


def native_patch(frame, board, region, row, col):
    points = region.pixel_points(frame.shape[1], frame.shape[0])
    dx = (points[1, 0] - points[0, 0]) / board.cols
    dy = (points[1, 1] - points[0, 1]) / board.rows
    inset = 3 * frame.shape[0] / 1080
    x1, y1 = np.rint(points[0] + (col * dx + inset, row * dy + inset)).astype(int)
    x2, y2 = np.rint(points[0] + ((col + 1) * dx - inset, (row + 1) * dy - inset)).astype(int)
    return frame[y1:y2, x1:x2]


@pytest.mark.parametrize("special_attack", (False, True))
def test_cross_sword_retains_complete_typed_preview_on_unlit_special(sample, special_attack):
    catalog, frame, board, background, region = sample
    reader = PlacementPreviewReader(catalog)
    preview = reader.read_frame(frame, region, "Cross", CARD, board,
        background=background, special_attack=special_attack)
    assert preview is not None, reader.last_diagnostics
    assert (preview.rotation, preview.origin, preview.score) == (2, (15, 9), 1.0)
    assert preview.cells == frozenset((row, 9) for row in range(15, 20))
    assert reader.last_diagnostics["matched_special_dots"] == 1
    assert (19, 9) in reader.last_diagnostics["dot_cells"]
    assert background.cells[19][9] is Cell.SELF_SPECIAL
    assert preview.valid is False  # Special attacks cannot cover existing specials.
    # The adjoining flame/stripe square has dot-like color fragments. It is
    # ordinary card ink, so clear stripes must take priority over color masks.
    assert (18, 9) in reader.last_diagnostics["candidate_cells"]
    assert (18, 9) not in reader.last_diagnostics["dot_cells"]


def test_static_specials_and_ordinary_ink_do_not_become_dot_evidence(sample):
    _, frame, board, background, region = sample
    for reading in board.readings:
        row, col = reading.row, reading.col
        cell = background.cells[row][col]
        if cell not in {Cell.SELF_SPECIAL, Cell.OPPONENT_SPECIAL, Cell.SELF_INK, Cell.OPPONENT_INK}:
            continue
        if (row, col) in {(row, 9) for row in range(15, 20)}:
            continue
        patch = native_patch(frame, board, region, row, col)
        assert preview_dot_score(patch) < 0.6, (row, col)
        assert context_dot_score(patch, cell) < 0.6, (row, col)
        if cell in {Cell.SELF_SPECIAL, Cell.OPPONENT_SPECIAL}:
            assert contextual_preview_dot_score(patch, cell, fiery=True) < 0.6, (row, col)


def test_missing_dot_does_not_invent_rotation_from_four_stripes(sample):
    catalog, frame, board, background, region = sample
    frame = frame.copy()
    target = native_patch(frame, board, region, 19, 9)
    flat_special = native_patch(frame, board, region, 12, 5)
    target[:] = cv2.resize(flat_special, (target.shape[1], target.shape[0]))
    reader = PlacementPreviewReader(catalog)
    assert reader.read_frame(frame, region, "Cross", CARD, board,
        background=background, special_attack=True) is None
    assert reader.last_diagnostics["failure"] == "AMBIGUOUS_SPECIAL_ORIENTATION"


def test_three_original_mask_components_pass_without_lowering_texture_requirement():
    hsv = np.full((34, 34, 3), (35, 255, 255), dtype=np.uint8)
    for row, col in ((6, 6), (6, 22), (22, 6)):
        hsv[row:row + 3, col:col + 4] = (30, 90, 240)
    patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert preview_diagonal_score(patch) < 0.15
    assert contextual_preview_dot_score(patch, Cell.SELF_SPECIAL, fiery=True) >= 0.6
    hsv[22:25, 6:10] = (35, 255, 255)
    patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert contextual_preview_dot_score(patch, Cell.SELF_SPECIAL, fiery=True) == 0.0


@pytest.mark.parametrize("height", (1080, 720))
def test_target_stripes_are_not_confused_with_special_dots(target_sample, height):
    catalog, frame, board, background, region = target_sample
    if height != frame.shape[0]:
        frame = cv2.resize(frame, (round(frame.shape[1] * height / frame.shape[0]), height),
                           interpolation=cv2.INTER_AREA)
    for row in range(3, 7):
        patch = native_patch(frame, board, region, row, 7)
        assert preview_diagonal_score(patch) > 0.15
        assert preview_dot_score(patch) < 0.6
    assert preview_dot_score(native_patch(frame, board, region, 7, 7)) >= 0.6
    reader = PlacementPreviewReader(catalog)
    preview = reader.read_frame(frame, region, "Cross", CARD, board,
        background=background, special_attack=True)
    assert preview is not None, reader.last_diagnostics
    assert (preview.rotation, preview.origin, preview.score) == (2, (3, 7), 1.0)
    assert preview.cells == frozenset((row, 7) for row in range(3, 8))
    assert preview.valid is True


def test_target_animation_does_not_accumulate_false_dot_orientation():
    reader = None
    for name in (*TARGET_PHASES, *TARGET_PHASES):
        catalog, frame, board, background, region = load_sample(name, TARGET_SAMPLES)
        if reader is None:
            reader = PlacementPreviewReader(catalog)
        preview = reader.read_frame(frame, region, "Cross", CARD, board,
            background=background, special_attack=True, temporal=True)
        assert preview is not None, reader.last_diagnostics
        assert (preview.rotation, preview.origin, preview.score) == (2, (3, 7), 1.0)
