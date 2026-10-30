"""Live ordinary and special previews over activated flame squares."""
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.card import CardShape
from tableturf_rebuild.domain.grid import Cell, PLAYABLE_CELLS
from tableturf_rebuild.vision.board_reader import BoardCellReading, BoardObservation
from tableturf_rebuild.vision.calibration import VisionRegion
from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader
from tableturf_rebuild.vision.preview_features import (
    contextual_preview_dot_score, preview_dot_score,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "artifacts/diagnostics/special-two-cell-stall"
CARD = "MiniGame_TakoPod"
PHASES = ("161503-495576", "161516-584502", "161529-840153", "161540-874177")


def load_sample(name):
    if name == "ordinary":
        directory = ROOT / "artifacts/diagnostics/ordinary-two-cell-fiery-stall"
        snapshot_path, frame_path = directory / "latest.json", directory / "analysis.png"
    else:
        snapshot_path, frame_path = SAMPLES / f"{name}.json", SAMPLES / f"{name}.png"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    frame = cv2.imread(str(frame_path))
    assert frame is not None
    saved = snapshot["observation"]["board"]
    readings = []
    for item in saved["readings"]:
        item = dict(item)
        for key in ("stage_cell", "inferred_cell"):
            item[key] = Cell(item[key])
        for key in ("mean_bgr", "median_hsv"):
            item[key] = tuple(item[key])
        readings.append(BoardCellReading(**item))
    cells = tuple(tuple(Cell(x) for x in row) for row in saved["cells"])
    board = BoardObservation(
        saved["stage_id"], saved["rows"], saved["cols"], cells, tuple(readings),
        saved["confidence"], 0.0, saved["preview_present"],
        sum(x in PLAYABLE_CELLS for row in cells for x in row),
    )
    executor = snapshot["executor"]
    background = replace(board, cells=tuple(tuple(Cell(x) for x in row)
        for row in executor["selection_board"]))
    points = executor["preview_attempts"][0]["region"]
    region = VisionRegion("native", "rect", tuple(map(tuple, points)), board.rows, board.cols)
    catalog = GameCatalog.from_package()
    card = replace(catalog.cards[CARD], shape=CardShape.from_matrix(executor["visual_shapes"][CARD]))
    catalog = replace(catalog, cards={**catalog.cards, CARD: card})
    return catalog, frame, board, background, region


def native_patch(frame, region, board, row, col):
    points = region.pixel_points(frame.shape[1], frame.shape[0])
    dx, dy = (points[1] - points[0]) / (board.cols, board.rows)
    inset = 3 * frame.shape[0] / 1080
    x1, y1 = np.rint(points[0] + (col * dx + inset, row * dy + inset)).astype(int)
    x2, y2 = np.rint(points[0] + ((col + 1) * dx - inset, (row + 1) * dy - inset)).astype(int)
    return frame[y1:y2, x1:x2]


@pytest.mark.parametrize("special_attack", (False, True))
@pytest.mark.parametrize("name", (*PHASES, "ordinary"))
def test_live_flame_dot_and_stripe_keep_complete_typed_two_cell_shape(name, special_attack):
    catalog, frame, board, background, region = load_sample(name)
    reader = PlacementPreviewReader(catalog)
    preview = reader.read_frame(frame, region, board.stage_id, CARD, board,
                                background=background, special_attack=special_attack)
    assert preview is not None, reader.last_diagnostics
    assert (preview.origin, preview.rotation, preview.score) == ((19, 9), 0, 1.0)
    assert preview.cells == frozenset({(19, 9), (19, 10)})
    assert [19, 9] in [list(cell) for cell in reader.last_diagnostics["dot_cells"]]
    assert reader.last_diagnostics["matched_special_dots"] == 1
    assert (18, 10) not in reader.last_diagnostics["dot_cells"]
    assert (19, 10) not in reader.last_diagnostics["dot_cells"]
    # A cursor can be read while invalid. Existing special squares must stay
    # forbidden, so the executor can move it before considering confirmation.
    assert preview.valid is False
    assert background.cells[19][9] is Cell.SELF_SPECIAL


@pytest.mark.parametrize("name", (*PHASES, "ordinary"))
def test_existing_flames_and_stripes_do_not_become_special_dot_evidence(name):
    _, frame, board, _, region = load_sample(name)
    real_dot = native_patch(frame, region, board, 19, 9)
    assert preview_dot_score(real_dot) == 0.0
    assert contextual_preview_dot_score(real_dot, Cell.SELF_SPECIAL,
        fiery=True) >= 0.6
    for row, col in ((19, 6), (11, 5)):
        patch = native_patch(frame, region, board, row, col)
        assert contextual_preview_dot_score(patch, Cell.SELF_SPECIAL,
            fiery=True) < 0.6


@pytest.mark.parametrize("special_attack", (False, True))
def test_three_animation_phases_retain_dot_and_orientation(special_attack):
    reader = None
    for name in PHASES[:3]:
        catalog, frame, board, background, region = load_sample(name)
        if reader is None:
            reader = PlacementPreviewReader(catalog)
        preview = reader.read_frame(frame, region, board.stage_id, CARD, board,
            background=background, special_attack=special_attack, temporal=True)
        assert preview is not None, reader.last_diagnostics
        assert (preview.rotation, preview.origin) == (0, (19, 9))
    assert reader.last_diagnostics["temporal_frames"] == 3


@pytest.mark.parametrize("special_attack", (False, True))
@pytest.mark.parametrize("name", (PHASES[1], "ordinary"))
def test_flame_without_dot_cannot_complete_a_two_cell_card(special_attack, name):
    catalog, frame, board, background, region = load_sample(name)
    # Replace only the real dotted square with another existing flame patch.
    # Matching must wait instead of borrowing committed orange ink.
    frame = frame.copy()
    dot = native_patch(frame, region, board, 19, 9)
    flame = native_patch(frame, region, board, 19, 6)
    dot[:] = cv2.resize(flame, (dot.shape[1], dot.shape[0]))
    reader = PlacementPreviewReader(catalog)
    assert reader.read_frame(frame, region, board.stage_id, CARD, board,
        background=background, special_attack=special_attack) is None


def test_fiery_mask_is_not_applied_to_an_unlit_or_unrelated_background():
    _, frame, board, _, region = load_sample(PHASES[1])
    patch = native_patch(frame, region, board, 19, 9)
    assert contextual_preview_dot_score(patch, Cell.SELF_SPECIAL,
        fiery=False) < 0.6
    assert contextual_preview_dot_score(patch, Cell.WALL,
        fiery=False) < 0.6
