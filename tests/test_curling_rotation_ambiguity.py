"""A current preview dot outranks an old special tile's accumulated hint."""
from dataclasses import replace
import json
from pathlib import Path
import time

import cv2
import pytest

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.card import CardShape
from tableturf_rebuild.domain.grid import Cell, PLAYABLE_CELLS
from tableturf_rebuild.vision.board_reader import BoardCellReading, BoardObservation
from tableturf_rebuild.vision.calibration import VisionRegion
from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader


SAMPLES = Path(__file__).resolve().parents[1] / "artifacts/diagnostics/latest-stall-audit"
CARD = "MiniGame_BombCurling"
STRIPES = frozenset({(19, 8), (19, 9), (19, 10)})
OLD = (18, 9)
ACTUAL = (20, 9)


@pytest.fixture(scope="module")
def capture():
    saved = json.loads((SAMPLES / "latest.json").read_text(encoding="utf-8"))
    source = saved["observation"]["board"]
    readings = []
    for source_reading in source["readings"]:
        item = dict(source_reading)
        for name in ("stage_cell", "inferred_cell"): item[name] = Cell(item[name])
        for name in ("mean_bgr", "median_hsv"): item[name] = tuple(item[name])
        readings.append(BoardCellReading(**item))
    cells = tuple(tuple(Cell(value) for value in row) for row in source["cells"])
    board = BoardObservation(source["stage_id"], source["rows"], source["cols"], cells,
        tuple(readings), source["confidence"], 0.0, source["preview_present"],
        sum(cell in PLAYABLE_CELLS for row in cells for cell in row))
    background = replace(board, cells=tuple(tuple(Cell(value) for value in row)
        for row in saved["executor"]["selection_board"]))
    catalog = GameCatalog.from_package()
    shape = CardShape.from_matrix(saved["executor"]["visual_shapes"][CARD])
    catalog = replace(catalog, cards={**catalog.cards, CARD: replace(catalog.cards[CARD], shape=shape)})
    points = saved["executor"]["preview_attempts"][0]["region"]
    region = VisionRegion("native", "rect", tuple(map(tuple, points)), board.rows, board.cols)
    frame = cv2.imread(str(SAMPLES / "analysis.png"))
    assert frame is not None
    return catalog, board, background, region, frame


def test_actual_frame_keeps_r2_when_a_previous_context_hint_is_in_temporal_history(capture):
    catalog, board, background, region, frame = capture
    reader = PlacementPreviewReader(catalog)
    initial = reader.read_frame(frame, region, "Cross", CARD, board, background=background, temporal=True)
    assert initial is not None and initial.rotation == 2 and initial.origin == (19, 8)
    # This is the provenance of the diagnostic's spurious opposite-end dot:
    # an earlier context-only animation phase, not current direct texture.
    reader._history.append((time.perf_counter(), STRIPES, frozenset({OLD}),
                            frozenset(), frozenset({OLD})))
    preview = reader.read_frame(frame, region, "Cross", CARD, board, background=background, temporal=True)
    assert preview is not None and (preview.rotation, preview.origin) == (2, (19, 8))
    assert preview.score == 1.0
    assert OLD in map(tuple, reader.last_diagnostics["dot_cells"])
    assert OLD in reader.last_diagnostics["historical_context_dot_cells"]
    assert reader.last_diagnostics["current_direct_dot_cells"] == [ACTUAL]
    assert reader.last_diagnostics["matched_current_direct_dots"] == 1


@pytest.mark.parametrize("direct", (frozenset(), frozenset({OLD, ACTUAL})))
def test_equal_evidence_for_opposite_special_positions_is_ambiguous(capture, direct):
    catalog, board, background, _, _ = capture
    reader = PlacementPreviewReader(catalog)
    result = reader._match_cells(STRIPES, CARD, board, frozenset({OLD, ACTUAL}),
        background=background, current_direct_special_cells=direct)
    assert result is None
    assert reader.last_diagnostics["failure"] == "AMBIGUOUS_SPECIAL_ORIENTATION"
    assert reader.last_diagnostics["best_score"] == 1.0


def test_current_direct_evidence_outweighs_a_previous_direct_orientation(capture):
    catalog, board, background, _, _ = capture
    reader = PlacementPreviewReader(catalog)
    preview = reader._match_cells(STRIPES, CARD, board, frozenset({OLD, ACTUAL}),
        background=background, current_direct_special_cells=frozenset({ACTUAL}),
        historical_direct_special_cells=frozenset({OLD}))
    assert preview is not None and (preview.rotation, preview.origin) == (2, (19, 8))


def test_historical_direct_can_disambiguate_a_context_only_current_phase(capture):
    catalog, board, background, _, _ = capture
    reader = PlacementPreviewReader(catalog)
    preview = reader._match_cells(STRIPES, CARD, board, frozenset({OLD, ACTUAL}),
        background=background, historical_direct_special_cells=frozenset({ACTUAL}))
    assert preview is not None and preview.rotation == 2


def test_board_fallback_preserves_its_current_dot_evidence(capture):
    catalog, board, _, _, _ = capture
    reader = PlacementPreviewReader(catalog)
    preview = reader.read(board, CARD)
    assert preview is not None and (preview.rotation, preview.origin) == (2, (19, 8))
    assert reader.last_diagnostics["matched_current_direct_dots"] == 1


def test_a_real_direct_dot_is_not_discarded_just_because_background_is_special(capture):
    catalog, board, background, _, _ = capture
    rows = [list(row) for row in background.cells]
    rows[ACTUAL[0]][ACTUAL[1]] = Cell.SELF_SPECIAL
    background = replace(background, cells=tuple(map(tuple, rows)))
    reader = PlacementPreviewReader(catalog)
    preview = reader._match_cells(STRIPES, CARD, board, frozenset({OLD, ACTUAL}),
        background=background, current_direct_special_cells=frozenset({ACTUAL}))
    assert preview is not None and preview.rotation == 2
    assert not preview.valid  # Reading the pose never grants placement legality.


def test_unique_context_only_dot_still_reads_without_generic_texture(capture):
    catalog, board, background, _, _ = capture
    reader = PlacementPreviewReader(catalog)
    preview = reader._match_cells(STRIPES, CARD, board, frozenset({ACTUAL}), background=background)
    assert preview is not None and (preview.rotation, preview.origin) == (2, (19, 8))
