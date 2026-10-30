"""Captured and typed-shape regressions for preview/background separation."""
import json
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.card import CardShape
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.board_reader import BoardGridReader
from tableturf_rebuild.vision.calibration import VisionRegion
from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader
from tableturf_rebuild.vision.selection_profiles import SelectionProfileResolver

ROOT = Path(__file__).resolve().parents[1]
CARD = "MiniGame_SaberLight00"


@pytest.fixture(scope="module")
def capture():
    sample = ROOT / "artifacts/diagnostics/long-card-stall"
    snapshot = json.loads((sample / "latest.json").read_text(encoding="utf-8"))
    frame = cv2.imread(str(sample / "analysis.png"))
    catalog = GameCatalog.from_package()
    card = replace(catalog.cards[CARD], shape=CardShape.from_matrix(
        snapshot["executor"]["visual_shapes"][CARD]))
    catalog = replace(catalog, cards={**catalog.cards, CARD: card})
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog, ROOT / "profiles/vision/对战场地-轻飘湖.json", "CenterHole")
    board = reader.read(frame, region, "CenterHole", reference, profile)
    background = replace(board, cells=tuple(tuple(Cell(x) for x in row)
        for row in snapshot["executor"]["selection_board"]))
    points = snapshot["executor"]["preview_attempts"][0]["region"]
    region = VisionRegion("native", "rect", tuple(tuple(p) for p in points), 16, 16)
    return catalog, frame, board, background, region


def test_long_card_uses_real_terminal_dot_instead_of_existing_orange(capture, monkeypatch):
    catalog, frame, board, background, region = capture
    import tableturf_rebuild.vision.placement_reader as module
    # Native rectangles must not first rescan a resized version of the board.
    monkeypatch.setattr(module, "warp_region", lambda *a, **kw: pytest.fail("redundant warped scan"))
    reader = PlacementPreviewReader(catalog)
    preview = reader.read_frame(frame, region, "CenterHole", CARD, board, background=background)
    assert preview is not None
    assert (preview.rotation, preview.origin, preview.score) == (2, (11, 3), 1.0)
    assert reader.last_diagnostics["matched_special_dots"] == 1
    # Several cells lie over committed ink; yellow pixels under the preview
    # must not make an overlapping ordinary placement appear legal.
    assert preview.valid is False


@pytest.mark.parametrize("sample, expected", [
    ("blocked-card", (1, 1)), ("gray-hand", (1, 1)),
    ("skip-button", (3, 1)), ("special-mode-off-available", (3, 1)),
    ("long-card-stall", None), ("two-cell-preview", None),
])
def test_animated_squid_identifies_card_or_bottom_button_but_not_placement(sample, expected):
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics" / sample / "analysis.png"))
    assert frame is not None
    resolver = SelectionProfileResolver.default()
    assert resolver.hand_cursor_slot(frame) == expected
    if expected is not None:
        assert resolver.resolve(frame).focus_slot == expected
        assert resolver.hand_visibility(frame).selection_visible is True


@pytest.mark.parametrize("rotation", range(4))
def test_old_special_beyond_other_end_cannot_reverse_a_straight_card(capture, rotation):
    catalog, _, board, background, _ = capture
    shape = catalog.cards[CARD].shape.rotated(rotation)
    origin = (7, 7)
    expected = {(origin[0] + r, origin[1] + c) for r, c in shape.occupied}
    special = {(origin[0] + r, origin[1] + c) for r, c in shape.occupied
               if int(shape.cells[r][c]) == 2}
    direction = (1, 0) if shape.height > 1 else (0, 1)
    low, high = min(expected), max(expected)
    old = ((high[0] + direction[0], high[1] + direction[1]) if low in special
           else (low[0] - direction[0], low[1] - direction[1]))
    cells = [[Cell.EMPTY] * board.cols for _ in range(board.rows)]
    cells[old[0]][old[1]] = Cell.SELF_SPECIAL
    background = replace(background, cells=tuple(map(tuple, cells)))
    board = replace(board, readings=tuple(replace(
        x, self_special_fraction=0.9 if (x.row, x.col) == old else 0.0,
        median_hsv=(20, 255, 255) if (x.row, x.col) == old else (32, 255, 255),
    ) for x in board.readings))
    reader = PlacementPreviewReader(catalog)
    preview = reader._match_cells(frozenset(expected - special), CARD, board,
                                  frozenset(special), background=background)
    assert preview is not None
    assert (preview.rotation, preview.origin) == (rotation, origin)
    assert preview.score == 1.0
    # Without a dot, both directions fit four stripes. Wait for another
    # animation phase instead of inventing which end is the special square.
    assert reader._match_cells(frozenset(expected - special), CARD, board,
                               background=background) is None
    assert reader.last_diagnostics["failure"] == "AMBIGUOUS_SPECIAL_ORIENTATION"
