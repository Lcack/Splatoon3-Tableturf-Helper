from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tableturf_rebuild.application.battle_executor import VisionMctsBattleExecutor
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.card import CardShape
from tableturf_rebuild.vision.preview_features import preview_diagonal_score, preview_dot_score


def test_yellow_white_stripes_detected_at_every_animation_phase():
    rows, cols = np.indices((32, 32))
    for phase in range(10):
        patch = np.zeros((32, 32, 3), dtype=np.uint8)
        patch[:] = (0, 255, 255)
        patch[(rows + cols + phase) % 10 < 4] = (255, 255, 255)
        assert preview_diagonal_score(patch) > 0.15
    assert preview_diagonal_score(np.full((32, 32, 3), (0, 255, 255), np.uint8)) == 0


def test_moving_dots_are_distinguished_from_flat_tiles():
    for phase in range(6):
        patch = np.full((32, 32, 3), 30, dtype=np.uint8)
        for row in range(-6, 38, 6):
            for col in range(-6, 38, 6):
                cv2.circle(patch, (col + phase, row + phase), 1, (230, 230, 230), -1)
        assert preview_dot_score(patch) >= 0.6
    assert preview_dot_score(np.full((32, 32, 3), 30, np.uint8)) == 0


def test_dots_dimmer_than_yellow_ink_remain_detectable():
    for phase in range(6):
        patch = np.full((34, 34, 3), (0, 255, 255), dtype=np.uint8)
        for row in range(-6, 40, 6):
            for col in range(-6, 40, 6):
                cv2.circle(patch, (col + phase, row + phase), 1, (230, 234, 234), -1)
        assert preview_dot_score(patch) >= 0.6
    assert preview_dot_score(np.full((34, 34, 3), (0, 255, 255), np.uint8)) == 0


@pytest.mark.parametrize("sample_name, origin, rotation, dots", [
    ("two-cell-preview", (21, 4), 0, [(21, 4)]),
    ("two-cell-rotation", (21, 9), 2, [(21, 10)]),
])
def test_live_dotted_special_and_striped_ink_form_complete_two_cell_preview(sample_name, origin, rotation, dots, monkeypatch):
    import json
    from tableturf_rebuild.domain.grid import Cell
    from tableturf_rebuild.vision.board_reader import BoardGridReader
    from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader

    root = Path(__file__).resolve().parents[1]
    sample = root / "artifacts/diagnostics" / sample_name
    snapshot = json.loads((sample / "latest.json").read_text(encoding="utf-8"))
    frame = cv2.imread(str(sample / "analysis.png"))
    catalog = GameCatalog.from_package()
    card_id = "MiniGame_TakoPod"
    card = replace(catalog.cards[card_id], shape=CardShape.from_matrix(snapshot['executor']['visual_shapes'][card_id]))
    catalog = replace(catalog, cards={**catalog.cards, card_id: card})
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog, root / "profiles/vision/对战场地-扭转河.json", "SShape")
    board = reader.read(frame, region, "SShape", reference, profile)
    base = replace(board, cells=tuple(tuple(Cell(x) for x in row)
                                      for row in snapshot['executor']['selection_board']))
    region = replace(region, kind="rect", points=tuple(tuple(p) for p in snapshot['executor']['preview_attempts'][0]['region']))
    preview_reader = PlacementPreviewReader(catalog)
    preview = preview_reader.read_frame(frame, region, "SShape", card_id, board, background=base)
    assert preview is not None
    assert preview.origin == origin
    assert preview.rotation == rotation
    assert preview.score == 1.0
    assert preview.cells == frozenset({origin, (origin[0], origin[1] + 1)})
    assert preview_reader.last_diagnostics['dot_cells'] == dots
    # Slow analysis and subpixel grid fitting must still retain adjacent
    # animation phases; a controller action explicitly clears this history.
    import tableturf_rebuild.vision.placement_reader as placement_module
    ticks = iter((10.0, 12.2))
    monkeypatch.setattr(placement_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    preview_reader.read_frame(frame, region, "SShape", card_id, board, background=base, temporal=True)
    jittered = replace(region, points=tuple((x + 0.5 / frame.shape[1], y) for x, y in region.points))
    preview_reader.read_frame(frame, jittered, "SShape", card_id, board, background=base, temporal=True)
    assert len(preview_reader._history) == 2
    preview_reader.reset_history()
    assert len(preview_reader._history) == 0


def test_visual_hand_shape_corrects_chirality_only_in_live_executor():
    catalog = GameCatalog.from_package()
    card = catalog.cards["MiniGame_ShooterFlash00"]
    actual = CardShape.from_matrix(reversed(card.shape.cells))
    executor = VisionMctsBattleExecutor(
        SimpleNamespace(catalog=catalog), DummyBackend(), (),
    )
    hand = SimpleNamespace(cards=[SimpleNamespace(card_id=card.card_id, observed_shape=actual.cells)])
    assert executor._bind_hand_shapes(hand) == {card.card_id}
    assert executor.catalog.cards[card.card_id].shape == actual
    assert catalog.cards[card.card_id].shape == card.shape
    assert executor.rules.catalog is executor.placement_reader.catalog


def test_core_reload_rebuilds_shared_types_consistently_in_isolated_process():
    code = """
from tableturf_rebuild.application.state_machine import GameState as OldState
from tableturf_rebuild.application.core_reload import reload_core
from tableturf_rebuild.application.battle_vision import BattleVisionRuntime as OldRuntime
bindings, versions = reload_core()
runtime = bindings['BattleVisionRuntime'].default()
assert bindings['GameState'] is not OldState
assert bindings['BattleVisionRuntime'] is not OldRuntime
assert runtime.state_machine.current is bindings['GameState'].UNKNOWN
assert bindings['GameState'].TURN_SELECT in bindings['TaskRunner'].BATTLE_STATES
assert 'tableturf_rebuild.planning.mcts' in versions
assert 'tableturf_rebuild.vision.preview_features' in versions
print('reload OK')
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                            env={**__import__('os').environ, "PYTHONPATH": str(root / 'src')}, timeout=30)
    assert result.returncode == 0, result.stderr
