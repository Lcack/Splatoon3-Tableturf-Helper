"""Submitted placements must survive reveal-like gray textures without motion."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import pytest

from tableturf_rebuild.application.battle_executor import BattleExecutorPhase, VisionMctsBattleExecutor
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.application.task_runner import TaskRunner
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action
from tableturf_rebuild.domain.card import CardShape
from tableturf_rebuild.domain.grid import Cell, PLAYABLE_CELLS
from tableturf_rebuild.vision.board_reader import BoardCellReading, BoardObservation
from tableturf_rebuild.vision.calibration import VisionRegion
from tableturf_rebuild.vision.hand_reader import HandObservation
from tableturf_rebuild.vision.selection_profiles import HandVisibilityObservation


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "artifacts/diagnostics/gray-target-130548"
CARD = "MiniGame_SpGreatBarrier"


@pytest.fixture(scope="module")
def catalog():
    return GameCatalog.from_package()


def fake_setup(catalog, *, submitted=True):
    backend = DummyBackend()
    backend.connect()
    ex = VisionMctsBattleExecutor(SimpleNamespace(catalog=catalog, stage_id="Cross"),
        backend, catalog.decks["MiniGame_Starter"].card_ids)
    action = Action.place("MiniGame_BombRobot", 0, (10, 5))
    ex._selected_action, ex._selected_card_id = action, action.card_id
    ex._last_state = GameState.TARGET_PLACEMENT
    ex._card_confirmed = True
    ex._card_confirmed_at = -100
    ex._place_sent = submitted
    ex._place_sent_at = time.perf_counter() - 2
    ex._placement_confirm_attempts = int(submitted)
    sent = []
    ex._pulse = lambda button: sent.append(button) or True
    board = SimpleNamespace(stage_id="Cross", rows=23, cols=19, placement_preview_present=True)
    ex._selection_board = board
    ex._build_battle_state = lambda _: object()
    ex.rules = SimpleNamespace(legal_actions=lambda *_: (action,))
    preview = SimpleNamespace(rotation=0, anchor=ex._action_anchor(action), origin=action.origin,
                              valid=True, score=1.0)
    reader = SimpleNamespace(read=lambda *_: reader.preview, preview=preview)
    ex.placement_reader = reader
    hand = SimpleNamespace(card_ids=(action.card_id, "a", "b", "c"), confidence=1.0)
    obs = SimpleNamespace(state=SimpleNamespace(state=GameState.TARGET_PLACEMENT), board=board,
        hand=hand, hand_visibility=HandVisibilityObservation(False, 135, 0, None),
        selection_profile=None, special_mode=False, reveal=None)
    return ex, reader, obs, sent


@pytest.mark.parametrize("change", ("gray", "partial", "rotated", "moved", "missing", "unknown-hand"))
def test_post_a_changed_preview_only_waits_and_never_repositions(catalog, change):
    ex, reader, obs, sent = fake_setup(catalog)
    if change == "gray": reader.preview.valid = False
    elif change == "partial": reader.preview.score = .8
    elif change == "rotated": reader.preview.rotation = 1
    elif change == "moved": reader.preview.anchor = (11, 5)
    elif change == "missing": reader.preview = None
    else: obs.hand = None
    for _ in range(5): ex.step(obs, None)
    assert ex.phase is BattleExecutorPhase.WAITING_RESULT
    assert ex._place_sent and not sent


def test_gray_before_the_first_a_is_still_rejected(catalog):
    ex, reader, obs, sent = fake_setup(catalog, submitted=False)
    reader.preview.valid = False
    for _ in range(3): ex.step(obs, None)
    assert ex.phase is BattleExecutorPhase.ERROR
    assert "灰色不可放" in ex.message and not sent


def test_ignored_a_can_retry_only_after_three_new_full_yellow_readings(catalog):
    ex, _, obs, sent = fake_setup(catalog)
    for _ in range(2): ex.step(obs, None)
    assert not sent
    ex.step(obs, None)
    assert sent == [Button.A] and ex._placement_confirm_attempts == 2
    assert ex._preview_signature_count == 0
    for _ in range(3):
        ex._place_sent_at = time.perf_counter() - 2
        ex.step(obs, None)
    assert sent == [Button.A, Button.A] and ex._placement_confirm_attempts == 3
    for _ in range(3):
        ex._place_sent_at = time.perf_counter() - 2
        ex.step(obs, None)
    assert ex.phase is BattleExecutorPhase.ERROR and len(sent) == 2


def test_post_a_timeout_is_about_missing_completion_not_gray_legality(catalog):
    ex, reader, obs, sent = fake_setup(catalog)
    ex._place_sent_at = time.perf_counter() - 21
    reader.preview.valid = False
    ex.step(obs, None)
    assert ex.phase is BattleExecutorPhase.ERROR
    assert "未确认回合推进" in ex.message and not sent


def load_observation(name):
    saved = json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    source = saved["observation"]
    b = source["board"]
    readings = []
    for reading in b["readings"]:
        item = dict(reading)
        for key in ("stage_cell", "inferred_cell"): item[key] = Cell(item[key])
        for key in ("mean_bgr", "median_hsv"): item[key] = tuple(item[key])
        readings.append(BoardCellReading(**item))
    cells = tuple(tuple(Cell(x) for x in row) for row in b["cells"])
    board = BoardObservation(b["stage_id"], b["rows"], b["cols"], cells, tuple(readings),
        b["confidence"], 0.0, b["preview_present"],
        sum(cell in PLAYABLE_CELLS for row in cells for cell in row))
    hand = HandObservation(tuple(SimpleNamespace(**item) for item in source["hand"]),
        source["hand_confidence"], "live") if source.get("hand") else None
    focus = tuple(source["focus"]) if source.get("focus") else None
    obs = SimpleNamespace(state=SimpleNamespace(state=GameState(source["state"])), board=board,
        hand=hand, hand_visibility=HandVisibilityObservation(**source["hand_visibility"]),
        selection_profile=SimpleNamespace(focus_slot=focus) if focus else None,
        special_mode=source.get("special_mode"), discard_mode=source.get("discard_mode"), reveal=None)
    return saved, obs


def test_actual_reveal_mislabeled_as_placement_waits_then_confirms_new_hand(catalog):
    saved, obs = load_observation("submitted-reveal-error")
    shape = CardShape.from_matrix(saved["executor"]["visual_shapes"][CARD])
    catalog = replace(catalog, cards={**catalog.cards, CARD: replace(catalog.cards[CARD], shape=shape)})
    points = saved["executor"]["preview_attempts"][0]["region"]
    region = VisionRegion("native", "rect", tuple(map(tuple, points)), 23, 19)
    backend = DummyBackend()
    backend.connect()
    ex = VisionMctsBattleExecutor(SimpleNamespace(catalog=catalog, stage_id="Cross", preview_region=region),
        backend, catalog.decks["MiniGame_Starter"].card_ids)
    ex._selected_action = Action.special(CARD, 0, (11, 6))
    ex._selected_card_id = CARD
    ex._last_state = GameState.TARGET_PLACEMENT
    ex._card_confirmed = ex._place_sent = True
    ex._place_sent_at = time.perf_counter() - 2
    ex._completed_turns = 5
    ex._selection_sp = (4, 3)
    ex._selection_board = replace(obs.board, cells=tuple(tuple(Cell(x) for x in row)
        for row in saved["executor"]["selection_board"]))
    frame = cv2.imread(str(SAMPLES / "submitted-reveal-error.png"))
    assert frame is not None
    ex.step(obs, frame)
    assert ex.phase is BattleExecutorPhase.WAITING_RESULT and ex._completed_turns == 5
    assert not any(state.buttons for state in backend.history)
    # Reload the old errored executor's receipt. It carries no trusted frozen
    # placement context and must only observe the successful hand change.
    runner = object.__new__(TaskRunner)
    runner.task, runner.battle_executor = object(), ex
    checkpoint = runner.progress_checkpoint()
    fresh = VisionMctsBattleExecutor(ex.runtime, backend, ex.own_deck_ids)
    runner.battle_executor = fresh
    runner.restore_progress(checkpoint)
    planned = []
    fresh._choose_and_select_card = lambda _: planned.append(fresh._selected_action) or False
    _, new = load_observation("new-hand-after-error")
    fresh.step(new, None)
    assert fresh._completed_turns == 5 and not planned
    _, new = load_observation("new-hand-after-error")
    fresh.step(new, None)
    assert fresh._completed_turns == 6 and fresh._revealed_own_cards == [CARD]
    assert planned == [None] and not fresh._place_sent
    runner.restore_progress(checkpoint)
    assert fresh._completed_turns == 6


def test_reloaded_receipt_cannot_retry_a_without_its_frozen_placement_context(catalog):
    ex, _, obs, _ = fake_setup(catalog)
    checkpoint = ex.pending_placement_checkpoint()
    fresh, reader, obs, sent = fake_setup(catalog)
    fresh.restore_pending_placement(checkpoint)
    for _ in range(4):
        fresh._place_sent_at = time.perf_counter() - 2
        fresh.step(obs, None)
    assert fresh._selection_board is None and fresh.phase is BattleExecutorPhase.WAITING_RESULT
    assert not sent
