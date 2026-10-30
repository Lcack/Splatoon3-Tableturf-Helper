from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from tableturf_rebuild.application.battle_executor import BattleExecutorPhase, VisionMctsBattleExecutor
from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.domain.actions import Action
from tableturf_rebuild.vision.hand_reader import HandObservation
from test_battle_regressions import executor_for, selection_observation
from tableturf_rebuild.data.catalog import GameCatalog


ROOT = Path(__file__).resolve().parents[1]


def test_mismatched_hand_cancels_phantom_card_before_any_navigation():
    executor, backend = executor_for(GameCatalog.from_package())
    executor._selected_action = Action.place("MiniGame_StringerShort00", 0, (1, 1))
    executor._selected_card_id = "MiniGame_StringerShort00"
    for _ in range(2):
        mismatch = HandObservation((), 0, "test", deck_mismatch=True,
                                   unknown_card_ids=("MiniGame_Denchinamazu",))
        executor.step(replace(selection_observation(executor), hand=mismatch), None)
    assert executor.phase is BattleExecutorPhase.ERROR
    assert executor._selected_action is None and executor._current_hand is None
    assert not any(state.buttons for state in backend.history)


def test_changing_identity_slot_cancels_action_instead_of_oscillating():
    executor, backend = executor_for(GameCatalog.from_package())
    executor._selected_action = Action.place("MiniGame_StringerShort00", 0, (1, 1))
    executor._selected_card_id = "MiniGame_StringerShort00"
    first = selection_observation(executor, focus=(2, 2))
    executor._choose_and_select_card(first)
    changed = replace(first, hand=SimpleNamespace(confidence=1.0,
        cards=(SimpleNamespace(card_id="MiniGame_StringerShort00", name="LACT-450",
                              slot_row=2, slot_col=1, available=True),)))
    executor._choose_and_select_card(changed)
    assert executor._selected_action is None
    assert len([s for s in backend.history if s.buttons]) == 1


def test_navigation_is_bounded_even_when_focus_never_reaches_target():
    executor, backend = executor_for(GameCatalog.from_package())
    executor._selected_action = Action.place("MiniGame_StringerShort00", 0, (1, 1))
    executor._selected_card_id = "MiniGame_StringerShort00"
    for _ in range(12):
        executor._choose_and_select_card(selection_observation(executor, focus=(2, 2)))
    assert executor.phase is BattleExecutorPhase.ERROR
    presses = [s.buttons for s in backend.history if s.buttons]
    assert len(presses) == 8 and all(Button.A not in s for s in presses)


def test_configured_deck_is_not_reused_as_visual_confirmation():
    runtime = BattleVisionRuntime.default()
    runtime.set_expected_deck("configured-only", runtime.catalog.decks["MiniGame_Starter"].card_ids)
    frame = cv2.imread(str(ROOT / "artifacts/state_capture/run-20261021-221749/session-20261021-222704_annotations/013_video_state.png"))
    runtime._refresh_deck(frame)
    assert runtime.deck_observation.source != "configured"
    assert runtime.deck_observation.deck_id != "configured-only"


def test_failed_hand_read_does_not_return_previous_turn_cards():
    runtime = BattleVisionRuntime.default()
    runtime.hand_observation = HandObservation((), 1, "previous-turn")
    def fail(*args, **kwargs):
        raise ValueError("transition/invalid geometry")
    runtime.hand_reader = SimpleNamespace(read=fail)
    assert runtime._read_hand(None, GameState.TURN_SELECT, None) is None
    assert runtime.hand_observation is None
