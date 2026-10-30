from dataclasses import replace
from types import SimpleNamespace

import pytest

from tableturf_rebuild.application.battle_executor import BattleExecutorPhase, VisionMctsBattleExecutor
from tableturf_rebuild.application.battle_vision import BattleVisionObservation
from tableturf_rebuild.application.state_machine import GameState, StateObservation
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action, ActionType
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.domain.state import Side
from tableturf_rebuild.vision.board_reader import BoardObservation


@pytest.fixture
def live():
    catalog = GameCatalog.from_package()
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(SimpleNamespace(catalog=catalog, stage_id="Square"),
        backend, catalog.decks["MiniGame_Starter"].card_ids,
        fallback_opponent_deck_id="MiniGame_Starter")
    cells = [list(row) for row in catalog.stages["Square"].cells]
    cells[5][5] = Cell.SELF_SPECIAL
    cells[5][6] = Cell.OPPONENT_INK
    board = BoardObservation("Square", len(cells), len(cells[0]), tuple(map(tuple, cells)),
        (), 1.0, 0.0, False, len(cells) * len(cells[0]), False)
    return executor, backend, board


def obs(board, *, state=GameState.TURN_SELECT, mode=False, focus=(2, 2), available=False, sp=1):
    hand = SimpleNamespace(cards=(SimpleNamespace(card_id="MiniGame_TakoPod", name="章鱼弹舱兵",
        slot_row=1, slot_col=1, available=available),), card_ids=("MiniGame_TakoPod",), confidence=1.0)
    return BattleVisionObservation(StateObservation(state, .95, (), (), ()), board,
        SimpleNamespace(focus_slot=focus) if focus else None, (), hand=hand,
        sp=SimpleNamespace(self_value=sp, opponent_value=2), special_mode=mode)


def test_gray_ordinary_card_can_be_planned_as_an_affordable_special(live):
    executor, backend, board = live
    choices = []
    executor.policy = SimpleNamespace(set_opponent_context=lambda *args: None,
        choose=lambda rules, state, side, actions: choices.extend(actions) or
            next(a for a in actions if a.action_type is ActionType.SPECIAL))
    executor._choose_and_select_card(obs(board))
    assert choices and all(a.action_type is ActionType.SPECIAL for a in choices)
    assert executor._selected_action.action_type is ActionType.SPECIAL
    assert [s.buttons for s in backend.history if s.buttons] == [frozenset({Button.DPAD_DOWN})]


def test_special_toggle_selection_and_placement_require_visual_mode_and_cached_sp(live):
    executor, backend, board = live
    action = Action.special("MiniGame_TakoPod", 0, (5, 6))
    executor._selected_action, executor._selected_card_id = action, action.card_id
    executor._selection_board = board
    executor._selection_sp = (1, 2)
    executor._last_state = GameState.TURN_SELECT
    executor.step(obs(board), None)  # Down to toggle.
    executor.step(obs(board, state=GameState.TARGET_PLACEMENT, focus=(3, 2)), None)  # Verified button focus.
    assert executor._card_confirmed is False
    executor.step(obs(board, state=GameState.SPECIAL_SELECT, mode=True, focus=(1, 1),
                      available=True, sp=None), None)
    assert executor._card_confirmed is True
    executor._card_confirmed_at = -10.0
    anchor = executor._action_anchor(action)
    preview = SimpleNamespace(rotation=0, origin=action.origin, anchor=anchor, score=1.0, valid=True)
    executor.placement_reader = SimpleNamespace(read=lambda *args, **kwargs: preview)
    for _ in range(3):
        executor.step(obs(replace(board, placement_preview_present=True), state=GameState.SPECIAL_PLACEMENT, mode=True,
                          available=True, sp=None), None)
    assert executor._place_sent is True
    assert executor.phase is BattleExecutorPhase.PLACING
    state = executor._build_battle_state(obs(board, mode=True, sp=None))
    assert state.player_a.sp == 1  # Burning/reserved HUD slots are not zero balance.
    assert action in executor.rules.legal_actions(state, Side.A)
    assert [s.buttons for s in backend.history if s.buttons] == [
        frozenset({Button.DPAD_DOWN}), frozenset({Button.A}),
        frozenset({Button.A}), frozenset({Button.A})]


def test_unknown_mode_never_confirms_a_special_card(live):
    executor, backend, board = live
    executor._selected_action = Action.special("MiniGame_TakoPod", 0, (5, 6))
    executor._selected_card_id = "MiniGame_TakoPod"
    executor._special_down_sent = True
    for _ in range(8):
        executor._special_activate_at = -10.0
        executor.step(obs(board, mode=None), None)
    assert executor.phase is BattleExecutorPhase.ERROR
    assert executor._card_confirmed is False
    assert not backend.history


def test_live_sp_comes_from_hud_balance_and_never_from_active_tile_count(live):
    executor, _, board = live
    observed = obs(board, sp=None)
    observed = replace(observed, active_specials=SimpleNamespace(self_count=9, opponent_count=9))
    state = executor._build_battle_state(observed)
    assert state.player_a.sp == 0
    assert state.player_b.sp == 2


def test_first_hud_frame_waits_instead_of_freezing_an_unconfirmed_zero(live):
    executor, backend, board = live
    observed = obs(board, sp=None)
    observed.sp.self_reason = "AWAITING_TWO_IDENTICAL_FRAMES"
    executor._choose_and_select_card(observed)
    assert executor._selected_action is None
    assert executor._selection_sp is None
    assert not backend.history
    assert "SP余额" in executor.message


@pytest.mark.parametrize("mode", [False, None])
def test_special_placement_never_confirms_after_mode_evidence_is_lost(live, mode):
    executor, backend, board = live
    executor._selected_action = Action.special("MiniGame_TakoPod", 0, (5, 6))
    executor._selected_card_id = "MiniGame_TakoPod"
    executor._card_confirmed = True
    executor._card_confirmed_at = -10.0
    executor._selection_board = board
    executor._selection_sp = (1, 2)
    preview = SimpleNamespace(rotation=0, origin=(5, 6), anchor=(5, 6), score=1.0, valid=True)
    executor.placement_reader = SimpleNamespace(read=lambda *args, **kwargs: preview)
    observed = obs(replace(board, placement_preview_present=True),
                   state=GameState.TARGET_PLACEMENT, mode=mode, available=True)
    for _ in range(8):
        executor.step(observed, None)
    assert executor._place_sent is False
    assert not backend.history
    assert executor.phase is BattleExecutorPhase.ERROR


@pytest.mark.parametrize("mode", [False, None])
def test_special_retry_never_confirms_after_mode_evidence_is_lost(live, mode):
    import time
    executor, backend, board = live
    executor._selected_action = Action.special("MiniGame_TakoPod", 0, (5, 6))
    executor._place_sent = True
    executor._place_sent_at = time.perf_counter() - 2.0
    executor.step(obs(board, state=GameState.TARGET_PLACEMENT, mode=mode), None)
    assert not backend.history
    assert executor.phase is BattleExecutorPhase.WAITING_RESULT


def test_toggle_retry_requires_current_button_focus_not_a_previous_down(live):
    executor, backend, board = live
    executor._selected_action = Action.special("MiniGame_TakoPod", 0, (5, 6))
    executor._special_down_sent = True
    executor._special_activation_attempts = 1
    executor._special_activate_at = -10.0
    executor._prepare_special_mode(obs(board, mode=None, focus=(1, 1), available=True))
    assert not backend.history
    assert executor._card_confirmed is False


@pytest.mark.parametrize("mode", [False, True])
def test_new_hand_clears_old_special_even_if_select_mode_label_lags(live, mode):
    from tableturf_rebuild.vision.selection_profiles import HandVisibilityObservation
    executor, _, board = live
    executor._selected_action = Action.special("MiniGame_TakoPod", 0, (5, 6))
    executor._selected_card_id = "MiniGame_TakoPod"
    executor._place_sent = True
    executor._last_state = GameState.TURN_REVEAL
    executor._turn_result_counted = True
    executor._completed_turns = 1
    observed = obs(board, state=GameState.SPECIAL_SELECT, mode=mode, available=True)
    observed = replace(observed,
        hand=SimpleNamespace(card_ids=("MiniGame_BombCurling",), confidence=1.0),
        hand_visibility=HandVisibilityObservation(True, 255, 1.0))
    executor._choose_and_select_card = lambda observation: False
    executor.step(observed, None)
    assert executor._selected_action is None
    assert executor._selected_card_id is None
    assert executor._place_sent is False
    assert executor._completed_turns == 1


def test_missing_preview_after_placement_waits_for_turn_instead_of_pressing_back(live):
    import time
    executor, backend, board = live
    executor._selected_action = Action.place("MiniGame_TakoPod", 0, (5, 6))
    executor._place_sent = True
    executor._place_sent_at = time.perf_counter() - 2.0
    executor.placement_reader = SimpleNamespace(read=lambda *args: None)
    observed = obs(replace(board, placement_preview_present=True),
                   state=GameState.TARGET_PLACEMENT)
    for _ in range(15):
        executor.step(observed, None)
    assert executor.phase is BattleExecutorPhase.WAITING_RESULT
    assert executor._preview_missing_reads == 0
    assert not backend.history
