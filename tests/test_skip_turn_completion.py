"""A consumed card starts a new turn even when the state label never changed."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tableturf_rebuild.application.battle_executor import BattleExecutorPhase, VisionMctsBattleExecutor
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.application.task_runner import TaskRunner
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action
from tableturf_rebuild.vision.hand_reader import HandObservation, HandCardObservation
from tableturf_rebuild.vision.selection_profiles import HandVisibilityObservation


ROOT = Path(__file__).resolve().parents[1]
OLD_CARD = "MiniGame_BombRobot"
NEW_CARDS = ("MiniGame_TakoPod", "MiniGame_ShockSonar", "MiniGame_ShooterFlash00", "MiniGame_TakoStrike")


@pytest.fixture(scope="module")
def catalog():
    return GameCatalog.from_package()


def make_hand(catalog, ids=NEW_CARDS, confidence=1.0):
    return HandObservation(tuple(HandCardObservation(
        i // 2 + 1, i % 2 + 1, card_id, catalog.cards[card_id].name_zh,
        catalog.cards[card_id].area, catalog.cards[card_id].sp_cost,
        catalog.cards[card_id].sp_cost, confidence, (), available=True,
    ) for i, card_id in enumerate(ids)), confidence, "live")


def setup(catalog, *, skip=True):
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(SimpleNamespace(catalog=catalog, stage_id="Cross"),
        backend, catalog.decks["MiniGame_Starter"].card_ids)
    executor._last_state = GameState.TURN_SELECT
    executor._selected_action = Action.skip(OLD_CARD) if skip else Action.place(OLD_CARD, 0, (0, 0))
    executor._selected_card_id = OLD_CARD
    executor._discard_confirm_sent = skip
    executor._skip_activate_sent = skip
    executor._place_sent = not skip
    executor._completed_turns = 10
    planned = []
    # Inspect the state handed to planning, without performing a new game action.
    executor._choose_and_select_card = lambda _: planned.append(executor._selected_action) or False
    return executor, backend, planned


def observation(catalog, *, ids=NEW_CARDS, confidence=1.0, focus=(1, 1), visible=True, discard=False):
    return SimpleNamespace(state=SimpleNamespace(state=GameState.TURN_SELECT), board=None,
        hand=make_hand(catalog, ids, confidence),
        selection_profile=SimpleNamespace(focus_slot=focus),
        hand_visibility=HandVisibilityObservation(visible, 135, 0, focus),
        discard_mode=discard, reveal=None)


@pytest.mark.parametrize("skip", (True, False))
def test_new_hand_without_reveal_clears_committed_action_and_counts_once(catalog, skip):
    executor, backend, planned = setup(catalog, skip=skip)
    executor.step(observation(catalog), None)
    assert executor._completed_turns == 10 and planned == []
    executor.step(observation(catalog), None)
    assert executor._completed_turns == 11
    assert executor._selected_action is None and planned == [None]
    assert executor._revealed_own_cards == [OLD_CARD]
    assert not executor._discard_confirm_sent and not executor._skip_activate_sent
    executor.step(observation(catalog), None)
    assert executor._completed_turns == 11 and executor._revealed_own_cards == [OLD_CARD]
    assert not any(state.buttons for state in backend.history)


def test_cached_hand_is_not_a_second_independent_read(catalog):
    executor, _, planned = setup(catalog)
    obs = observation(catalog)
    for _ in range(4):
        executor.step(obs, None)
    assert executor._completed_turns == 10 and not planned
    executor.step(replace_observation_hand(obs), None)
    assert executor._completed_turns == 11


def replace_observation_hand(obs):
    return SimpleNamespace(**{**vars(obs), "hand": replace(obs.hand)})


@pytest.mark.parametrize("changes", [
    {"ids": (OLD_CARD, *NEW_CARDS[1:])},
    {"confidence": .6}, {"focus": None}, {"visible": False},
    {"discard": True}, {"discard": None},
])
def test_incomplete_completion_evidence_does_not_count_or_replan(catalog, changes):
    executor, backend, planned = setup(catalog)
    for _ in range(3):
        executor.step(observation(catalog, **changes), None)
    assert executor._completed_turns == 10 and not planned
    assert executor._selected_card_id == OLD_CARD
    assert executor.phase is BattleExecutorPhase.WAITING_RESULT
    assert not any(state.buttons for state in backend.history)


def test_missing_commit_cannot_turn_a_changed_hand_into_an_extra_turn(catalog):
    executor, _, _ = setup(catalog)
    executor._discard_confirm_sent = False
    executor._skip_activate_sent = False
    executor._open_skip_dialog = lambda _: False
    for _ in range(3):
        executor.step(observation(catalog), None)
    assert executor._completed_turns == 10 and executor._revealed_own_cards == []


def test_already_counted_reveal_only_clears_old_action(catalog):
    executor, _, planned = setup(catalog)
    executor._completed_turns = 11
    executor._turn_result_counted = True
    executor._revealed_own_cards = [OLD_CARD]
    executor.step(observation(catalog), None)
    executor.step(observation(catalog), None)
    assert executor._completed_turns == 11 and executor._revealed_own_cards == [OLD_CARD]
    assert planned == [None]


def test_pending_discard_survives_core_reload_without_inventing_completion(catalog):
    old, _, _ = setup(catalog)
    old._discard_confirm_attempts = 1
    # This reproduces the actual failed state, after the new hand was read.
    old.phase = BattleExecutorPhase.ERROR
    runner = object.__new__(TaskRunner)
    runner.task = object()
    runner.battle_executor = old
    checkpoint = runner.progress_checkpoint()
    fresh, _, planned = setup(catalog)
    fresh.reset_match()
    fresh._choose_and_select_card = lambda _: planned.append(fresh._selected_action) or False
    runner.battle_executor = fresh
    runner.restore_progress(checkpoint)
    assert fresh._completed_turns == 10 and fresh._selected_card_id == OLD_CARD
    assert fresh.phase is BattleExecutorPhase.WAITING_RESULT
    fresh.step(observation(catalog), None)
    assert fresh._completed_turns == 10
    fresh.step(observation(catalog), None)
    assert fresh._completed_turns == 11 and fresh._revealed_own_cards == [OLD_CARD]
    runner.restore_progress(checkpoint)  # Reapplying an older checkpoint cannot rewind this turn.
    assert fresh._completed_turns == 11 and fresh._selected_action is None


def test_saved_live_case_has_a_real_new_hand_and_successfully_submitted_discard(catalog):
    path = ROOT / "artifacts/diagnostics/mcts-card-not-in-hand/latest.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["executor"]["action"]["card_id"] == OLD_CARD
    assert saved["executor"]["discard_confirm_attempts"] == 1
    assert tuple(item["card_id"] for item in saved["observation"]["hand"]) == NEW_CARDS
    assert all(item["match_score"] == 1.0 for item in saved["observation"]["hand"])
    assert saved["observation"]["discard_mode"] is False
    executor, _, planned = setup(catalog)
    for _ in range(2):
        executor.step(observation(catalog), None)
    assert executor._completed_turns == 11 and planned == [None]
