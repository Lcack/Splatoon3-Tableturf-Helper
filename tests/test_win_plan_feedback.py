from dataclasses import replace
from types import SimpleNamespace

import pytest

from tableturf_rebuild.application.battle_vision import BattleVisionObservation
from tableturf_rebuild.application.deck_config import DeckConfig
from tableturf_rebuild.application.state_machine import GameState, StateObservation, StateScore
from tableturf_rebuild.application.task_plan import NpcTaskDefinition, build_npc_plan
from tableturf_rebuild.application.task_runner import TaskRunner, TaskStatus
from tableturf_rebuild.control.buttons import Button, ControllerState
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.match_result_reader import (
    MatchOutcome, MatchResultObservation, PlayerResult,
)


NPC_ID = "MiniGame_WeaponShop"


class Runtime:
    def __init__(self):
        self.catalog = GameCatalog.from_package()
        self.observation = observation(GameState.UNKNOWN)

    def update(self, frame, **kwargs):
        return self.observation

    def clear_expected_deck(self):
        pass

    def set_expected_deck(self, name, card_ids):
        self.expected_deck = (name, card_ids)


def observation(state, *, result=None, npc_id=NPC_ID, difficulty=2, confidence=1.0, menu=None):
    return BattleVisionObservation(
        StateObservation(state, .95, (StateScore(state, .95),), (), ()), None, None, (),
        match_result=result, menu=menu,
        opponent_context=SimpleNamespace(npc_id=npc_id, npc_name="武器子",
            difficulty=difficulty, confidence=confidence),
    )


def result(outcome="WIN", confidence=1.0):
    return MatchResultObservation(MatchOutcome(outcome),
        PlayerResult.LOSE if outcome == "WIN" else PlayerResult.WIN,
        PlayerResult.WIN if outcome == "WIN" else PlayerResult.LOSE, confidence)


def menu(wins, *, npc_id=NPC_ID):
    return SimpleNamespace(npc_id=npc_id, npc_name="武器子", level_wins=wins,
        wins_confidence=1.0, difficulty=2)


@pytest.fixture
def runner(monkeypatch):
    def pulse(controller, buttons, *args):
        controller.send_state(ControllerState(frozenset(buttons)))
        controller.send_state(ControllerState.neutral())
    monkeypatch.setattr("tableturf_rebuild.application.task_runner.pulse_buttons", pulse)
    backend = DummyBackend()
    backend.connect()
    runtime = Runtime()
    runner = TaskRunner(runtime, backend, action_cooldown_seconds=0)
    task = NpcTaskDefinition(NPC_ID, "武器子", 2, 1)
    runner.configure(task, None)
    runner.plan = build_npc_plan(task, (3, 0, 0))
    runner._stable_level_wins = (3, 0, 0)
    runner._wins_npc_id = NPC_ID
    runner.start()
    tick(runner, observation(GameState.TURN_SELECT))
    return runner


def tick(runner, obs, now=10):
    runner.runtime.observation = obs
    return runner.tick(None, now=now)


def presses(runner):
    return [state.buttons for state in runner.controller.history if state.buttons]


def confirm_win(runner):
    tick(runner, observation(GameState.MATCH_RESULT, result=result()))
    return tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=11)


def test_result_updates_level_two_plan_immediately_and_exits_final_replay(runner):
    confirmed = confirm_win(runner)
    assert confirmed.plan.complete
    assert confirmed.plan.total_remaining_wins == 0
    assert confirmed.plan.stages[1].current_wins == 1
    assert confirmed.observed_level_wins == (3, 1, 0)
    assert presses(runner) == [frozenset({Button.A})]
    tick(runner, observation(GameState.MATCH_REWARD), now=12)
    tick(runner, observation(GameState.REPLAY_CHOICE), now=13)
    completed = tick(runner, observation(GameState.REPLAY_CHOICE), now=14)
    assert presses(runner)[-2:] == [frozenset({Button.DPAD_LEFT}), frozenset({Button.A})]
    assert completed.status is TaskStatus.WAITING_RECOGNITION
    tick(runner, observation(GameState.MATCH_SETUP, menu=menu((3, 1, 0))), now=15)
    completed = tick(runner, observation(GameState.MATCH_SETUP, menu=menu((3, 1, 0))), now=16)
    assert completed.status is TaskStatus.COMPLETED


def test_unknown_result_or_reward_never_skips_the_unread_result(runner):
    for state in (GameState.MATCH_RESULT, GameState.MATCH_REWARD):
        for _ in range(4):
            tick(runner, observation(state, result=result("UNKNOWN")))
    assert not presses(runner)
    assert runner.plan.total_remaining_wins == 1
    assert runner.status is TaskStatus.WAITING_RECOGNITION


def test_cached_reading_is_not_counted_as_two_frames(runner):
    cached = result()
    for _ in range(5):
        tick(runner, observation(GameState.MATCH_RESULT, result=cached))
    assert runner.plan.total_remaining_wins == 1
    assert not presses(runner)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()))
    assert runner.plan.complete


def test_conflicting_or_low_confidence_results_restart_confirmation(runner):
    tick(runner, observation(GameState.MATCH_RESULT, result=result()))
    tick(runner, observation(GameState.MATCH_RESULT, result=result("LOSS")))
    tick(runner, observation(GameState.MATCH_RESULT, result=result(confidence=.1)))
    tick(runner, observation(GameState.MATCH_RESULT, result=result()))
    assert not presses(runner)
    assert not runner.plan.complete
    tick(runner, observation(GameState.MATCH_RESULT, result=result()))
    assert runner.plan.complete


def test_repeated_result_and_state_flicker_only_credit_one_win(runner):
    confirm_win(runner)
    tick(runner, observation(GameState.MATCH_REWARD), now=12)
    tick(runner, observation(GameState.MATCH_REWARD), now=13)
    for index in range(3):
        tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=14 + index * 2)
        tick(runner, observation(GameState.MATCH_REWARD), now=15 + index * 2)
    assert runner._stable_level_wins == (3, 1, 0)
    assert runner.plan.total_remaining_wins == 0
    assert presses(runner) == [frozenset({Button.A}), frozenset({Button.A})]


def test_late_result_can_be_confirmed_at_the_reward_screen(runner):
    tick(runner, observation(GameState.MATCH_RESULT, result=result("UNKNOWN")))
    first = tick(runner, observation(GameState.MATCH_REWARD, result=result()), now=11)
    assert not first.plan.complete
    confirmed = tick(runner, observation(GameState.MATCH_REWARD, result=result()), now=12)
    assert confirmed.plan.complete
    assert presses(runner) == [frozenset({Button.A})]


@pytest.mark.parametrize("identity", [("other", 2), None])
def test_win_cannot_be_credited_to_unknown_or_wrong_npc(runner, identity):
    runner._match_identity = identity
    confirm_win(runner)
    assert runner.plan.total_remaining_wins == 1
    assert not presses(runner)


def test_result_uses_frozen_match_difficulty_instead_of_next_active_stage(runner):
    runner.task = replace(runner.task, target_level=3, target_wins=1)
    runner.plan = build_npc_plan(runner.task, (3, 2, 0))
    runner._stable_level_wins = (3, 2, 0)
    runner._match_identity = (NPC_ID, 2)
    confirm_win(runner)
    assert runner._stable_level_wins == (3, 3, 0)
    assert runner.plan.active_stage.level == 3
    tick(runner, observation(GameState.REPLAY_CHOICE), now=12)
    assert presses(runner)[-1] == frozenset({Button.DPAD_LEFT})


def test_continuing_replay_does_not_subtract_the_win_twice(runner):
    runner.task = replace(runner.task, target_wins=2)
    runner.plan = build_npc_plan(runner.task, (3, 0, 0))
    confirm_win(runner)
    assert runner.plan.total_remaining_wins == 1
    tick(runner, observation(GameState.REPLAY_CHOICE), now=12)
    assert presses(runner)[-1] == frozenset({Button.DPAD_RIGHT})


def test_replay_continue_arms_next_result_but_redraw_flicker_does_not(runner):
    runner.task = replace(runner.task, target_wins=3)
    runner.plan = build_npc_plan(runner.task, (3, 0, 0))
    confirm_win(runner)
    tick(runner, observation(GameState.REDRAW), now=12)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=13)
    assert runner._stable_level_wins == (3, 1, 0)
    tick(runner, observation(GameState.REPLAY_CHOICE), now=14)
    tick(runner, observation(GameState.REPLAY_CHOICE), now=15)
    # Replay opens a new match only after that match's actual 15-card deck
    # has been seen and confirmed. It cannot reuse the preceding win's hand.
    runner.deck = DeckConfig("本局视觉确认牌组",
        runner.runtime.catalog.decks["MiniGame_Starter"].card_ids)
    visible = SimpleNamespace(source="auto", cards=tuple(
        SimpleNamespace(card_id=card_id, sp_cost=runner.runtime.catalog.cards[card_id].sp_cost)
        for card_id in runner.deck.card_ids))
    deck_screen = replace(observation(GameState.DECK_SELECT), deck=visible)
    tick(runner, deck_screen, now=16)
    tick(runner, deck_screen, now=18)
    assert runner._deck_selection_confirmed and not runner._awaiting_deck_reselection
    tick(runner, observation(GameState.REDRAW), now=19)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=20)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=21)
    assert runner._stable_level_wins == (3, 2, 0)
    assert runner.plan.total_remaining_wins == 1


def test_stale_menu_cannot_erase_a_credited_win_or_consume_pending_confirmation(runner):
    confirm_win(runner)
    for _ in range(3):
        assert runner._confirmed_level_wins(menu((3, 0, 0))) == (3, 1, 0)
    assert runner._wins_change_allowed is True
    for _ in range(3):
        assert runner._confirmed_level_wins(menu((3, 1, 0))) == (3, 1, 0)
    assert runner._wins_change_allowed is False
    assert runner.plan.with_observed_wins((3, 0, 0)).total_remaining_wins == 0


def test_result_and_plan_survive_core_reload_without_recrediting(runner):
    confirm_win(runner)
    checkpoint = runner.progress_checkpoint()
    replacement = TaskRunner(runner.runtime, runner.controller, action_cooldown_seconds=0)
    replacement.configure(runner.task, None)
    replacement.plan = runner.plan
    replacement.restore_progress(checkpoint)
    replacement.start()
    tick(replacement, observation(GameState.MATCH_RESULT, result=result()))
    assert replacement._stable_level_wins == (3, 1, 0)
    assert replacement.plan.complete


def test_completed_plan_cannot_start_playing_another_match(runner):
    confirm_win(runner)
    before = presses(runner)
    stopped = tick(runner, observation(GameState.REDRAW))
    assert stopped.status is TaskStatus.WAITING_RECOGNITION
    assert presses(runner) == before
