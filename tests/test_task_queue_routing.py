"""Task handoff waits for a real menu, and changes NPC without replay inputs."""
from dataclasses import replace

import pytest

from tableturf_rebuild.application.state_machine import GameState, StateScore
from tableturf_rebuild.application.task_plan import NpcTaskDefinition
from tableturf_rebuild.application.task_runner import TaskRunner, TaskStatus
from test_win_plan_feedback import (
    NPC_ID, Runtime, confirm_win, menu, observation, presses, runner, tick,
)
from tableturf_rebuild.control.buttons import Button


def safe_menu(state=GameState.MATCH_SETUP, *, wins=(3, 1, 0), npc_id=NPC_ID):
    return observation(state, npc_id=npc_id, menu=menu(wins, npc_id=npc_id))


def test_exit_a_does_not_complete_until_two_independent_safe_menu_observations(runner):
    confirm_win(runner)
    tick(runner, observation(GameState.REPLAY_CHOICE), now=12)
    snapshot = tick(runner, observation(GameState.REPLAY_CHOICE), now=13)
    assert snapshot.status is TaskStatus.WAITING_RECOGNITION
    assert runner._replay_exit_attempts == 1
    before = presses(runner)
    cached = safe_menu()
    tick(runner, cached, now=14)
    tick(runner, cached, now=15)
    assert runner.status is TaskStatus.WAITING_RECOGNITION
    tick(runner, replace(cached), now=16)
    assert runner.status is TaskStatus.COMPLETED
    assert presses(runner) == before


@pytest.mark.parametrize("state", (GameState.UNKNOWN, GameState.TURN_SELECT,
                                   GameState.REDRAW, GameState.DECK_SELECT))
def test_complete_plan_cannot_handoff_from_battle_or_unknown_scenes(runner, state):
    confirm_win(runner)
    before = presses(runner)
    for now in (12, 13, 14):
        tick(runner, observation(state), now=now)
    assert runner.status is TaskStatus.WAITING_RECOGNITION
    assert presses(runner) == before


def test_hysteresis_retained_menu_label_is_not_a_safe_handoff(runner):
    confirm_win(runner)
    stale = safe_menu()
    stale = replace(stale, state=replace(stale.state, confidence=1,
        candidates=(StateScore(GameState.REPLAY_CHOICE, 1),)))
    for now in (12, 13, 14):
        tick(runner, replace(stale), now=now)
    assert runner.status is TaskStatus.WAITING_RECOGNITION
    assert runner._completion_menu_reads == 0


def test_ignored_exit_reasserts_cancel_and_stops_after_four_confirmations(runner):
    confirm_win(runner)
    for now in (12, 13, 14, 15, 16, 18, 19, 21, 22, 24):
        tick(runner, observation(GameState.REPLAY_CHOICE), now=now)
    assert runner._replay_exit_attempts == 4 and runner.status is TaskStatus.ERROR
    assert presses(runner)[1:] == [frozenset({button})
        for _ in range(4) for button in (Button.DPAD_LEFT, Button.A)]


def test_exit_budget_survives_reload_but_safe_menu_reads_do_not(runner):
    confirm_win(runner)
    runner._replay_exit_attempts = 4
    runner._replay_exit_at = 20
    tick(runner, safe_menu(), now=21)
    saved = runner.progress_checkpoint()
    fresh = TaskRunner(runner.runtime, runner.controller, action_cooldown_seconds=0)
    fresh.configure(runner.task, None)
    fresh.plan = runner.plan
    fresh.restore_progress(saved)
    fresh.start()
    assert fresh._completion_menu_reads == 0
    before = presses(runner)
    tick(fresh, observation(GameState.REPLAY_CHOICE), now=23)
    assert fresh.status is TaskStatus.ERROR and presses(runner) == before


def test_new_npc_exits_old_setup_once_then_waits_for_real_roster(runner):
    next_id = "MiniGame_MiniGameKurage0"
    runner.configure(NpcTaskDefinition(next_id, "水母", 3, 30), None)
    runner.start()
    old_setup = safe_menu(wins=(3, 3, 30))
    tick(runner, old_setup, now=10)
    assert presses(runner) == [frozenset({Button.B})]
    for now in (11, 12):
        tick(runner, replace(old_setup), now=now)
    assert presses(runner) == [frozenset({Button.B})]
    fake_roster = safe_menu(GameState.NPC_ROSTER)
    fake_roster = replace(fake_roster, state=replace(fake_roster.state,
        candidates=(StateScore(GameState.MATCH_SETUP, .95),)))
    tick(runner, fake_roster, now=13)
    assert runner._setup_return_sent
    tick(runner, safe_menu(GameState.NPC_ROSTER), now=14)
    assert not runner._setup_return_sent
    assert presses(runner)[-1] == frozenset({Button.DPAD_DOWN})


def test_cross_npc_b_receipt_survives_reload_without_resending(runner):
    runner.configure(NpcTaskDefinition("MiniGame_MiniGameKurage0", "水母", 3, 30), None)
    runner.start()
    tick(runner, safe_menu(wins=(3, 3, 30)), now=10)
    saved = runner.progress_checkpoint()
    fresh = TaskRunner(runner.runtime, runner.controller, action_cooldown_seconds=0)
    fresh.configure(runner.task, None)
    fresh.restore_progress(saved)
    fresh.start()
    for now in (11, 13, 23):
        tick(fresh, safe_menu(wins=(3, 3, 30)), now=now)
    assert fresh.status is TaskStatus.ERROR
    assert presses(runner) == [frozenset({Button.B})]


def test_same_npc_higher_goal_uses_actual_cumulative_wins_without_b(runner):
    runner.configure(NpcTaskDefinition(NPC_ID, "武器子", 2, 3), None)
    runner.start()
    for now in (10, 11, 12):
        tick(runner, safe_menu(), now=now)
    assert runner.plan.stages[1].current_wins == 1
    assert runner.plan.stages[1].remaining_wins == 2
    assert frozenset({Button.B}) not in presses(runner)


def test_same_npc_goal_already_met_completes_in_menu_without_starting_match(runner):
    runner.configure(NpcTaskDefinition(NPC_ID, "武器子", 2, 1), None)
    runner.start()
    for now in (10, 11, 12):
        tick(runner, safe_menu(wins=(3, 2, 0)), now=now)
    assert runner.status is TaskStatus.COMPLETED and not presses(runner)
    assert runner._stable_level_wins == (3, 2, 0)


def test_configure_clears_expected_deck_before_resetting_old_runtime_context(runner):
    events = []
    runtime = runner.runtime
    runtime.opponent_context = object()
    runtime.expected_deck = object()
    def clear():
        events.append("clear")
        runtime.expected_deck = None
    def reset():
        assert runtime.expected_deck is None
        events.append("reset")
        runtime.opponent_context = None
    runtime.clear_expected_deck, runtime.reset = clear, reset
    runner.configure(NpcTaskDefinition(NPC_ID, "武器子", 3, 30), None)
    assert events == ["clear", "reset"] and runtime.opponent_context is None


def test_configure_can_clear_context_on_a_legacy_runtime_without_reset(runner):
    runner.runtime.opponent_context = object()
    runner.configure(NpcTaskDefinition(NPC_ID, "武器子", 3, 30), None)
    assert runner.runtime.opponent_context is None
