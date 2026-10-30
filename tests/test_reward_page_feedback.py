"""Reward evidence advances known pages without inferring or recounting a win."""
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.state_machine import GameState, StateScore
from tableturf_rebuild.application.task_runner import TaskRunner, TaskStatus
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.vision.reward_page_reader import RewardPageObservation
from test_win_plan_feedback import confirm_win, observation, presses, runner, tick


def reward(kind="LEVEL_XP", visible=True):
    obs = observation(GameState.MATCH_REWARD)
    return replace(obs, state=replace(obs.state, confidence=.7917, candidates=(
        StateScore(GameState.MATCH_REWARD, .7917), StateScore(GameState.REPLAY_CHOICE, .6659),
    )), reward_page_evidence=RewardPageObservation(visible, .99 if visible else 0,
        kind if visible else None, .99 if visible else .3, None if visible else "TITLE_UNCERTAIN"))


def test_known_reward_passes_local_evidence_with_two_independent_frames(runner):
    confirm_win(runner)
    obs = reward()
    tick(runner, obs, now=12)
    tick(runner, obs, now=13)
    assert presses(runner) == [frozenset({Button.A})]
    tick(runner, replace(obs), now=14)
    assert presses(runner) == [frozenset({Button.A}), frozenset({Button.A})]
    assert runner._stable_level_wins == (3, 1, 0)


def test_reward_presence_does_not_supply_a_match_outcome(runner):
    for index in range(3):
        tick(runner, reward(), now=12 + index)
    assert not presses(runner) and not runner._result_confirmed
    assert runner._stable_level_wins == (3, 0, 0)


def test_unknown_reward_cannot_use_a_confident_full_frame_as_fallback(runner):
    confirm_win(runner)
    unknown = reward(visible=False)
    unknown = replace(unknown, state=replace(unknown.state, confidence=1,
        candidates=(StateScore(GameState.MATCH_REWARD, 1),)))
    for index in range(3):
        tick(runner, replace(unknown), now=12 + index)
    assert presses(runner) == [frozenset({Button.A})]
    assert "TITLE_UNCERTAIN" in runner.message


def test_distinct_recognized_reward_subpage_gets_fresh_evidence_and_retry_budget(runner):
    confirm_win(runner)
    tick(runner, reward(), now=12)
    tick(runner, reward(), now=13)
    runner._reward_advance_attempts = 4
    before = presses(runner)
    # A future reader may identify another title; no item layout is assumed.
    tick(runner, reward(kind="OTHER_KNOWN_REWARD"), now=15)
    assert presses(runner) == before and runner._settlement_evidence_reads == 1
    assert runner._reward_advance_attempts == 4
    tick(runner, reward(kind="OTHER_KNOWN_REWARD"), now=16)
    assert presses(runner) == before + [frozenset({Button.A})]
    assert runner._reward_advance_attempts == 1
    assert runner._stable_level_wins == (3, 1, 0)


def test_changing_reward_kinds_cannot_exhaust_an_unbounded_a_budget(runner):
    confirm_win(runner)
    for index in range(13):
        kind = "LEVEL_XP" if index % 2 == 0 else "OTHER_KNOWN_REWARD"
        tick(runner, reward(kind), now=12 + index * 3)
        tick(runner, reward(kind), now=13 + index * 3)
    assert runner.status is TaskStatus.ERROR
    assert runner._reward_total_attempts == runner.REWARD_TOTAL_MAX_ATTEMPTS
    assert len(presses(runner)) == 1 + runner.REWARD_TOTAL_MAX_ATTEMPTS


def test_reward_kind_and_used_retry_budget_survive_reload(runner):
    confirm_win(runner)
    tick(runner, reward(), now=12)
    tick(runner, reward(), now=13)
    runner._reward_advance_attempts = 4
    checkpoint = runner.progress_checkpoint()
    replacement = TaskRunner(runner.runtime, runner.controller, action_cooldown_seconds=0)
    replacement.configure(runner.task, None)
    replacement.plan = runner.plan
    replacement.restore_progress(checkpoint)
    replacement.start()
    before = presses(runner)
    tick(replacement, reward(), now=20)
    tick(replacement, reward(), now=21)
    assert replacement.status is TaskStatus.ERROR
    assert replacement._reward_page_kind == "LEVEL_XP"
    assert presses(runner) == before


def test_actual_low_scoring_xp_frame_overrides_wrong_global_state_without_reading_hand(runner):
    image = cv2.imread(str(Path(__file__).resolve().parents[1] /
        "artifacts/live_task/run-20261030T013150Z-11628-f6eda8/evidence/061610-726000.png"))
    assert image is not None
    runtime = BattleVisionRuntime.default()
    raw = observation(GameState.TURN_SELECT).state
    runtime.state_machine.recognizer.classify = lambda _: raw
    runtime._read_hand = lambda *args: (_ for _ in ()).throw(AssertionError("no reward hand read"))
    confirm_win(runner)
    for index in range(2):
        obs = runtime.update(image, include_selection=True)
        assert obs.state.state is GameState.MATCH_REWARD
        assert obs.reward_page_evidence.visible
        tick(runner, obs, now=12 + index)
    assert presses(runner) == [frozenset({Button.A}), frozenset({Button.A})]
    assert runner._stable_level_wins == (3, 1, 0)


@pytest.mark.parametrize("name", ("023746-528005", "034750-792978"))
def test_actual_single_confirm_notifications_use_the_reward_chain(runner, name):
    image = cv2.imread(str(Path(__file__).resolve().parents[1] /
        f"artifacts/live_task/run-20261030T013150Z-11628-f6eda8/evidence/{name}.png"))
    assert image is not None
    runtime = BattleVisionRuntime.default()
    # Before a result receipt, even a real notice cannot acknowledge a match.
    obs = runtime.update(image)
    assert obs.reward_page_evidence.page_kind == "SINGLE_CONFIRM_NOTICE"
    tick(runner, obs, now=9)
    assert not presses(runner)
    confirm_win(runner)
    for index in range(2):
        obs = runtime.update(image)
        assert obs.state.state is GameState.MATCH_REWARD
        tick(runner, obs, now=12 + index)
    assert presses(runner) == [frozenset({Button.A}), frozenset({Button.A})]
    assert runner._reward_page_kind == "SINGLE_CONFIRM_NOTICE"
    assert runner._stable_level_wins == (3, 1, 0)
