"""A sent confirm is retried only while its original result page is visible."""
from dataclasses import replace
from pathlib import Path

import cv2

from tableturf_rebuild.application.state_machine import GameState, StateScore
from tableturf_rebuild.application.task_runner import TaskRunner, TaskStatus
from tableturf_rebuild.control.buttons import Button
from test_win_plan_feedback import runner, observation, result, tick, presses, confirm_win


def test_result_early_a_is_retried_slowly_then_stops_after_a_finite_budget(runner):
    confirm_win(runner)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=12)
    assert len(presses(runner)) == 1
    for now in (12.5, 14.0, 15.5):
        tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=now)
    assert presses(runner) == [frozenset({Button.A})] * 4
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=17)
    assert runner.status is TaskStatus.ERROR
    assert runner._result_advance_attempts == 4
    assert runner._stable_level_wins == (3, 1, 0)
    assert len(presses(runner)) == 4


def test_first_result_confirm_waits_for_the_page_to_settle(runner):
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=10)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=10.1)
    assert runner._result_confirmed and not presses(runner)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=10.75)
    assert presses(runner) == [frozenset({Button.A})]


def test_confirmed_result_does_not_authorize_a_on_unknown_or_conflicting_frames(runner):
    confirm_win(runner)
    for reading in (result("UNKNOWN"), result("LOSS"), result(confidence=.3)):
        tick(runner, observation(GameState.MATCH_RESULT, result=reading), now=20)
        tick(runner, observation(GameState.MATCH_RESULT, result=reading), now=21)
    assert len(presses(runner)) == 1
    assert runner.status is TaskStatus.WAITING_RECOGNITION


def test_reward_requires_two_independent_confident_scene_observations(runner):
    confirm_win(runner)
    low = observation(GameState.MATCH_REWARD)
    low = replace(low, state=replace(low.state, confidence=.5))
    tick(runner, low, now=12)
    tick(runner, low, now=13)
    assert len(presses(runner)) == 1
    reward = observation(GameState.MATCH_REWARD)
    tick(runner, reward, now=14)
    tick(runner, reward, now=15)
    assert len(presses(runner)) == 1
    tick(runner, observation(GameState.MATCH_REWARD), now=16)
    assert len(presses(runner)) == 2


def test_reward_can_retry_but_never_retries_after_replay_is_seen(runner):
    confirm_win(runner)
    tick(runner, observation(GameState.MATCH_REWARD), now=12)
    tick(runner, observation(GameState.MATCH_REWARD), now=13)
    tick(runner, observation(GameState.MATCH_REWARD), now=14.5)
    assert len(presses(runner)) == 3
    assert runner._result_advance_acknowledged
    tick(runner, observation(GameState.REPLAY_CHOICE), now=15)
    tick(runner, observation(GameState.REPLAY_CHOICE), now=16)
    before = presses(runner)
    assert runner._reward_advance_acknowledged
    for state in (GameState.MATCH_RESULT, GameState.MATCH_REWARD):
        tick(runner, observation(state, result=result()), now=20)
        tick(runner, observation(state, result=result()), now=21)
    assert presses(runner) == before


def test_result_retry_budget_survives_core_reload(runner):
    confirm_win(runner)
    for now in (12.5, 14, 15.5):
        tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=now)
    checkpoint = runner.progress_checkpoint()
    replacement = TaskRunner(runner.runtime, runner.controller, action_cooldown_seconds=0)
    replacement.configure(runner.task, None)
    replacement.plan = runner.plan
    replacement.restore_progress(checkpoint)
    replacement.start()
    before = presses(runner)
    tick(replacement, observation(GameState.MATCH_RESULT, result=result()), now=17)
    tick(replacement, observation(GameState.MATCH_RESULT, result=result()), now=18)
    assert replacement.status is TaskStatus.ERROR
    assert presses(runner) == before
    assert replacement._stable_level_wins == (3, 1, 0)


def test_legacy_sent_latch_allows_bounded_visual_retry(runner):
    confirm_win(runner)
    runner._result_advance_attempts = 0
    runner._result_advance_at = 0
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=20)
    assert runner._result_advance_attempts == 2
    assert len(presses(runner)) == 2


def test_low_quality_or_one_transient_replay_frame_cannot_lock_result_retries(runner):
    confirm_win(runner)
    low = observation(GameState.REPLAY_CHOICE)
    low = replace(low, state=replace(low.state, confidence=.4))
    tick(runner, low, now=12)
    assert not runner._result_advance_acknowledged
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=13)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=14)
    assert len(presses(runner)) == 2
    tick(runner, observation(GameState.REPLAY_CHOICE), now=15)
    assert not runner._result_advance_acknowledged
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=16)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=17)
    assert sum(Button.A in buttons for buttons in presses(runner)) == 3


def test_previous_replay_cannot_acknowledge_the_next_match_pages(runner):
    from test_win_plan_feedback import NPC_ID
    runner.task = replace(runner.task, target_wins=3)
    from tableturf_rebuild.application.task_plan import build_npc_plan
    runner.plan = build_npc_plan(runner.task, (3, 0, 0))
    confirm_win(runner)
    tick(runner, observation(GameState.REPLAY_CHOICE), now=12)
    tick(runner, observation(GameState.REPLAY_CHOICE), now=13)
    assert runner._awaiting_deck_reselection and not runner._result_confirmed
    # The previous menu can remain visible while Continue opens the next deck.
    tick(runner, observation(GameState.REPLAY_CHOICE), now=14)
    tick(runner, observation(GameState.REPLAY_CHOICE), now=15)
    assert not runner._result_advance_acknowledged
    assert not runner._reward_advance_acknowledged
    runner._awaiting_deck_reselection = False  # Actual deck confirmation is tested separately.
    runner._match_identity = (NPC_ID, 2)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=20)
    tick(runner, observation(GameState.MATCH_RESULT, result=result()), now=21)
    assert runner._result_advance_sent and runner._result_advance_attempts == 1


def test_replay_candidate_hidden_by_reward_hysteresis_never_authorizes_reward_a(runner):
    confirm_win(runner)
    delayed = observation(GameState.MATCH_REWARD)
    delayed = replace(delayed, state=replace(delayed.state, confidence=1.0,
        candidates=(StateScore(GameState.REPLAY_CHOICE, 1.07), StateScore(GameState.MATCH_REWARD, .806))))
    tick(runner, delayed, now=12)
    tick(runner, replace(delayed), now=13)
    assert presses(runner) == [frozenset({Button.A})]
    assert not runner._result_advance_acknowledged


def test_real_replay_image_under_reward_hysteresis_cannot_receive_a_reward_confirm(runner):
    from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
    root = Path(__file__).resolve().parents[1]
    frame = cv2.imread(str(root / "artifacts/state_capture/run-20261021-221749/session-20261021-222704_annotations/032_video_state.png"))
    assert frame is not None
    runtime = BattleVisionRuntime.default()
    runtime.state_machine.force_state(GameState.MATCH_REWARD)
    confirm_win(runner)
    delayed_count = 0
    for index in range(5):
        obs = runtime.update(frame)
        before = presses(runner)
        tick(runner, obs, now=20 + index)
        if obs.state.state is GameState.MATCH_REWARD:
            delayed_count += 1
            assert obs.state.candidates[0].state is GameState.REPLAY_CHOICE
            assert presses(runner) == before
    assert delayed_count > 0


def test_current_replay_with_distinct_candidate_but_lower_full_scene_score_can_continue(runner):
    from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
    root = Path(__file__).resolve().parents[1]
    frame = cv2.imread(str(root / "artifacts/diagnostics/loss-result-stall/replay-capture.jpg"))
    assert frame is not None
    runtime = BattleVisionRuntime.default()
    runtime.state_machine.force_state(GameState.REPLAY_CHOICE)
    confirm_win(runner)
    for index in range(2):
        obs = runtime.update(frame)
        assert obs.state.state is GameState.REPLAY_CHOICE
        tick(runner, obs, now=20 + index)
    assert presses(runner)[-2:] == [frozenset({Button.DPAD_LEFT}), frozenset({Button.A})]


def test_ambiguous_reward_candidates_do_not_authorize_retry(runner):
    confirm_win(runner)
    obs = observation(GameState.MATCH_REWARD)
    obs = replace(obs, state=replace(obs.state, confidence=1.0,
        candidates=(StateScore(GameState.MATCH_REWARD, .84), StateScore(GameState.REPLAY_CHOICE, .82))))
    tick(runner, obs, now=12)
    tick(runner, replace(obs), now=13)
    assert presses(runner) == [frozenset({Button.A})]
