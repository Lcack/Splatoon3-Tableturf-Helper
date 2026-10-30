"""Confirm the replay modal independently of the final board background."""
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.state_machine import GameState, StateScore
from tableturf_rebuild.application.task_plan import build_npc_plan
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.vision.replay_choice_reader import ReplayChoiceObservation
from test_win_plan_feedback import confirm_win, observation, presses, runner, tick


def modal(visible=True):
    return ReplayChoiceObservation(visible, .99 if visible else .4, .99, .99,
                                   .99 if visible else .4,
                                   None if visible else "BUTTON_TEXT_UNCERTAIN")


def replay(*, visible=True, score=.775, margin=.29):
    obs = observation(GameState.REPLAY_CHOICE)
    return replace(obs, state=replace(obs.state, confidence=score, candidates=(
        StateScore(GameState.REPLAY_CHOICE, score),
        StateScore(GameState.MATCH_REWARD, score - margin),
    )), replay_choice_evidence=modal(visible))


def continue_plan(runner):
    runner.task = replace(runner.task, target_wins=3)
    runner.plan = build_npc_plan(runner.task, (3, 0, 0))
    confirm_win(runner)


def test_two_local_modal_confirmations_continue_and_preserve_win_progress(runner):
    continue_plan(runner)
    obs = replay()
    tick(runner, obs, now=12)
    tick(runner, obs, now=13)  # Cached observation is not a second confirmation.
    assert presses(runner) == [frozenset({Button.A}), frozenset({Button.DPAD_RIGHT})]
    tick(runner, replace(obs), now=14)
    assert presses(runner)[-1] == frozenset({Button.A})
    assert runner._awaiting_deck_reselection
    assert runner._stable_level_wins == (3, 1, 0)


@pytest.mark.parametrize("visible,score,margin", [(False, .99, .4), (True, .99, .02)])
def test_negative_modal_or_ambiguous_scene_never_confirms(runner, visible, score, margin):
    confirm_win(runner)
    before = presses(runner)
    for index in range(3):
        tick(runner, replay(visible=visible, score=score, margin=margin), now=12 + index)
    assert presses(runner) == before
    assert runner._replay_evidence_reads == 0
    assert "标题=" in runner.message


def test_local_modal_cannot_authorize_a_state_retained_by_hysteresis(runner):
    confirm_win(runner)
    obs = replay()
    obs = replace(obs, state=replace(obs.state, state=GameState.MATCH_REWARD, confidence=1))
    for index in range(3):
        tick(runner, replace(obs), now=12 + index)
    assert presses(runner) == [frozenset({Button.A})]
    assert not runner._result_advance_acknowledged


def test_actual_win_replay_below_whole_frame_threshold_can_continue(runner):
    frame = cv2.imread(str(Path(__file__).resolve().parents[1] /
        "artifacts/diagnostics/win-replay-stall-140226/capture.jpg"))
    assert frame is not None
    runtime = BattleVisionRuntime.default()
    runtime.state_machine.force_state(GameState.REPLAY_CHOICE)
    continue_plan(runner)
    for index in range(2):
        obs = runtime.update(frame)
        assert obs.state.state is GameState.REPLAY_CHOICE
        assert obs.state.confidence < runner.SETTLEMENT_SCENE_CONFIDENCE
        assert obs.replay_choice_evidence.visible
        tick(runner, obs, now=12 + index)
    assert presses(runner)[-2:] == [frozenset({Button.DPAD_RIGHT}), frozenset({Button.A})]
    assert runner._awaiting_deck_reselection
    assert runner._stable_level_wins == (3, 1, 0)
