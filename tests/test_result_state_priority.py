"""Result words override whole-page guesses before hand/board interpretation."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.state_machine import GameState, StateObservation, StateScore
from tableturf_rebuild.vision.match_result_reader import MatchOutcome, MatchResultObservation, PlayerResult


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "artifacts/diagnostics/win-misread-hand-20261030"
STATES = ROOT / "artifacts/state_capture/run-20261021-221749/session-20261021-222704_annotations"
WINS = (SAMPLE / "analysis.png", SAMPLE / "capture.jpg",
        *(SAMPLE / f"phase-{i:02d}.jpg" for i in range(4)))
LOSS = STATES / "011_video_state.png"


def load(path):
    frame = cv2.imread(str(path))
    assert frame is not None, path
    return frame


def fail_battle_read(*args, **kwargs):
    pytest.fail("known result page must not be decoded as a hand or board")


@pytest.mark.parametrize("path", WINS)
def test_actual_win_bypasses_false_hand_focus_and_stale_battle_state(path):
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("Cross")
    runtime.state_machine.force_state(GameState.TURN_SELECT)
    runtime.hand_observation = object()  # A prior turn must not leak into the result.
    runtime.board_reader.read = fail_battle_read
    runtime.selection_resolver.hand_visibility = fail_battle_read
    runtime.selection_resolver.match = fail_battle_read
    runtime._read_hand = fail_battle_read
    frame = load(path)

    observation = runtime.update(frame, include_selection=True, force_selection=True,
                                 placement_expected=True)

    assert observation.state.state is GameState.MATCH_RESULT
    assert observation.state.candidates[0].state is GameState.MATCH_RESULT
    assert observation.match_result.outcome is MatchOutcome.WIN
    assert observation.match_result.confidence >= runtime.RESULT_OVERRIDE_CONFIDENCE
    assert runtime.state_machine.current is GameState.MATCH_RESULT
    assert runtime.hand_observation is None
    assert observation.hand is None and observation.hand_visibility is None
    assert observation.selection_profile is None and observation.selection_candidates == ()
    assert observation.board is None and observation.reveal is None


def test_real_loss_also_bypasses_card_interpretation_and_results_are_not_cached():
    runtime = BattleVisionRuntime.default()
    runtime.state_machine.force_state(GameState.TARGET_PLACEMENT)
    runtime.selection_resolver.match = fail_battle_read
    first = runtime.update(load(LOSS), include_selection=True)
    second = runtime.update(load(WINS[0]), include_selection=True)
    assert first.state.state is second.state.state is GameState.MATCH_RESULT
    assert first.match_result.outcome is MatchOutcome.LOSS
    assert second.match_result.outcome is MatchOutcome.WIN
    assert first.match_result is not second.match_result


@pytest.mark.parametrize("index", (1, 10, 12, 17, 19, 24, 29, 31, 32, 39, 42, 45))
def test_real_battle_and_menu_pages_do_not_trigger_result_priority(index):
    runtime = BattleVisionRuntime.default()
    observation = runtime.update(load(STATES / f"{index:03d}_video_state.png"))
    assert not (observation.match_result is not None
                and observation.match_result.outcome in {MatchOutcome.WIN, MatchOutcome.LOSS})


@pytest.mark.parametrize("quality", (0.0, 0.84))
def test_uncertain_result_cannot_override_real_card_selection(quality):
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("Rectangle")
    runtime.state_machine.force_state(GameState.TURN_SELECT)
    runtime.match_result_reader = SimpleNamespace(read=lambda frame: MatchResultObservation(
        MatchOutcome.WIN if quality else MatchOutcome.UNKNOWN,
        PlayerResult.LOSE, PlayerResult.WIN, quality))
    observation = runtime.update(load(STATES / "042_video_state.png"), include_selection=True)
    assert observation.state.state is GameState.TURN_SELECT
    assert observation.match_result is None
    assert observation.hand is not None


def test_confirmed_dim_placement_skips_hand_work_even_with_stale_selection_class():
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("CenterHole")
    runtime._grid_fit_checked = True
    runtime.state_machine.force_state(GameState.TURN_SELECT)
    raw = StateObservation(GameState.TURN_SELECT, .95, (
        StateScore(GameState.TURN_SELECT, .95), StateScore(GameState.TARGET_PLACEMENT, .3),
    ), (), ())
    runtime.state_machine.recognizer = SimpleNamespace(classify=lambda frame: raw)
    runtime.selection_resolver.match = fail_battle_read
    runtime._read_hand = fail_battle_read
    frame = load(ROOT / "artifacts/diagnostics/latency-investigation/analysis.png")
    observation = runtime.update(frame, include_selection=True, placement_expected=True)
    assert observation.hand_visibility.selection_visible is False
    assert observation.board is not None
    assert observation.hand is None and observation.selection_profile is None


def test_visible_card_selection_is_reread_even_if_executor_expected_placement():
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("Rectangle")
    runtime.state_machine.force_state(GameState.TURN_SELECT)
    calls = []
    original = runtime._read_hand
    def counted_read(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    runtime._read_hand = counted_read
    observation = runtime.update(load(STATES / "042_video_state.png"),
                                 include_selection=True, placement_expected=True)
    assert observation.hand_visibility.selection_visible is True
    assert observation.selection_profile is not None
    assert observation.hand is not None and calls
