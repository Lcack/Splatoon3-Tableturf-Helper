from pathlib import Path
from dataclasses import replace
import warnings

import cv2
import numpy as np
import pytest

from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.state_machine import (
    GameState, StateEvidence, StateMachine, StateObservation, StateScore,
)
from tableturf_rebuild.vision.preview_features import _edge_diagonal_score
from tableturf_rebuild.vision.sp_reader import SpObservation


ROOT = Path(__file__).resolve().parents[1]


def legacy_diagonal_score(edges):
    edges = np.rot90(cv2.dilate(edges, np.ones((2, 2), np.uint8)))
    half = edges.shape[1] // 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = [float(np.mean(np.diag(edges, offset) == 255))
                  for offset in range(-half, half)]
    return sum(value > 0.8 for value in values) / len(values) if values else 0.0


@pytest.mark.parametrize("shape", [(32, 32), (34, 34), (20, 40), (40, 20), (8, 34), (1, 8)])
def test_vectorized_diagonal_reduction_preserves_detector(shape):
    rng = np.random.default_rng(43)
    for density in (0, .05, .2, .5, .8, 1):
        for _ in range(4):
            edges = (rng.random(shape) < density).astype(np.uint8) * 255
            assert _edge_diagonal_score(edges) == legacy_diagonal_score(edges)


def reveal_like_observation():
    return StateObservation(GameState.TURN_REVEAL, .93, (
        StateScore(GameState.TURN_REVEAL, .93),
        StateScore(GameState.TARGET_PLACEMENT, .80),
        StateScore(GameState.TURN_SELECT, .53),
    ), (), ())


def test_two_white_cursor_observations_end_reveal_without_generic_hysteresis():
    machine = StateMachine(stable_frames=4)
    machine.force_state(GameState.TURN_REVEAL)
    visible = StateEvidence(placement_preview_present=False,
                            hand_selection_visible=True, hand_cursor_visible=True)
    raw = reveal_like_observation()
    assert machine.update_observation(raw, visible).state is GameState.TURN_REVEAL
    assert machine.update_observation(raw, visible).state is GameState.TURN_SELECT


def test_no_cursor_keeps_normal_reveal_hysteresis():
    machine = StateMachine(stable_frames=4)
    machine.force_state(GameState.TURN_REVEAL)
    weak = StateEvidence(placement_preview_present=False, hand_selection_visible=True)
    for _ in range(4):
        assert machine.update_observation(reveal_like_observation(), weak).state is GameState.TURN_REVEAL


def test_confirmed_placement_does_not_reidentify_four_dim_hand_cards():
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("CenterHole")
    runtime._grid_fit_checked = True
    runtime.state_machine.force_state(GameState.TARGET_PLACEMENT)
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/latency-investigation/analysis.png"))
    assert frame is not None
    def unexpected_read(*args, **kwargs):
        pytest.fail("confirmed dim placement should reuse the executor's cached hand")
    runtime._read_hand = unexpected_read
    observation = runtime.update(frame, include_selection=True, placement_expected=True)
    assert observation.hand_visibility.selection_visible is False
    assert observation.hand is None
    assert observation.board is not None
    assert observation.timings_ms["board"] > 0
    assert observation.timings_ms["total"] >= observation.timings_ms["board"]


def test_live_bright_selection_clears_reveal_with_two_independent_frames():
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("SShape")
    runtime.state_machine.force_state(GameState.TURN_REVEAL)
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/blocked-card/analysis.png"))
    assert frame is not None
    first = runtime.update(frame, include_selection=True)
    second = runtime.update(frame, include_selection=True)
    assert second.hand_visibility.cursor_slot == (1, 1)
    assert second.state.state is GameState.TURN_SELECT
    assert second.hand is not None


def test_sp_confirms_sides_independently_and_never_reuses_spent_balance():
    from types import SimpleNamespace
    runtime = BattleVisionRuntime.default()
    raw = SpObservation(3, 2, 1.0, 1.0, (), (), 255)
    runtime.sp_reader = SimpleNamespace(read=lambda frame: raw)
    frame = np.zeros((1, 1, 3), np.uint8)
    first = runtime._read_sp(frame, GameState.TURN_SELECT)
    assert first.self_value is None and first.opponent_value is None
    second = runtime._read_sp(frame, GameState.TURN_SELECT)
    assert (second.self_value, second.opponent_value) == (3, 2)
    raw = replace(raw, self_value=0)
    changed = runtime._read_sp(frame, GameState.TARGET_PLACEMENT)
    assert changed.self_value is None  # never publish the old spendable 3
    assert changed.opponent_value == 2
    confirmed = runtime._read_sp(frame, GameState.TARGET_PLACEMENT)
    assert confirmed.self_value == 0
    raw = replace(raw, self_value=None, self_confidence=0, self_reason="HUD_NOT_VISIBLE")
    assert runtime._read_sp(frame, GameState.TURN_SELECT).self_value is None
    raw = replace(raw, self_value=0, self_confidence=1, self_reason=None)
    assert runtime._read_sp(frame, GameState.TURN_SELECT).self_value is None
    assert runtime._read_sp(frame, GameState.TURN_SELECT).self_value == 0
    runtime._read_sp(frame, GameState.MATCH_RESULT)
    assert runtime._read_sp(frame, GameState.TURN_SELECT).opponent_value is None


def test_sp_low_confidence_cannot_become_spendable_even_after_repetition():
    from types import SimpleNamespace
    runtime = BattleVisionRuntime.default()
    raw = SpObservation(4, 2, 0.79, 1.0, (), (), 255)
    runtime.sp_reader = SimpleNamespace(read=lambda frame: raw)
    frame = np.zeros((1, 1, 3), np.uint8)
    for _ in range(4):
        assert runtime._read_sp(frame, GameState.TURN_SELECT).self_value is None


def test_button_label_and_bottom_cursor_recover_stale_finishing_state():
    from types import SimpleNamespace
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("CenterHole")
    runtime._grid_fit_checked = True
    runtime.state_machine.force_state(GameState.FINISHING)
    raw = StateObservation(GameState.FINISHING, .95, (
        StateScore(GameState.FINISHING, .95), StateScore(GameState.TURN_SELECT, .35),
    ), (), ())
    runtime.state_machine.recognizer = SimpleNamespace(classify=lambda frame: raw)
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/special-mode-off-available/analysis.png"))
    assert frame is not None
    original = runtime.mode_reader.read
    calls = []
    def counted_read(frame):
        calls.append(1)
        return original(frame)
    runtime.mode_reader.read = counted_read
    first = runtime.update(frame, include_selection=True)
    second = runtime.update(frame, include_selection=True)
    assert first.board is not None and second.board is not None
    assert first.hand_visibility.cursor_slot == (3, 1)
    assert second.state.state is GameState.TURN_SELECT
    assert second.selection_profile.focus_slot == (3, 1)
    assert second.hand is not None
    assert len(calls) == 2  # one early reading per frame, reused in the result
