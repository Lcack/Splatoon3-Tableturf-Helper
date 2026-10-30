"""Rotation commands need visual acknowledgement before another correction."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tableturf_rebuild.application.battle_executor import BattleExecutorPhase, VisionMctsBattleExecutor
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action
from tableturf_rebuild.domain.card import CardShape, CardSpec
from tableturf_rebuild.vision.placement_reader import PlacementPreviewObservation


@pytest.fixture(scope="module")
def catalog():
    base = GameCatalog.from_package()
    probe = CardSpec("rotation_probe", "probe", "旋转测试", 1,
                     CardShape.from_matrix([[2], [1], [1], [1], [1]]), 5)
    return replace(base, cards={**base.cards, probe.card_id: probe})


def make_executor(catalog, *, target=0, card_id="rotation_probe", target_origin=(10, 5)):
    backend = DummyBackend()
    backend.connect()
    runtime = SimpleNamespace(catalog=catalog, stage_id="Cross")
    executor = VisionMctsBattleExecutor(runtime, backend, catalog.decks["MiniGame_Starter"].card_ids)
    executor._selected_action = Action.place(card_id, target, target_origin)
    executor._selected_card_id = card_id
    executor._card_confirmed = True
    executor._card_confirmed_at = -100
    sent = []
    executor._pulse = lambda button: sent.append(button) or True
    reader = SimpleNamespace(preview=None, reset_history=lambda: None)
    reader.read = lambda *_: reader.preview
    executor.placement_reader = reader
    board = SimpleNamespace(placement_preview_present=True)
    observation = SimpleNamespace(state=SimpleNamespace(state=GameState.TARGET_PLACEMENT),
        board=board, selection_profile=None, hand_visibility=None, hand=None, reveal=None)
    return executor, reader, sent, observation


def feed(setup, rotation, count=1, *, origin=(10, 5), valid=True):
    executor, reader, _, observation = setup
    shape = executor.catalog.cards[executor._selected_card_id].shape.rotated(rotation)
    cells = frozenset((origin[0] + row, origin[1] + col) for row, col in shape.occupied)
    reader.preview = PlacementPreviewObservation(cells, rotation, origin, min(cells), 1.0, valid, float(valid))
    for _ in range(count):
        executor._preview_response_at = 0  # Feed a frame after the real response delay.
        executor.step(observation, None)


def test_reverse_quarter_turn_uses_one_x_then_requires_visual_confirmation(catalog):
    setup = make_executor(catalog)
    executor, _, sent, _ = setup
    original_action = executor._selected_action
    feed(setup, 1, 2)
    assert sent == [Button.X]
    assert executor._pending_rotation["expected_rotation"] == 0
    feed(setup, 0, 2)
    assert sent == [Button.X]  # The two acknowledgement frames also count toward final stability.
    assert executor._pending_rotation is None
    feed(setup, 0)
    assert sent == [Button.X, Button.A]
    assert executor._selected_action is original_action


def test_half_turn_sends_two_y_only_after_each_observed_quarter_turn(catalog):
    setup = make_executor(catalog, target=2)
    _, _, sent, _ = setup
    feed(setup, 0, 2)
    assert sent == [Button.Y]
    feed(setup, 1)
    assert sent == [Button.Y]
    feed(setup, 1)
    assert sent == [Button.Y, Button.Y]
    feed(setup, 2, 3)
    assert sent == [Button.Y, Button.Y, Button.A]


def test_ambiguous_pixel_orientation_never_falls_back_to_a_board_guess(catalog):
    import numpy as np
    executor, reader, sent, observation = make_executor(catalog, target=2)
    executor.runtime.preview_region = object()
    reader.last_diagnostics = {"failure": "AMBIGUOUS_SPECIAL_ORIENTATION"}
    reader.read_frame = lambda *args, **kwargs: None
    def forbidden_fallback(*args, **kwargs):
        pytest.fail("ambiguous pixel evidence must not become a guessed board orientation")
    reader.read = forbidden_fallback
    executor._pending_rotation = {"card_id": "rotation_probe", "button": "Y",
                                  "before_rotation": 1, "expected_rotation": 2}
    for _ in range(3):
        executor.step(observation, np.zeros((2, 2, 3), dtype=np.uint8))
    assert executor.phase is BattleExecutorPhase.WAITING and not sent
    assert executor._preview_missing_reads == 3
    assert executor._pending_rotation["expected_rotation"] == 2


@pytest.mark.parametrize("wrong_rotation", (1, 2, 3))
def test_wrong_feedback_never_triggers_opposite_rotation_or_placement(catalog, wrong_rotation):
    setup = make_executor(catalog)
    executor, _, sent, _ = setup
    feed(setup, 1, 2)  # One X should lead to 0.
    feed(setup, wrong_rotation, executor.MAX_ROTATION_FEEDBACK_READS)
    assert sent == [Button.X]
    assert executor.phase is BattleExecutorPhase.ERROR
    assert "期望r=0" in executor.message


def test_uncommanded_rotation_change_is_rechecked_instead_of_corrected(catalog):
    setup = make_executor(catalog)
    executor, _, sent, _ = setup
    feed(setup, 0, 2)  # Correct orientation, waiting for the third placement frame.
    feed(setup, 1, executor.MAX_ROTATION_FEEDBACK_READS)
    assert sent == []
    assert executor.phase is BattleExecutorPhase.ERROR


def test_equivalent_symmetric_rotations_do_not_reset_stability(catalog):
    setup = make_executor(catalog, card_id="MiniGame_ShockSonar")
    _, _, sent, _ = setup
    feed(setup, 1)
    feed(setup, 3)
    feed(setup, 2)
    assert sent == [Button.A]


def test_manual_feedback_reset_keeps_plan_and_card_memory(catalog):
    setup = make_executor(catalog)
    executor, _, sent, _ = setup
    action = executor._selected_action
    executor._revealed_own_cards = ["MiniGame_Tamaire"]
    executor._completed_turns = 3
    feed(setup, 1, 2)
    executor.reset_preview_feedback()
    assert executor._selected_action is action
    assert executor._revealed_own_cards == ["MiniGame_Tamaire"] and executor._completed_turns == 3
    assert executor._pending_rotation is None and executor._verified_rotation is None
    feed(setup, 0, 3)
    assert sent == [Button.X, Button.A]
