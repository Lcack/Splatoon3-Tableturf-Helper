"""Bounded directional batches retain real visual feedback and fine placement."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tableturf_rebuild.application import battle_executor as execution
from tableturf_rebuild.application.battle_executor import BattleExecutorPhase, VisionMctsBattleExecutor
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action
from tableturf_rebuild.domain.card import CardShape, CardSpec
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.placement_reader import PlacementPreviewObservation


@pytest.fixture(scope="module")
def catalog():
    base = GameCatalog.from_package()
    card = CardSpec("motion_probe", "probe", "移动测试", 1,
                    CardShape.from_matrix([[2], [1], [1], [1], [1]]), 5)
    return replace(base, cards={**base.cards, card.card_id: card})


def setup_executor(catalog, monkeypatch, *, target=(10, 13), backend=None, cells=None):
    backend = backend or DummyBackend()
    backend.connect()
    runtime = SimpleNamespace(catalog=catalog, stage_id="Cross")
    executor = VisionMctsBattleExecutor(runtime, backend, catalog.decks["MiniGame_Starter"].card_ids)
    executor._selected_action = Action.place("motion_probe", 0, target)
    executor._selected_card_id = "motion_probe"
    executor._card_confirmed = True
    executor._card_confirmed_at = -100
    cells = cells or tuple(tuple(Cell.EMPTY for _ in range(25)) for _ in range(25))
    board = SimpleNamespace(placement_preview_present=True, rows=len(cells), cols=len(cells[0]), cells=cells)
    observation = SimpleNamespace(state=SimpleNamespace(state=GameState.TARGET_PLACEMENT),
        board=board, selection_profile=None, hand_visibility=None, hand=None, reveal=None)
    reader = SimpleNamespace(preview=None, reset_history=lambda: None)
    reader.read = lambda *_: reader.preview
    executor.placement_reader = reader
    sleeps = []
    original = execution.pulse_buttons
    def pulse(controller, buttons):
        original(controller, buttons, sleep=sleeps.append)
    monkeypatch.setattr(execution, "pulse_buttons", pulse)
    return executor, reader, observation, backend, sleeps


def feed(setup, anchor, count=1, *, score=1.0, valid=False):
    executor, reader, observation, _, _ = setup
    cells = frozenset((anchor[0] + row, anchor[1]) for row in range(5))
    reader.preview = PlacementPreviewObservation(cells, 0, anchor, anchor, score, valid, float(valid))
    for _ in range(count):
        executor._preview_response_at = 0
        executor.step(observation, None)


def presses(backend):
    return [next(iter(state.buttons)) for state in backend.history if state.buttons]


def test_coarse_batch_waits_for_orientation_and_real_feedback_before_more_keys(catalog, monkeypatch):
    setup = setup_executor(catalog, monkeypatch)
    executor, _, _, backend, sleeps = setup
    feed(setup, (10, 5))
    assert presses(backend) == []
    feed(setup, (10, 5))
    assert presses(backend) == [Button.DPAD_RIGHT] * 3
    assert sleeps == [0.05, 0.35] * 3  # No increased pulse frequency or held key.
    assert executor._pending_movement["expected_anchor"] == (10, 8)
    assert executor._last_preview.anchor == (10, 5)  # Sending is not an observation.
    feed(setup, (10, 8))
    assert presses(backend) == [Button.DPAD_RIGHT] * 3
    feed(setup, (10, 8))
    assert presses(backend) == [Button.DPAD_RIGHT] * 6
    assert executor._pending_movement["before_anchor"] == (10, 8)
    assert executor._pending_movement["expected_anchor"] == (10, 11)
    feed(setup, (10, 11), 2)
    assert presses(backend) == [Button.DPAD_RIGHT] * 7
    feed(setup, (10, 12), 2)
    assert presses(backend) == [Button.DPAD_RIGHT] * 8
    feed(setup, (10, 13), 2, valid=True)
    assert Button.A not in presses(backend)
    feed(setup, (10, 13), valid=True)
    assert presses(backend) == [Button.DPAD_RIGHT] * 8 + [Button.A]


@pytest.mark.parametrize("distance", (1, 2, 3, 4))
def test_near_target_stays_single_step(catalog, monkeypatch, distance):
    setup = setup_executor(catalog, monkeypatch, target=(10, 5 + distance))
    feed(setup, (10, 5), 2)
    assert presses(setup[3]) == [Button.DPAD_RIGHT]
    assert setup[0]._pending_movement is None


@pytest.mark.parametrize("blocked_col", (6, 7, 8, 9))
def test_wall_in_short_path_or_one_square_buffer_disables_batch(catalog, monkeypatch, blocked_col):
    cells = [list([Cell.EMPTY] * 25) for _ in range(25)]
    cells[10][blocked_col] = Cell.WALL
    setup = setup_executor(catalog, monkeypatch, cells=tuple(map(tuple, cells)))
    feed(setup, (10, 5), 2)
    assert presses(setup[3]) == [Button.DPAD_RIGHT]


@pytest.mark.parametrize("anchor,target", [((14, 16), (21, 16)), ((10, 16), (10, 22))])
def test_cross_void_and_edge_are_not_batched(catalog, monkeypatch, anchor, target):
    setup = setup_executor(catalog, monkeypatch, target=target, cells=catalog.stages["Cross"].cells)
    feed(setup, anchor, 2)
    assert len(presses(setup[3])) == 1
    assert setup[0]._pending_movement is None


def test_feedback_shortfall_disables_coarse_until_this_action_finishes(catalog, monkeypatch):
    setup = setup_executor(catalog, monkeypatch)
    executor, _, _, backend, _ = setup
    feed(setup, (10, 5), 2)
    feed(setup, (10, 7), 2)  # One of the three pulses was lost.
    assert presses(backend) == [Button.DPAD_RIGHT] * 3
    assert executor._coarse_movement_disabled
    assert executor._last_movement_batch["difference"] == (0, -1)
    assert "本张牌改为逐格精调" in executor.message
    feed(setup, (10, 7))
    assert presses(backend) == [Button.DPAD_RIGHT] * 4
    assert Button.A not in presses(backend)


def test_low_shape_score_uses_single_step_even_when_orientation_is_stable(catalog, monkeypatch):
    setup = setup_executor(catalog, monkeypatch)
    feed(setup, (10, 5), 2, score=0.8)
    assert presses(setup[3]) == [Button.DPAD_RIGHT]


class HookBackend(DummyBackend):
    on_release = None
    fail_on_press = None
    attempts = 0

    def send_state(self, state):
        if state.buttons:
            self.attempts += 1
            if self.attempts == self.fail_on_press:
                return False
        result = super().send_state(state)
        if not state.buttons and self.on_release is not None:
            self.on_release()
        return result


def test_pause_cancels_remaining_batch_pulses_and_resume_needs_new_observations(catalog, monkeypatch):
    backend = HookBackend()
    setup = setup_executor(catalog, monkeypatch, backend=backend)
    executor = setup[0]
    backend.on_release = executor.cancel_pending_motion
    feed(setup, (10, 5), 2)
    assert presses(backend) == [Button.DPAD_RIGHT]
    assert executor._pending_movement["sent_steps"] == 1
    assert executor._pending_movement["cancelled"]
    feed(setup, (10, 6), 3)
    assert presses(backend) == [Button.DPAD_RIGHT]
    backend.on_release = None
    executor.resume_pending_motion()
    feed(setup, (10, 6))
    assert presses(backend) == [Button.DPAD_RIGHT]
    feed(setup, (10, 6))
    assert presses(backend) == [Button.DPAD_RIGHT] * 4


def test_controller_failure_stops_batch_without_sending_later_pulses(catalog, monkeypatch):
    backend = HookBackend()
    backend.fail_on_press = 2
    setup = setup_executor(catalog, monkeypatch, backend=backend)
    feed(setup, (10, 5), 2)
    assert backend.attempts == 2
    assert presses(backend) == [Button.DPAD_RIGHT]
    assert setup[0].phase is BattleExecutorPhase.ERROR
    assert setup[0]._pending_movement["sent_steps"] == 1


def test_pause_does_not_clear_an_existing_executor_error(catalog, monkeypatch):
    setup = setup_executor(catalog, monkeypatch)
    executor = setup[0]
    executor.phase = BattleExecutorPhase.ERROR
    executor.cancel_pending_motion()
    feed(setup, (10, 5), 2)
    assert executor.phase is BattleExecutorPhase.ERROR
    assert not presses(setup[3])
