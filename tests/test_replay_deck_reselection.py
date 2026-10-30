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


def obs(state, *, deck=None):
    return BattleVisionObservation(StateObservation(state, .99, (StateScore(state, .99),), (), ()),
        None, None, (), deck=deck)


class Runtime:
    def __init__(self, catalog):
        self.catalog = catalog
        self.observation = obs(GameState.UNKNOWN)
        self.expected_deck = None
        self.clears = 0

    def update(self, frame, **kwargs):
        return self.observation

    def clear_expected_deck(self):
        self.expected_deck = None
        self.clears += 1

    def set_expected_deck(self, name, ids):
        self.expected_deck = (name, ids)


@pytest.fixture
def runner(monkeypatch):
    def pulse(controller, buttons, *args):
        controller.send_state(ControllerState(frozenset(buttons)))
        controller.send_state(ControllerState.neutral())
    monkeypatch.setattr("tableturf_rebuild.application.task_runner.pulse_buttons", pulse)
    catalog = GameCatalog.from_package()
    backend = DummyBackend()
    backend.connect()
    runtime = Runtime(catalog)
    runner = TaskRunner(runtime, backend, action_cooldown_seconds=0, deck_probe_seconds=0)
    runner.configure(NpcTaskDefinition(NPC_ID, "武器子", 2, 3),
        DeckConfig("指定卡组", catalog.decks["MiniGame_Starter"].card_ids))
    runner.plan = build_npc_plan(runner.task, (3, 0, 0))
    runner.start()
    return runner


def visual(runner, source="auto"):
    return SimpleNamespace(source=source, confidence=1.0, cards=tuple(
        SimpleNamespace(card_id=card_id, sp_cost=runner.runtime.catalog.cards[card_id].sp_cost)
        for card_id in runner.deck.card_ids))


def tick(runner, observation, now=10):
    runner.runtime.observation = observation
    return runner.tick(None, now=now)


def presses(runner):
    return [s.buttons for s in runner.controller.history if s.buttons]


def continue_replay(runner, now=20):
    runner._last_match_outcome = "LOSS"
    runner._result_confirmed = True
    runner._match_identity = (NPC_ID, 2)
    tick(runner, obs(GameState.REPLAY_CHOICE), now=now)
    tick(runner, obs(GameState.REPLAY_CHOICE), now=now + 1)


def test_initial_and_replay_decks_each_need_a_new_visual_confirmation(runner):
    tick(runner, obs(GameState.DECK_SELECT, deck=visual(runner)))
    assert runner._deck_selection_confirmed
    assert not runner._awaiting_deck_reselection
    assert presses(runner) == [frozenset({Button.A})]
    assert runner.runtime.expected_deck is not None
    runner._deck_navigation_steps = 7
    continue_replay(runner)
    assert runner._awaiting_deck_reselection
    assert not runner._deck_selection_confirmed
    assert runner._deck_navigation_steps == 0
    assert runner.runtime.expected_deck is None
    # No visual observation from the previous match can confirm this one.
    tick(runner, obs(GameState.DECK_SELECT), now=22)
    assert runner._awaiting_deck_reselection
    assert presses(runner).count(frozenset({Button.A})) == 2  # first deck + replay only
    tick(runner, obs(GameState.DECK_SELECT, deck=visual(runner)), now=23)
    assert not runner._awaiting_deck_reselection
    assert runner._deck_selection_confirmed
    assert presses(runner).count(frozenset({Button.A})) == 3


def test_configured_fifteen_cards_cannot_confirm_the_selected_deck(runner):
    tick(runner, obs(GameState.DECK_SELECT, deck=visual(runner, "configured")))
    assert runner.status is TaskStatus.WAITING_FOR_DECK
    assert runner._awaiting_deck_reselection
    assert not runner._deck_selection_confirmed
    assert not presses(runner)


@pytest.mark.parametrize("state", [GameState.BATTLE_INTRO, GameState.REDRAW,
    GameState.TURN_SELECT, GameState.TARGET_PLACEMENT, GameState.FINISHING])
def test_missing_deck_screen_blocks_every_automatic_battle_action(runner, state):
    continue_replay(runner)
    before = presses(runner)
    battle = SimpleNamespace(step=lambda *args: pytest.fail("deck lock must block executor"))
    runner.battle_executor = battle
    snapshot = tick(runner, obs(state), now=23)
    assert snapshot.status is TaskStatus.WAITING_FOR_DECK
    assert "重新确认" in snapshot.message
    assert presses(runner) == before


@pytest.mark.parametrize("state", [GameState.MATCH_RESULT, GameState.MATCH_REWARD])
def test_old_win_after_replay_cannot_be_credited_before_new_deck_confirmation(runner, state):
    continue_replay(runner)
    before = presses(runner)
    initial_plan = runner.plan
    for now in (22, 23, 24):
        # Independent readings are deliberately reliable. The deck lock,
        # rather than weak OCR or reading reuse, must reject the stale win.
        old_result = MatchResultObservation(MatchOutcome.WIN,
            PlayerResult.LOSE, PlayerResult.WIN, 1.0)
        snapshot = tick(runner, replace(obs(state), match_result=old_result), now=now)
        assert snapshot.status is TaskStatus.WAITING_FOR_DECK
    assert runner.plan == initial_plan
    assert not runner._result_confirmed
    assert not runner._match_win_applied
    assert presses(runner) == before


def test_failed_deck_confirm_keeps_reselection_locked(runner):
    runner._pulse = lambda *args: False
    tick(runner, obs(GameState.DECK_SELECT, deck=visual(runner)))
    assert runner._awaiting_deck_reselection
    assert not runner._deck_selection_confirmed
    assert runner.runtime.expected_deck is None


def test_partial_or_different_visual_deck_does_not_clear_lock(runner):
    partial = visual(runner)
    partial.cards = partial.cards[:-1]
    tick(runner, obs(GameState.DECK_SELECT, deck=partial))
    assert runner._awaiting_deck_reselection
    assert not runner._deck_selection_confirmed
    assert all(Button.A not in item for item in presses(runner))


def test_reselection_lock_and_navigation_survive_reload(runner):
    continue_replay(runner)
    tick(runner, obs(GameState.DECK_SELECT), now=23)
    assert runner._deck_navigation_steps == 1
    checkpoint = runner.progress_checkpoint()
    replacement = TaskRunner(runner.runtime, runner.controller,
        action_cooldown_seconds=0, deck_probe_seconds=0)
    replacement.configure(runner.task, runner.deck)
    replacement.plan = runner.plan
    replacement.restore_progress(checkpoint)
    replacement.start()
    before = presses(replacement)
    blocked = tick(replacement, obs(GameState.REDRAW), now=24)
    assert blocked.status is TaskStatus.WAITING_FOR_DECK
    assert replacement._deck_navigation_steps == 1
    assert presses(replacement) == before


def test_confirmed_current_deck_survives_reload_without_another_a(runner):
    tick(runner, obs(GameState.DECK_SELECT, deck=visual(runner)))
    checkpoint = runner.progress_checkpoint()
    replacement = TaskRunner(runner.runtime, runner.controller,
        action_cooldown_seconds=0, deck_probe_seconds=0)
    replacement.configure(runner.task, runner.deck)
    replacement.plan = runner.plan
    replacement.restore_progress(checkpoint)
    replacement.start()
    before = presses(replacement)
    tick(replacement, obs(GameState.DECK_SELECT), now=24)
    assert replacement._deck_selection_confirmed
    assert not replacement._awaiting_deck_reselection
    assert replacement.runtime.expected_deck is not None
    assert presses(replacement) == before


def test_starting_mid_battle_does_not_invent_a_reselection_requirement(runner):
    assert not runner._awaiting_deck_reselection
    tick(runner, obs(GameState.REDRAW), now=10)
    snapshot = tick(runner, obs(GameState.REDRAW), now=12)
    assert snapshot.status is TaskStatus.RUNNING
    assert presses(runner) == [frozenset({Button.DPAD_LEFT})]


def test_initial_menu_start_requires_the_same_deck_selection_as_a_replay(runner):
    menu = SimpleNamespace(npc_id=NPC_ID, npc_name="武器子", level_wins=(3, 0, 0),
        wins_confidence=1.0, difficulty=2)
    runner._stable_level_wins = (3, 0, 0)
    runner._wins_npc_id = NPC_ID
    runner._wins_change_allowed = False
    setup = replace(obs(GameState.MATCH_SETUP), menu=menu)
    tick(runner, setup, now=10)
    tick(runner, setup, now=11)
    assert runner._awaiting_deck_reselection
    assert not runner._deck_selection_confirmed
    assert runner.runtime.expected_deck is None
