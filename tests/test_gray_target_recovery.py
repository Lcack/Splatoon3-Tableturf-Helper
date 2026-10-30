"""A rejected move cancels once, re-reads and never retries the same footprint."""
from dataclasses import replace
from types import SimpleNamespace
import time

import pytest

from tableturf_rebuild.application.battle_executor import BattleExecutorPhase
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action
from tableturf_rebuild.vision.selection_profiles import HandVisibilityObservation
from test_post_placement_feedback import fake_setup


@pytest.fixture(scope="module")
def catalog():
    return GameCatalog.from_package()


def setup(catalog):
    ex, reader, obs, sent = fake_setup(catalog, submitted=False)
    ids = ("MiniGame_BombRobot", "MiniGame_BombCurling", "MiniGame_TakoPod", "MiniGame_Tower")
    cards = tuple(SimpleNamespace(card_id=card, slot_row=1 + i // 2, slot_col=1 + i % 2,
                                  match_score=1, name=card) for i, card in enumerate(ids))
    hand = SimpleNamespace(cards=cards, card_ids=ids, confidence=1)
    ex._current_hand = obs.hand = hand
    reader.preview.valid = False
    return ex, reader, obs, sent


def cancel_view(obs):
    return SimpleNamespace(**{**obs.__dict__,
        "state": SimpleNamespace(state=GameState.TURN_SELECT),
        "hand": SimpleNamespace(**obs.hand.__dict__),
        "board": SimpleNamespace(**{**obs.board.__dict__, "placement_preview_present": False}),
        "hand_visibility": HandVisibilityObservation(True, 230, .8, (1, 1)),
        "selection_profile": SimpleNamespace(focus_slot=(1, 1)), "discard_mode": False})


def reject(ex, obs):
    for _ in range(3):
        ex.step(obs, None)


def test_gray_cancels_once_and_requires_original_hand_and_two_fresh_reads(catalog):
    ex, _, obs, sent = setup(catalog)
    reject(ex, obs)
    assert sent == [Button.B] and ex._gray_recovery_count == 1
    assert ex._completed_turns == 0 and not ex._revealed_own_cards
    assert ex._selected_action is None and not ex._place_sent
    for _ in range(3): ex.step(obs, None)  # The old gray frame cannot confirm B.
    assert sent == [Button.B]
    fresh = cancel_view(obs)
    ex.step(fresh, None)
    ex.step(fresh, None)  # Cached board/hand is not a second reading.
    assert ex._gray_recovery["reads"] == 1
    different = cancel_view(obs)
    different.hand.card_ids = ("wrong",) * 4
    ex.step(different, None)
    assert ex._gray_recovery["reads"] == 0
    ex.step(cancel_view(obs), None)
    ex.step(cancel_view(obs), None)
    assert ex._gray_recovery is None and sent == [Button.B]


def test_replanning_filters_the_rejected_footprint(catalog):
    ex, _, obs, sent = setup(catalog)
    bad = ex._selected_action
    reject(ex, obs)
    ex.step(cancel_view(obs), None)
    ex.step(cancel_view(obs), None)
    good = Action.place(bad.card_id, bad.rotation, (bad.origin[0], bad.origin[1] + 1))
    offered = []
    ex.rules = SimpleNamespace(legal_actions=lambda *_: (bad, good))
    ex._bind_hand_shapes = lambda hand: set(hand.card_ids)
    ex.policy = SimpleNamespace(set_opponent_context=lambda _: None,
        choose=lambda rules, state, side, candidates: offered.extend(candidates) or good)
    fresh = cancel_view(obs)
    fresh.opponent_context = None
    ex.step(fresh, None)
    assert bad not in offered and good in offered
    assert ex._selected_action == good
    assert ex._completed_turns == 0


def test_reload_preserves_exclusion_budget_and_pending_cancel_without_resending_b(catalog):
    ex, _, obs, _ = setup(catalog)
    reject(ex, obs)
    saved = ex.battle_memory_checkpoint()
    fresh, _, _, sent = setup(catalog)
    fresh.restore_battle_memory(saved)
    for _ in range(3): fresh.step(obs, None)
    assert not sent and fresh._gray_recovery_count == 1
    assert fresh._rejected_placements == ex._rejected_placements
    fresh.step(cancel_view(obs), None)
    fresh.step(cancel_view(obs), None)
    assert fresh._gray_recovery is None and not sent


def test_recovery_timeout_stops_without_extra_a_or_b(catalog):
    ex, _, obs, sent = setup(catalog)
    reject(ex, obs)
    ex._gray_recovery["at"] = time.perf_counter() - 13
    ex.step(obs, None)
    assert ex.phase is BattleExecutorPhase.ERROR and sent == [Button.B]


def test_recovery_budget_is_bounded_and_only_a_real_turn_change_clears_it(catalog):
    ex, _, obs, sent = setup(catalog)
    ex._gray_recovery_count = ex.MAX_GRAY_RECOVERIES
    reject(ex, obs)
    assert ex.phase is BattleExecutorPhase.ERROR and not sent
    assert "恢复3次" in ex.message
    ex._completed_turns += 1
    ex.step(obs, None)
    assert ex._gray_recovery_count == 0 and not ex._rejected_placements


def test_submitted_a_never_enters_gray_recovery(catalog):
    ex, _, obs, sent = setup(catalog)
    ex._place_sent = True
    ex._place_sent_at = time.perf_counter() - 2
    reject(ex, obs)
    assert ex.phase is BattleExecutorPhase.WAITING_RESULT
    assert ex._gray_recovery is None and not sent


def test_a_new_card_does_not_inherit_the_rejected_previews_stability_count(catalog):
    ex, reader, obs, sent = setup(catalog)
    hand = ex._current_hand
    reject(ex, obs)
    ex.step(cancel_view(obs), None)
    ex.step(cancel_view(obs), None)
    shape = ex.catalog.cards["MiniGame_BombCurling"].shape
    first = min(shape.occupied)
    anchor = reader.preview.anchor
    ex._selected_action = Action.place("MiniGame_BombCurling", 0,
                                      (anchor[0] - first[0], anchor[1] - first[1]))
    ex._selected_card_id = ex._selected_action.card_id
    ex._current_hand = hand
    ex._card_confirmed = True
    ex._card_confirmed_at = ex._preview_response_at = -100
    ex._position_selected_card(obs, None)
    assert ex._preview_signature_count == 1
    assert sent == [Button.B] and ex._gray_recovery_count == 1


def test_checkpoint_immediately_after_turn_increment_does_not_carry_old_rejections(catalog):
    ex, _, obs, _ = setup(catalog)
    reject(ex, obs)
    ex._completed_turns += 1  # The commit-completion branch advances mid-step.
    saved = ex.battle_memory_checkpoint()  # Reload before the next step starts.
    assert saved["rejected_placements"] == () and saved["gray_recoveries"] == 0
    assert saved["gray_recovery_hand"] is None
    fresh, _, _, _ = setup(catalog)
    fresh.restore_battle_memory(saved)
    assert fresh._completed_turns == 1
    assert not fresh._rejected_placements and fresh._gray_recovery is None
