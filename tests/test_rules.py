import pytest

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action, ActionType
from tableturf_rebuild.domain.card import CardCell
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.domain.rules import BattleRules, RulesError
from tableturf_rebuild.domain.state import Side


@pytest.fixture(scope="module")
def catalog() -> GameCatalog:
    return GameCatalog.from_package()


@pytest.fixture(scope="module")
def rules(catalog: GameCatalog) -> BattleRules:
    return BattleRules(catalog)


def two_hands(catalog: GameCatalog):
    deck = catalog.decks["MiniGame_Aori"].card_ids
    return deck[:4], deck[4:8]


def test_new_battle_is_reproducible(rules: BattleRules) -> None:
    first = rules.new_battle("Square", "MiniGame_Aori", "MiniGame_Aori", seed=42)
    second = rules.new_battle("Square", "MiniGame_Aori", "MiniGame_Aori", seed=42)

    assert first == second
    assert len(first.player_a.hand) == 4
    assert len(first.player_b.hand) == 4
    assert len(first.player_a.draw_pile) == 11
    assert len(first.player_b.draw_pile) == 11
    assert first.turns_remaining == 12


def test_card_rotation_is_reversible(catalog: GameCatalog) -> None:
    for card in catalog.cards.values():
        for rotation in range(4):
            rotated = card.shape.rotated(rotation)
            assert rotated.size == card.shape.size
            assert rotated.rotated(4 - rotation) == card.shape


def test_redraw_is_available_once(
    rules: BattleRules, catalog: GameCatalog
) -> None:
    state = rules.new_battle(
        "Square",
        "MiniGame_Aori",
        "MiniGame_Aori",
        seed=11,
    )
    redrawn = rules.redraw(state, Side.A, seed=9)

    assert not redrawn.player_a.redraw_available
    assert len(redrawn.player_a.hand) == 4
    assert len(redrawn.player_a.draw_pile) == 11
    with pytest.raises(RulesError):
        rules.redraw(redrawn, Side.A, seed=10)


def test_simultaneous_turn_and_skip_discard(
    rules: BattleRules, catalog: GameCatalog
) -> None:
    hand, opponent_hand = two_hands(catalog)
    state = rules.with_hand(
        "Square",
        hand,
        opponent_hand=opponent_hand,
        self_sp=2,
    )
    legal_a = rules.legal_actions(state, Side.A)
    legal_b = rules.legal_actions(state, Side.B)
    skip_a = next(
        action
        for action in legal_a
        if action.action_type is ActionType.SKIP and action.card_id == hand[0]
    )
    skip_b = next(
        action
        for action in legal_b
        if action.action_type is ActionType.SKIP and action.card_id == opponent_hand[0]
    )
    next_state = rules.apply_turn(
        state,
        skip_a,
        skip_b,
        legal_a=legal_a,
        legal_b=legal_b,
    )

    assert next_state.turns_remaining == 11
    assert skip_a.card_id not in next_state.player_a.hand
    assert skip_b.card_id not in next_state.player_b.hand
    assert next_state.player_a.sp == state.player_a.sp + 1
    assert next_state.player_b.sp == state.player_b.sp + 1


def test_every_generated_action_is_applicable(
    rules: BattleRules, catalog: GameCatalog
) -> None:
    hand, opponent_hand = two_hands(catalog)
    state = rules.with_hand("Square", hand, opponent_hand=opponent_hand)
    legal_a = rules.legal_actions(state, Side.A)
    legal_b = rules.legal_actions(state, Side.B)
    placements = [
        action
        for action in legal_a
        if action.action_type is ActionType.PLACE
    ]

    assert placements
    skip_b = legal_b[0]
    for action in placements[:20]:
        next_state = rules.apply_turn(
            state,
            action,
            skip_b,
            legal_a=legal_a,
            legal_b=legal_b,
        )
        assert next_state.turns_remaining == 11


def test_every_placement_touches_current_ink_frontier(
    rules: BattleRules,
    catalog: GameCatalog,
) -> None:
    hand, opponent_hand = two_hands(catalog)
    state = rules.with_hand(
        "Square",
        hand,
        opponent_hand=opponent_hand,
    )
    frontier = rules._empty_frontier(state, Side.A)
    placements = [
        action
        for action in rules.legal_actions(state, Side.A)
        if action.action_type is ActionType.PLACE
    ]

    assert placements
    for action in placements:
        targets = rules.action_targets(state, action)
        assert any(
            (row, col) in frontier
            for row, col, _ in targets
        )


def test_illegal_action_is_rejected(
    rules: BattleRules, catalog: GameCatalog
) -> None:
    hand, opponent_hand = two_hands(catalog)
    state = rules.with_hand("Square", hand, opponent_hand=opponent_hand)
    illegal = Action.place(hand[0], 0, (0, 0))

    with pytest.raises(RulesError):
        rules.apply_turn(
            state,
            illegal,
            Action.skip(opponent_hand[0]),
        )


def test_draw_from_pile_refills_both_hands(
    rules: BattleRules, catalog: GameCatalog
) -> None:
    state = rules.new_battle("Square", "MiniGame_Aori", "MiniGame_Aori", seed=1)
    legal_a = rules.legal_actions(state, Side.A)
    legal_b = rules.legal_actions(state, Side.B)
    action_a = next(
        action
        for action in legal_a
        if action.action_type is ActionType.PLACE
    )
    action_b = next(
        action
        for action in legal_b
        if action.action_type is ActionType.PLACE
    )
    old_a = state.player_a
    old_b = state.player_b

    next_state = rules.apply_turn(
        state,
        action_a,
        action_b,
        legal_a=legal_a,
        legal_b=legal_b,
    )

    assert len(next_state.player_a.hand) == 4
    assert len(next_state.player_b.hand) == 4
    assert action_a.card_id not in next_state.player_a.hand
    assert action_b.card_id not in next_state.player_b.hand
    assert old_a.draw_pile[0] in next_state.player_a.hand
    assert old_b.draw_pile[0] in next_state.player_b.hand


def test_terminal_score_uses_both_players(
    rules: BattleRules, catalog: GameCatalog
) -> None:
    hand, opponent_hand = two_hands(catalog)
    state = rules.with_hand(
        "Painted",
        hand,
        opponent_hand=opponent_hand,
        round_index=1,
    )
    legal_a = rules.legal_actions(state, Side.A)
    legal_b = rules.legal_actions(state, Side.B)
    terminal = rules.apply_turn(
        state,
        next(action for action in legal_a if action.action_type is ActionType.SKIP),
        next(action for action in legal_b if action.action_type is ActionType.SKIP),
        legal_a=legal_a,
        legal_b=legal_b,
    )

    assert terminal.terminal
    a_score, b_score = terminal.scores()
    assert a_score > 0
    assert b_score > 0
    assert terminal.winner() in {"A", "B", "draw"}


def test_twelve_turn_smoke_on_every_stage(
    rules: BattleRules, catalog: GameCatalog
) -> None:
    for stage_id in catalog.stages:
        state = rules.new_battle(
            stage_id,
            "MiniGame_Aori",
            "MiniGame_Aori",
            seed=17,
        )
        for _ in range(12):
            legal_a = rules.legal_actions(state, Side.A)
            legal_b = rules.legal_actions(state, Side.B)
            action_a = next(
                (
                    action
                    for action in legal_a
                    if action.action_type is ActionType.PLACE
                ),
                legal_a[0],
            )
            action_b = next(
                (
                    action
                    for action in legal_b
                    if action.action_type is ActionType.PLACE
                ),
                legal_b[0],
            )
            state = rules.apply_turn(
                state,
                action_a,
                action_b,
                legal_a=legal_a,
                legal_b=legal_b,
            )
        assert state.terminal


def test_overlap_rules(rules: BattleRules, catalog: GameCatalog) -> None:
    cards = sorted(catalog.cards.values(), key=lambda card: card.shape.size)
    smaller, larger = cards[0], cards[-1]
    state = rules.new_battle("Square", "MiniGame_Aori", "MiniGame_Aori", seed=3)
    small_action = Action.place(smaller.card_id, 0, (0, 0))
    large_action = Action.place(larger.card_id, 0, (0, 0))
    same_small_action = Action.place(smaller.card_id, 0, (0, 0))

    assert rules._resolve_overlap(
        state,
        {Side.A: large_action, Side.B: small_action},
        [(Side.A, CardCell.INK), (Side.B, CardCell.INK)],
    ) is Cell.OPPONENT_INK
    assert rules._resolve_overlap(
        state,
        {Side.A: small_action, Side.B: same_small_action},
        [(Side.A, CardCell.INK), (Side.B, CardCell.INK)],
    ) is Cell.WALL
    assert rules._resolve_overlap(
        state,
        {Side.A: large_action, Side.B: small_action},
        [(Side.A, CardCell.INK), (Side.B, CardCell.SPECIAL)],
    ) is Cell.OPPONENT_SPECIAL

