from dataclasses import replace

import pytest

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action, ActionType
from tableturf_rebuild.domain.card import CardShape, CardSpec
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.domain.rules import BattleRules
from tableturf_rebuild.domain.stage import StageSpec
from tableturf_rebuild.domain.state import Side
from tableturf_rebuild.planning.mcts import MCTSConfig, MCTSDecisionPolicy
from tableturf_rebuild.planning.policies import GreedyPolicy


@pytest.fixture(scope="module")
def catalog():
    catalog = GameCatalog.from_package()
    probes = [
        CardSpec("probe_ink", "ink", "墨", 1, CardShape.from_matrix([[1]]), 1),
        CardSpec("probe_special", "special", "特殊", 1, CardShape.from_matrix([[2]]), 1),
        CardSpec("probe_bar", "bar", "长条", 2, CardShape.from_matrix([[1, 1, 1, 1]]), 4),
        CardSpec("probe_long", "long", "更长条", 2, CardShape.from_matrix([[1] * 6]), 6),
    ]
    return replace(catalog, cards={**catalog.cards, **{card.card_id: card for card in probes}})


def setup_state(catalog, grid, *, turns=8, sp=0):
    rules = BattleRules(catalog)
    cells = tuple(map(tuple, grid))
    stage = StageSpec("probe", "测试", cells, (), len(grid) * len(grid[0]))
    state = rules.with_hand("Square", ("probe_ink", "probe_special", "probe_bar", "probe_long"),
                            opponent_hand=("probe_ink",), self_sp=sp)
    return rules, replace(state, stage=stage, board=cells, turns_remaining=turns)


def almost_closed_grid(*, missing=((3, 4),)):
    grid = [[Cell.EMPTY] * 7 for _ in range(7)]
    for row in range(2, 5):
        for col in range(2, 5):
            grid[row][col] = Cell.SELF_INK
    grid[3][3] = Cell.SELF_SPECIAL
    for row, col in missing:
        grid[row][col] = Cell.EMPTY
    grid[5][2] = Cell.SELF_INK
    return grid


def test_closing_last_neighbor_matches_sp_gain_in_rules(catalog):
    rules, state = setup_state(catalog, almost_closed_grid())
    closing = Action.place("probe_ink", 0, (3, 4))
    remote = Action.place("probe_ink", 0, (5, 3))
    legal = rules.legal_actions(state, Side.A)
    assert closing in legal and remote in legal
    prior = GreedyPolicy()
    assert prior.activation_features(rules, state, Side.A, closing).activated == 1
    assert prior.activation_features(rules, state, Side.A, remote).activated == 0
    assert prior.choose(rules, state, Side.A, (remote, closing)) == closing
    after = rules.apply_turn(state, closing, Action.skip("probe_ink"))
    assert after.player_a.sp == state.player_a.sp + 1


def test_partial_closure_has_small_progress_without_inventing_sp(catalog):
    rules, state = setup_state(catalog, almost_closed_grid(missing=((3, 4), (4, 4))))
    action = Action.place("probe_ink", 0, (3, 4))
    features = GreedyPolicy().activation_features(rules, state, Side.A, action)
    assert features.activated == 0
    assert 0 < features.closure_progress < 0.5
    after = rules.apply_turn(state, action, Action.skip("probe_ink"))
    assert after.player_a.sp == state.player_a.sp


def test_new_special_can_activate_using_edges_wall_and_enemy_ink(catalog):
    grid = [[Cell.EMPTY] * 7 for _ in range(7)]
    grid[0][1] = Cell.SELF_INK
    grid[1][0] = Cell.WALL
    grid[1][1] = Cell.OPPONENT_INK
    rules, state = setup_state(catalog, grid)
    action = Action.place("probe_special", 0, (0, 0))
    assert action in rules.legal_actions(state, Side.A)
    assert GreedyPolicy().activation_features(rules, state, Side.A, action).activated == 1
    after = rules.apply_turn(state, action, Action.skip("probe_ink"))
    assert after.player_a.sp == 1


def test_active_point_is_not_rewarded_twice_and_cache_keeps_side_separate(catalog):
    grid = almost_closed_grid()
    grid[3][5] = Cell.OPPONENT_INK
    rules, state = setup_state(catalog, grid)
    action = Action.place("probe_ink", 0, (3, 4))
    prior = GreedyPolicy()
    assert action in rules.legal_actions(state, Side.A)
    assert action in rules.legal_actions(state, Side.B)
    assert prior._score_only(rules, state, Side.A, action) > prior._score_only(rules, state, Side.B, action)
    after = rules.apply_turn(state, action, Action.skip("probe_ink"))
    remote = Action.place("probe_ink", 0, (5, 3))
    assert prior.activation_features(rules, after, Side.A, remote).activated == 0


def test_final_turn_has_no_unspendable_sp_bonus(catalog):
    rules, state = setup_state(catalog, almost_closed_grid(), turns=1)
    action = Action.place("probe_ink", 0, (3, 4))
    old = GreedyPolicy(activation_weight=0, closure_progress_weight=0)
    new = GreedyPolicy()
    assert new.activation_features(rules, state, Side.A, action).activated == 1
    assert old._score_only(rules, state, Side.A, action) == new._score_only(rules, state, Side.A, action)


def test_activation_prior_does_not_pay_three_cells_for_one_sp(catalog):
    rules, state = setup_state(catalog, almost_closed_grid())
    closing = Action.place("probe_ink", 0, (3, 4))
    expansion = Action.place("probe_bar", 0, (6, 0))
    assert closing in rules.legal_actions(state, Side.A)
    assert expansion in rules.legal_actions(state, Side.A)
    assert GreedyPolicy().choose(rules, state, Side.A, (closing, expansion)) == expansion


def test_pruned_mcts_candidates_keep_comparable_closure_move(catalog):
    rules, state = setup_state(catalog, almost_closed_grid())
    closing = Action.place("probe_ink", 0, (3, 4))
    bars = tuple(action for action in rules.legal_actions(state, Side.A)
                 if action.action_type is ActionType.PLACE and action.card_id == "probe_bar")
    assert len(bars) >= 4
    actions = bars + (closing,)
    baseline = MCTSDecisionPolicy(config=MCTSConfig(preserve_strategic_candidates=False))
    improved = MCTSDecisionPolicy()
    assert closing not in baseline._candidate_actions(rules, state, Side.A, actions, 4)
    retained = improved._candidate_actions(rules, state, Side.A, actions, 4)
    assert closing in retained
    assert len(set(retained)) == 4


def test_pruned_mcts_candidates_compare_enemy_overwriting_special(catalog):
    grid = almost_closed_grid(missing=())
    grid[3][4] = Cell.OPPONENT_INK
    rules, state = setup_state(catalog, grid, sp=1)
    special = Action.special("probe_ink", 0, (3, 4))
    legal = rules.legal_actions(state, Side.A)
    assert special in legal
    bars = tuple(action for action in legal
                 if action.action_type is ActionType.PLACE and action.card_id == "probe_bar")
    retained = MCTSDecisionPolicy()._candidate_actions(rules, state, Side.A, bars + (special,), 4)
    assert special in retained


def test_score_cache_does_not_reuse_old_catalog_geometry(catalog):
    rules, state = setup_state(catalog, almost_closed_grid())
    action = Action.place("probe_ink", 0, (3, 4))
    prior = GreedyPolicy()
    first = prior._score_only(rules, state, Side.A, action)
    larger = replace(catalog.cards["probe_ink"], shape=CardShape.from_matrix([[1, 1]]), area=2)
    changed = BattleRules(replace(catalog, cards={**catalog.cards, larger.card_id: larger}))
    second = prior._score_only(changed, state, Side.A, action)
    assert second > first
