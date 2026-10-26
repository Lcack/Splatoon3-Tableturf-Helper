from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.rules import BattleRules
from tableturf_rebuild.domain.state import Side
from tableturf_rebuild.planning.mcts import MCTSConfig, MCTSDecisionPolicy
from tableturf_rebuild.planning.context import OpponentContext
from tableturf_rebuild.planning.policies import GreedyPolicy


def test_mcts_returns_legal_action_and_is_deterministic() -> None:
    catalog = GameCatalog.from_package()
    rules = BattleRules(catalog)
    state = rules.new_battle(
        "Square",
        "MiniGame_Aori",
        "MiniGame_Aori",
        seed=21,
    )
    actions = rules.legal_actions(state, Side.A)
    config = MCTSConfig(simulations=8, time_limit_seconds=2.0)
    first = MCTSDecisionPolicy(config=config, seed=123)
    second = MCTSDecisionPolicy(config=config, seed=123)

    action_first = first.choose(rules, state, Side.A, actions)
    action_second = second.choose(rules, state, Side.A, actions)

    assert action_first == action_second
    assert action_first in actions
    assert first.last_stats is not None
    assert first.last_stats.simulations > 0
    assert first.last_stats.elapsed_seconds <= 2.0


def test_mcts_retains_opponent_context() -> None:
    catalog = GameCatalog.from_package()
    rules = BattleRules(catalog)
    state = rules.new_battle(
        "Rectangle",
        "MiniGame_WeaponShop",
        "MiniGame_Aori",
        seed=9,
    )
    context = OpponentContext(
        npc_id="MiniGame_WeaponShop",
        npc_name="武器子",
        difficulty=2,
        ai_type="Balance",
        deck_id="MiniGamePresetDeck_00",
        stage_id="Rectangle",
        stage_name="正直大道",
        confidence=0.9,
    )
    mcts = MCTSDecisionPolicy(
        config=MCTSConfig(simulations=2, time_limit_seconds=1.0),
        seed=4,
        opponent_context=context,
    )

    mcts.choose(rules, state, Side.A, rules.legal_actions(state, Side.A))

    assert mcts.last_stats is not None
    assert mcts.last_stats.opponent_context == context


def test_mcts_can_plan_a_complete_turn_against_greedy() -> None:
    catalog = GameCatalog.from_package()
    rules = BattleRules(catalog)
    state = rules.new_battle(
        "SmallSquare",
        "MiniGame_Aori",
        "MiniGame_Aori",
        seed=8,
    )
    mcts = MCTSDecisionPolicy(
        config=MCTSConfig(simulations=4, time_limit_seconds=2.0),
        seed=7,
    )
    opponent = GreedyPolicy()
    legal_a = rules.legal_actions(state, Side.A)
    legal_b = rules.legal_actions(state, Side.B)

    action_a = mcts.choose(rules, state, Side.A, legal_a)
    action_b = opponent.choose(rules, state, Side.B, legal_b)
    next_state = rules.apply_turn(
        state,
        action_a,
        action_b,
        legal_a=legal_a,
        legal_b=legal_b,
    )

    assert next_state.turns_remaining == 11
