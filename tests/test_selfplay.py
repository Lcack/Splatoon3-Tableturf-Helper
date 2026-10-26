from tableturf_rebuild.application.selfplay import SelfPlayRunner
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.planning.policies import GreedyPolicy, RandomPolicy


def test_selfplay_game_is_complete_and_reproducible() -> None:
    catalog = GameCatalog.from_package()
    runner = SelfPlayRunner(catalog)
    first = runner.play_game(
        game_index=0,
        seed=123,
        stage_id="Square",
        deck_a_id="MiniGame_Aori",
        deck_b_id="MiniGame_Aori",
        policy_a=GreedyPolicy(),
        policy_b=GreedyPolicy(),
    )
    second = runner.play_game(
        game_index=0,
        seed=123,
        stage_id="Square",
        deck_a_id="MiniGame_Aori",
        deck_b_id="MiniGame_Aori",
        policy_a=GreedyPolicy(),
        policy_b=GreedyPolicy(),
    )

    assert first.actions_played == 24
    assert tuple(
        (
            item.turn,
            item.side,
            item.action,
            item.card_id,
            item.rotation,
            item.origin,
        )
        for item in first.action_trace
    ) == tuple(
        (
            item.turn,
            item.side,
            item.action,
            item.card_id,
            item.rotation,
            item.origin,
        )
        for item in second.action_trace
    )
    assert first.winner in {"A", "B", "draw"}
    assert first.score_a >= 0
    assert first.score_b >= 0


def test_npc_scenario_batch() -> None:
    catalog = GameCatalog.from_package()
    runner = SelfPlayRunner(catalog)
    records = runner.run_scenarios(
        games=4,
        scenarios=catalog.npc_scenarios(),
        seed=99,
        policy_a=GreedyPolicy(),
        policy_b=RandomPolicy(seed=7),
    )
    summary = runner.summarize(records)

    assert summary.games == 4
    assert summary.wins_a + summary.wins_b + summary.draws == 4
    assert all(record.actions_played == 24 for record in records)
