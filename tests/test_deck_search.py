import json

from tableturf_rebuild.application.deck_search import (
    DeckSearchConfig,
    DeckSearchRunner,
    generate_candidate_decks,
)
from tableturf_rebuild.data.catalog import GameCatalog


def test_generated_decks_are_unique_and_follow_template() -> None:
    catalog = GameCatalog.from_package()
    decks = generate_candidate_decks(catalog, count=40, seed=123)

    assert len(decks) == 40
    assert len({deck.card_ids for deck in decks}) == 40
    for deck in decks:
        assert len(deck.card_ids) == 15
        assert len(set(deck.card_ids)) == 15
        cards = [catalog.cards[card_id] for card_id in deck.card_ids]
        assert sum(card.area >= 15 for card in cards) == 2
        assert sum(
            card.area in {12, 13} and card.sp_cost != 3 for card in cards
        ) == 4
        assert sum(card.area == 12 and card.sp_cost == 3 for card in cards) == 1
        assert sum(card.area == 8 for card in cards) == 3
        assert "MiniGame_KoJudgekun" in deck.card_ids
        assert sum(card.area in {4, 5} for card in cards) == 4
        assert sum(card.area == 3 for card in cards) == 1


def test_deck_search_streams_and_resumes(tmp_path) -> None:
    catalog = GameCatalog.from_package()
    scenarios = tuple(
        scenario
        for scenario in catalog.npc_scenarios()
        if scenario.level == 3
    )[:2]
    config = DeckSearchConfig(
        candidate_count=1,
        games_per_deck=2,
        seed=99,
        simulations=1,
        time_limit_seconds=0.2,
    )
    runner = DeckSearchRunner(catalog)
    first = runner.run(tmp_path / "search", config, scenarios)

    assert first.games_run == 2
    assert first.decks_completed == 1
    assert (tmp_path / "search" / "games.jsonl").exists()
    leaderboard = json.loads(
        (tmp_path / "search" / "leaderboard.json").read_text(encoding="utf-8")
    )
    assert leaderboard["decks_completed"] == 1
    progress = json.loads(
        (tmp_path / "search" / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "complete"
    assert progress["games_completed"] == 2

    resumed = runner.run(
        tmp_path / "search",
        config,
        scenarios,
        resume=True,
    )
    assert resumed.games_run == 0
    assert resumed.decks_completed == 1
