from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tableturf_rebuild.application.batch import append_json_record, iter_json_records
from tableturf_rebuild.application.selfplay import GameRecord, SelfPlayRunner
from tableturf_rebuild.data.catalog import GameCatalog, NpcScenario
from tableturf_rebuild.planning.mcts import MCTSConfig, MCTSDecisionPolicy
from tableturf_rebuild.planning.policies import GreedyPolicy


@dataclass(frozen=True, slots=True)
class DeckCandidate:
    candidate_id: str
    card_ids: tuple[str, ...]


@dataclass(slots=True)
class DeckStats:
    candidate_id: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    score_for: int = 0
    score_against: int = 0
    elapsed_seconds: float = 0.0

    def add(self, record: GameRecord) -> None:
        self.games += 1
        self.wins += record.winner == "A"
        self.losses += record.winner == "B"
        self.draws += record.winner == "draw"
        self.score_for += record.score_a
        self.score_against += record.score_b
        self.elapsed_seconds += record.elapsed_seconds

    def as_dict(self, card_ids: tuple[str, ...]) -> dict[str, Any]:
        games = max(1, self.games)
        return {
            "candidate_id": self.candidate_id,
            "card_ids": list(card_ids),
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / games,
            "average_score_for": self.score_for / games,
            "average_score_against": self.score_against / games,
            "average_score_margin": (self.score_for - self.score_against) / games,
            "elapsed_seconds": self.elapsed_seconds,
        }


def generate_candidate_decks(
    catalog: GameCatalog,
    count: int,
    seed: int,
) -> tuple[DeckCandidate, ...]:
    judge = catalog.cards["MiniGame_KoJudgekun"]
    del judge
    pools = {
        "large": [
            card.card_id for card in catalog.cards.values() if card.area >= 15
        ],
        "medium": [
            card.card_id
            for card in catalog.cards.values()
            if card.area in {12, 13} and card.sp_cost != 3
        ],
        "special": [
            card.card_id
            for card in catalog.cards.values()
            if card.area == 12 and card.sp_cost == 3
        ],
        "eight": [
            card.card_id for card in catalog.cards.values() if card.area == 8
        ],
        "small": [
            card.card_id
            for card in catalog.cards.values()
            if card.area in {4, 5}
        ],
        "size3": [
            card.card_id for card in catalog.cards.values() if card.area == 3
        ],
    }
    judge_id = "MiniGame_KoJudgekun"
    eight_others = [card_id for card_id in pools["eight"] if card_id != judge_id]
    rng = random.Random(seed)
    decks: list[DeckCandidate] = []
    seen: set[tuple[str, ...]] = set()

    while len(decks) < count:
        cards = []
        cards.extend(rng.sample(pools["large"], 2))
        cards.extend(rng.sample(pools["medium"], 4))
        cards.extend(rng.sample(pools["special"], 1))
        cards.append(judge_id)
        cards.extend(rng.sample(eight_others, 2))
        cards.extend(rng.sample(pools["small"], 4))
        cards.extend(rng.sample(pools["size3"], 1))
        key = tuple(sorted(cards))
        if len(set(key)) != 15 or key in seen:
            continue
        seen.add(key)
        decks.append(
            DeckCandidate(
                candidate_id=f"deck-{len(decks) + 1:04d}",
                card_ids=key,
            )
        )
    return tuple(decks)


def select_level3_scenarios(
    catalog: GameCatalog, games_per_deck: int
) -> tuple[NpcScenario, ...]:
    scenarios = [
        scenario for scenario in catalog.npc_scenarios() if scenario.level == 3
    ]
    by_type: dict[str, list[NpcScenario]] = {}
    for scenario in scenarios:
        by_type.setdefault(scenario.ai_type, []).append(scenario)
    for group in by_type.values():
        group.sort(key=lambda scenario: scenario.npc_id)

    selected: list[NpcScenario] = []
    for ai_type, target in (
        ("AccumulateSpecial", 2),
        ("Aggressive", 6),
        ("Balance", games_per_deck - 8),
    ):
        selected.extend(by_type.get(ai_type, [])[: max(0, target)])
    for scenario in scenarios:
        if len(selected) >= games_per_deck:
            break
        if scenario not in selected:
            selected.append(scenario)
    if len(selected) < games_per_deck:
        while len(selected) < games_per_deck:
            selected.extend(scenarios[: games_per_deck - len(selected)])
    return tuple(selected[:games_per_deck])


@dataclass(frozen=True, slots=True)
class DeckSearchConfig:
    candidate_count: int
    games_per_deck: int
    seed: int
    simulations: int
    time_limit_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DeckSearchResult:
    games_run: int
    decks_completed: int
    elapsed_seconds: float
    stopped_reason: str
    leaderboard: list[dict[str, Any]]


class DeckSearchRunner:
    def __init__(self, catalog: GameCatalog):
        self.catalog = catalog
        self.selfplay = SelfPlayRunner(catalog)

    def run(
        self,
        output_dir: Path,
        config: DeckSearchConfig,
        scenarios: tuple[NpcScenario, ...],
        resume: bool = False,
        max_seconds: float | None = None,
    ) -> DeckSearchResult:
        if len(scenarios) != config.games_per_deck:
            raise ValueError("scenario count must equal games_per_deck")
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / "meta.json"
        decks_path = output_dir / "decks.json"
        games_path = output_dir / "games.jsonl"
        leaderboard_path = output_dir / "leaderboard.json"
        progress_path = output_dir / "progress.json"
        expected_meta = {
            "format_version": 1,
            **config.as_dict(),
            "scenarios": [asdict(scenario) for scenario in scenarios],
        }

        decks = generate_candidate_decks(
            self.catalog,
            config.candidate_count,
            config.seed,
        )
        if resume:
            if not meta_path.exists() or not decks_path.exists():
                raise ValueError("resume requires existing meta.json and decks.json")
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = existing_meta.pop("created_at", None)
            if existing_meta != expected_meta:
                raise ValueError("resume configuration does not match")
            stored_decks = json.loads(decks_path.read_text(encoding="utf-8"))
            expected_decks = [
                {
                    "candidate_id": deck.candidate_id,
                    "card_ids": list(deck.card_ids),
                }
                for deck in decks
            ]
            if stored_decks != expected_decks:
                raise ValueError("generated deck list does not match saved run")
        else:
            if games_path.exists() and games_path.stat().st_size:
                raise ValueError("games.jsonl already contains data")
            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
            self._write_json(meta_path, {**expected_meta, "created_at": created_at})
            self._write_json(decks_path, [asdict(deck) for deck in decks])

        deck_by_id = {deck.candidate_id: deck for deck in decks}
        stats = {deck.candidate_id: DeckStats(deck.candidate_id) for deck in decks}
        completed: set[tuple[str, int]] = set()
        if games_path.exists():
            for value in iter_json_records(games_path):
                record = GameRecord(**value)
                key = (record.deck_a_id, record.game_index)
                if key in completed:
                    continue
                completed.add(key)
                stats[record.deck_a_id].add(record)

        started = time.monotonic()
        games_run = 0
        stopped_reason = "decks"
        try:
            for deck_index, deck in enumerate(decks):
                policy_a = MCTSDecisionPolicy(
                    config=MCTSConfig(
                        simulations=config.simulations,
                        time_limit_seconds=config.time_limit_seconds,
                    ),
                    seed=config.seed ^ (deck_index * 0x9E3779B1),
                    name="candidate-mcts",
                )
                policy_b = GreedyPolicy(name="npc-greedy-proxy")
                deck_done = True
                for game_index, scenario in enumerate(scenarios):
                    if (deck.candidate_id, game_index) in completed:
                        continue
                    if (
                        max_seconds is not None
                        and time.monotonic() - started >= max_seconds
                    ):
                        stopped_reason = "time_limit"
                        deck_done = False
                        break
                    record = self.selfplay.play_game(
                        game_index=game_index,
                        seed=config.seed + game_index * 104729,
                        stage_id=scenario.stage_id,
                        deck_a_id=deck.candidate_id,
                        deck_b_id=scenario.deck_id,
                        policy_a=policy_a,
                        policy_b=policy_b,
                        scenario_b=scenario,
                        deck_a_cards=deck.card_ids,
                    )
                    append_json_record(games_path, record)
                    stats[deck.candidate_id].add(record)
                    completed.add((deck.candidate_id, game_index))
                    games_run += 1
                    self._write_progress(
                        progress_path,
                        config,
                        stats,
                        deck_by_id,
                        started_at=created_at,
                        invocation_started=started,
                        current_deck=deck.candidate_id,
                        current_scenario=scenario.npc_id,
                        current_game=game_index + 1,
                        status="running",
                    )
                if not deck_done:
                    self._write_progress(
                        progress_path,
                        config,
                        stats,
                        deck_by_id,
                        started_at=created_at,
                        invocation_started=started,
                        current_deck=deck.candidate_id,
                        current_scenario=scenario.npc_id,
                        current_game=game_index + 1,
                        status="time_limit",
                    )
                    break
                if stats[deck.candidate_id].games == config.games_per_deck:
                    self._write_leaderboard(
                        leaderboard_path,
                        stats,
                        deck_by_id,
                        config.games_per_deck,
                    )
        except KeyboardInterrupt:
            stopped_reason = "interrupted"
        leaderboard = self._write_leaderboard(
            leaderboard_path,
            stats,
            deck_by_id,
            config.games_per_deck,
        )
        self._write_progress(
            progress_path,
            config,
            stats,
            deck_by_id,
            started_at=created_at,
            invocation_started=started,
            current_deck="",
            current_scenario="",
            current_game=0,
            status="complete" if stopped_reason == "decks" else stopped_reason,
        )
        decks_completed = sum(
            value.games == config.games_per_deck for value in stats.values()
        )
        return DeckSearchResult(
            games_run=games_run,
            decks_completed=decks_completed,
            elapsed_seconds=time.monotonic() - started,
            stopped_reason=stopped_reason,
            leaderboard=leaderboard,
        )

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_name(f"{path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)

    def _write_leaderboard(
        self,
        path: Path,
        stats: dict[str, DeckStats],
        deck_by_id: dict[str, DeckCandidate],
        expected_games: int,
    ) -> list[dict[str, Any]]:
        values = [
            stats[deck_id].as_dict(deck_by_id[deck_id].card_ids)
            for deck_id, value in stats.items()
            if value.games > 0
        ]
        values.sort(
            key=lambda item: (
                item["win_rate"],
                item["average_score_margin"],
            ),
            reverse=True,
        )
        payload = {
            "expected_games_per_deck": expected_games,
            "decks_with_results": len(values),
            "decks_completed": sum(
                item["games"] == expected_games for item in values
            ),
            "ranking": values,
        }
        self._write_json(path, payload)
        return values

    def _write_progress(
        self,
        path: Path,
        config: DeckSearchConfig,
        stats: dict[str, DeckStats],
        deck_by_id: dict[str, DeckCandidate],
        started_at: str | None,
        invocation_started: float,
        current_deck: str,
        current_scenario: str,
        current_game: int,
        status: str,
    ) -> None:
        completed_games = sum(value.games for value in stats.values())
        completed_decks = sum(
            value.games == config.games_per_deck for value in stats.values()
        )
        ranking = [
            stats[deck_id].as_dict(deck_by_id[deck_id].card_ids)
            for deck_id, value in stats.items()
            if value.games > 0
        ]
        ranking.sort(
            key=lambda item: (
                item["win_rate"],
                item["average_score_margin"],
            ),
            reverse=True,
        )
        self._write_json(
            path,
            {
                "status": status,
                "created_at": started_at,
                "invocation_elapsed_seconds": time.monotonic() - invocation_started,
                "updated_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "decks_total": config.candidate_count,
                "decks_completed": completed_decks,
                "games_total": config.candidate_count * config.games_per_deck,
                "games_completed": completed_games,
                "current_deck": current_deck,
                "current_scenario": current_scenario,
                "current_game": current_game,
                "leaderboard": ranking[:20],
            },
        )

