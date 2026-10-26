from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from tableturf_rebuild.data.catalog import GameCatalog, NpcScenario
from tableturf_rebuild.domain.actions import Action
from tableturf_rebuild.domain.rules import BattleRules
from tableturf_rebuild.domain.state import Side
from tableturf_rebuild.planning.mcts import MCTSDecisionPolicy
from tableturf_rebuild.planning.policies import Policy


@dataclass(frozen=True, slots=True)
class ActionRecord:
    turn: int
    side: str
    action: str
    card_id: str | None
    rotation: int | None
    origin: tuple[int, int] | None
    decision_seconds: float
    search_simulations: int | None
    search_value: float | None


@dataclass(frozen=True, slots=True)
class GameRecord:
    game_index: int
    seed: int
    stage_id: str
    deck_a_id: str
    deck_b_id: str
    scenario_a_npc: str
    scenario_a_level: int
    scenario_a_ai_type: str
    scenario_b_npc: str
    scenario_b_level: int
    scenario_b_ai_type: str
    policy_a: str
    policy_b: str
    winner: str
    score_a: int
    score_b: int
    actions_played: int
    elapsed_seconds: float
    action_trace: tuple[ActionRecord, ...]


@dataclass(frozen=True, slots=True)
class MatchSummary:
    games: int
    wins_a: int
    wins_b: int
    draws: int
    average_score_a: float
    average_score_b: float
    elapsed_seconds: float


class SelfPlayRunner:
    def __init__(self, catalog: GameCatalog):
        self.catalog = catalog
        self.rules = BattleRules(catalog)

    def play_game(
        self,
        game_index: int,
        seed: int,
        stage_id: str,
        deck_a_id: str,
        deck_b_id: str,
        policy_a: Policy,
        policy_b: Policy,
        scenario_a: NpcScenario | None = None,
        scenario_b: NpcScenario | None = None,
        deck_a_cards: tuple[str, ...] | None = None,
    ) -> GameRecord:
        started = time.perf_counter()
        if deck_a_cards is None:
            state = self.rules.new_battle(
                stage_id=stage_id,
                deck_a_id=deck_a_id,
                deck_b_id=deck_b_id,
                seed=seed,
            )
        else:
            state = self.rules.new_battle_from_cards(
                stage_id=stage_id,
                deck_a=deck_a_cards,
                deck_b=self.catalog.decks[deck_b_id].card_ids,
                seed=seed,
            )
        trace: list[ActionRecord] = []

        while not state.terminal:
            legal_a = self.rules.legal_actions(state, Side.A)
            legal_b = self.rules.legal_actions(state, Side.B)
            if not legal_a or not legal_b:
                raise RuntimeError("non-terminal state has no legal actions")
            turn = 12 - state.turns_remaining + 1
            action_a, record_a = self._choose(
                policy_a, state, Side.A, legal_a, turn
            )
            action_b, record_b = self._choose(
                policy_b, state, Side.B, legal_b, turn
            )
            trace.extend((record_a, record_b))
            state = self.rules.apply_turn(
                state,
                action_a,
                action_b,
                legal_a=legal_a,
                legal_b=legal_b,
            )
            if len(trace) > 24:
                raise RuntimeError("game exceeded 24 player actions")

        score_a, score_b = state.scores()
        return GameRecord(
            game_index=game_index,
            seed=seed,
            stage_id=stage_id,
            deck_a_id=deck_a_id,
            deck_b_id=deck_b_id,
            scenario_a_npc=scenario_a.npc_id if scenario_a else "",
            scenario_a_level=scenario_a.level if scenario_a else 0,
            scenario_a_ai_type=scenario_a.ai_type if scenario_a else "",
            scenario_b_npc=scenario_b.npc_id if scenario_b else "",
            scenario_b_level=scenario_b.level if scenario_b else 0,
            scenario_b_ai_type=scenario_b.ai_type if scenario_b else "",
            policy_a=policy_a.name,
            policy_b=policy_b.name,
            winner=state.winner() or "unknown",
            score_a=score_a,
            score_b=score_b,
            actions_played=len(trace),
            elapsed_seconds=time.perf_counter() - started,
            action_trace=tuple(trace),
        )

    def run_scenarios(
        self,
        games: int,
        scenarios: Iterable[NpcScenario],
        seed: int,
        policy_a: Policy,
        policy_b: Policy,
    ) -> tuple[GameRecord, ...]:
        scenario_list = tuple(scenarios)
        if not scenario_list:
            raise ValueError("at least one NPC scenario is required")

        records: list[GameRecord] = []
        for index in range(games):
            scenario_a = scenario_list[index % len(scenario_list)]
            scenario_b = scenario_list[(index + 1) % len(scenario_list)]
            records.append(
                self.play_game(
                    game_index=index,
                    seed=seed + index * 104729,
                    stage_id=scenario_a.stage_id,
                    deck_a_id=scenario_a.deck_id,
                    deck_b_id=scenario_b.deck_id,
                    policy_a=policy_a,
                    policy_b=policy_b,
                    scenario_a=scenario_a,
                    scenario_b=scenario_b,
                )
            )
        return tuple(records)

    @staticmethod
    def summarize(records: Iterable[GameRecord]) -> MatchSummary:
        record_list = tuple(records)
        if not record_list:
            raise ValueError("cannot summarize an empty set")
        return MatchSummary(
            games=len(record_list),
            wins_a=sum(record.winner == Side.A.value for record in record_list),
            wins_b=sum(record.winner == Side.B.value for record in record_list),
            draws=sum(record.winner == "draw" for record in record_list),
            average_score_a=sum(record.score_a for record in record_list)
            / len(record_list),
            average_score_b=sum(record.score_b for record in record_list)
            / len(record_list),
            elapsed_seconds=sum(record.elapsed_seconds for record in record_list),
        )

    @staticmethod
    def write_jsonl(path: Path, records: Iterable[GameRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                json.dump(asdict(record), file, ensure_ascii=False)
                file.write("\n")

    def _choose(
        self,
        policy: Policy,
        state,
        side: Side,
        actions: tuple[Action, ...],
        turn: int,
    ) -> tuple[Action, ActionRecord]:
        started = time.perf_counter()
        action = policy.choose(self.rules, state, side, actions)
        elapsed = time.perf_counter() - started
        stats = policy.last_stats if isinstance(policy, MCTSDecisionPolicy) else None
        record = ActionRecord(
            turn=turn,
            side=side.value,
            action=action.action_type.value,
            card_id=action.card_id,
            rotation=action.rotation,
            origin=action.origin,
            decision_seconds=elapsed,
            search_simulations=stats.simulations if stats else None,
            search_value=stats.best_value if stats else None,
        )
        return action, record

