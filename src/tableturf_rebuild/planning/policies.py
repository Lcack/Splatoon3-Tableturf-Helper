from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol

from tableturf_rebuild.domain.actions import Action, ActionType
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.domain.rules import BattleRules
from tableturf_rebuild.domain.state import BattleState, Side


class Policy(Protocol):
    name: str

    def choose(
        self,
        rules: BattleRules,
        state: BattleState,
        side: Side,
        actions: Sequence[Action],
    ) -> Action:
        ...


class RandomPolicy:
    def __init__(self, seed: int = 0, name: str = "random"):
        self.name = name
        self._rng = random.Random(seed)

    def choose(
        self,
        rules: BattleRules,
        state: BattleState,
        side: Side,
        actions: Sequence[Action],
    ) -> Action:
        del rules, state, side
        return self._rng.choice(actions)


class FastRandomPolicy:
    """Legal rollout policy that avoids enumerating every board position."""

    def __init__(self, seed: int = 0, name: str = "fast-random"):
        self.name = name
        self._rng = random.Random(seed)

    def choose(
        self,
        rules: BattleRules,
        state: BattleState,
        side: Side,
        actions: Sequence[Action],
    ) -> Action:
        del actions
        return self.sample(rules, state, side)

    def sample(
        self,
        rules: BattleRules,
        state: BattleState,
        side: Side,
    ) -> Action:
        return rules.sample_action(state, side, self._rng)


class GreedyPolicy:
    """Deterministic one-ply policy used as a trustworthy baseline."""

    def __init__(self, name: str = "greedy"):
        self.name = name
        self._score_cache: dict[tuple[BattleState, Action], float] = {}

    def choose(
        self,
        rules: BattleRules,
        state: BattleState,
        side: Side,
        actions: Sequence[Action],
    ) -> Action:
        return self.rank_actions(rules, state, side, actions)[0]

    def rank_actions(
        self,
        rules: BattleRules,
        state: BattleState,
        side: Side,
        actions: Sequence[Action],
    ) -> tuple[Action, ...]:
        return tuple(
            sorted(
                actions,
                key=lambda action: self._score_only(rules, state, side, action),
                reverse=True,
            )
        )

    def _score_only(
        self,
        rules: BattleRules,
        state: BattleState,
        side: Side,
        action: Action,
    ) -> float:
        own_special = (
            Cell.SELF_SPECIAL if side is Side.A else Cell.OPPONENT_SPECIAL
        )
        opponent_ink = {Cell.OPPONENT_INK, Cell.OPPONENT_SPECIAL}
        if side is Side.B:
            opponent_ink = {Cell.SELF_INK, Cell.SELF_SPECIAL}
        key = (state, action)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        if action.action_type is ActionType.SKIP:
            return 0.25
        targets = rules.action_targets(state, action)
        assert action.card_id is not None
        card = rules.catalog.cards[action.card_id]
        board = state.board
        painted = sum(board[row][col] == Cell.EMPTY for row, col, _ in targets)
        overwritten = sum(board[row][col] in opponent_ink for row, col, _ in targets)
        specials = sum(card_cell.value == 2 for _, _, card_cell in targets)
        value = float(painted + overwritten * 1.35 + specials * 0.6)
        if action.action_type is ActionType.SPECIAL:
            value -= card.sp_cost * 0.45
            value += overwritten * 0.5
        value += sum(
            0.08
            for row, col, _ in targets
            if any(
                0 <= row + dr < state.stage.rows
                and 0 <= col + dc < state.stage.cols
                and board[row + dr][col + dc] == own_special
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if dr != 0 or dc != 0
            )
        )
        if len(self._score_cache) >= 32768:
            self._score_cache.clear()
        self._score_cache[key] = value
        return value

