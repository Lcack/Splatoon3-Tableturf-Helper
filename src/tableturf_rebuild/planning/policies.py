from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tableturf_rebuild.domain.actions import Action, ActionType
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.domain.rules import BattleRules, NEIGHBORS_8
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


@dataclass(frozen=True, slots=True)
class ActivationFeatures:
    """Own-action effect before resolving the simultaneous opponent move."""

    activated: int = 0
    closure_progress: float = 0.0


class GreedyPolicy:
    """Deterministic one-ply policy used as a trustworthy baseline."""

    def __init__(
        self, name: str = "greedy", *,
        activation_weight: float = 2.0,
        closure_progress_weight: float = 0.5,
    ):
        self.name = name
        self.activation_weight = activation_weight
        self.closure_progress_weight = closure_progress_weight
        self._score_cache: dict[tuple[BattleState, Side, Action], float] = {}
        self._score_cache_rules = None
        self._special_board = None
        self._special_topology: dict[Side, dict[tuple[int, int], frozenset[tuple[int, int]]]] = {}
        self._activation_state = None
        self._activation_rules = None
        self._activation_cache: dict[tuple[Side, Action], ActivationFeatures] = {}

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
        if self._score_cache_rules is not rules:
            self._score_cache.clear()
            self._score_cache_rules = rules
        key = (state, side, action)
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
        # Newly surrounded points yield SP after this turn. Keep the board
        # score primary and use a bounded resource prior to distinguish close
        # placements. SP acquired on the final turn cannot be spent later.
        if state.turns_remaining > 1 and (self.activation_weight or self.closure_progress_weight):
            activation = self.activation_features(rules, state, side, action, targets=targets)
            value += (activation.activated * self.activation_weight
                      + activation.closure_progress * self.closure_progress_weight)
        if len(self._score_cache) >= 32768:
            self._score_cache.clear()
        self._score_cache[key] = value
        return value

    def activation_features(
        self, rules: BattleRules, state: BattleState, side: Side, action: Action,
        *, targets=None,
    ) -> ActivationFeatures:
        """Compute closure locally using the same eight neighbors as rules.

        Opponent ink, walls, voids and board edges already close a side. Only
        EMPTY neighbors remain to be filled. Old active points earn no bonus;
        a new special tile can itself become active in its placement turn.
        """
        if action.action_type is ActionType.SKIP:
            return ActivationFeatures()
        if self._activation_state is not state or self._activation_rules is not rules:
            self._activation_state = state
            self._activation_rules = rules
            self._activation_cache.clear()
        cached = self._activation_cache.get((side, action))
        if cached is not None:
            return cached
        own_special = Cell.SELF_SPECIAL if side is Side.A else Cell.OPPONENT_SPECIAL
        board = state.board
        rows, cols = state.stage.rows, state.stage.cols
        if self._special_board is not board:
            self._special_board = board
            self._special_topology.clear()
        topology = self._special_topology.get(side)
        if topology is None:
            topology = {
                (row, col): frozenset(
                    (row + dr, col + dc) for dr, dc in NEIGHBORS_8
                    if 0 <= row + dr < rows and 0 <= col + dc < cols
                    and board[row + dr][col + dc] is Cell.EMPTY
                )
                for row in range(rows) for col in range(cols)
                if board[row][col] is own_special
            }
            self._special_topology[side] = topology
        if targets is None:
            targets = rules.action_targets(state, action)
        filled = frozenset((row, col) for row, col, _ in targets
                           if board[row][col] is Cell.EMPTY)
        new_specials = {(row, col) for row, col, cell in targets if cell.value == 2}
        affected = set(new_specials)
        for row, col in filled:
            affected.update((row + dr, col + dc) for dr, dc in NEIGHBORS_8
                            if (row + dr, col + dc) in topology)
        activated = 0
        progress = 0.0
        for row, col in affected:
            old_empty = topology.get((row, col))
            if old_empty is not None:
                if not old_empty:  # Already active: there is no second SP.
                    continue
                remaining = len(old_empty - filled)
                activated += int(remaining == 0)
                progress += 1 / (remaining + 1) - 1 / (len(old_empty) + 1)
            else:
                remaining = sum(
                    board[row + dr][col + dc] is Cell.EMPTY
                    and (row + dr, col + dc) not in filled
                    for dr, dc in NEIGHBORS_8
                    if 0 <= row + dr < rows and 0 <= col + dc < cols
                )
                activated += int(remaining == 0)
                progress += 1 / (remaining + 1)
        result = ActivationFeatures(activated, progress)
        self._activation_cache[(side, action)] = result
        return result

