from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Sequence

from tableturf_rebuild.domain.actions import Action, ActionType
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.domain.rules import BattleRules
from tableturf_rebuild.domain.state import BattleState, Side
from tableturf_rebuild.planning.context import OpponentContext
from tableturf_rebuild.planning.policies import FastRandomPolicy, GreedyPolicy, Policy


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 128
    time_limit_seconds: float = 1.0
    exploration_weight: float = math.sqrt(2.0)
    root_action_limit: int = 24
    tree_action_limit: int = 12
    preserve_strategic_candidates: bool = True


@dataclass(frozen=True, slots=True)
class SearchStats:
    simulations: int
    elapsed_seconds: float
    root_visits: int
    best_value: float
    best_action: Action
    opponent_context: OpponentContext | None = None
    prior_seconds: float = 0.0


@dataclass(slots=True)
class MCTSNode:
    state: BattleState
    parent: "MCTSNode | None" = None
    action_from_parent: Action | None = None
    visits: int = 0
    value: float = 0.0
    untried_actions: list[Action] = field(default_factory=list)
    children: dict[Action, "MCTSNode"] = field(default_factory=dict)


class MCTSDecisionPolicy:
    """Single-agent MCTS against a deterministic opponent model.

    Tableturf turns are simultaneous. The tree searches the acting player's
    actions, while the opponent chooses through the supplied policy on the same
    pre-turn state. Rollouts continue both sides to the actual terminal state.
    """

    def __init__(
        self,
        config: MCTSConfig | None = None,
        seed: int = 0,
        opponent_model: Policy | None = None,
        rollout_policy: Policy | None = None,
        prior_policy: GreedyPolicy | None = None,
        opponent_context: OpponentContext | None = None,
        name: str = "mcts",
    ):
        self.name = name
        self.config = config or MCTSConfig()
        self._rng = random.Random(seed)
        self.opponent_model = opponent_model or GreedyPolicy(name="mcts-opponent")
        self.rollout_policy = rollout_policy or FastRandomPolicy(
            seed=seed ^ 0xC0FFEE,
            name="mcts-rollout",
        )
        self.prior_policy = prior_policy or GreedyPolicy(name="mcts-prior")
        self.opponent_context = opponent_context
        self.last_stats: SearchStats | None = None

    def set_opponent_context(self, context: OpponentContext | None) -> None:
        self.opponent_context = context

    def choose(
        self,
        rules: BattleRules,
        state: BattleState,
        side: Side,
        actions: Sequence[Action],
    ) -> Action:
        if not actions:
            raise ValueError("MCTS requires at least one legal action")
        started = time.perf_counter()
        effective_root_limit = min(
            self.config.root_action_limit,
            max(4, self.config.simulations // 4),
        )
        candidates = self._candidate_actions(
            rules,
            state,
            side,
            actions,
            effective_root_limit,
        )
        prior_seconds = time.perf_counter() - started
        if len(candidates) == 1:
            action = candidates[0]
            self.last_stats = SearchStats(
                0,
                prior_seconds,
                0,
                0.0,
                action,
                self.opponent_context,
                prior_seconds,
            )
            return action

        root = MCTSNode(
            state=state,
            untried_actions=list(candidates),
        )
        completed = 0
        for _ in range(self.config.simulations):
            if completed and (
                time.perf_counter() - started >= self.config.time_limit_seconds
            ):
                break

            path = [root]
            node = root
            while not node.state.terminal:
                if node.untried_actions:
                    action = node.untried_actions.pop(
                        self._rng.randrange(len(node.untried_actions))
                    )
                    node = self._expand(rules, node, side, action)
                    path.append(node)
                    break
                node = self._select_child(node)
                path.append(node)

            reward = self._rollout(rules, node.state, side)
            for visited in reversed(path):
                visited.visits += 1
                visited.value += reward
            completed += 1

        if not root.children:
            action = candidates[0]
            self.last_stats = SearchStats(
                completed,
                time.perf_counter() - started,
                root.visits,
                0.0,
                action,
                self.opponent_context,
                prior_seconds,
            )
            return action

        best_action, best_child = max(
            root.children.items(),
            key=lambda item: (
                item[1].visits,
                item[1].value / max(1, item[1].visits),
            ),
        )
        best_value = best_child.value / max(1, best_child.visits)
        self.last_stats = SearchStats(
            simulations=completed,
            elapsed_seconds=time.perf_counter() - started,
            root_visits=root.visits,
            best_value=best_value,
            best_action=best_action,
            opponent_context=self.opponent_context,
            prior_seconds=prior_seconds,
        )
        return best_action

    def _expand(
        self,
        rules: BattleRules,
        node: MCTSNode,
        side: Side,
        action: Action,
    ) -> MCTSNode:
        legal_a = rules.legal_actions(node.state, Side.A)
        legal_b = rules.legal_actions(node.state, Side.B)
        legal_opponent = legal_b if side is Side.A else legal_a
        if not legal_opponent:
            raise RuntimeError("opponent has no legal action in non-terminal state")
        opponent_action = self.opponent_model.choose(
            rules,
            node.state,
            side.other,
            legal_opponent,
        )
        action_a, action_b = (
            (action, opponent_action)
            if side is Side.A
            else (opponent_action, action)
        )
        child_state = rules.apply_turn(
            node.state,
            action_a,
            action_b,
            legal_a=legal_a,
            legal_b=legal_b,
        )
        child = MCTSNode(
            state=child_state,
            parent=node,
            action_from_parent=action,
            untried_actions=list(
                self._candidate_actions(
                    rules,
                    child_state,
                    side,
                    rules.legal_actions(child_state, side),
                    min(
                        self.config.tree_action_limit,
                        max(2, self.config.simulations // 16),
                    ),
                )
            )
            if not child_state.terminal
            else [],
        )
        node.children[action] = child
        return child

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        log_parent = math.log(max(1, node.visits))

        def uct(child: MCTSNode) -> float:
            exploitation = child.value / max(1, child.visits)
            exploration = self.config.exploration_weight * math.sqrt(
                log_parent / max(1, child.visits)
            )
            return exploitation + exploration

        return max(node.children.values(), key=uct)

    def _rollout(self, rules: BattleRules, state: BattleState, side: Side) -> float:
        current = state
        while not current.terminal:
            fast_rollout = isinstance(self.rollout_policy, FastRandomPolicy)
            if fast_rollout:
                action_a = self.rollout_policy.sample(rules, current, Side.A)
                action_b = self.rollout_policy.sample(rules, current, Side.B)
                legal_a = None
                legal_b = None
            else:
                legal_a = rules.legal_actions(current, Side.A)
                legal_b = rules.legal_actions(current, Side.B)
                action_a = self.rollout_policy.choose(
                    rules, current, Side.A, legal_a
                )
                action_b = self.rollout_policy.choose(
                    rules, current, Side.B, legal_b
                )
            current = rules.apply_turn(
                current,
                action_a,
                action_b,
                legal_a=legal_a,
                legal_b=legal_b,
                validate=not fast_rollout,
            )

        score_a, score_b = current.scores()
        if score_a == score_b:
            return 0.0
        winner = Side.A if score_a > score_b else Side.B
        return 1.0 if winner is side else -1.0

    def _candidate_actions(
        self,
        rules: BattleRules,
        state: BattleState,
        side: Side,
        actions: Sequence[Action],
        limit: int,
    ) -> tuple[Action, ...]:
        if limit <= 0 or len(actions) <= limit:
            return tuple(actions)
        ranked = self.prior_policy.rank_actions(rules, state, side, actions)
        if not self.config.preserve_strategic_candidates:
            return ranked[:limit]
        retained = list(ranked[:limit])
        # Small live searches keep only four roots. Reserve room to compare a
        # real SP-closing move and an enemy-overwriting special attack instead
        # of filling every slot with near-identical expansion placements.
        activation = next((action for action in ranked
                           if state.turns_remaining > 1
                           and self.prior_policy.activation_features(rules, state, side, action).activated), None)
        if (activation is not None
            and self.prior_policy._score_only(rules, state, side, activation)
            < self.prior_policy._score_only(rules, state, side, ranked[0]) - 2.0):
            activation = None
        opponent_ink = ({Cell.OPPONENT_INK, Cell.OPPONENT_SPECIAL} if side is Side.A
                        else {Cell.SELF_INK, Cell.SELF_SPECIAL})
        special = next((action for action in ranked
                        if action.action_type is ActionType.SPECIAL
                        and any(state.board[row][col] in opponent_ink
                                for row, col, _ in rules.action_targets(state, action))), None)
        reserves = {action for action in (activation, special) if action is not None}
        protected = {ranked[0]}
        for action in (item for item in ranked if item in reserves):
            if action not in retained:
                replace_at = next((i for i in range(len(retained) - 1, -1, -1)
                                   if retained[i] not in protected), None)
                if replace_at is None:
                    break
                retained[replace_at] = action
            protected.add(action)
        return tuple(retained)

