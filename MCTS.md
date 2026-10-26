# Opponent-Model MCTS

The current implementation is an opponent-model Monte Carlo tree search for
the local player. It is not a joint-action Nash solver.

## Search model

At every node:

1. The tree selects an action for the searched player.
2. The opponent selects an immediate action with the configured opponent
   policy on the same pre-turn state.
3. The joint turn is resolved using the simultaneous-turn rules.
4. A fast legal rollout continues both sides to the final score.
5. The terminal reward is `+1`, `0`, or `-1` from the searched player's
   perspective.

The default opponent and rollout policies are Greedy and FastRandom
respectively. They are configuration points for later opponent-pool and
league evaluation.

## Search controls

- `MCTSConfig.simulations`: maximum simulations per decision.
- `MCTSConfig.time_limit_seconds`: soft wall-clock limit. One simulation may
  finish after the limit.
- `root_action_limit`: number of high-prior root actions considered.
- `tree_action_limit`: number of high-prior actions retained at deeper nodes.
- `exploration_weight`: UCT exploration coefficient.
- `seed`: determines exploration order and rollout randomness.

Every decision exposes `last_stats` with simulations, elapsed time, root visits,
best value, and selected action.

The effective root and deeper-node limits are further capped from the simulation
budget so that a small search does not spread one visit across every possible
action:

```text
root candidates = min(root_action_limit, max(4, simulations // 4))
deep candidates = min(tree_action_limit, max(2, simulations // 16))
```

This uses the Greedy heuristic as a prior and lets MCTS compare the strongest
candidate subset more deeply.

## Current limitations

- In offline exact simulation both hands are visible. Live play requires a
  hidden-card belief state.
- The opponent is a fixed policy model, not a full search of all opponent
  actions.
- Root/deep action pruning uses the Greedy prior and can discard a move that
  MCTS would otherwise discover.
- The rollout policy is intentionally cheap. Its estimates are noisy and need
  calibration against stronger rollouts and recorded games.
- Search quality must be measured over many fixed-seed games. A single result
  is not evidence of strength.

