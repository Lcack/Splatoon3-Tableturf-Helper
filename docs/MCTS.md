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
- `preserve_strategic_candidates`: keeps room in the pruned candidates for a
  comparable SP-closing action and an enemy-overwriting special attack.

Every decision exposes `last_stats` with simulations, elapsed time, root visits,
best value, and selected action. `prior_seconds` measures root prior ranking;
`elapsed_seconds` and the soft time budget include that ranking. Generating the
legal-action list before calling `choose` is still measured by the caller.

The effective root and deeper-node limits are further capped from the simulation
budget so that a small search does not spread one visit across every possible
action:

```text
root candidates = min(root_action_limit, max(4, simulations // 4))
deep candidates = min(tree_action_limit, max(2, simulations // 16))
```

This uses the Greedy heuristic as a prior and lets MCTS compare the strongest
candidate subset more deeply.

## Special-point closure prior

The Greedy prior computes the effect of the proposed placement on the eight
neighbors of existing own special tiles and on newly placed special tiles.
Only empty neighbors need filling: opponent ink, walls, void cells and board
edges already close a side. Filling the last empty neighbor is distinguished
from merely being adjacent, and an already active point is not rewarded again.

Before the final turn, a newly active point contributes 2 prior score units.
Partial closure contributes a smaller potential difference, weighted by 0.5,
using `1 / (empty_neighbors + 1)`. New special tiles use the corresponding
post-placement potential. These are explicit heuristic weights, not learned
values or a claim of optimal play. SP earned on the final turn gets no bonus
because it cannot fund a later special attack.

When action pruning is enabled, one closure move may be retained if its prior
score is at most 2 below the strongest candidate. An enemy-overwriting SPECIAL
candidate may also be retained. The strongest prior action is always retained;
MCTS still evaluates terminal wins/losses rather than adding activation points
to the final game score. All candidates must come from the caller's supplied
legal-action list. Live controller support for SPECIAL is a separate concern.

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

