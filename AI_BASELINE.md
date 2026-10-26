# AI Baseline Smoke Results

Date: 2026-10-21

These are engineering smoke tests, not estimates of performance against the
real in-game NPC policy.

## Configuration

- NPC level scenario filter: level 3
- MCTS simulations per decision: 32
- MCTS soft time limit per decision: 1 second
- MCTS opponent model: Greedy
- MCTS rollout: FastRandom
- Seed: 20261022
- Games per role: 10

## Results

| MCTS side | MCTS wins | Greedy wins | Draws | Average score |
|---|---:|---:|---:|---|
| A | 10 | 0 | 0 | 81.10 : 71.90 |
| B | 6 | 4 | 0 | 75.10 : 78.20 |

Combined policy wins across the role-swapped smoke test:

| Policy | Wins |
|---|---:|
| MCTS | 16 |
| Greedy | 4 |

The sample is too small for a confidence claim. Both role directions must
continue to be tested with more seeds, held-out scenarios, and stronger
opponent models before MCTS becomes the default policy.

## Runtime

- 10 MCTS-vs-Greedy games: about 44 seconds
- Approximate MCTS game time in this configuration: 4 to 5 seconds

Runtime depends heavily on board complexity, action count, simulations, and
the time limit.

