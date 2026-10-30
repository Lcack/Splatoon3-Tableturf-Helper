# Tableturf Battle Rules Used by the Simulator

Rule reference checked on 2026-10-21:

- [SplatoonWiki: Tableturf Battle](https://splatoonwiki.org/wiki/Tableturf_Battle)
- [MediaWiki API extract](https://splatoonwiki.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=Tableturf_Battle&format=json&formatversion=2)

## Implemented rules

- A game has 12 turns.
- Both players choose one of four hand cards and commit an action
  simultaneously each turn.
- Each player starts with a unique 15-card deck, draws four cards, and may
  redraw the hand once before the first turn.
- A played card is removed from the hand and the next card is drawn when
  available.
- Passing discards one hand card and grants one Special Point.
- A normal placement must touch existing own ink orthogonally or diagonally
  and cannot overlap existing ink.
- A Special Attack can replace normal ink, cannot replace a Special Space or
  Wall, and must touch the player's own Special Space.
- A Special Space becomes active when no adjacent cell is empty. Newly active
  Special Spaces grant Special Points.
- At the end of the game, more inked cells wins.

## Same-turn overlap resolution

- If the cards have different pattern sizes, the smaller card is resolved
  last and normally covers the larger card.
- A Special cell cannot be overwritten by a normal cell. If both overlapping
  cells are Special, the smaller card covers the larger one.
- If both cards have the same size and overlap on normal cells, the shared
  cells become walls.
- For equal-size overlap, a Special cell survives against a normal cell; two
  Special cells become a wall.

## Explicit simulation boundaries

- The NPC JSON provides decks, stages, levels, and AI type labels, but not the
  game's internal NPC policy. Offline agents are therefore policy proxies, not
  verified recreations of the CPU opponent.
- Resigning is not part of self-play evaluation.
- Online multiplayer-specific behavior is outside the current model.
- The initial redraw policy is available in the rules layer but is not chosen
  by the current baseline agents.
