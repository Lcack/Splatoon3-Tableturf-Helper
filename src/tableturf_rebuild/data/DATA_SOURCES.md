# Game Data Sources

The canonical files under `data/game/` are generated from
`recognation_preparation/` by `tools/sync_game_data.py`.

The original source directory is never modified. The importer strips harmless
trailing commas from the extracted JSON, converts output to strict UTF-8 JSON,
and records SHA-256 hashes in `SOURCE_MANIFEST.json`.

## Matrix encodings

Card `square_matrix` values:

```text
0 = empty
1 = normal ink
2 = special ink
```

Stage `square_matrix` values:

```text
1  = empty playable cell
2  = player A normal ink
4  = player A special
8  = player B normal ink
16 = player B special
32 = non-playable neutral/void
64 = wall
```

Player A corresponds to the local player and player B to the opponent.

Canonical JSON matrices are stored top-to-bottom, then left-to-right, using
zero-based `(row, column)` coordinates. Visual descriptions that count from the
bottom can be converted with:

```text
zero_based_row = rows - bottom_up_row_number
zero_based_col = left_to_right_column_number - 1
```

All extracted stages have centrally symmetric player A/player B special-point
sets. This is enforced by the data validator.

## Known source repairs

The importer raises an error on unknown inconsistencies and applies only the
following explicit repairs:

1. `Rectangle` has metadata `26x9` and `MaxCells=234`, but its extracted matrix
   was `22x11` and byte-for-byte equivalent in content to `Painted`. The
   canonical matrix is rebuilt as a `26x9` playable rectangle and special cells
   are placed from `SpecialPoints`.
2. `MaxCells` is recomputed from the canonical matrix. This changes `Stairs`
   from 236 to 240 because its matrix contains 240 non-wall cells. The value
   240 was also confirmed by direct visual inspection on 2026-10-21.

## Images

`assets/reference/stage/` contains copies of the extracted stage images:

- `CenterHole.png`, `Cross.png`, ... are the named stage preview maps.
- `AreaPiece_00.png` through `AreaPiece_04.png` are 42x42 UI sprites. The
  visual role mapping in `assets/reference/stage/semantic_manifest.json` is
  SELF_INK, OPPONENT_INK, SELF_SPECIAL, OPPONENT_SPECIAL, and INVALID;
  `Mng_Point_02.png` is used there as an EMPTY reference.
- Sprite filename suffixes are not numeric matrix encodings. The visual
  reference manifest and the canonical data encodings serve different roles.

`Rectangle` is reconstructed rather than visually traced cell-by-cell. Its
metadata, stage image aspect ratio, spawn points, and `MaxCells=234` agree on a
26x9 playable rectangle. On 2026-10-21 the user visually confirmed the shape
and the bottom-up special points `(4,5)` for player A and `(23,5)` for player B.
Those convert to canonical top-down coordinates `(22,4)` and `(3,4)`.


