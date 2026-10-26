# AutoTableTurf Rebuild

Clean-room rebuild of the AutoTableTurf stack.

The current implementation covers the data layer and deterministic battle
rules. Vision, controller integration, and the application state machine are
later phases.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Validate data

```powershell
.\.venv\Scripts\tableturf.exe validate-data
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Offline self-play

```powershell
.\.venv\Scripts\tableturf.exe selfplay --games 100 --seed 42 --level 3
.\.venv\Scripts\tableturf.exe selfplay --games 10 --seed 42 --level 3 --policy-a mcts --policy-b greedy --simulations 32 --time-limit 1
```

The command cycles through NPC decks and stages extracted from the game data
and writes a JSONL trace under `artifacts/selfplay/`.

See `RULES.md`, `MCTS.md`, and `AI_OBJECTIVES.md` for rule assumptions, search
limits, and the distinction between policy training and counter-deck search.

For resumable and crash-safe long runs, see `RUN_LOOP.md`.

For EasyCon controller hardware validation, see `CONTROLLER_DEBUG.md`.

For capture-card diagnostics, see `CAPTURE_DEBUG.md`.

For long-running candidate-deck evaluation, see `DECK_SEARCH.md`.

For precise visual ROI calibration, see `ROI_CALIBRATION_SPEC.md`.

## Data encoding

Card matrices use:

```text
0 = empty
1 = normal ink
2 = special ink
```

Stage matrices use cell-type bit values:

```text
1  = empty and playable
2  = player A normal ink
4  = player A special
8  = player B normal ink
16 = player B special
32 = non-playable neutral/void
64 = wall
```

These two encodings are intentionally not shared.

See `PROJECT_PLAN.md` and `src/tableturf_rebuild/data/DATA_SOURCES.md` for the
rebuild strategy and data provenance.

