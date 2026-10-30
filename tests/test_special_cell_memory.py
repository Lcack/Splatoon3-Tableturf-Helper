"""An already committed special square cannot vanish with its flame phase."""
from dataclasses import replace
import json
from pathlib import Path

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action
from tableturf_rebuild.domain.card import CardShape
from tableturf_rebuild.domain.grid import Cell, PLAYABLE_CELLS
from tableturf_rebuild.domain.rules import BattleRules
from tableturf_rebuild.domain.state import Side
from tableturf_rebuild.vision.board_reader import BoardCellReading, BoardObservation, SpecialCellMemory


SAMPLES = Path(__file__).resolve().parents[1] / "artifacts/diagnostics/current-special-stall"


def sample(index):
    saved = json.loads((SAMPLES / f"clear-{index}.json").read_text(encoding="utf-8"))
    return board_from_saved(saved), saved


def board_from_saved(saved):
    source = saved["observation"]["board"]
    readings = []
    for reading in source["readings"]:
        item = dict(reading)
        for name in ("stage_cell", "inferred_cell"): item[name] = Cell(item[name])
        for name in ("mean_bgr", "median_hsv"): item[name] = tuple(item[name])
        readings.append(BoardCellReading(**item))
    cells = tuple(tuple(Cell(value) for value in row) for row in source["cells"])
    return BoardObservation(source["stage_id"], source["rows"], source["cols"], cells,
        tuple(readings), source["confidence"], 0.0, source["preview_present"],
        sum(cell in PLAYABLE_CELLS for row in cells for cell in row))


def change_cell(board, row, col, value):
    cells = [list(items) for items in board.cells]
    cells[row][col] = value
    readings = tuple(replace(item, inferred_cell=value) if (item.row, item.col) == (row, col)
                     else item for item in board.readings)
    return replace(board, cells=tuple(map(tuple, cells)), readings=readings)


def test_real_flame_phases_keep_the_committed_special_position():
    memory = SpecialCellMemory()
    boards = [sample(index)[0] for index in range(3)]
    assert [board.cells[15][6] for board in boards] == [Cell.SELF_INK, Cell.SELF_SPECIAL, Cell.SELF_INK]
    corrected = [memory.update(board, allow_new=True) for board in boards]
    assert [board.cells[15][6] for board in corrected] == [Cell.SELF_INK, Cell.SELF_SPECIAL, Cell.SELF_SPECIAL]
    assert next(item for item in corrected[-1].readings if (item.row, item.col) == (15, 6)).inferred_cell is Cell.SELF_SPECIAL


def test_memory_removes_the_actual_illegal_great_barrier_target_from_legal_actions():
    first, _ = sample(1)
    saved = json.loads((SAMPLES / "latest.json").read_text(encoding="utf-8"))
    missed, _ = sample(2)
    memory = SpecialCellMemory()
    memory.update(first, allow_new=True)
    corrected = memory.update(missed, allow_new=True)
    catalog = GameCatalog.from_package()
    card_id = "MiniGame_SpGreatBarrier"
    shape = CardShape.from_matrix(saved["executor"]["visual_shapes"][card_id])
    catalog = replace(catalog, cards={**catalog.cards, card_id: replace(catalog.cards[card_id], shape=shape)})
    rules = BattleRules(catalog)
    state = rules.with_hand("Cross", (card_id,), self_sp=3, round_index=6)
    action = Action.special(card_id, 0, (11, 5))
    assert action in rules.legal_actions(replace(state, board=missed.cells), Side.A)
    assert action not in rules.legal_actions(replace(state, board=corrected.cells), Side.A)


def test_preview_or_untrusted_screen_never_learns_new_specials():
    board, _ = sample(1)
    missed, _ = sample(2)
    for first, allowed in ((replace(board, placement_preview_present=True), True), (board, False)):
        memory = SpecialCellMemory()
        memory.update(first, allow_new=allowed)
        assert memory.update(missed, allow_new=False).cells[15][6] is Cell.SELF_INK


def test_low_global_preview_flag_cannot_learn_a_local_dot():
    board, _ = sample(1)
    readings = tuple(replace(item, preview_dot_score=.2) if (item.row, item.col) == (15, 6)
                     else item for item in board.readings)
    memory = SpecialCellMemory()
    memory.update(replace(board, readings=readings), allow_new=True)
    missed, _ = sample(2)
    assert memory.update(missed, allow_new=False).cells[15][6] is Cell.SELF_INK


def test_preview_frames_retain_known_positions_without_learning_new_ones():
    board, _ = sample(1)
    memory = SpecialCellMemory()
    memory.update(board, allow_new=True)
    preview = change_cell(board, 15, 6, Cell.SELF_INK)
    preview = change_cell(preview, 14, 6, Cell.SELF_SPECIAL)
    preview = replace(preview, placement_preview_present=True)
    remembered = memory.update(preview, allow_new=True)
    assert remembered.cells[15][6] is Cell.SELF_SPECIAL
    cleared = change_cell(preview, 14, 6, Cell.SELF_INK)
    assert memory.update(cleared, allow_new=False).cells[14][6] is Cell.SELF_INK


def test_reset_and_map_change_drop_previous_match_positions():
    board, _ = sample(1)
    missed, _ = sample(2)
    memory = SpecialCellMemory()
    memory.update(board, allow_new=True)
    memory.reset()
    assert memory.update(missed, allow_new=False).cells[15][6] is Cell.SELF_INK
    memory.update(board, allow_new=True)
    other = replace(missed, stage_id="other")
    assert memory.update(other, allow_new=False).cells[15][6] is Cell.SELF_INK


REPEATED = SAMPLES.parent / "repeated-stall-audit"
HISTORY = SAMPLES.parents[1] / "live_task/run-20261030T013150Z-11628-f6eda8/evidence"


def historical_opponent_special():
    saved = json.loads((HISTORY / "074143-561907.json").read_text(encoding="utf-8"))
    return board_from_saved(saved)


def robot_preview_exclusions():
    # Observed Robot stripes surround origin(18,8); the title dot is at(18,8).
    # Exclude the complete 3x3 card box and one further square on every side.
    return frozenset((row, col) for row in range(17, 22) for col in range(7, 12))


def test_uncovered_real_special_survives_two_opposing_flame_colors():
    history = historical_opponent_special()
    assert history.placement_preview_present and history.cells[10][12] is Cell.OPPONENT_SPECIAL
    memory = SpecialCellMemory()
    memory.update(history, allow_new=False, trusted_preview_exclusions=robot_preview_exclusions())
    raw = []
    remembered = []
    for index in (0, 1, 2, 3):
        saved = json.loads((REPEATED / f"clear-{index}.json").read_text(encoding="utf-8"))
        board = board_from_saved(saved)
        raw.append(board.cells[10][12])
        remembered.append(memory.update(board, allow_new=True).cells[10][12])
    assert raw == [Cell.EMPTY, Cell.EMPTY, Cell.SELF_SPECIAL, Cell.SELF_SPECIAL]
    assert remembered == [Cell.OPPONENT_SPECIAL] * 4


def test_global_preview_without_trusted_footprint_still_never_learns():
    history = historical_opponent_special()
    later = board_from_saved(json.loads((REPEATED / "clear-0.json").read_text(encoding="utf-8")))
    for exclusions in (None, frozenset()):
        memory = SpecialCellMemory()
        memory.update(history, allow_new=True, trusted_preview_exclusions=exclusions)
        assert memory.update(later, allow_new=False).cells[10][12] is Cell.EMPTY


def test_exclusion_zone_never_learns_even_strong_special_colored_pixels():
    history = historical_opponent_special()
    later = board_from_saved(json.loads((REPEATED / "clear-0.json").read_text(encoding="utf-8")))
    memory = SpecialCellMemory()
    exclusions = robot_preview_exclusions() | {(10, 12)}
    memory.update(history, allow_new=False, trusted_preview_exclusions=exclusions)
    assert memory.update(later, allow_new=False).cells[10][12] is Cell.EMPTY


def test_uncovered_preview_still_rejects_a_local_dot_false_positive():
    history = historical_opponent_special()
    readings = tuple(replace(item, preview_dot_score=.2) if (item.row, item.col) == (10, 12)
                     else item for item in history.readings)
    memory = SpecialCellMemory()
    memory.update(replace(history, readings=readings), allow_new=False,
                  trusted_preview_exclusions=robot_preview_exclusions())
    later = board_from_saved(json.loads((REPEATED / "clear-0.json").read_text(encoding="utf-8")))
    assert memory.update(later, allow_new=False).cells[10][12] is Cell.EMPTY


def test_explicit_exclusions_also_protect_when_global_preview_detection_misses():
    history = replace(historical_opponent_special(), placement_preview_present=False)
    later = board_from_saved(json.loads((REPEATED / "clear-0.json").read_text(encoding="utf-8")))
    memory = SpecialCellMemory()
    memory.update(history, allow_new=True,
                  trusted_preview_exclusions=robot_preview_exclusions() | {(10, 12)})
    assert memory.update(later, allow_new=False).cells[10][12] is Cell.EMPTY
