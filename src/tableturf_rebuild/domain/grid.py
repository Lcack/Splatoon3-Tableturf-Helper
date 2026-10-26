from enum import IntEnum


class Cell(IntEnum):
    """Stage cell values used by the extracted MiniGameStage data."""

    EMPTY = 1
    SELF_INK = 2
    SELF_SPECIAL = 4
    OPPONENT_INK = 8
    OPPONENT_SPECIAL = 16
    VOID = 32
    WALL = 64


PLAYABLE_CELLS = frozenset(
    {
        Cell.EMPTY,
        Cell.SELF_INK,
        Cell.SELF_SPECIAL,
        Cell.OPPONENT_INK,
        Cell.OPPONENT_SPECIAL,
    }
)

SELF_INK_CELLS = frozenset({Cell.SELF_INK, Cell.SELF_SPECIAL})
OPPONENT_INK_CELLS = frozenset({Cell.OPPONENT_INK, Cell.OPPONENT_SPECIAL})
BLOCKED_CELLS = frozenset({Cell.VOID, Cell.WALL})


def stage_cell_from_int(value: int) -> Cell:
    """Convert an extracted stage value to a Cell.

    The stage file already stores bit-style cell types. It must not use the
    card encoding where 1 means ink and 2 means special.
    """

    try:
        return Cell(value)
    except ValueError as error:
        raise ValueError(f"unsupported stage cell value: {value}") from error

