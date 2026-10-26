from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from typing import Iterable


Position = tuple[int, int]


class CardCell(IntEnum):
    EMPTY = 0
    INK = 1
    SPECIAL = 2


@dataclass(frozen=True, slots=True)
class CardShape:
    """A trimmed, immutable card pattern."""

    cells: tuple[tuple[CardCell, ...], ...]
    occupied_cells: tuple[Position, ...] = field(init=False, repr=False)
    cell_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        occupied = tuple(
            (row, col)
            for row, cells in enumerate(self.cells)
            for col, cell in enumerate(cells)
            if cell != CardCell.EMPTY
        )
        if not occupied:
            raise ValueError("card shape must contain at least one inked cell")
        object.__setattr__(self, "occupied_cells", occupied)
        object.__setattr__(self, "cell_count", len(occupied))

    @classmethod
    def from_matrix(cls, matrix: Iterable[Iterable[int]]) -> "CardShape":
        rows = tuple(tuple(CardCell(value) for value in row) for row in matrix)
        if not rows or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("card matrix must be a non-empty rectangle")

        occupied_rows = [
            row_index
            for row_index, row in enumerate(rows)
            if any(cell != CardCell.EMPTY for cell in row)
        ]
        occupied_cols = [
            col_index
            for col_index in range(len(rows[0]))
            if any(row[col_index] != CardCell.EMPTY for row in rows)
        ]
        if not occupied_rows or not occupied_cols:
            raise ValueError("card matrix must contain at least one inked cell")

        top, bottom = occupied_rows[0], occupied_rows[-1] + 1
        left, right = occupied_cols[0], occupied_cols[-1] + 1
        trimmed = tuple(row[left:right] for row in rows[top:bottom])
        return cls(trimmed)

    @property
    def height(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0])

    @property
    def occupied(self) -> tuple[Position, ...]:
        return self.occupied_cells

    @property
    def size(self) -> int:
        return self.cell_count

    @lru_cache(maxsize=None)
    def rotated(self, turns: int) -> "CardShape":
        """Return a counter-clockwise rotation in 90-degree steps."""

        turns %= 4
        if turns == 0:
            return self

        matrix = [[CardCell.EMPTY for _ in range(self.height)] for _ in range(self.width)]
        for row, cells in enumerate(self.cells):
            for col, cell in enumerate(cells):
                target_row = self.width - 1 - col
                target_col = row
                matrix[target_row][target_col] = cell

        result = CardShape(tuple(tuple(row) for row in matrix))
        return result.rotated(turns - 1)


@dataclass(frozen=True, slots=True)
class CardSpec:
    card_id: str
    name: str
    name_zh: str
    sp_cost: int
    shape: CardShape
    area: int
    category: str = ""
    rarity: str = ""
    season: int = 0

    def __post_init__(self) -> None:
        if not self.card_id:
            raise ValueError("card_id cannot be empty")
        if not 1 <= self.sp_cost <= 6:
            raise ValueError(f"{self.card_id}: invalid SP cost {self.sp_cost}")
        if self.area != self.shape.size:
            raise ValueError(
                f"{self.card_id}: area {self.area} does not match shape {self.shape.size}"
            )

