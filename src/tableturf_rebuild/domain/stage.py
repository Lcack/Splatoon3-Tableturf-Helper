from __future__ import annotations

from dataclasses import dataclass, field

from tableturf_rebuild.domain.grid import PLAYABLE_CELLS, Cell


@dataclass(frozen=True, slots=True)
class SpecialPoint:
    row: int
    col: int
    owner: str
    kind: str


@dataclass(frozen=True, slots=True)
class StageSpec:
    stage_id: str
    name_zh: str
    cells: tuple[tuple[Cell, ...], ...]
    special_points: tuple[SpecialPoint, ...]
    source_max_cells: int
    rows: int = field(init=False)
    cols: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.cells or not self.cells[0]:
            raise ValueError(f"{self.stage_id}: stage matrix cannot be empty")
        object.__setattr__(self, "rows", len(self.cells))
        object.__setattr__(self, "cols", len(self.cells[0]))

    @property
    def playable_count(self) -> int:
        return sum(cell in PLAYABLE_CELLS for row in self.cells for cell in row)

    @property
    def max_cells(self) -> int:
        return self.playable_count


@dataclass(frozen=True, slots=True)
class PresetDeck:
    deck_id: str
    card_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NpcSpec:
    npc_id: str
    name: str
    name_zh: str
    ai_levels: tuple[int, ...]
    ai_types: tuple[str, ...]
    deck_ids: tuple[str, ...]
    map_ids: tuple[str, ...]
    card_sleeve_id: str

