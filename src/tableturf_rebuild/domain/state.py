from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tableturf_rebuild.domain.grid import OPPONENT_INK_CELLS, SELF_INK_CELLS, Cell
from tableturf_rebuild.domain.stage import StageSpec


class Side(str, Enum):
    A = "A"
    B = "B"

    @property
    def other(self) -> "Side":
        return Side.B if self is Side.A else Side.A


@dataclass(frozen=True, slots=True)
class PlayerState:
    side: Side
    hand: tuple[str, ...]
    draw_pile: tuple[str, ...]
    sp: int = 0
    redraw_available: bool = True

    def __post_init__(self) -> None:
        if len(self.hand) > 4:
            raise ValueError("hand cannot contain more than four cards")
        if len(set(self.hand)) != len(self.hand):
            raise ValueError("hand card IDs must be unique")
        if self.sp < 0:
            raise ValueError("SP cannot be negative")


@dataclass(frozen=True, slots=True)
class BattleState:
    stage: StageSpec
    board: tuple[tuple[Cell, ...], ...]
    player_a: PlayerState
    player_b: PlayerState
    turns_remaining: int = 12

    def __post_init__(self) -> None:
        if len(self.board) != self.stage.rows:
            raise ValueError("board row count does not match stage")
        if any(len(row) != self.stage.cols for row in self.board):
            raise ValueError("board must be rectangular and match stage width")
        if self.player_a.side is not Side.A or self.player_b.side is not Side.B:
            raise ValueError("player state sides do not match player_a/player_b")
        if not 0 <= self.turns_remaining <= 12:
            raise ValueError("turns_remaining must be between 0 and 12")

    @property
    def terminal(self) -> bool:
        return self.turns_remaining == 0

    def player(self, side: Side) -> PlayerState:
        return self.player_a if side is Side.A else self.player_b

    def scores(self) -> tuple[int, int]:
        a_score = sum(cell in SELF_INK_CELLS for row in self.board for cell in row)
        b_score = sum(cell in OPPONENT_INK_CELLS for row in self.board for cell in row)
        return a_score, b_score

    def winner(self) -> str | None:
        if not self.terminal:
            return None
        a_score, b_score = self.scores()
        if a_score > b_score:
            return Side.A.value
        if b_score > a_score:
            return Side.B.value
        return "draw"
