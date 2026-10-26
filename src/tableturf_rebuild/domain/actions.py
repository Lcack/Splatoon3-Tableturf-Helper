from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    PLACE = "place"
    SPECIAL = "special"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Action:
    action_type: ActionType
    card_id: str | None = None
    rotation: int | None = None
    origin: tuple[int, int] | None = None

    @classmethod
    def skip(cls, card_id: str) -> "Action":
        return cls(ActionType.SKIP, card_id)

    @classmethod
    def place(
        cls, card_id: str, rotation: int, origin: tuple[int, int]
    ) -> "Action":
        return cls(ActionType.PLACE, card_id, rotation % 4, origin)

    @classmethod
    def special(
        cls, card_id: str, rotation: int, origin: tuple[int, int]
    ) -> "Action":
        return cls(ActionType.SPECIAL, card_id, rotation % 4, origin)

    def __post_init__(self) -> None:
        if self.action_type is ActionType.SKIP:
            if not self.card_id:
                raise ValueError("skip action requires the discarded card_id")
            if self.rotation is not None or self.origin is not None:
                raise ValueError("skip action cannot carry placement data")
            return

        if not self.card_id:
            raise ValueError("place/special action requires a card_id")
        if self.rotation not in {0, 1, 2, 3}:
            raise ValueError("rotation must be 0, 1, 2, or 3")
        if self.origin is None or len(self.origin) != 2:
            raise ValueError("place/special action requires a two-dimensional origin")

