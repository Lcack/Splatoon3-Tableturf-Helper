from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpponentContext:
    npc_id: str
    npc_name: str
    difficulty: int
    ai_type: str
    deck_id: str
    stage_id: str
    stage_name: str
    confidence: float
