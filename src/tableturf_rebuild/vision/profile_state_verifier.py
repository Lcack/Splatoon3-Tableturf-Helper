from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tableturf_rebuild.vision.profile_matcher import VisionProfileMatcher


STATE_PROFILE_FILES = {
    "NPC_ROSTER": ("npc选择.json",),
    "MATCH_SETUP": ("难度选择.json",),
    "DECK_SELECT": ("卡组选择.json",),
    "TURN_SELECT": (
        "battle_select.json",
        "battle_select11.json",
        "battle_select12.json",
        "battle_select21.json",
    ),
    "REDRAW": (
        "battle_select_clean.json",
    ),
    "SKIP_DISCARD": ("battle_select_22.json",),
    "TURN_REVEAL": ("结算阶段开牌.json",),
}


@dataclass(frozen=True, slots=True)
class ProfileStateScore:
    state: str
    score: float
    profile: str
    region_count: int


class VisionProfileStateVerifier:
    def __init__(
        self,
        profile_dir: Path,
        state_files: dict[str, tuple[str, ...]] | None = None,
    ):
        self.profile_dir = profile_dir
        configured_state_files = state_files or _load_state_files(profile_dir)
        self.matcher = VisionProfileMatcher(
            {
                state: tuple(profile_dir / name for name in names)
                for state, names in configured_state_files.items()
            }
        )

    @classmethod
    def default(cls) -> "VisionProfileStateVerifier":
        profile_dir = Path(__file__).resolve().parents[3] / "profiles" / "vision"
        return cls(profile_dir)

    def score(
        self,
        frame: np.ndarray,
        states: set[str] | None = None,
    ) -> tuple[ProfileStateScore, ...]:
        return tuple(
            ProfileStateScore(
                state=item.key,
                score=item.score,
                profile=item.profile,
                region_count=item.region_count,
            )
            for item in self.matcher.score(frame, keys=states)
        )


def _load_state_files(profile_dir: Path) -> dict[str, tuple[str, ...]]:
    index_path = profile_dir / "index.json"
    if not index_path.exists():
        return STATE_PROFILE_FILES
    value = json.loads(index_path.read_text(encoding="utf-8"))
    configured = value.get("state_profiles", {})
    result = {
        state: tuple(str(name) for name in names)
        for state, names in configured.items()
    }
    return result or STATE_PROFILE_FILES
