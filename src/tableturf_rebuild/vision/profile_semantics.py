from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tableturf_rebuild.vision.calibration import VisionProfile, VisionRegion


class RegionRole(StrEnum):
    BOARD = "board"
    HAND_CARD = "hand_card"
    HAND_GRID = "hand_grid"
    HAND_AREA_NUMBER = "hand_area_number"
    HAND_SPECIAL_COST = "hand_special_cost"
    HAND_CARD_NAME = "hand_card_name"
    SKIP_BUTTON = "skip_button"
    SPECIAL_BUTTON = "special_button"
    OWN_SP_INDICATOR = "own_sp_indicator"
    OPPONENT_SP_INDICATOR = "opponent_sp_indicator"
    SELF_SCORE = "self_score"
    OPPONENT_SCORE = "opponent_score"
    REVEAL_CARD = "reveal_card"
    REVEAL_TITLE = "reveal_title"
    REVEAL_AREA_NUMBER = "reveal_area_number"
    REVEAL_GRID = "reveal_grid"
    REVEAL_SPECIAL_COST = "reveal_special_cost"
    NPC_NAME = "npc_name"
    NPC_LIST = "npc_list"
    WIN_COUNT = "win_count"
    DIFFICULTY_SELECTOR = "difficulty_selector"
    STAGE_PREVIEW = "stage_preview"
    STAGE_NAME = "stage_name"
    DECK_LIST_AREA = "deck_list_area"
    DECK_CARDS_AREA = "deck_cards_area"
    DECK_CARD_GRID = "deck_card_grid"
    DECK_CARD_COST = "deck_card_cost"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class SemanticRegion:
    role: RegionRole
    region: VisionRegion
    slot_row: int | None = None
    slot_col: int | None = None


@dataclass(frozen=True, slots=True)
class ProfileSemantics:
    path: Path
    profile: VisionProfile
    regions: tuple[SemanticRegion, ...]

    @classmethod
    def load(cls, path: Path) -> "ProfileSemantics":
        profile = VisionProfile.load(path)
        return cls(
            path=path,
            profile=profile,
            regions=tuple(
                _semantic_region(region) for region in profile.regions
            ),
        )

    def by_role(self, role: RegionRole) -> tuple[SemanticRegion, ...]:
        return tuple(item for item in self.regions if item.role is role)

    def by_slot(
        self,
        role: RegionRole,
        row: int,
        col: int,
    ) -> SemanticRegion | None:
        return next(
            (
                item
                for item in self.regions
                if item.role is role
                and item.slot_row == row
                and item.slot_col == col
            ),
            None,
        )

    def board_region(self) -> VisionRegion | None:
        regions = self.by_role(RegionRole.BOARD)
        return regions[0].region if regions else None


def _semantic_region(region: VisionRegion) -> SemanticRegion:
    name = region.name.strip()
    compact = re.sub(r"\s+", "", name)
    slot = _parse_slot(compact)
    slot_row = slot[0] if slot else None
    slot_col = slot[1] if slot else None
    role = _classify_region(compact, region, slot)
    return SemanticRegion(
        role=role,
        region=region,
        slot_row=slot_row,
        slot_col=slot_col,
    )


def _classify_region(
    name: str,
    region: VisionRegion,
    slot: tuple[int, int] | None,
) -> RegionRole:
    if (
        region.grid_rows == 26
        and region.grid_cols == 9
        or "对战场地" in name
        and region.grid_rows > 0
        and region.grid_cols > 0
    ):
        return RegionRole.BOARD
    if "跳过按钮" in name:
        return RegionRole.SKIP_BUTTON
    if name == "特殊攻击按钮" or "特殊攻击按钮" in name:
        return RegionRole.SPECIAL_BUTTON
    if "我方特殊攻击点数" in name and slot is None:
        return RegionRole.OWN_SP_INDICATOR
    if "对方特殊攻击点数" in name and slot is None:
        return RegionRole.OPPONENT_SP_INDICATOR
    if "我方分数" in name or "我方占地面积" in name:
        return RegionRole.SELF_SCORE
    if "对方分数" in name or "敌方分数" in name or "对方占地面积" in name:
        return RegionRole.OPPONENT_SCORE
    if (
        name.startswith("我方卡牌特殊攻击点数")
        or name.startswith("对方卡牌特殊攻击点数")
    ):
        return RegionRole.REVEAL_SPECIAL_COST
    if "卡牌特殊攻击点数" in name:
        return (
            RegionRole.DECK_CARD_COST
            if name.startswith("卡牌特殊攻击点数")
            else RegionRole.HAND_SPECIAL_COST
        )
    if "特殊攻击点数" in name and slot is not None:
        return RegionRole.HAND_SPECIAL_COST
    if name in {"对方卡牌", "我方卡牌"}:
        return RegionRole.REVEAL_CARD
    if "卡牌标题" in name:
        return RegionRole.REVEAL_TITLE
    if "卡牌点数" in name:
        return RegionRole.REVEAL_AREA_NUMBER
    if "卡牌示意" in name:
        return (
            RegionRole.DECK_CARD_GRID
            if slot is not None
            else RegionRole.REVEAL_GRID
        )
    if "卡牌内数字" in name:
        return RegionRole.HAND_AREA_NUMBER
    if "卡牌内网格" in name:
        return RegionRole.HAND_GRID
    if "卡牌外边框" in name or name.startswith("board（"):
        return RegionRole.HAND_CARD
    if "卡牌名" in name:
        return RegionRole.HAND_CARD_NAME
    if "胜利数" in name:
        return RegionRole.WIN_COUNT
    if name == "npc名字":
        return RegionRole.NPC_NAME
    if "npc列表" in name:
        return RegionRole.NPC_LIST
    if "难度选择" in name:
        return RegionRole.DIFFICULTY_SELECTOR
    if "对战场地名" in name:
        return RegionRole.STAGE_NAME
    if "对战场地" in name or "地图区域" in name:
        return RegionRole.STAGE_PREVIEW
    if "卡组区域" in name:
        return RegionRole.DECK_LIST_AREA
    if "卡牌区域" in name:
        return RegionRole.DECK_CARDS_AREA
    return RegionRole.OTHER


def _parse_slot(name: str) -> tuple[int, int] | None:
    match = re.search(
        r"[（(](\d+)[，,](\d+)[）)]",
        name,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))
