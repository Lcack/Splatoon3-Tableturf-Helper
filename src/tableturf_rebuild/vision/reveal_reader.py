from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.calibration import VisionProfile, warp_region
from tableturf_rebuild.vision.deck_reader import (
    CardShapeMatch,
    DeckGridReader,
)


@dataclass(frozen=True, slots=True)
class RevealedCardOption:
    card_id: str
    name: str
    area: int
    sp_cost: int
    score: float


@dataclass(frozen=True, slots=True)
class RevealedCardObservation:
    card_id: str | None
    name: str | None
    area: int | None
    sp_cost: int | None
    detected_cells: int
    visible_cost_lights: int
    match_score: float
    confidence: float
    candidates: tuple[RevealedCardOption, ...]


@dataclass(frozen=True, slots=True)
class RevealObservation:
    own: RevealedCardObservation
    opponent: RevealedCardObservation


class RevealReader:
    def __init__(
        self,
        catalog: GameCatalog,
        profile_dir: Path,
        shape_reader: DeckGridReader | None = None,
    ):
        self.catalog = catalog
        self.profile = VisionProfile.load(profile_dir / "结算阶段开牌.json")
        self.regions = self.profile.region_map()
        self.shape_reader = shape_reader or DeckGridReader(
            catalog,
            profile_dir,
        )

    def read(
        self,
        frame: np.ndarray,
        *,
        own_card_ids: tuple[str, ...] | None = None,
        opponent_card_ids: tuple[str, ...] | None = None,
    ) -> RevealObservation:
        return RevealObservation(
            own=self._read_card(
                frame,
                region_name="我方卡牌示意",
                cost_region_name="我方卡牌特殊攻击点数",
                side="self",
                candidate_ids=own_card_ids,
            ),
            opponent=self._read_card(
                frame,
                region_name="对方卡牌示意",
                cost_region_name="对方卡牌特殊攻击点数",
                side="opponent",
                candidate_ids=opponent_card_ids,
            ),
        )

    def _read_card(
        self,
        frame: np.ndarray,
        *,
        region_name: str,
        cost_region_name: str,
        side: str,
        candidate_ids: tuple[str, ...] | None,
    ) -> RevealedCardObservation:
        region = self.regions[region_name]
        pattern = self.shape_reader.pattern_from_region(
            frame,
            region,
            side=side,
        )
        lights = self.shape_reader.cost_lights_from_region(
            frame,
            self.regions[cost_region_name],
        )
        candidates = self.shape_reader.match_shape(
            pattern,
            candidate_ids or tuple(self.catalog.cards),
            lights,
        )
        detected_cells = int(np.count_nonzero(pattern))
        options = self._options(candidates)
        if (
            not candidates
            or detected_cells == 0
            or candidates[0].score < 0.55
        ):
            return RevealedCardObservation(
                card_id=None,
                name=None,
                area=None,
                sp_cost=None,
                detected_cells=detected_cells,
                visible_cost_lights=lights,
                match_score=candidates[0].score if candidates else 0.0,
                confidence=candidates[0].score if candidates else 0.0,
                candidates=options,
            )
        best = self.catalog.cards[candidates[0].card_id]
        return RevealedCardObservation(
            card_id=best.card_id,
            name=best.name_zh or best.name,
            area=best.area,
            sp_cost=best.sp_cost,
            detected_cells=detected_cells,
            visible_cost_lights=lights,
            match_score=candidates[0].score,
            confidence=candidates[0].score,
            candidates=options,
        )

    def _options(
        self,
        matches: tuple[CardShapeMatch, ...],
    ) -> tuple[RevealedCardOption, ...]:
        result = []
        for match in matches[:5]:
            card = self.catalog.cards[match.card_id]
            result.append(
                RevealedCardOption(
                    card_id=card.card_id,
                    name=card.name_zh or card.name,
                    area=card.area,
                    sp_cost=card.sp_cost,
                    score=match.score,
                )
            )
        return tuple(result)
