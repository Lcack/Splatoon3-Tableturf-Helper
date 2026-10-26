from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.deck_reader import (
    DeckGridReader,
    _maximum_weight_assignment,
)
from tableturf_rebuild.vision.profile_semantics import (
    ProfileSemantics,
    RegionRole,
)


@dataclass(frozen=True, slots=True)
class HandCardCandidate:
    card_id: str
    name: str
    area: int
    sp_cost: int
    score: float


@dataclass(frozen=True, slots=True)
class HandCardObservation:
    slot_row: int
    slot_col: int
    card_id: str
    name: str
    area: int
    sp_cost: int
    visible_cost_lights: int
    match_score: float
    candidates: tuple[HandCardCandidate, ...]


@dataclass(frozen=True, slots=True)
class HandObservation:
    cards: tuple[HandCardObservation, ...]
    confidence: float
    profile: str

    @property
    def card_ids(self) -> tuple[str, ...]:
        return tuple(card.card_id for card in self.cards)


class HandReader:
    def __init__(
        self,
        catalog: GameCatalog,
        profile_dir: Path,
        shape_reader: DeckGridReader | None = None,
    ):
        self.catalog = catalog
        self.shape_reader = shape_reader or DeckGridReader(catalog, profile_dir)
        self._semantics: dict[Path, ProfileSemantics] = {}

    def read(
        self,
        frame: np.ndarray,
        profile_path: Path,
        candidate_ids: tuple[str, ...],
    ) -> HandObservation:
        semantics = self._semantics.get(profile_path)
        if semantics is None:
            semantics = ProfileSemantics.load(profile_path)
            self._semantics[profile_path] = semantics
        detected = {}
        for row in (1, 2):
            for col in (1, 2):
                grid = semantics.by_slot(RegionRole.HAND_GRID, row, col)
                cost = semantics.by_slot(
                    RegionRole.HAND_SPECIAL_COST,
                    row,
                    col,
                )
                if grid is None or cost is None:
                    continue
                pattern = self.shape_reader.pattern_from_region(
                    frame,
                    grid.region,
                )
                lights = self.shape_reader.cost_lights_from_region(
                    frame,
                    cost.region,
                )
                detected[(row, col)] = (pattern, lights)
        if len(detected) != 4:
            raise ValueError("hand profile did not produce four card slots")

        scores = {
            slot: {
                match.card_id: match.score
                for match in self.shape_reader.match_shape(
                    pattern,
                    candidate_ids,
                    lights,
                )
            }
            for slot, (pattern, lights) in detected.items()
        }
        slots = sorted(detected)
        assignments = _maximum_weight_assignment(
            slots,
            tuple(dict.fromkeys(candidate_ids)),
            scores,
        )
        cards = []
        for slot in slots:
            card_id, score = assignments[slot]
            card = self.catalog.cards[card_id]
            ranked = sorted(
                candidate_ids,
                key=lambda candidate: scores[slot][candidate],
                reverse=True,
            )
            cards.append(
                HandCardObservation(
                    slot_row=slot[0],
                    slot_col=slot[1],
                    card_id=card_id,
                    name=card.name_zh or card.name,
                    area=card.area,
                    sp_cost=card.sp_cost,
                    visible_cost_lights=detected[slot][1],
                    match_score=score,
                    candidates=tuple(
                        HandCardCandidate(
                            card_id=candidate_id,
                            name=(
                                self.catalog.cards[candidate_id].name_zh
                                or self.catalog.cards[candidate_id].name
                            ),
                            area=self.catalog.cards[candidate_id].area,
                            sp_cost=self.catalog.cards[candidate_id].sp_cost,
                            score=scores[slot][candidate_id],
                        )
                        for candidate_id in ranked[:4]
                    ),
                )
            )
        return HandObservation(
            cards=tuple(cards),
            confidence=float(np.mean([card.match_score for card in cards])),
            profile=profile_path.name,
        )
