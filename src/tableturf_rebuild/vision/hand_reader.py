from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.deck_reader import (
    DeckGridReader,
    _maximum_weight_assignment,
)
from tableturf_rebuild.vision.profile_semantics import (
    ProfileSemantics,
    RegionRole,
)
from tableturf_rebuild.vision.calibration import warp_region
from tableturf_rebuild.vision.selection_profiles import hand_ink_visibility


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
    observed_shape: tuple[tuple[int, ...], ...] | None = None
    available: bool | None = None
    ink_value: float = 0.0


@dataclass(frozen=True, slots=True)
class HandObservation:
    cards: tuple[HandCardObservation, ...]
    confidence: float
    profile: str
    deck_mismatch: bool = False
    unknown_card_ids: tuple[str, ...] = ()

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
        typed_patterns = {}
        availability = {}
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
                step = self.shape_reader.GRID_SAMPLE_SIZE
                hsv = cv2.cvtColor(warp_region(frame, grid.region, 8 * step, 8 * step), cv2.COLOR_BGR2HSV)
                availability[(row, col)] = hand_ink_visibility(hsv)
                dim = availability[(row, col)][0] is False
                pattern = self.shape_reader.pattern_from_region(
                    frame,
                    grid.region,
                    minimum_value=80 if dim else None,
                )
                lights = self._cost_lights(
                    warp_region(frame, cost.region, 100, 20),
                    minimum_value=90 if dim else 150,
                )
                detected[(row, col)] = (pattern, lights)
                typed = pattern.copy()
                for cell_row, cell_col in zip(*np.nonzero(pattern)):
                    patch = hsv[
                        cell_row * step + 5 : (cell_row + 1) * step - 5,
                        cell_col * step + 5 : (cell_col + 1) * step - 5,
                    ]
                    orange = ((patch[:, :, 0] >= 15) & (patch[:, :, 0] < 27)
                              & (patch[:, :, 1] >= 100) & (patch[:, :, 2] >= (80 if dim else 140)))
                    if float(orange.mean()) > 0.2:
                        typed[cell_row, cell_col] = 2
                typed_patterns[(row, col)] = typed
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
        unknown_card_ids = set()
        for slot, (pattern, lights) in detected.items():
            expected_best = max(scores[slot].values(), default=0.0)
            if expected_best >= 0.85:
                continue
            # A configured deck is a prior, not evidence that a different
            # visible card is one of its members. Only a near-exact external
            # match can disprove it; noisy low scores alone cannot.
            global_matches = self.shape_reader.match_shape(
                pattern, tuple(self.catalog.cards), lights,
            )
            if not global_matches or global_matches[0].score < 0.98:
                continue
            if global_matches[0].score - expected_best < 0.15:
                continue
            # Shape and cost cannot distinguish certain weapon editions.
            # Preserve all equally exact outside IDs as unresolved evidence.
            unknown_card_ids.update(
                match.card_id for match in global_matches
                if match.score >= 0.98 and match.card_id not in scores[slot]
            )
        if unknown_card_ids:
            return HandObservation(
                cards=(), confidence=0.0, profile=profile_path.name,
                deck_mismatch=True, unknown_card_ids=tuple(sorted(unknown_card_ids)),
            )
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
                    observed_shape=tuple(tuple(int(value) for value in row) for row in typed_patterns[slot]),
                    available=availability[slot][0],
                    ink_value=availability[slot][1],
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

    @staticmethod
    def _cost_lights(patch: np.ndarray, *, minimum_value: int) -> int:
        # The selected card has a light gray frame: V alone read its empty
        # fifth slot (V=153, S=20 in a live capture) as a fifth SP light.
        # Filled lights are yellow, including disabled cards' dim lights.
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        yellow = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 55)
                  & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= minimum_value))
        return sum(float(yellow[:, index * 20 + 5:(index + 1) * 20 - 5].mean()) > 0.5
                   for index in range(5))
