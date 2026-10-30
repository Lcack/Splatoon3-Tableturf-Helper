from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.card import CardSpec
from tableturf_rebuild.vision.calibration import (
    VisionProfile,
    VisionRegion,
    warp_region,
)
from tableturf_rebuild.vision.profile_semantics import (
    ProfileSemantics,
    RegionRole,
)


@dataclass(frozen=True, slots=True)
class DeckSlotCandidate:
    card_id: str
    name: str
    area: int
    sp_cost: int
    score: float


@dataclass(frozen=True, slots=True)
class DeckSlotObservation:
    slot_row: int
    slot_col: int
    card_id: str
    name: str
    area: int
    sp_cost: int
    visible_cost_lights: int
    match_score: float
    candidates: tuple[DeckSlotCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class DeckObservation:
    deck_id: str
    deck_score: float
    deck_margin: float
    total_area: int
    total_sp_cost: int
    cards: tuple[DeckSlotObservation, ...]
    confidence: float
    source: str = "auto"


@dataclass(frozen=True, slots=True)
class DeckFingerprint:
    patterns: tuple[bytes, ...]
    lights: tuple[int, ...]

    def changed_slots(
        self,
        other: "DeckFingerprint",
    ) -> int:
        if len(self.patterns) != len(other.patterns):
            return max(len(self.patterns), len(other.patterns))
        changed = 0
        for current, candidate, current_lights, candidate_lights in zip(
            self.patterns,
            other.patterns,
            self.lights,
            other.lights,
            strict=True,
        ):
            left = np.frombuffer(current, dtype=np.uint8)
            right = np.frombuffer(candidate, dtype=np.uint8)
            if (
                int(np.count_nonzero(left != right)) >= 2
                or current_lights != candidate_lights
            ):
                changed += 1
        return changed

    def cell_difference(self, other: "DeckFingerprint") -> int:
        if len(self.patterns) != len(other.patterns):
            return 64 * max(len(self.patterns), len(other.patterns))
        difference = 0
        for current, candidate in zip(
            self.patterns,
            other.patterns,
            strict=True,
        ):
            difference += int(
                np.count_nonzero(
                    np.frombuffer(current, dtype=np.uint8)
                    != np.frombuffer(candidate, dtype=np.uint8)
                )
            )
        difference += 4 * sum(
            abs(current - candidate)
            for current, candidate in zip(
                self.lights,
                other.lights,
                strict=True,
            )
        )
        return difference

    def materially_different(
        self,
        other: "DeckFingerprint",
        *,
        minimum_cells: int = 4,
        minimum_slots: int = 1,
    ) -> bool:
        return (
            self.cell_difference(other) >= minimum_cells
            and self.changed_slots(other) >= minimum_slots
        )


@dataclass(frozen=True, slots=True)
class CardShapeMatch:
    card_id: str
    score: float


class DeckGridReader:
    CELL_GRID_SIZE = 8
    GRID_SAMPLE_SIZE = 24

    def __init__(self, catalog: GameCatalog, profile_dir: Path):
        self.catalog = catalog
        self.semantics = ProfileSemantics.load(profile_dir / "卡组选择.json")
        self.variants = {
            card_id: self._shape_variants(card)
            for card_id, card in catalog.cards.items()
        }

    def read(self, frame: np.ndarray) -> DeckObservation:
        return self._read_detected(self._detect_cards(frame))

    def fingerprint(self, frame: np.ndarray) -> DeckFingerprint:
        detected = self._detect_cards(frame)
        slots = sorted(detected)
        return DeckFingerprint(
            patterns=tuple(
                np.asarray(detected[slot][0], dtype=np.uint8).tobytes()
                for slot in slots
            ),
            lights=tuple(detected[slot][1] for slot in slots),
        )

    def pattern_from_region(
        self,
        frame: np.ndarray,
        region: VisionRegion,
        *,
        side: str = "self",
        minimum_value: int | None = None,
    ) -> np.ndarray:
        if side not in {"self", "opponent"}:
            raise ValueError("card pattern side must be self or opponent")
        patch = warp_region(
            frame,
            region,
            self.CELL_GRID_SIZE * self.GRID_SAMPLE_SIZE,
            self.CELL_GRID_SIZE * self.GRID_SAMPLE_SIZE,
        )
        return self._pattern(patch, side=side, minimum_value=minimum_value)

    def cost_lights_from_region(
        self,
        frame: np.ndarray,
        region: VisionRegion,
        *,
        minimum_value: int = 150,
    ) -> int:
        return self._cost_lights(warp_region(frame, region, 100, 20), minimum_value=minimum_value)

    def match_shape(
        self,
        pattern: np.ndarray,
        candidate_ids: tuple[str, ...] | list[str],
        visible_cost_lights: int | None = None,
    ) -> tuple[CardShapeMatch, ...]:
        scores = [
            CardShapeMatch(
                card_id=card_id,
                score=self._score(
                    pattern,
                    visible_cost_lights,
                    card_id,
                ),
            )
            for card_id in dict.fromkeys(candidate_ids)
            if card_id in self.catalog.cards
        ]
        scores.sort(key=lambda item: item.score, reverse=True)
        return tuple(scores)

    def _detect_cards(
        self,
        frame: np.ndarray,
    ) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
        detected = {}
        for row in range(1, 6):
            for col in range(1, 4):
                grid = self.semantics.by_slot(
                    RegionRole.DECK_CARD_GRID,
                    row,
                    col,
                )
                cost = self.semantics.by_slot(
                    RegionRole.DECK_CARD_COST,
                    row,
                    col,
                )
                if grid is None or cost is None:
                    continue
                pattern = self.pattern_from_region(frame, grid.region)
                lights = self._cost_lights(
                    warp_region(frame, cost.region, 100, 20)
                )
                detected[(row, col)] = (pattern, lights)
        if len(detected) != 15:
            raise ValueError("deck profile did not produce 15 card slots")
        return detected

    def _read_detected(
        self,
        detected: dict[tuple[int, int], tuple[np.ndarray, int]],
    ) -> DeckObservation:
        slots = sorted(detected)
        card_ids = tuple(self.catalog.cards)
        scores = {
            slot: {
                card_id: self._score(pattern, lights, card_id)
                for card_id in self.catalog.cards
            }
            for slot, (pattern, lights) in detected.items()
        }
        assignments = _maximum_weight_assignment(
            slots,
            card_ids,
            scores,
        )

        cards = []
        slot_margins = []
        for slot in slots:
            card_id, score = assignments[slot]
            card = self.catalog.cards[card_id]
            lights = detected[slot][1]
            ranked = sorted(
                card_ids,
                key=lambda candidate: scores[slot][candidate],
                reverse=True,
            )
            second_score = (
                scores[slot][ranked[1]]
                if len(ranked) > 1
                else 0.0
            )
            slot_margins.append(score - second_score)
            candidates = tuple(
                DeckSlotCandidate(
                    card_id=candidate_id,
                    name=(
                        self.catalog.cards[candidate_id].name_zh
                        or self.catalog.cards[candidate_id].name
                    ),
                    area=self.catalog.cards[candidate_id].area,
                    sp_cost=self.catalog.cards[candidate_id].sp_cost,
                    score=scores[slot][candidate_id],
                )
                for candidate_id in ranked[:5]
            )
            cards.append(
                DeckSlotObservation(
                    slot_row=slot[0],
                    slot_col=slot[1],
                    card_id=card_id,
                    name=card.name_zh or card.name,
                    area=card.area,
                    sp_cost=card.sp_cost,
                    visible_cost_lights=lights,
                    match_score=score,
                    candidates=candidates,
                )
            )
        deck_score = float(np.mean([card.match_score for card in cards]))
        return DeckObservation(
            deck_id="CUSTOM_AUTO",
            deck_score=deck_score,
            deck_margin=min(slot_margins, default=0.0),
            total_area=sum(card.area for card in cards),
            total_sp_cost=sum(card.sp_cost for card in cards),
            cards=tuple(cards),
            confidence=min(1.0, max(0.0, deck_score)),
            source="auto",
        )

    def _pattern(
        self,
        patch: np.ndarray,
        *,
        side: str = "self",
        minimum_value: int | None = None,
    ) -> np.ndarray:
        if side == "self":
            lower_hue, upper_hue = 15, 55
            minimum_saturation, default_value = 100, 140
        else:
            lower_hue, upper_hue = 95, 170
            minimum_saturation, default_value = 110, 130
        if minimum_value is None:
            minimum_value = default_value
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        bits = np.zeros((8, 8), dtype=np.uint8)
        step = self.GRID_SAMPLE_SIZE
        margin = 5
        for row in range(8):
            for col in range(8):
                cell = hsv[
                    row * step + margin : (row + 1) * step - margin,
                    col * step + margin : (col + 1) * step - margin,
                ]
                mask = (
                    (cell[:, :, 0] >= lower_hue)
                    & (cell[:, :, 0] <= upper_hue)
                    & (cell[:, :, 1] >= minimum_saturation)
                    & (cell[:, :, 2] >= minimum_value)
                )
                bits[row, col] = int(float(mask.mean()) > 0.2)
        return bits

    @staticmethod
    def _cost_lights(patch: np.ndarray, *, minimum_value: int = 150) -> int:
        value = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
        return sum(
            value[:, index * 20 + 5 : (index + 1) * 20 - 5].mean() > minimum_value
            for index in range(5)
        )

    def _score(
        self,
        pattern: np.ndarray,
        lights: int | None,
        card_id: str,
    ) -> float:
        best = 0.0
        for variant in self.variants[card_id]:
            intersection = int(np.count_nonzero(pattern & variant))
            union = int(np.count_nonzero(pattern | variant))
            if union:
                best = max(best, intersection / union)
        card = self.catalog.cards[card_id]
        if lights is None:
            return best
        compatible = card.sp_cost == lights or (
            card.sp_cost == 6 and lights == 5
        )
        return best - (0.0 if compatible else 0.25)

    def _shape_variants(self, card: CardSpec) -> tuple[np.ndarray, ...]:
        base = np.array(
            [
                [1 if cell > 0 else 0 for cell in row]
                for row in card.shape.cells
            ],
            dtype=np.uint8,
        )
        results = set()
        for transform in (
            lambda value: value,
            lambda value: np.rot90(value, 1),
            lambda value: np.rot90(value, 2),
            lambda value: np.rot90(value, 3),
            lambda value: np.fliplr(value),
            lambda value: np.flipud(value),
            lambda value: value.T,
            lambda value: np.rot90(value.T, 2),
        ):
            shape = transform(base)
            height, width = shape.shape
            for row in range(8 - height + 1):
                for col in range(8 - width + 1):
                    canvas = np.zeros((8, 8), dtype=np.uint8)
                    canvas[row : row + height, col : col + width] = shape
                    results.add(canvas.tobytes())
        return tuple(
            np.frombuffer(value, dtype=np.uint8).reshape(8, 8)
            for value in results
        )


def _maximum_weight_assignment(
    slots: list[tuple[int, int]],
    card_ids: tuple[str, ...],
    scores: dict[tuple[int, int], dict[str, float]],
) -> dict[tuple[int, int], tuple[str, float]]:
    """Assign one unique card to every slot with a rectangular Hungarian solver."""

    if len(slots) > len(card_ids):
        raise ValueError("not enough candidate cards for deck slots")
    row_count = len(slots)
    column_count = len(card_ids)
    costs = [
        [
            1.0 - scores[slot][card_id]
            for card_id in card_ids
        ]
        for slot in slots
    ]

    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        column0 = 0
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column0] = True
            current_row = matched_row[column0]
            delta = float("inf")
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current = (
                    costs[current_row - 1][column - 1]
                    - u[current_row]
                    - v[column]
                )
                if current < minimum[column]:
                    minimum[column] = current
                    previous_column[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = next_column
            if matched_row[column0] == 0:
                break
        while True:
            previous = previous_column[column0]
            matched_row[column0] = matched_row[previous]
            column0 = previous
            if column0 == 0:
                break

    column_for_row = [0] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            column_for_row[matched_row[column] - 1] = column - 1
    return {
        slot: (
            card_ids[column_for_row[index]],
            scores[slot][card_ids[column_for_row[index]]],
        )
        for index, slot in enumerate(slots)
    }
