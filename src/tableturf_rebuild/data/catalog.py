from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tableturf_rebuild.domain.card import CardShape, CardSpec
from tableturf_rebuild.domain.grid import Cell, stage_cell_from_int
from tableturf_rebuild.domain.stage import NpcSpec, PresetDeck, SpecialPoint, StageSpec


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NpcScenario:
    npc_id: str
    level: int
    ai_type: str
    deck_id: str
    stage_id: str


def normalize_row_id(reference: str) -> str:
    return reference.rsplit("/", 1)[-1].split(".", 1)[0]


def normalize_card_sleeve_id(reference: str) -> str:
    stem = normalize_row_id(reference)
    if stem.startswith("MiniGameCardSleeve_"):
        return "MngCardSleeve_" + stem.removeprefix("MiniGameCardSleeve_")
    return stem


@dataclass(frozen=True, slots=True)
class GameCatalog:
    cards: dict[str, CardSpec]
    decks: dict[str, PresetDeck]
    npcs: dict[str, NpcSpec]
    stages: dict[str, StageSpec]

    @classmethod
    def from_package(cls) -> "GameCatalog":
        data_dir = Path(__file__).resolve().parent / "game"
        return cls.from_directory(data_dir)

    @classmethod
    def from_directory(cls, data_dir: Path) -> "GameCatalog":
        cards_raw = cls._load_list(data_dir / "MiniGameCardInfo_Matrix_ZH.json")
        decks_raw = cls._load_list(data_dir / "MiniGamePresetDeck.json")
        npcs_raw = cls._load_list(data_dir / "MiniGameGameNpcData_ZH.json")
        stages_raw = cls._load_list(data_dir / "MiniGameStage.json")

        cards = {entry["__RowId"]: cls._parse_card(entry) for entry in cards_raw}
        decks = {entry["__RowId"]: cls._parse_deck(entry) for entry in decks_raw}
        stages = {entry["Name"]: cls._parse_stage(entry) for entry in stages_raw}
        npcs = {entry["__RowId"]: cls._parse_npc(entry) for entry in npcs_raw}

        catalog = cls(cards=cards, decks=decks, npcs=npcs, stages=stages)
        catalog.validate()
        return catalog

    @staticmethod
    def _load_list(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8-sig") as file:
            value = json.load(file)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise DataValidationError(f"{path.name}: expected a list of objects")
        return value

    @classmethod
    def _parse_card(cls, entry: dict[str, Any]) -> CardSpec:
        matrix = entry.get("square_matrix")
        if not isinstance(matrix, list):
            raise DataValidationError(f"{entry.get('Name')}: missing square_matrix")
        shape = CardShape.from_matrix(matrix)
        return CardSpec(
            card_id=entry["__RowId"],
            name=entry["Name"],
            name_zh=entry.get("name_zh", ""),
            sp_cost=int(entry["SpecialCost"]),
            shape=shape,
            area=int(entry.get("area", shape.size)),
            category=entry.get("Category", ""),
            rarity=entry.get("Rarity", ""),
            season=int(entry.get("Season", 0)),
        )

    @staticmethod
    def _parse_deck(entry: dict[str, Any]) -> PresetDeck:
        card_ids = tuple(normalize_row_id(card) for card in entry.get("Card", ()))
        return PresetDeck(deck_id=entry["__RowId"], card_ids=card_ids)

    @staticmethod
    def _parse_stage(entry: dict[str, Any]) -> StageSpec:
        matrix = entry.get("square_matrix")
        if not isinstance(matrix, list) or not matrix:
            raise DataValidationError(f"{entry.get('Name')}: missing square_matrix")
        cells = tuple(
            tuple(stage_cell_from_int(int(value)) for value in row) for row in matrix
        )
        special_points = tuple(
            SpecialPoint(
                row=int(point["rows"]),
                col=int(point["cols"]),
                owner=point["player"],
                kind=point.get("type", ""),
            )
            for point in entry.get("SpecialPoints", ())
        )
        return StageSpec(
            stage_id=entry["Name"],
            name_zh=entry.get("name_zh", ""),
            cells=cells,
            special_points=special_points,
            source_max_cells=int(entry.get("MaxCells", 0)),
        )

    @staticmethod
    def _parse_npc(entry: dict[str, Any]) -> NpcSpec:
        return NpcSpec(
            npc_id=entry["__RowId"],
            name=entry["Name"],
            name_zh=entry.get("name_zh", ""),
            ai_levels=tuple(int(level) for level in entry.get("AILevel", ())),
            ai_types=tuple(str(ai_type) for ai_type in entry.get("AIType", ())),
            deck_ids=tuple(normalize_row_id(deck) for deck in entry.get("Deck", ())),
            map_ids=tuple(str(stage) for stage in entry.get("Map", ())),
            card_sleeve_id=normalize_card_sleeve_id(entry.get("CardSleeve", "")),
        )

    def validate(self) -> None:
        errors: list[str] = []
        if len(self.cards) != 266:
            errors.append(f"expected 266 cards, got {len(self.cards)}")
        if len(self.decks) != 31:
            errors.append(f"expected 31 decks, got {len(self.decks)}")
        if len(self.npcs) != 30:
            errors.append(f"expected 30 NPCs, got {len(self.npcs)}")
        if len(self.stages) != 15:
            errors.append(f"expected 15 stages, got {len(self.stages)}")

        for deck in self.decks.values():
            if len(deck.card_ids) != 15:
                errors.append(f"{deck.deck_id}: expected 15 cards")
            for card_id in deck.card_ids:
                if card_id not in self.cards:
                    errors.append(f"{deck.deck_id}: unknown card {card_id}")

        for npc in self.npcs.values():
            if not (len(npc.ai_levels) == len(npc.deck_ids) == len(npc.map_ids)):
                errors.append(f"{npc.npc_id}: level/deck/map count mismatch")
            for deck_id in npc.deck_ids:
                if deck_id not in self.decks:
                    errors.append(f"{npc.npc_id}: unknown deck {deck_id}")
            for map_id in npc.map_ids:
                if map_id not in self.stages:
                    errors.append(f"{npc.npc_id}: unknown stage {map_id}")
            if not npc.card_sleeve_id:
                errors.append(f"{npc.npc_id}: missing card sleeve")

        for stage in self.stages.values():
            if any(len(row) != stage.cols for row in stage.cells):
                errors.append(f"{stage.stage_id}: non-rectangular matrix")
            if stage.playable_count != stage.source_max_cells:
                errors.append(
                    f"{stage.stage_id}: playable count {stage.playable_count} "
                    f"does not match MaxCells {stage.source_max_cells}"
                )
            for point in stage.special_points:
                if not (0 <= point.row < stage.rows and 0 <= point.col < stage.cols):
                    errors.append(
                        f"{stage.stage_id}: special point outside matrix "
                        f"({point.row}, {point.col})"
                    )
                    continue
                expected = Cell.SELF_SPECIAL if point.owner == "A" else Cell.OPPONENT_SPECIAL
                if stage.cells[point.row][point.col] != expected:
                    errors.append(
                        f"{stage.stage_id}: special point ({point.row}, {point.col}) "
                        f"does not match matrix cell"
                    )

            self_points = {
                (point.row, point.col)
                for point in stage.special_points
                if point.owner == "A"
            }
            opponent_points = {
                (point.row, point.col)
                for point in stage.special_points
                if point.owner == "B"
            }
            rotated = {
                (stage.rows - 1 - row, stage.cols - 1 - col)
                for row, col in self_points
            }
            if rotated != opponent_points:
                errors.append(
                    f"{stage.stage_id}: special points are not centrally symmetric"
                )

        if errors:
            raise DataValidationError("\n".join(errors))

    def deck_for_npc(self, npc_id: str, level: int = 1) -> PresetDeck:
        npc = self.npcs[npc_id]
        if not 1 <= level <= len(npc.deck_ids):
            raise KeyError(f"{npc_id}: level {level} does not exist")
        return self.decks[npc.deck_ids[level - 1]]

    def stage_for_npc(self, npc_id: str, level: int = 1) -> StageSpec:
        npc = self.npcs[npc_id]
        if not 1 <= level <= len(npc.map_ids):
            raise KeyError(f"{npc_id}: level {level} does not exist")
        return self.stages[npc.map_ids[level - 1]]

    def npc_scenarios(self) -> tuple[NpcScenario, ...]:
        scenarios: list[NpcScenario] = []
        for npc in sorted(self.npcs.values(), key=lambda item: item.npc_id):
            for index, (deck_id, stage_id) in enumerate(
                zip(npc.deck_ids, npc.map_ids, strict=True),
                start=1,
            ):
                scenarios.append(
                    NpcScenario(
                        npc_id=npc.npc_id,
                        level=index,
                        ai_type=(
                            npc.ai_types[index - 1]
                            if index - 1 < len(npc.ai_types)
                            else ""
                        ),
                        deck_id=deck_id,
                        stage_id=stage_id,
                    )
                )
        return tuple(scenarios)

