from tableturf_rebuild.data.catalog import (
    GameCatalog,
    normalize_card_sleeve_id,
)
from tableturf_rebuild.domain.card import CardCell
from tableturf_rebuild.domain.grid import PLAYABLE_CELLS, Cell


def test_catalog_counts_and_references() -> None:
    catalog = GameCatalog.from_package()

    assert len(catalog.cards) == 266
    assert len(catalog.decks) == 31
    assert len(catalog.npcs) == 30
    assert len(catalog.stages) == 15
    assert all(len(deck.card_ids) == 15 for deck in catalog.decks.values())
    assert all(
        card_id in catalog.cards
        for deck in catalog.decks.values()
        for card_id in deck.card_ids
    )
    assert all(
        deck_id in catalog.decks
        for npc in catalog.npcs.values()
        for deck_id in npc.deck_ids
    )


def test_card_sleeve_name_is_normalized_to_extracted_asset_name() -> None:
    assert (
        normalize_card_sleeve_id(
            "Work/Gyml/MiniGameCardSleeve_Aori.spl__MiniGameCardSleeve.gyml"
        )
        == "MngCardSleeve_Aori"
    )


def test_card_matrix_encoding_and_area() -> None:
    catalog = GameCatalog.from_package()

    for card in catalog.cards.values():
        assert card.area == card.shape.size
        assert all(
            cell in {CardCell.EMPTY, CardCell.INK, CardCell.SPECIAL}
            for row in card.shape.cells
            for cell in row
        )


def test_stage_encoding_and_playable_count() -> None:
    catalog = GameCatalog.from_package()

    for stage in catalog.stages.values():
        assert stage.playable_count == stage.source_max_cells
        assert all(
            cell in PLAYABLE_CELLS or cell in {Cell.VOID, Cell.WALL}
            for row in stage.cells
            for cell in row
        )


def test_rectangle_source_repair() -> None:
    catalog = GameCatalog.from_package()
    rectangle = catalog.stages["Rectangle"]
    painted = catalog.stages["Painted"]

    assert (rectangle.rows, rectangle.cols) == (26, 9)
    assert rectangle.playable_count == 234
    assert rectangle.cells != painted.cells
    assert rectangle.cells[22][4] == Cell.SELF_SPECIAL
    assert rectangle.cells[3][4] == Cell.OPPONENT_SPECIAL


def test_all_special_points_are_centrally_symmetric() -> None:
    catalog = GameCatalog.from_package()

    for stage in catalog.stages.values():
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
        rotated_self_points = {
            (stage.rows - 1 - row, stage.cols - 1 - col)
            for row, col in self_points
        }
        assert rotated_self_points == opponent_points


def test_stage_uses_all_extended_cell_values() -> None:
    catalog = GameCatalog.from_package()
    values = {
        int(cell)
        for stage in catalog.stages.values()
        for row in stage.cells
        for cell in row
    }

    assert {1, 2, 4, 8, 16, 32, 64}.issubset(values)


def test_npc_level_resolves_deck_and_stage() -> None:
    catalog = GameCatalog.from_package()

    for npc in catalog.npcs.values():
        for level in range(1, len(npc.deck_ids) + 1):
            deck = catalog.deck_for_npc(npc.npc_id, level)
            stage = catalog.stage_for_npc(npc.npc_id, level)
            assert deck.deck_id == npc.deck_ids[level - 1]
            assert stage.stage_id == npc.map_ids[level - 1]
            assert npc.card_sleeve_id.startswith("MngCardSleeve_")

