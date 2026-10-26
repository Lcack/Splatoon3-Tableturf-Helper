import json

import pytest

from tableturf_rebuild.application.deck_config import (
    DeckConfigError,
    DeckPresetStore,
    load_deck_config,
)
from tableturf_rebuild.data.catalog import GameCatalog


def test_load_deck_config_accepts_card_ids(tmp_path) -> None:
    catalog = GameCatalog.from_package()
    card_ids = tuple(catalog.cards)[:15]
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(
            {"name": "我的自定义牌组", "cards": card_ids},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    config = load_deck_config(path, catalog)

    assert config.name == "我的自定义牌组"
    assert config.card_ids == card_ids


def test_load_deck_config_rejects_duplicate_cards(tmp_path) -> None:
    catalog = GameCatalog.from_package()
    card_id = next(iter(catalog.cards))
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps({"cards": [card_id] * 15}),
        encoding="utf-8",
    )

    with pytest.raises(DeckConfigError, match="重复"):
        load_deck_config(path, catalog)


def test_deck_preset_store_round_trip(tmp_path) -> None:
    catalog = GameCatalog.from_package()
    store = DeckPresetStore(tmp_path, catalog)
    card_ids = tuple(catalog.cards)[:15]

    preset = store.save("我的:卡组", card_ids)
    loaded = store.list()

    assert preset.path.exists()
    assert len(loaded) == 1
    assert loaded[0].config.name == "我的:卡组"
    assert loaded[0].config.card_ids == card_ids
    store.delete(loaded[0])
    assert store.list() == ()
