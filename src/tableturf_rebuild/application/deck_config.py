from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tableturf_rebuild.data.catalog import GameCatalog


class DeckConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DeckConfig:
    name: str
    card_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeckPreset:
    path: Path
    config: DeckConfig


def load_deck_config(path: Path, catalog: GameCatalog) -> DeckConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DeckConfigError("牌组配置必须是JSON对象")
    name = str(value.get("name") or path.stem).strip()
    cards = value.get("cards")
    if not isinstance(cards, list) or len(cards) != 15:
        raise DeckConfigError("牌组配置的 cards 必须正好包含15项")

    by_name: dict[str, list[str]] = {}
    for card in catalog.cards.values():
        for label in {card.name, card.name_zh}:
            if label:
                by_name.setdefault(label, []).append(card.card_id)

    resolved = []
    for entry in cards:
        text = str(entry).strip()
        if text in catalog.cards:
            card_id = text
        else:
            matches = by_name.get(text, [])
            if len(matches) != 1:
                raise DeckConfigError(
                    f"无法唯一解析卡牌: {text!r}，请改用 card_id"
                )
            card_id = matches[0]
        resolved.append(card_id)
    if len(set(resolved)) != 15:
        raise DeckConfigError("牌组配置不能包含重复卡牌")
    return DeckConfig(name=name, card_ids=tuple(resolved))


def save_deck_config(
    path: Path,
    name: str,
    card_ids: tuple[str, ...],
) -> DeckConfig:
    if not name.strip():
        raise DeckConfigError("牌组名称不能为空")
    if len(card_ids) != 15 or len(set(card_ids)) != 15:
        raise DeckConfigError("牌组必须包含15张不重复卡牌")
    value = {
        "name": name.strip(),
        "cards": list(card_ids),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return DeckConfig(name=name.strip(), card_ids=card_ids)


class DeckPresetStore:
    def __init__(self, root: Path, catalog: GameCatalog):
        self.root = root
        self.catalog = catalog

    def list(self) -> tuple[DeckPreset, ...]:
        if not self.root.exists():
            return ()
        presets = []
        for path in sorted(self.root.glob("*.json")):
            try:
                config = load_deck_config(path, self.catalog)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            presets.append(DeckPreset(path=path, config=config))
        return tuple(presets)

    def save(self, name: str, card_ids: tuple[str, ...]) -> DeckPreset:
        slug = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name.strip())
        slug = slug.strip(" .") or "deck"
        path = self.root / f"{slug}.json"
        config = save_deck_config(path, name, card_ids)
        return DeckPreset(path=path, config=config)

    @staticmethod
    def delete(preset: DeckPreset) -> None:
        preset.path.unlink(missing_ok=True)
