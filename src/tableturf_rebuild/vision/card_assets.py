from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tableturf_rebuild.data.catalog import GameCatalog


@dataclass(frozen=True, slots=True)
class CardFaceAsset:
    card_id: str
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class CardSleeveAsset:
    sleeve_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class CardAssetIndex:
    faces: dict[str, CardFaceAsset]
    sleeves: dict[str, CardSleeveAsset]

    @classmethod
    def load(
        cls,
        catalog: GameCatalog,
        face_dir: Path,
        sleeve_dir: Path,
    ) -> "CardAssetIndex":
        faces = {}
        missing = []
        for card in catalog.cards.values():
            path = face_dir / f"{card.name}.png"
            if not path.exists():
                missing.append(f"{card.card_id}: {path}")
                continue
            faces[card.card_id] = CardFaceAsset(card.card_id, card.name, path)
        if missing:
            raise FileNotFoundError("missing card faces:\n" + "\n".join(missing))

        sleeves = {}
        for npc in catalog.npcs.values():
            path = sleeve_dir / f"{npc.card_sleeve_id}.png"
            if not path.exists():
                raise FileNotFoundError(
                    f"{npc.npc_id}: missing sleeve {path}"
                )
            sleeves[npc.card_sleeve_id] = CardSleeveAsset(
                npc.card_sleeve_id,
                path,
            )
        return cls(faces=faces, sleeves=sleeves)

    def face(self, card_id: str) -> CardFaceAsset:
        return self.faces[card_id]

    def sleeve(self, sleeve_id: str) -> CardSleeveAsset:
        return self.sleeves[sleeve_id]

