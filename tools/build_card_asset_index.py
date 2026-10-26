"""Validate and index extracted card-face and sleeve images."""

from __future__ import annotations

import json
from pathlib import Path

from tableturf_rebuild import GameCatalog
from tableturf_rebuild.vision.card_assets import CardAssetIndex


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    card_dir = Path(r"E:\Splatoon\splat3\images\minigame\card")
    sleeve_dir = Path(r"E:\Splatoon\splat3\images\minigame\sleeve")
    catalog = GameCatalog.from_package()
    assets = CardAssetIndex.load(catalog, card_dir, sleeve_dir)
    output = {
        "format_version": 1,
        "card_source": str(card_dir),
        "sleeve_source": str(sleeve_dir),
        "cards": {
            card_id: {
                "name": asset.name,
                "file": asset.path.name,
            }
            for card_id, asset in sorted(assets.faces.items())
        },
        "sleeves": {
            npc.npc_id: {
                "sleeve_id": npc.card_sleeve_id,
                "file": assets.sleeve(npc.card_sleeve_id).path.name,
            }
            for npc in sorted(catalog.npcs.values(), key=lambda item: item.npc_id)
        },
    }
    target = root / "assets" / "card_assets" / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(
        f"wrote {len(assets.faces)} card faces and "
        f"{len(assets.sleeves)} sleeves to {target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

