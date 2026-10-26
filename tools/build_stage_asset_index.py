"""Validate and index extracted stage preview images."""

from __future__ import annotations

import json
from pathlib import Path

from tableturf_rebuild import GameCatalog
from tableturf_rebuild.vision.stage_assets import StageAssetIndex


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    stage_dir = root / "assets" / "reference" / "stage"
    catalog = GameCatalog.from_package()
    assets = StageAssetIndex.load(catalog, stage_dir)
    target = root / "assets" / "stage_assets" / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            {
                "source": str(stage_dir),
                "stages": {
                    stage_id: {
                        "name_zh": asset.name_zh,
                        "file": asset.path.name,
                    }
                    for stage_id, asset in sorted(assets.stages.items())
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")
    print(f"wrote {len(assets.stages)} stage assets to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

