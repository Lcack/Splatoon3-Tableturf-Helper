from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tableturf_rebuild.data.catalog import GameCatalog


@dataclass(frozen=True, slots=True)
class StageAsset:
    stage_id: str
    name_zh: str
    path: Path


@dataclass(frozen=True, slots=True)
class StageAssetIndex:
    stages: dict[str, StageAsset]

    @classmethod
    def load(cls, catalog: GameCatalog, stage_dir: Path) -> "StageAssetIndex":
        values = {}
        missing = []
        for stage_id, stage in catalog.stages.items():
            path = stage_dir / f"{stage_id}.png"
            if not path.exists():
                missing.append(str(path))
                continue
            values[stage_id] = StageAsset(stage_id, stage.name_zh, path)
        if missing:
            raise FileNotFoundError(
                "missing stage preview images:\n" + "\n".join(missing)
            )
        return cls(values)

