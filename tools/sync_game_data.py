"""Normalize the unpacked game data into the package data directory.

The source files are kept unchanged. Repairs are explicit and are also
recorded in the generated source manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


PLAYABLE_VALUES = {1, 2, 4, 8, 16}


def load_tolerant_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair_stages(stages: list[dict[str, Any]]) -> list[str]:
    repairs: list[str] = []
    for stage in stages:
        name = stage["Name"]
        rows = int(stage["Dimensions"]["rows"])
        cols = int(stage["Dimensions"]["cols"])
        matrix = stage["square_matrix"]
        actual_shape = (len(matrix), len(matrix[0]) if matrix else 0)

        if actual_shape != (rows, cols):
            if name != "Rectangle":
                raise ValueError(
                    f"{name}: matrix shape {actual_shape} does not match {(rows, cols)}"
                )

            matrix = [[1 for _ in range(cols)] for _ in range(rows)]
            for point in stage.get("SpecialPoints", ()):
                row, col = int(point["rows"]), int(point["cols"])
                matrix[row][col] = 4 if point["player"] == "A" else 16
            stage["square_matrix"] = matrix
            repairs.append(
                f"{name}: rebuilt {rows}x{cols} matrix from dimensions and "
                f"special points; original had shape {actual_shape}"
            )

        playable = sum(
            value in PLAYABLE_VALUES for row in stage["square_matrix"] for value in row
        )
        if int(stage["MaxCells"]) != playable:
            old_value = int(stage["MaxCells"])
            stage["MaxCells"] = playable
            repairs.append(
                f"{name}: MaxCells corrected from {old_value} to {playable}"
            )

    return repairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1].parent / "recognation_preparation",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src"
        / "tableturf_rebuild"
        / "data"
        / "game",
    )
    args = parser.parse_args()

    source: Path = args.source.resolve()
    destination: Path = args.destination.resolve()
    project_root = Path(__file__).resolve().parents[1]
    destination.mkdir(parents=True, exist_ok=True)

    names = [
        "MiniGameCardInfo_Matrix_ZH.json",
        "MiniGamePresetDeck.json",
        "MiniGameGameNpcData_ZH.json",
        "MiniGameStage.json",
    ]
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}

    for name in names:
        source_path = source / name
        value = load_tolerant_json(source_path)
        source_hashes[name] = sha256(source_path)
        counts[name] = len(value)

        if name == "MiniGameStage.json":
            repairs = repair_stages(value)
        else:
            repairs = []

        output_path = destination / name
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
        output_hashes[name] = sha256(output_path)

        if repairs:
            for repair in repairs:
                print(f"[repair] {repair}")

    stage_source = source / "stage"
    stage_destination = project_root / "assets" / "reference" / "stage"
    if stage_source.is_dir():
        stage_destination.mkdir(parents=True, exist_ok=True)
        for image in stage_source.glob("*.png"):
            shutil.copy2(image, stage_destination / image.name)

    manifest = {
        "source": str(source),
        "generated_by": "tools/sync_game_data.py",
        "counts": counts,
        "files": {
            name: {
                "source_sha256": source_hashes[name],
                "canonical_sha256": output_hashes[name],
            }
            for name in names
        },
        "stage_image_directory": "assets/reference/stage",
    }
    with (destination / "SOURCE_MANIFEST.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"wrote canonical data to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

