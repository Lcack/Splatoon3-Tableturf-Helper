from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class AnnotationShape:
    kind: str
    start: tuple[float, float]
    end: tuple[float, float]
    color: str = "#ff3b30"


def sanitize_filename(value: str, fallback: str = "state") -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*\\x00-\\x1f]+', "_", value.strip())
    cleaned = re.sub(r"[\s_]+", "_", cleaned)
    cleaned = cleaned.strip(" ._")
    return cleaned[:80] or fallback


def render_annotations(
    image: Image.Image,
    shapes: Iterable[AnnotationShape],
) -> Image.Image:
    source = image.convert("RGBA")
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = source.size
    for shape in shapes:
        x1 = int(max(0.0, min(1.0, shape.start[0])) * width)
        y1 = int(max(0.0, min(1.0, shape.start[1])) * height)
        x2 = int(max(0.0, min(1.0, shape.end[0])) * width)
        y2 = int(max(0.0, min(1.0, shape.end[1])) * height)
        red, green, blue = _hex_rgb(shape.color)
        bounds = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if shape.kind == "rectangle":
            draw.rectangle(
                bounds,
                outline=(red, green, blue, 255),
                fill=(red, green, blue, 35),
                width=5,
            )
        elif shape.kind == "ellipse":
            draw.ellipse(
                bounds,
                outline=(red, green, blue, 255),
                fill=(red, green, blue, 35),
                width=5,
            )
        elif shape.kind == "arrow":
            _draw_arrow(draw, (x1, y1), (x2, y2), (red, green, blue, 255))
        else:
            raise ValueError(f"unknown annotation kind: {shape.kind}")
    return Image.alpha_composite(source, overlay).convert("RGB")


def shape_to_dict(shape: AnnotationShape) -> dict[str, Any]:
    return {
        "kind": shape.kind,
        "start": list(shape.start),
        "end": list(shape.end),
        "color": shape.color,
    }


def save_annotation_bundle(
    original: Image.Image,
    target_dir: Path,
    sequence: int,
    state_name: str,
    description: str,
    attention: str,
    actions: str,
    next_state: str,
    shapes: list[AnnotationShape],
    capture_info: dict[str, Any],
) -> dict[str, Any]:
    import json
    from datetime import datetime

    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(state_name)
    actual_sequence = sequence
    while True:
        stem = f"{actual_sequence:03d}_{safe_name}"
        if not any(
            (target_dir / f"{stem}{suffix}").exists()
            for suffix in (".png", ".annotated.png", ".json")
        ):
            break
        actual_sequence += 1
    original_path = target_dir / f"{stem}.png"
    annotated_path = target_dir / f"{stem}.annotated.png"
    metadata_path = target_dir / f"{stem}.json"

    original.save(original_path)
    render_annotations(original, shapes).save(annotated_path)
    metadata = {
        "sequence": actual_sequence,
        "state_name": state_name.strip(),
        "description": description.strip(),
        "attention": attention.strip(),
        "actions": actions.strip(),
        "next_state": next_state.strip(),
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "original_image": original_path.name,
        "annotated_image": annotated_path.name,
        "annotations": [shape_to_dict(shape) for shape in shapes],
        "capture": capture_info,
    }
    with metadata_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with (target_dir / "annotations.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False)
        file.write("\n")
    return metadata


def _hex_rgb(value: str) -> tuple[int, int, int]:
    clean = value.lstrip("#")
    if len(clean) != 6:
        raise ValueError(f"invalid color: {value}")
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


def _draw_arrow(draw: ImageDraw.ImageDraw, start, end, color) -> None:
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=(0, 0, 0, 220), width=9)
    draw.line((x1, y1, x2, y2), fill=color, width=5)
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 24
    spread = 0.55
    points = (
        (x2, y2),
        (
            x2 - length * math.cos(angle - spread),
            y2 - length * math.sin(angle - spread),
        ),
        (
            x2 - length * math.cos(angle + spread),
            y2 - length * math.sin(angle + spread),
        ),
    )
    draw.polygon(points, fill=color)

