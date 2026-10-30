from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


RegionHandle = Literal[
    "nw",
    "n",
    "ne",
    "e",
    "se",
    "s",
    "sw",
    "w",
    "vertex:0",
    "vertex:1",
    "vertex:2",
    "vertex:3",
    "edge:0",
    "edge:1",
    "edge:2",
    "edge:3",
]


@dataclass(frozen=True, slots=True)
class VisionRegion:
    name: str
    kind: str
    points: tuple[tuple[float, float], ...]
    grid_rows: int = 0
    grid_cols: int = 0

    def __post_init__(self) -> None:
        expected = 2 if self.kind == "rect" else 4
        if self.kind not in {"rect", "quad"}:
            raise ValueError("region kind must be rect or quad")
        if len(self.points) != expected:
            raise ValueError(f"{self.kind} requires {expected} points")
        if self.kind == "quad":
            object.__setattr__(
                self,
                "points",
                order_quad_corners(self.points),
            )
        if any(not (0 <= value <= 1) for point in self.points for value in point):
            raise ValueError("region points must be normalized to 0..1")
        if self.grid_rows < 0 or self.grid_cols < 0:
            raise ValueError("grid dimensions cannot be negative")

    def pixel_points(self, width: int, height: int) -> np.ndarray:
        return np.array(
            [[x * (width - 1), y * (height - 1)] for x, y in self.points],
            dtype=np.float32,
        )

    @classmethod
    def from_pixel_points(
        cls,
        *,
        name: str,
        kind: str,
        points: np.ndarray,
        width: int,
        height: int,
        grid_rows: int = 0,
        grid_cols: int = 0,
    ) -> "VisionRegion":
        if width <= 1 or height <= 1:
            raise ValueError("image dimensions must be greater than one")
        clipped = np.clip(
            np.asarray(points, dtype=np.float64),
            [0.0, 0.0],
            [width - 1.0, height - 1.0],
        )
        normalized = clipped / [width - 1.0, height - 1.0]
        return cls(
            name=name,
            kind=kind,
            points=tuple(
                (float(point[0]), float(point[1])) for point in normalized
            ),
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )


@dataclass(frozen=True, slots=True)
class CellMaskOverride:
    row: int
    col: int
    cell: int

    def __post_init__(self) -> None:
        if self.row < 0 or self.col < 0:
            raise ValueError("cell mask coordinates cannot be negative")
        if self.cell not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError("cell mask override contains unsupported stage cell")


@dataclass(frozen=True, slots=True)
class VisionProfile:
    profile_id: str
    source_width: int
    source_height: int
    stage_id: str | None = None
    regions: tuple[VisionRegion, ...] = field(default_factory=tuple)
    notes: str = ""
    cell_overrides: tuple[CellMaskOverride, ...] = field(default_factory=tuple)

    def region_map(self) -> dict[str, VisionRegion]:
        return {region.name: region for region in self.regions}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(asdict(self), file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "VisionProfile":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["regions"] = tuple(
            VisionRegion(
                name=region["name"],
                kind=region["kind"],
                points=tuple(tuple(point) for point in region["points"]),
                grid_rows=region.get("grid_rows", 0),
                grid_cols=region.get("grid_cols", 0),
            )
            for region in value.get("regions", ())
        )
        value["cell_overrides"] = tuple(
            CellMaskOverride(
                row=int(item["row"]),
                col=int(item["col"]),
                cell=int(item["cell"]),
            )
            for item in value.get("cell_overrides", ())
        )
        return cls(**value)


def region_grid_cells(
    region: VisionRegion,
    width: int,
    height: int,
) -> tuple[tuple[np.ndarray, str], ...]:
    if region.grid_rows <= 0 or region.grid_cols <= 0:
        return ()
    points = region.pixel_points(width, height)
    if region.kind == "rect":
        x1, y1 = points[0]
        x2, y2 = points[1]
        corners = np.float32(
            [
                [min(x1, x2), min(y1, y2)],
                [max(x1, x2), min(y1, y2)],
                [max(x1, x2), max(y1, y2)],
                [min(x1, x2), max(y1, y2)],
            ]
        )
    else:
        corners = np.float32(order_quad_corners(tuple(map(tuple, points))))
    source = np.float32(
        [
            [0, 0],
            [region.grid_cols, 0],
            [region.grid_cols, region.grid_rows],
            [0, region.grid_rows],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, corners)
    result = []
    for row in range(region.grid_rows):
        for col in range(region.grid_cols):
            cell = np.float32(
                [[col, row], [col + 1, row], [col + 1, row + 1], [col, row + 1]]
            ).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(cell, transform).reshape(4, 2)
            result.append((mapped, f"{row},{col}"))
    return tuple(result)


def warp_region(
    image: np.ndarray,
    region: VisionRegion,
    output_width: int,
    output_height: int,
    *,
    padding: int = 0,
) -> np.ndarray:
    if output_width <= 0 or output_height <= 0:
        raise ValueError("warp output dimensions must be positive")
    if padding < 0:
        raise ValueError("warp padding cannot be negative")
    if image.ndim not in (2, 3):
        raise ValueError("image must be grayscale or color")
    points = region.pixel_points(image.shape[1], image.shape[0])
    if region.kind == "rect":
        x1, y1 = points[0]
        x2, y2 = points[1]
        corners = np.float32(
            [
                [min(x1, x2), min(y1, y2)],
                [max(x1, x2), min(y1, y2)],
                [max(x1, x2), max(y1, y2)],
                [min(x1, x2), max(y1, y2)],
            ]
        )
    else:
        corners = np.float32(order_quad_corners(tuple(map(tuple, points))))
    destination = np.float32(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ]
    )
    destination += padding
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (output_width + 2 * padding, output_height + 2 * padding),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def order_quad_corners(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if len(points) != 4:
        raise ValueError("quad ordering requires four points")
    values = np.asarray(points, dtype=np.float64)
    coordinate_sum = values[:, 0] + values[:, 1]
    coordinate_difference = values[:, 0] - values[:, 1]
    indices = (
        int(np.argmin(coordinate_sum)),
        int(np.argmax(coordinate_difference)),
        int(np.argmax(coordinate_sum)),
        int(np.argmin(coordinate_difference)),
    )
    if len(set(indices)) != 4:
        center = values.mean(axis=0)
        angles = np.arctan2(
            values[:, 1] - center[1],
            values[:, 0] - center[0],
        )
        ordered_indices = np.argsort(angles)
        points_by_angle = values[ordered_indices]
        indices = (
            0,
            1,
            2,
            3,
        )
        values = points_by_angle
    return tuple(
        (float(values[index, 0]), float(values[index, 1]))
        for index in indices
    )


def region_pixel_bounds(
    region: VisionRegion,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    points = region.pixel_points(width, height)
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def region_pixel_size(
    region: VisionRegion,
    width: int,
    height: int,
) -> tuple[float, float]:
    left, top, right, bottom = region_pixel_bounds(region, width, height)
    return right - left, bottom - top


def resize_region_to_pixels(
    region: VisionRegion,
    width: int,
    height: int,
    target_width: float,
    target_height: float,
) -> VisionRegion:
    if target_width <= 0 or target_height <= 0:
        raise ValueError("region dimensions must be positive")
    points = region.pixel_points(width, height)
    left, top, right, bottom = region_pixel_bounds(region, width, height)
    source_width = right - left
    source_height = bottom - top
    if source_width <= 0 or source_height <= 0:
        raise ValueError("cannot resize a region with zero width or height")
    center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    scale = np.array(
        [target_width / source_width, target_height / source_height],
        dtype=np.float64,
    )
    resized = center + (points.astype(np.float64) - center) * scale
    return VisionRegion.from_pixel_points(
        name=region.name,
        kind=region.kind,
        points=resized,
        width=width,
        height=height,
        grid_rows=region.grid_rows,
        grid_cols=region.grid_cols,
    )


def translate_region_pixels(
    region: VisionRegion,
    width: int,
    height: int,
    delta_x: float,
    delta_y: float,
) -> VisionRegion:
    points = region.pixel_points(width, height).astype(np.float64)
    left, top, right, bottom = region_pixel_bounds(region, width, height)
    applied_x = min(max(delta_x, -left), (width - 1) - right)
    applied_y = min(max(delta_y, -top), (height - 1) - bottom)
    points += [applied_x, applied_y]
    return VisionRegion.from_pixel_points(
        name=region.name,
        kind=region.kind,
        points=points,
        width=width,
        height=height,
        grid_rows=region.grid_rows,
        grid_cols=region.grid_cols,
    )


def drag_resize_region(
    region: VisionRegion,
    width: int,
    height: int,
    handle: RegionHandle,
    pointer_start: tuple[float, float],
    pointer_end: tuple[float, float],
) -> VisionRegion:
    start = np.asarray(pointer_start, dtype=np.float64)
    end = np.asarray(pointer_end, dtype=np.float64)
    delta = end - start
    if region.kind == "rect":
        left, top, right, bottom = region_pixel_bounds(region, width, height)
        minimum_span = 1.0
        if "w" in handle:
            left = min(max(0.0, left + delta[0]), right - minimum_span)
        if "e" in handle:
            right = max(min(width - 1.0, right + delta[0]), left + minimum_span)
        if "n" in handle:
            top = min(max(0.0, top + delta[1]), bottom - minimum_span)
        if "s" in handle:
            bottom = max(
                min(height - 1.0, bottom + delta[1]),
                top + minimum_span,
            )
        points = np.array([[left, top], [right, bottom]], dtype=np.float64)
    else:
        points = region.pixel_points(width, height).astype(np.float64)
        if handle.startswith("vertex:"):
            index = int(handle.split(":", maxsplit=1)[1])
            if not 0 <= index < len(points):
                raise ValueError("quad vertex handle is out of range")
            points[index] += delta
        elif handle.startswith("edge:"):
            edge = int(handle.split(":", maxsplit=1)[1])
            if not 0 <= edge < 4:
                raise ValueError("quad edge handle is out of range")
            first = edge
            second = (edge + 1) % 4
            if edge in (0, 2):
                points[[first, second], 1] += delta[1]
            else:
                points[[first, second], 0] += delta[0]
        else:
            raise ValueError("quad regions require vertex or edge handles")
    return VisionRegion.from_pixel_points(
        name=region.name,
        kind=region.kind,
        points=points,
        width=width,
        height=height,
        grid_rows=region.grid_rows,
        grid_cols=region.grid_cols,
    )

