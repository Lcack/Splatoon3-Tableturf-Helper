"""Empty-cell grid fitting adapted from AutoTableTurf's detection/stage.py.

Upstream: fga401/AutoTableTurf, GPL-3.0. Keep the native cell centers and
sample their interiors, as the original executor does.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from tableturf_rebuild.domain.grid import PLAYABLE_CELLS
from tableturf_rebuild.vision.calibration import VisionRegion


@dataclass(frozen=True, slots=True)
class GridFit:
    stage_id: str
    region: VisionRegion
    empty_cells: int
    confidence: float
    step_x: float
    step_y: float


def _axis(values, step):
    groups = []
    for value in sorted(values):
        if not groups or value - float(np.mean(groups[-1])) > step * 0.45:
            groups.append([value])
        else:
            groups[-1].append(value)
    means = np.array([np.mean(group) for group in groups])
    if len(means) < 4:
        return None
    spacing = float(np.median(np.diff(means)))
    indices = np.rint((means - means[0]) / spacing).astype(int)
    slope, origin = np.polyfit(indices, means, 1)
    if np.max(np.abs(means - (origin + indices * slope))) > step * 0.2:
        return None
    return float(origin), float(slope), int(indices[-1]) + 1


def fit_empty_cell_grid(frame, catalog) -> GridFit | None:
    height, width = frame.shape[:2]
    scale = height / 1080.0
    left, right = round(width * 0.38), round(width * 0.82)
    hsv = cv2.cvtColor(frame[:, left:right], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 46))
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    good = ((stats[:, 2] > 30 * scale) & (stats[:, 2] < 45 * scale)
            & (stats[:, 3] > 30 * scale) & (stats[:, 3] < 45 * scale)
            & (stats[:, 4] > 900 * scale ** 2) & (stats[:, 4] < 2025 * scale ** 2))
    centers = centers[good]
    if len(centers) < 32:
        return None
    centers[:, 0] += left
    x = _axis(centers[:, 0], 40 * scale)
    y = _axis(centers[:, 1], 40 * scale)
    if x is None or y is None:
        return None
    x0, dx, cols = x
    y0, dy, rows = y
    empty = {(int(round((cy - y0) / dy)), int(round((cx - x0) / dx))) for cx, cy in centers}
    candidates = []
    for stage in catalog.stages.values():
        if (stage.rows, stage.cols) != (rows, cols):
            continue
        matched = sum(stage.cells[row][col] in PLAYABLE_CELLS for row, col in empty)
        precision = matched / len(empty)
        coverage = matched / stage.playable_count
        candidates.append((0.8 * precision + 0.2 * coverage, precision, stage.stage_id))
    candidates.sort(reverse=True)
    if not candidates or candidates[0][1] < 0.96:
        return None
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.03:
        return None
    points = ((max(0, x0 - dx / 2) / (width - 1), max(0, y0 - dy / 2) / (height - 1)),
              (min(width - 1, x0 + (cols - 0.5) * dx) / (width - 1),
               min(height - 1, y0 + (rows - 0.5) * dy) / (height - 1)))
    return GridFit(candidates[0][2], VisionRegion("live_native_grid", "rect", points, rows, cols),
                   len(empty), candidates[0][0], dx, dy)
