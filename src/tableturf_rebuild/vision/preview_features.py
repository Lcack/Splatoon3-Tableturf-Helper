from __future__ import annotations

import cv2
import numpy as np
from functools import lru_cache

from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.legacy_preview import DOT_RANGES


PREVIEW_CANNY_THRESHOLD = (50, 80)
PREVIEW_DIAGONAL_RATIO = 0.80
PREVIEW_INK_RATIO = 0.15


def preview_diagonal_score(patch: np.ndarray) -> float:
    if patch.ndim == 3:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # Yellow/white stripes can have almost identical luminance while
        # saturation changes sharply. Inspect both channels at every phase.
        return max(
            _edge_diagonal_score(cv2.Canny(patch, *PREVIEW_CANNY_THRESHOLD)),
            preview_diagonal_score(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)),
            preview_diagonal_score(hsv[:, :, 1]),
        )
    gray = (
        cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if patch.ndim == 3
        else patch
    )
    return _edge_diagonal_score(cv2.Canny(gray, *PREVIEW_CANNY_THRESHOLD))


def _edge_diagonal_score(edges: np.ndarray) -> float:
    edges = cv2.dilate(
        edges,
        kernel=np.ones((2, 2), dtype=np.uint8),
    )
    edges = np.rot90(edges)
    half = edges.shape[1] // 2
    if not half:
        return 0.0
    labels, lengths, offsets = _diagonal_layout(*edges.shape)
    # One reduction replaces a Python loop and np.mean for each diagonal.
    # The offsets, dilation, rotation and >0.80 test match the old detector.
    counts = np.bincount(labels, weights=(edges == 255).ravel(), minlength=len(lengths))
    fractions = counts[offsets] / lengths[offsets]
    return float(np.count_nonzero(fractions > PREVIEW_DIAGONAL_RATIO) / (2 * half))


@lru_cache(maxsize=32)
def _diagonal_layout(height: int, width: int):
    rows, cols = np.indices((height, width))
    labels = (cols - rows + height - 1).ravel()
    lengths = np.bincount(labels, minlength=height + width - 1)
    offsets = np.arange(-(width // 2), width // 2) + height - 1
    # np.diag returns an empty array for offsets outside a rectangular image.
    # Those diagonals never pass the old mean>threshold test but still count
    # in its denominator.
    offsets = offsets[(offsets >= 0) & (offsets < len(lengths))]
    return labels, lengths, offsets


def is_diagonal_preview_patch(patch: np.ndarray) -> bool:
    return preview_diagonal_score(patch) > PREVIEW_INK_RATIO


def preview_dot_score(patch: np.ndarray) -> float:
    """Find repeated small bright dots, independent of their animation phase."""
    if patch.ndim == 3:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        white = ((hsv[:, :, 1] < 100)
                 | ((hsv[:, :, 0] >= 10) & (hsv[:, :, 0] < 27)
                    & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 160)))
    else:
        value = patch
        white = np.ones(value.shape, dtype=bool)
    value = value[2:-2, 2:-2]
    white = white[2:-2, 2:-2]
    # On yellow occupied cells, pale dots have V around 230 while the ink
    # beneath has V=255. They are less saturated, not brighter than the ink.
    threshold = max(100.0, min(200.0, float(value.mean() + 0.5 * value.std())))
    mask = ((value > threshold) & white).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dots = sum(
        2 <= area <= 35
        and min(width, height) >= 2
        and 0.4 <= width / height <= 2.5
        # Antialiased stripes can break into many 1x2 specks or sparse
        # diagonal chains. Genuine animated dots form compact components.
        and area / (width * height) >= 0.4
        for _, _, width, height, area in stats[1:count]
    )
    return min(1.0, dots / 10.0) if dots >= 5 else 0.0


def contextual_preview_dot_score(
    patch: np.ndarray,
    underlying: Cell,
    *,
    fiery: bool = False,
    fire_below: Cell | None = None,
) -> float:
    """Original preview dot masks for burning special squares and their flames.

    AutoTableTurf's stage.py uses different masks on activated special squares
    and on squares immediately above their flames.
    Connected small components, rather than a bright flame alone, are required.
    """
    ranges = DOT_RANGES.get(underlying, ())
    if fiery and underlying is Cell.SELF_SPECIAL:
        ranges = (
            ((25, 50, 230), (35, 140, 250), 4, 3),
            ((5, 40, 230), (25, 100, 250), 4, 3),
        )
    elif fiery and underlying is Cell.OPPONENT_SPECIAL:
        ranges = (
            ((80, 10, 230), (90, 15, 245), 4, 3),
            ((90, 70, 235), (100, 90, 245), 4, 3),
            ((80, 25, 235), (90, 45, 245), 4, 3),
        )
    if not ranges:
        return 0.0
    extras = ()
    if fire_below is Cell.SELF_SPECIAL:
        extras = (((20, 125, 240), (30, 140, 255)),)
    elif fire_below is Cell.OPPONENT_SPECIAL:
        extras = (((80, 25, 235), (90, 45, 245)),)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    scale = (min(patch.shape[:2]) / 34.0) ** 2
    # The original ORs ranges with the same component thresholds before
    # labeling them. Keep that union: one animated dot can span both ranges.
    groups: dict[tuple[int, int], list] = {}
    for low, high, minimum_area, minimum_count in ranges:
        groups.setdefault((minimum_area, minimum_count), []).append((low, high))
    best = 0.0
    for (minimum_area, minimum_count), colors in groups.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in (*colors, *extras):
            mask |= cv2.inRange(hsv, low, high)
        if float(np.mean(mask > 0)) < 0.025:
            continue
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        count = int(np.count_nonzero(
            (stats[1:, 4] >= max(2, minimum_area * scale))
            & (stats[1:, 4] < 50 * scale)
        ))
        if count >= minimum_count:
            best = max(best, min(1.0, max(0.6, count / max(6, minimum_count))))
    return best
