"""Background-specific dot masks ported from AutoTableTurf (GPL-3.0).

Source: tableturf/manager/detection/stage.py, preview/__is_special_preview.
"""
import cv2
import numpy as np

from tableturf_rebuild.domain.grid import Cell


DOT_RANGES = {
    Cell.EMPTY: [((0, 0, 180), (179, 50, 220), 4, 3),
                 ((10, 130, 160), (20, 255, 220), 6, 5)],
    Cell.WALL: [((120, 10, 150), (130, 80, 255), 4, 3)],
    Cell.VOID: [((120, 10, 150), (130, 80, 255), 4, 3)],
    Cell.SELF_INK: [((25, 90, 230), (35, 190, 240), 4, 3)],
    Cell.OPPONENT_INK: [((110, 70, 240), (120, 100, 255), 4, 3)],
    Cell.SELF_SPECIAL: [((10, 95, 240), (20, 130, 255), 4, 3)],
    Cell.OPPONENT_SPECIAL: [((90, 60, 230), (100, 80, 250), 4, 3)],
}


def context_dot_score(patch, underlying: Cell) -> float:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    scale = (min(patch.shape[:2]) / 34.0) ** 2
    best = 0.0
    for low, high, minimum_area, minimum_count in DOT_RANGES.get(underlying, ()):
        mask = cv2.inRange(hsv, low, high)
        if float(np.mean(mask > 0)) < 0.025:
            continue
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        count = int(np.count_nonzero((stats[1:, 4] >= max(2, minimum_area * scale))
                                     & (stats[1:, 4] < 50 * scale)))
        if count >= minimum_count:
            # The original accepts three components for these masks. Keep
            # that positive evidence above the reader's 0.6 dot threshold.
            best = max(best, min(1.0, max(0.6, count / max(6, minimum_count))))
    return best
