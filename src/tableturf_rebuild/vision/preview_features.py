from __future__ import annotations

import cv2
import numpy as np


PREVIEW_CANNY_THRESHOLD = (50, 80)
PREVIEW_DIAGONAL_RATIO = 0.80
PREVIEW_INK_RATIO = 0.15


def preview_diagonal_score(patch: np.ndarray) -> float:
    gray = (
        cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if patch.ndim == 3
        else patch
    )
    edges = cv2.Canny(gray, *PREVIEW_CANNY_THRESHOLD)
    edges = cv2.dilate(
        edges,
        kernel=np.ones((2, 2), dtype=np.uint8),
    )
    edges = np.rot90(edges)
    half = edges.shape[1] // 2
    diagonal_scores = [
        float(np.mean(np.diag(edges, offset) == 255))
        for offset in range(-half, half)
    ]
    if not diagonal_scores:
        return 0.0
    return float(
        sum(
            score > PREVIEW_DIAGONAL_RATIO
            for score in diagonal_scores
        )
        / len(diagonal_scores)
    )


def is_diagonal_preview_patch(patch: np.ndarray) -> bool:
    return preview_diagonal_score(patch) > PREVIEW_INK_RATIO
