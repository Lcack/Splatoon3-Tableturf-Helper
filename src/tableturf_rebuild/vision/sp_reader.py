"""Read spendable Special Points from the battle HUD.

The slot coordinates and HSV masks are adapted from AutoTableTurf (GPL-3.0),
``tableturf/manager/detection/stage.py:sp``. An activated board special square
is not the SP balance: passing earns SP and Special Attacks spend SP. Keep this
observation separate from ``ActiveSpecialObservation``.

Coordinates assume an uncropped 16:9 capture. This reader has no temporal
state; callers should confirm consecutive observations before spending SP.
Confidence is a conservative image-quality score, not a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SpObservation:
    self_value: int | None
    opponent_value: int | None
    self_confidence: float
    opponent_confidence: float
    self_slot_ratios: tuple[float, ...]
    opponent_slot_ratios: tuple[float, ...]
    frame_brightness: float
    self_reason: str | None = None
    opponent_reason: str | None = None


class SpReader:
    REFERENCE_WIDTH = 1920
    REFERENCE_HEIGHT = 1080
    SLOT_SIZE = 15
    SELF_STARTS = ((987, 61), (987, 231), (987, 399))  # (row, column)
    OPPONENT_STARTS = ((64, 50), (64, 183), (64, 314))
    SELF_STEP = 32
    OPPONENT_STEP = 25
    SELF_HSV = ((10, 220, 220), (25, 255, 255))
    OPPONENT_HSV = ((80, 80, 220), (100, 255, 255))
    EMPTY_MAX_RATIO = 0.15
    FILLED_MIN_RATIO = 0.55
    MIN_FRAME_BRIGHTNESS = 200.0

    def read(self, frame: np.ndarray) -> SpObservation:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or frame.dtype != np.uint8):
            return self._unknown("INVALID_FRAME")
        height, width = frame.shape[:2]
        if height < 360 or abs(width / height - 16 / 9) > 0.02:
            return self._unknown("UNSUPPORTED_CAPTURE_GEOMETRY")
        # The Switch idle dimming affects the whole capture. A dark empty bar
        # must not become a confident zero merely because every color mask is 0.
        brightness = float(np.percentile(np.max(frame[::8, ::8], axis=2), 99.5))
        own = self._counter(frame, self.SELF_STARTS, self.SELF_STEP, self.SELF_HSV)
        opponent = self._counter(frame, self.OPPONENT_STARTS, self.OPPONENT_STEP,
                                 self.OPPONENT_HSV)
        if brightness < self.MIN_FRAME_BRIGHTNESS:
            return SpObservation(None, None, 0.0, 0.0, own[2], opponent[2],
                                 brightness, "FRAME_TOO_DARK", "FRAME_TOO_DARK")
        return SpObservation(own[0], opponent[0], own[1], opponent[1], own[2],
                             opponent[2], brightness, own[3], opponent[3])

    @classmethod
    def _counter(cls, frame, starts, step, bounds):
        height, width = frame.shape[:2]
        scale_x, scale_y = width / cls.REFERENCE_WIDTH, height / cls.REFERENCE_HEIGHT
        ratios = []
        values = []
        for row, col in starts:
            for offset in range(5):
                left = col + offset * step
                y1, y2 = round(row * scale_y), round((row + cls.SLOT_SIZE) * scale_y)
                x1, x2 = round(left * scale_x), round((left + cls.SLOT_SIZE) * scale_x)
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
                    return None, 0.0, tuple(ratios), "HUD_OUTSIDE_FRAME"
                hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
                ratios.append(float(np.mean(cv2.inRange(hsv, *bounds) > 0)))
                values.append(float(np.median(hsv[:, :, 2])))
        observed = tuple(ratios)
        if any(cls.EMPTY_MAX_RATIO < value < cls.FILLED_MIN_RATIO for value in ratios):
            return None, 0.0, observed, "AMBIGUOUS_SLOT_COLOR"
        filled = tuple(value >= cls.FILLED_MIN_RATIO for value in ratios)
        count = sum(filled)
        if filled != (True,) * count + (False,) * (len(filled) - count):
            return None, 0.0, observed, "NONCONTIGUOUS_SP_SLOTS"
        # The first two HUD slots have black interiors when unlit. Warning
        # banners and the result screen can cover these positions; their zero
        # color masks do not constitute evidence of a zero SP balance.
        if any(not filled[i] and values[i] > 20 for i in range(2)):
            return None, 0.0, observed, "HUD_NOT_VISIBLE"
        confidences = (
            min(1.0, 0.7 + 0.3 * (value - 0.4) / 0.4) if is_filled
            else 1.0 - 0.2 * value / cls.EMPTY_MAX_RATIO
            for value, is_filled in zip(ratios, filled)
        )
        return count, float(min(confidences)), observed, None

    @staticmethod
    def _unknown(reason: str) -> SpObservation:
        return SpObservation(None, None, 0.0, 0.0, (), (), 0.0, reason, reason)
