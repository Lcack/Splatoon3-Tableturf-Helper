"""Read the Special Attack toggle from its button label and background.

Button coordinates and HSV ranges originate from AutoTableTurf (GPL-3.0),
``tableturf/manager/detection/ui.py:special_on``. Its dark color mask alone
also matches the ordinary disabled button in the current capture. Distinguish
the actual Chinese ON/OFF labels before declaring a mode, including while the
ON button is dimmed during placement. This does not infer mode from SP.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class BattleModeObservation:
    special_mode: bool | None
    confidence: float
    on_score: float
    off_score: float
    yellow_fraction: float
    dark_fraction: float
    ordinary_blue_fraction: float
    gray_fraction: float
    frame_brightness: float
    reason: str | None = None


class BattleModeReader:
    BUTTON_ROI = (295, 840, 495, 890)
    TEXT_SEARCH_ROI = (330, 847, 410, 897)
    TEXT_MINIMUM = 0.88
    TEXT_MARGIN = 0.15
    TEXT_TEMPLATE_SCALES = (1.0, 1.05, 1.10)
    MIN_FRAME_BRIGHTNESS = 200.0

    def __init__(self, template_path: Path | None = None):
        path = template_path or Path(__file__).resolve().parents[3] / "assets/battle_mode/button_text.npz"
        self.templates = {}
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                # Focusing the bottom toggle enlarges the button and its label
                # by about 10% (confirmed in special-button-stall/analysis.png).
                # Cover that animation range instead of lowering the text or
                # background confidence thresholds for an unmatched glyph.
                self.templates = {
                    key: tuple(cv2.GaussianBlur(
                        reference if scale == 1.0 else cv2.resize(reference, None, fx=scale, fy=scale),
                        (3, 3), .8)
                        for reference in data[key] for scale in self.TEXT_TEMPLATE_SCALES)
                    for key in ("on", "off")
                }

    def read(self, frame: np.ndarray) -> BattleModeObservation:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
            or frame.shape[2] != 3 or frame.dtype != np.uint8):
            return self._unknown("INVALID_FRAME")
        height, width = frame.shape[:2]
        if height < 360 or abs(width / height - 16 / 9) > 0.02:
            return self._unknown("UNSUPPORTED_CAPTURE_GEOMETRY")
        brightness = float(np.percentile(np.max(frame[::8, ::8], axis=2), 99.5))
        if brightness < self.MIN_FRAME_BRIGHTNESS:
            return self._unknown("FRAME_TOO_DARK", brightness)
        if not self.templates:
            return self._unknown("MISSING_BUTTON_TEXT_TEMPLATES", brightness)
        button = self._crop(frame, self.BUTTON_ROI)
        hsv = cv2.cvtColor(button, cv2.COLOR_BGR2HSV)
        yellow = float(np.mean(cv2.inRange(hsv, (20, 150, 200), (40, 255, 240)) > 0))
        dark = float(np.mean(cv2.inRange(hsv, (70, 0, 100), (115, 50, 150)) > 0))
        blue = float(np.mean((hsv[:, :, 0] >= 110) & (hsv[:, :, 0] <= 145)
                             & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 80)))
        gray = float(np.mean((hsv[:, :, 1] < 70) & (hsv[:, :, 2] >= 65)
                             & (hsv[:, :, 2] <= 200)))
        search = cv2.cvtColor(self._crop(frame, self.TEXT_SEARCH_ROI), cv2.COLOR_BGR2GRAY)
        # A small common blur preserves glyph strokes when a 720p/540p capture
        # is normalized to the reference coordinates.
        search = cv2.GaussianBlur(search, (3, 3), .8)
        scores = {key: max(float(cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED).max())
                           for reference in references)
                  for key, references in self.templates.items()}
        on_score, off_score = scores["on"], scores["off"]
        special_mode = None
        reason = "BUTTON_LABEL_UNCERTAIN"
        if on_score >= self.TEXT_MINIMUM and on_score - off_score >= self.TEXT_MARGIN:
            if yellow >= 0.25 or gray >= 0.6:
                special_mode, reason = True, None
            else:
                reason = "ON_LABEL_WITHOUT_BUTTON_BACKGROUND"
        elif off_score >= self.TEXT_MINIMUM and off_score - on_score >= self.TEXT_MARGIN:
            if blue >= 0.5 or gray >= 0.6:
                special_mode, reason = False, None
            else:
                reason = "OFF_LABEL_WITHOUT_BUTTON_BACKGROUND"
        confidence = max(on_score, off_score) if special_mode is not None else 0.0
        return BattleModeObservation(special_mode, confidence, on_score, off_score,
                                     yellow, dark, blue, gray, brightness, reason)

    @staticmethod
    def _crop(frame: np.ndarray, bounds):
        height, width = frame.shape[:2]
        left, top, right, bottom = bounds
        patch = frame[round(top * height / 1080):round(bottom * height / 1080),
                      round(left * width / 1920):round(right * width / 1920)]
        return cv2.resize(patch, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _unknown(reason, brightness=0.0):
        return BattleModeObservation(None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     brightness, reason)
