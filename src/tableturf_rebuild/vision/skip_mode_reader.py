"""Read the skip dialog from the actual Chinese button label.

The yellow squid identifies focus separately. Only the full ``返回`` label
acknowledges the discard dialog; ``跳过`` means it has not opened. Button
color is a generic visibility check, never evidence that a key took effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SkipModeObservation:
    discard_mode: bool | None
    confidence: float
    discard_score: float
    normal_score: float
    frame_brightness: float
    reason: str | None = None


class SkipModeReader:
    BUTTON_ROI = (55, 850, 245, 895)
    TEXT_SEARCH_ROI = (108, 840, 202, 912)
    TEXT_MINIMUM = 0.88
    TEXT_MARGIN = 0.15
    TEXT_TEMPLATE_SCALES = (1.0, 1.05, 1.10)
    MIN_FRAME_BRIGHTNESS = 200.0

    def __init__(self, template_path: Path | None = None):
        path = template_path or Path(__file__).resolve().parents[3] / "assets/battle_skip/button_text.npz"
        self.templates = {}
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                self.templates = {
                    key: tuple(cv2.GaussianBlur(
                        reference if scale == 1.0 else cv2.resize(reference, None, fx=scale, fy=scale),
                        (3, 3), .8)
                        for reference in data[key] for scale in self.TEXT_TEMPLATE_SCALES)
                    for key in ("discard", "normal")
                }

    def read(self, frame: np.ndarray) -> SkipModeObservation:
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
        search = cv2.GaussianBlur(
            cv2.cvtColor(self._crop(frame, self.TEXT_SEARCH_ROI), cv2.COLOR_BGR2GRAY),
            (3, 3), .8,
        )
        scores = {
            key: max(float(cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED).max())
                     for reference in references)
            for key, references in self.templates.items()
        }
        discard_score, normal_score = scores["discard"], scores["normal"]
        best = max(discard_score, normal_score)
        if best < self.TEXT_MINIMUM or abs(discard_score - normal_score) < self.TEXT_MARGIN:
            return SkipModeObservation(None, 0.0, discard_score, normal_score, brightness,
                                       "BUTTON_LABEL_UNCERTAIN")
        # All three observed button backgrounds are valid. Their colors do not
        # decide which mode is active and do not stand in for a focus cursor.
        hsv = cv2.cvtColor(self._crop(frame, self.BUTTON_ROI), cv2.COLOR_BGR2HSV)
        blue = ((hsv[:, :, 0] >= 110) & (hsv[:, :, 0] <= 145)
                & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 80))
        yellow = ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 40)
                  & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 150))
        gray = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] >= 65) & (hsv[:, :, 2] <= 200))
        if float(np.mean(blue | yellow | gray)) < 0.6:
            return SkipModeObservation(None, 0.0, discard_score, normal_score, brightness,
                                       "LABEL_WITHOUT_BUTTON_BACKGROUND")
        return SkipModeObservation(discard_score > normal_score, best, discard_score,
                                   normal_score, brightness)

    @staticmethod
    def _crop(frame: np.ndarray, bounds):
        height, width = frame.shape[:2]
        left, top, right, bottom = bounds
        patch = frame[round(top * height / 1080):round(bottom * height / 1080),
                      round(left * width / 1920):round(right * width / 1920)]
        return cv2.resize(patch, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _unknown(reason, brightness=0.0):
        return SkipModeObservation(None, 0.0, 0.0, 0.0, brightness, reason)
