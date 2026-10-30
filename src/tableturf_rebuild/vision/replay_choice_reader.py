"""Recognize the replay dialog from its three Chinese text labels.

Only the title and both buttons are matched. NPCs, result banners, ink,
button animation and the moving squid outside these labels are irrelevant.
This reader establishes the dialog's presence; it does not infer its focus.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ReplayChoiceObservation:
    visible: bool
    confidence: float
    title_score: float
    cancel_score: float
    continue_score: float
    reason: str | None = None


class ReplayChoiceReader:
    # Coordinates refer to an uncropped 1920 x 1080 capture.
    TEXT_SEARCH_ROIS = {
        "title": (828, 479, 1095, 555),
        "cancel": (745, 766, 842, 843),
        "continue": (1087, 760, 1195, 839),
    }
    TITLE_MINIMUM = 0.90
    BUTTON_MINIMUM = 0.88
    MIN_BRIGHTNESS = 200.0

    def __init__(self, template_path: Path | None = None):
        path = template_path or (
            Path(__file__).resolve().parents[3] / "assets/replay_choice/prompt_text.npz"
        )
        self.templates = {}
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                self.templates = {
                    key: cv2.GaussianBlur(data[key], (3, 3), .8)
                    for key in self.TEXT_SEARCH_ROIS
                }

    def read(self, frame: np.ndarray) -> ReplayChoiceObservation:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
            or frame.shape[2] != 3 or frame.dtype != np.uint8):
            return self._unknown("INVALID_FRAME")
        height, width = frame.shape[:2]
        if height < 360 or abs(width / height - 16 / 9) > .02:
            return self._unknown("UNSUPPORTED_CAPTURE_GEOMETRY")
        if not self.templates:
            return self._unknown("MISSING_DIALOG_TEXT_TEMPLATES")
        # Correlation itself is invariant to dimming. Do not acknowledge the
        # dialog during a dark transition just because its text remains faint.
        brightness = np.percentile(np.max(frame[::8, ::8], axis=2), 99.5)
        if brightness < self.MIN_BRIGHTNESS:
            return self._unknown("FRAME_TOO_DARK")
        scores = {}
        for key, bounds in self.TEXT_SEARCH_ROIS.items():
            search = cv2.GaussianBlur(
                cv2.cvtColor(self._crop(frame, bounds), cv2.COLOR_BGR2GRAY),
                (3, 3), .8,
            )
            if key == "title" and np.percentile(search, 97) < self.MIN_BRIGHTNESS:
                return self._unknown("TITLE_TOO_DARK")
            matches = cv2.matchTemplate(search, self.templates[key], cv2.TM_CCOEFF_NORMED)
            # Buttons switch between white-on-purple and dark-on-yellow. Their
            # text shape must match in either polarity; title polarity is fixed.
            scores[key] = float((matches if key == "title" else np.abs(matches)).max())
        title, cancel, continue_ = (scores[key] for key in self.TEXT_SEARCH_ROIS)
        if title < self.TITLE_MINIMUM:
            reason = "TITLE_UNCERTAIN"
        elif min(cancel, continue_) < self.BUTTON_MINIMUM:
            reason = "BUTTON_LABELS_UNCERTAIN"
        else:
            reason = None
        visible = reason is None
        return ReplayChoiceObservation(visible, min(scores.values()) if visible else 0.0,
                                       title, cancel, continue_, reason)

    @staticmethod
    def _crop(frame: np.ndarray, bounds):
        height, width = frame.shape[:2]
        left, top, right, bottom = bounds
        patch = frame[round(top * height / 1080):round(bottom * height / 1080),
                      round(left * width / 1920):round(right * width / 1920)]
        return cv2.resize(patch, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _unknown(reason):
        return ReplayChoiceObservation(False, 0.0, 0.0, 0.0, 0.0, reason)
