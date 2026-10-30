"""Read observed XP and single-confirm notification layouts.

Both layouts require their full local structure. The single-confirm reader
does not identify the reward's contents or infer a popup from an A glyph alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class RewardPageObservation:
    visible: bool
    confidence: float
    page_kind: str | None
    title_score: float
    reason: str | None = None


class RewardPageReader:
    TITLE_SEARCH_ROI = (630, 455, 917, 541)
    BADGE_ROI = (1170, 399, 1318, 539)
    BAR_ROI = (618, 520, 1308, 651)
    LEVEL_ROI = (705, 556, 801, 626)
    PROGRESS_ROI = (1005, 553, 1235, 618)
    TITLE_MINIMUM = .90
    MIN_BRIGHTNESS = 200.0
    NOTICE_TITLE_ROI = (600, 260, 1320, 365)
    NOTICE_BODY_ROI = (650, 680, 1290, 813)
    NOTICE_CONFIRM_ROI = (1343, 820, 1478, 913)
    NOTICE_PANEL_ROIS = (
        (425, 350, 675, 680), (1250, 400, 1490, 680),
        (650, 215, 850, 260), (500, 815, 1200, 865),
    )
    NOTICE_CONFIRM_MINIMUM = .94

    def __init__(self, template_path: Path | None = None):
        path = template_path or (
            Path(__file__).resolve().parents[3] / "assets/match_reward/level_title.npz"
        )
        self.templates = ()
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                self.templates = tuple(cv2.GaussianBlur(reference, (3, 3), .8)
                                       for reference in data["title"])
        notice_path = Path(__file__).resolve().parents[3] / "assets/match_reward/notice_confirm.npz"
        self.notice_templates = ()
        if notice_path.is_file():
            with np.load(notice_path, allow_pickle=False) as data:
                self.notice_templates = tuple(cv2.GaussianBlur(reference, (3, 3), .8)
                                              for reference in data["confirm"])

    def read(self, frame: np.ndarray) -> RewardPageObservation:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
            or frame.shape[2] != 3 or frame.dtype != np.uint8):
            return self._unknown("INVALID_FRAME")
        height, width = frame.shape[:2]
        if height < 360 or abs(width / height - 16 / 9) > .02:
            return self._unknown("UNSUPPORTED_CAPTURE_GEOMETRY")
        if not self.templates:
            return self._unknown("MISSING_LEVEL_TITLE_TEMPLATES")
        if np.percentile(np.max(frame[::8, ::8], axis=2), 99.5) < self.MIN_BRIGHTNESS:
            return self._unknown("FRAME_TOO_DARK")
        notice = self._read_single_confirm_notice(frame)
        if notice is not None:
            return notice
        title_patch = self._crop(frame, self.TITLE_SEARCH_ROI)
        search = cv2.GaussianBlur(cv2.cvtColor(title_patch, cv2.COLOR_BGR2GRAY), (3, 3), .8)
        title_score = max(float(cv2.matchTemplate(search, reference,
                         cv2.TM_CCOEFF_NORMED).max()) for reference in self.templates)
        if title_score < self.TITLE_MINIMUM:
            return self._unknown("LEVEL_TITLE_UNCERTAIN", title_score)
        # Correlation alone can match dim text. The sampled title is magenta;
        # the XP popup must retain its bright title and green points badge.
        title_hsv = cv2.cvtColor(title_patch, cv2.COLOR_BGR2HSV)
        magenta = ((title_hsv[:, :, 0] > 135) & (title_hsv[:, :, 0] < 170)
                   & (title_hsv[:, :, 1] > 100) & (title_hsv[:, :, 2] > 120))
        if float(magenta.mean()) < .18:
            return self._unknown("LEVEL_TITLE_TOO_DIM", title_score)
        badge = cv2.cvtColor(self._crop(frame, self.BADGE_ROI), cv2.COLOR_BGR2HSV)
        green = ((badge[:, :, 0] > 40) & (badge[:, :, 0] < 100)
                 & (badge[:, :, 1] > 90) & (badge[:, :, 2] > 100))
        if float(green.mean()) < .20 or self._white_fraction(badge) < .025:
            return self._unknown("POINTS_BADGE_INCOMPLETE", title_score)
        bar = cv2.cvtColor(self._crop(frame, self.BAR_ROI), cv2.COLOR_BGR2HSV)
        if float((bar[:, :, 2] < 60).mean()) < .30:
            return self._unknown("XP_BAR_INCOMPLETE", title_score)
        level = cv2.cvtColor(self._crop(frame, self.LEVEL_ROI), cv2.COLOR_BGR2HSV)
        progress = cv2.cvtColor(self._crop(frame, self.PROGRESS_ROI), cv2.COLOR_BGR2HSV)
        if self._white_fraction(level) < .09 or self._white_fraction(progress) < .10:
            return self._unknown("XP_NUMBERS_INCOMPLETE", title_score)
        return RewardPageObservation(True, title_score, "LEVEL_XP", title_score)

    def _read_single_confirm_notice(self, frame: np.ndarray) -> RewardPageObservation | None:
        if not self.notice_templates:
            return None
        # Check separated areas of the actual large panel. XP/replay dialogs
        # use a lighter, smaller panel and cannot pass this shell check.
        dark_fractions = []
        for bounds in self.NOTICE_PANEL_ROIS:
            patch = self._crop(frame, bounds)
            dark_fractions.append(float((np.max(patch, axis=2) < 65).mean()))
        if min(dark_fractions) < .94:
            return None
        title_hsv = cv2.cvtColor(self._crop(frame, self.NOTICE_TITLE_ROI), cv2.COLOR_BGR2HSV)
        title = ((title_hsv[:, :, 1] < 70) & (title_hsv[:, :, 2] > 205)).astype(np.uint8)
        rows = np.flatnonzero(np.count_nonzero(title, axis=1) > 2)
        columns = np.flatnonzero(np.count_nonzero(title, axis=0) > 2)
        if not rows.size or not columns.size:
            return None
        title_height = int(rows[-1] - rows[0] + 1)
        title_width = int(columns[-1] - columns[0] + 1)
        glyphs = cv2.connectedComponentsWithStats(title)[2][1:]
        glyphs = glyphs[(glyphs[:, 4] >= 20) & (glyphs[:, 3] >= 16)
                        & (glyphs[:, 3] <= 70)]
        density = float(title[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1].mean())
        centered = 1 - abs(float(columns[0] + columns[-1]) / 2 - title.shape[1] / 2) / 80
        # This is a spatial text score, not an OCR claim about its contents.
        # A filled white rectangle, a lone icon, or a partial title is rejected.
        if (not 30 <= title_height <= 75 or not 120 <= title_width <= 680
            or not 4 <= len(glyphs) <= 48 or not .18 <= density <= .75 or centered < .90):
            return None
        title_score = min(1.0, centered)
        body = cv2.cvtColor(self._crop(frame, self.NOTICE_BODY_ROI), cv2.COLOR_BGR2HSV)
        if self._white_fraction(body) < .012:
            return self._unknown("NOTICE_BODY_INCOMPLETE", title_score)
        search = cv2.GaussianBlur(
            cv2.cvtColor(self._crop(frame, self.NOTICE_CONFIRM_ROI), cv2.COLOR_BGR2GRAY),
            (3, 3), .8,
        )
        confirm_score = max(float(cv2.matchTemplate(search, reference,
                           cv2.TM_CCOEFF_NORMED).max()) for reference in self.notice_templates)
        if confirm_score < self.NOTICE_CONFIRM_MINIMUM:
            return self._unknown("NOTICE_CONFIRM_UNCERTAIN", title_score)
        confidence = min(title_score, confirm_score, *dark_fractions)
        return RewardPageObservation(True, confidence, "SINGLE_CONFIRM_NOTICE", title_score)

    @staticmethod
    def _white_fraction(hsv):
        return float(((hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 190)).mean())

    @staticmethod
    def _crop(frame: np.ndarray, bounds):
        height, width = frame.shape[:2]
        left, top, right, bottom = bounds
        patch = frame[round(top * height / 1080):round(bottom * height / 1080),
                      round(left * width / 1920):round(right * width / 1920)]
        return cv2.resize(patch, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _unknown(reason, title_score=0.0):
        return RewardPageObservation(False, 0.0, None, title_score, reason)
