from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.vision.card_assets import CardAssetIndex


@dataclass(frozen=True, slots=True)
class CardMatch:
    card_id: str
    score: float
    inliers: int
    good_matches: int


@dataclass(slots=True)
class _Descriptors:
    image: np.ndarray
    keypoints: tuple
    descriptors: np.ndarray | None


def rectify_card(
    frame: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] = (512, 714),
) -> np.ndarray:
    source = np.asarray(corners, dtype=np.float32)
    if source.shape != (4, 2):
        raise ValueError("card corners must have shape (4, 2)")
    width, height = output_size
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(frame, transform, (width, height))


class CardArtMatcher:
    """Match a rectified in-game card against extracted card artwork.

    Candidate filtering from the known deck should happen before calling this
    matcher. Matching a few cards is fast; matching all 266 is intended for
    offline diagnostics only.
    """

    def __init__(
        self,
        assets: CardAssetIndex,
        max_features: int = 700,
        ratio: float = 0.72,
    ):
        self.assets = assets
        self.ratio = ratio
        self._sift = cv2.SIFT_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._cache: dict[str, _Descriptors] = {}

    def match(
        self,
        card_image: np.ndarray,
        candidate_ids: list[str] | tuple[str, ...],
        top_k: int = 5,
    ) -> tuple[CardMatch, ...]:
        target = self._apply_clahe(card_image)
        target_keypoints, target_descriptors = self._sift.detectAndCompute(
            target, None
        )
        if target_descriptors is None or len(target_keypoints) < 4:
            return ()

        scored = []
        for card_id in candidate_ids:
            reference = self._reference(card_id)
            if reference.descriptors is None:
                continue
            pairs = self._matcher.knnMatch(
                target_descriptors,
                reference.descriptors,
                k=2,
            )
            good = [
                first
                for pair in pairs
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < self.ratio * second.distance
            ]
            inliers = self._homography_inliers(
                target_keypoints,
                reference.keypoints,
                good,
            )
            denominator = max(12, min(len(target_keypoints), len(reference.keypoints)))
            score = 0.72 * (inliers / denominator) + 0.28 * (
                len(good) / denominator
            )
            scored.append(
                CardMatch(
                    card_id=card_id,
                    score=score,
                    inliers=inliers,
                    good_matches=len(good),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return tuple(scored[:top_k])

    def _reference(self, card_id: str) -> _Descriptors:
        cached = self._cache.get(card_id)
        if cached is not None:
            return cached
        image = cv2.imread(
            str(self.assets.face(card_id).path),
            cv2.IMREAD_UNCHANGED,
        )
        if image is None:
            raise RuntimeError(f"cannot read card asset: {card_id}")
        image, mask = self._crop_alpha(image)
        image = self._apply_clahe(image)
        keypoints, descriptors = self._sift.detectAndCompute(image, mask)
        value = _Descriptors(image, tuple(keypoints), descriptors)
        self._cache[card_id] = value
        return value

    @staticmethod
    def _crop_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            ys, xs = np.where(alpha > 16)
            if len(xs):
                x1, x2 = xs.min(), xs.max() + 1
                y1, y2 = ys.min(), ys.max() + 1
                cropped = image[y1:y2, x1:x2, :3]
                mask = alpha[y1:y2, x1:x2]
                return cropped, mask
        return image[:, :, :3], None

    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if gray.dtype != np.uint8:
            gray = cv2.convertScaleAbs(gray)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    @staticmethod
    def _homography_inliers(target_keypoints, reference_keypoints, good) -> int:
        if len(good) < 4:
            return 0
        source = np.float32(
            [target_keypoints[match.queryIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        target = np.float32(
            [reference_keypoints[match.trainIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        return int(mask.sum()) if mask is not None else 0

