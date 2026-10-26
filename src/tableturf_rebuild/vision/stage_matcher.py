from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tableturf_rebuild.vision.stage_assets import StageAssetIndex


@dataclass(frozen=True, slots=True)
class StageMatch:
    stage_id: str
    score: float
    inliers: int
    good_matches: int


@dataclass(slots=True)
class _StageDescriptors:
    keypoints: tuple
    descriptors: np.ndarray | None
    shape_mask: np.ndarray


class StagePreviewMatcher:
    def __init__(self, assets: StageAssetIndex, max_features: int = 600):
        self.assets = assets
        self._sift = cv2.SIFT_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._cache: dict[str, _StageDescriptors] = {}

    def match(self, preview_image: np.ndarray, top_k: int = 3) -> tuple[StageMatch, ...]:
        target = self._prepare(preview_image)
        target_shape = self._shape_mask(preview_image)
        target_keypoints, target_descriptors = self._sift.detectAndCompute(
            target, None
        )
        if target_descriptors is None or len(target_keypoints) < 4:
            return ()
        results = []
        for stage_id in self.assets.stages:
            reference = self._reference(stage_id)
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
                if first.distance < 0.72 * second.distance
            ]
            inliers = 0
            if len(good) >= 4:
                source = np.float32(
                    [target_keypoints[item.queryIdx].pt for item in good]
                ).reshape(-1, 1, 2)
                destination = np.float32(
                    [reference.keypoints[item.trainIdx].pt for item in good]
                ).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
                inliers = int(mask.sum()) if mask is not None else 0
            denominator = max(
                10,
                min(len(target_keypoints), len(reference.keypoints)),
            )
            feature_score = 0.72 * (inliers / denominator) + 0.28 * (
                len(good) / denominator
            )
            intersection = np.logical_and(target_shape, reference.shape_mask).sum()
            union = np.logical_or(target_shape, reference.shape_mask).sum()
            shape_score = float(intersection / union) if union else 0.0
            score = 0.35 * feature_score + 0.65 * shape_score
            results.append(
                StageMatch(stage_id, score, inliers, len(good))
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return tuple(results[:top_k])

    def _reference(self, stage_id: str) -> _StageDescriptors:
        cached = self._cache.get(stage_id)
        if cached is not None:
            return cached
        image = cv2.imread(
            str(self.assets.stages[stage_id].path),
            cv2.IMREAD_UNCHANGED,
        )
        if image is None:
            raise RuntimeError(f"cannot read stage preview: {stage_id}")
        image, mask = self._crop_alpha(image)
        prepared = self._prepare(image)
        prepared_mask = (
            cv2.resize(
                mask,
                (prepared.shape[1], prepared.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if mask is not None
            else None
        )
        keypoints, descriptors = self._sift.detectAndCompute(
            prepared,
            prepared_mask,
        )
        value = _StageDescriptors(
            tuple(keypoints),
            descriptors,
            self._shape_from_alpha(mask, image.shape),
        )
        self._cache[stage_id] = value
        return value

    @staticmethod
    def _crop_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            ys, xs = np.where(alpha > 16)
            if len(xs):
                x1, x2 = xs.min(), xs.max() + 1
                y1, y2 = ys.min(), ys.max() + 1
                return image[y1:y2, x1:x2, :3], alpha[y1:y2, x1:x2]
        return image[:, :, :3], None

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image
        )
        if gray.dtype != np.uint8:
            gray = cv2.convertScaleAbs(gray)
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    @classmethod
    def _shape_mask(cls, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 4:
            return cls._shape_from_alpha(image[:, :, 3], image.shape[:2])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mask = (gray > 12).astype(np.uint8) * 255
        ys, xs = np.where(mask > 0)
        if not len(xs):
            return np.zeros((128, 128), dtype=bool)
        crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        return cls._fit_mask(crop)

    @classmethod
    def _shape_from_alpha(cls, alpha: np.ndarray | None, shape) -> np.ndarray:
        if alpha is None:
            return np.ones((128, 128), dtype=bool)
        mask = (alpha > 16).astype(np.uint8) * 255
        ys, xs = np.where(mask > 0)
        if not len(xs):
            return np.zeros((128, 128), dtype=bool)
        crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        return cls._fit_mask(crop)

    @staticmethod
    def _fit_mask(mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        scale = min(128 / width, 128 / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(
            mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        output = np.zeros((128, 128), dtype=np.uint8)
        x = (128 - resized_width) // 2
        y = (128 - resized_height) // 2
        output[y : y + resized_height, x : x + resized_width] = resized
        return output > 0


def validate_expected_stage(
    expected_stage_id: str,
    observed_matches: tuple[StageMatch, ...],
    minimum_score: float = 0.10,
) -> tuple[bool, str]:
    if not observed_matches:
        return False, "没有识别到地图预览"
    best = observed_matches[0]
    if best.score < minimum_score:
        return False, f"地图匹配置信度不足: {best.score:.3f}"
    if best.stage_id != expected_stage_id:
        return (
            False,
            f"地图不匹配: 期望 {expected_stage_id}, 识别 {best.stage_id}",
        )
    return True, f"地图匹配: {expected_stage_id}"

