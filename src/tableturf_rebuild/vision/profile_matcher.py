from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.vision.calibration import VisionProfile, VisionRegion, warp_region


@dataclass(frozen=True, slots=True)
class ProfileMatchScore:
    key: str
    score: float
    profile: str
    region_count: int


@dataclass(frozen=True, slots=True)
class _RegionDescriptor:
    region: VisionRegion
    gray: np.ndarray
    edges: np.ndarray
    weight: float


@dataclass(frozen=True, slots=True)
class _ProfileReference:
    key: str
    profile_name: str
    descriptors: tuple[_RegionDescriptor, ...]

    def score(self, frame: np.ndarray) -> float:
        if not self.descriptors:
            return 0.0
        values = []
        for reference in self.descriptors:
            gray, edges = descriptor_features(
                warp_region(frame, reference.region, 48, 48)
            )
            gray_score = (float(np.dot(gray, reference.gray)) + 1.0) / 2.0
            edge_score = (float(np.dot(edges, reference.edges)) + 1.0) / 2.0
            values.append(
                reference.weight * (0.72 * gray_score + 0.28 * edge_score)
            )
        total_weight = sum(item.weight for item in self.descriptors)
        return float(sum(values) / total_weight) if total_weight else 0.0


class VisionProfileMatcher:
    def __init__(
        self,
        profile_paths: dict[str, tuple[Path, ...]],
    ):
        self.references = tuple(
            reference
            for key, paths in profile_paths.items()
            for path in paths
            if (
                reference := self._load_reference(
                    key,
                    path,
                )
            )
            is not None
        )

    def score(
        self,
        frame: np.ndarray,
        keys: set[str] | None = None,
    ) -> tuple[ProfileMatchScore, ...]:
        best_by_key: dict[str, ProfileMatchScore] = {}
        for reference in self.references:
            if keys is not None and reference.key not in keys:
                continue
            value = reference.score(frame)
            current = best_by_key.get(reference.key)
            if current is None or value > current.score:
                best_by_key[reference.key] = ProfileMatchScore(
                    key=reference.key,
                    score=value,
                    profile=reference.profile_name,
                    region_count=len(reference.descriptors),
                )
        return tuple(
            sorted(
                best_by_key.values(),
                key=lambda item: item.score,
                reverse=True,
            )
        )

    @staticmethod
    def _load_reference(
        key: str,
        profile_path: Path,
    ) -> _ProfileReference | None:
        if not profile_path.exists():
            return None
        profile = VisionProfile.load(profile_path)
        source_text = profile.notes.removeprefix("source=").strip()
        if not source_text:
            return None
        source_path = Path(source_text)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        descriptors = []
        for region in profile.regions:
            gray, edges = descriptor_features(
                warp_region(image, region, 48, 48)
            )
            descriptors.append(
                _RegionDescriptor(
                    region=region,
                    gray=gray,
                    edges=edges,
                    weight=_region_weight(region),
                )
            )
        return _ProfileReference(
            key=key,
            profile_name=profile_path.name,
            descriptors=tuple(descriptors),
        )


def descriptor_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    edges = cv2.Canny(gray.astype(np.uint8), 40, 100).astype(np.float32)
    return _normalize(gray), _normalize(edges)


def _normalize(value: np.ndarray) -> np.ndarray:
    flat = value.reshape(-1).astype(np.float64)
    flat -= flat.mean()
    length = float(np.linalg.norm(flat))
    if length <= 1e-9:
        return np.zeros_like(flat)
    return flat / length


def _region_weight(region: VisionRegion) -> float:
    xs = [point[0] for point in region.points]
    ys = [point[1] for point in region.points]
    area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
    return max(0.25, min(2.0, area * 40.0))
