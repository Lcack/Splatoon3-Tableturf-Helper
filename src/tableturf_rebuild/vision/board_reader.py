from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.grid import PLAYABLE_CELLS, Cell
from tableturf_rebuild.domain.stage import SpecialPoint, StageSpec
from tableturf_rebuild.vision.calibration import (
    VisionProfile,
    VisionRegion,
    warp_region,
)
from tableturf_rebuild.vision.preview_features import (
    PREVIEW_INK_RATIO,
    preview_diagonal_score,
)


@dataclass(frozen=True, slots=True)
class BoardColorConfig:
    # Hue ranges are derived from assets/reference/stage/semantic_manifest.json:
    # self ink 32, self special 19, opponent ink 117, opponent special 92.
    self_ink_hue_range: tuple[int, int] = (27, 55)
    self_special_hue_range: tuple[int, int] = (0, 27)
    opponent_ink_hue_range: tuple[int, int] = (105, 165)
    opponent_special_hue_range: tuple[int, int] = (75, 105)
    minimum_saturation: int = 70
    minimum_value: int = 55
    minimum_ink_fraction: float = 0.20
    change_threshold: float = 0.12


@dataclass(frozen=True, slots=True)
class BoardCellReading:
    row: int
    col: int
    stage_cell: Cell
    inferred_cell: Cell
    mean_bgr: tuple[int, int, int]
    median_hsv: tuple[int, int, int]
    self_fraction: float
    opponent_fraction: float
    self_special_fraction: float
    opponent_special_fraction: float
    self_special_pattern_delta: float
    opponent_special_pattern_delta: float
    center_brightness_contrast: float
    center_saturation: float
    special_active: bool
    color_change: float
    edge_density: float
    preview_diagonal_score: float = 0.0


@dataclass(frozen=True, slots=True)
class BoardObservation:
    stage_id: str
    rows: int
    cols: int
    cells: tuple[tuple[Cell, ...], ...]
    readings: tuple[BoardCellReading, ...]
    confidence: float
    placement_preview_score: float
    placement_preview_present: bool
    playable_cells: int
    reference_available: bool = False

    @property
    def self_count(self) -> int:
        return sum(
            cell in {Cell.SELF_INK, Cell.SELF_SPECIAL}
            for row in self.cells
            for cell in row
        )

    @property
    def opponent_count(self) -> int:
        return sum(
            cell in {Cell.OPPONENT_INK, Cell.OPPONENT_SPECIAL}
            for row in self.cells
            for cell in row
        )

    @property
    def self_special_count(self) -> int:
        return sum(
            cell is Cell.SELF_SPECIAL
            for row in self.cells
            for cell in row
        )

    @property
    def opponent_special_count(self) -> int:
        return sum(
            cell is Cell.OPPONENT_SPECIAL
            for row in self.cells
            for cell in row
        )

    @property
    def self_active_special_count(self) -> int:
        return sum(
            reading.special_active
            and reading.inferred_cell is Cell.SELF_SPECIAL
            for reading in self.readings
        )

    @property
    def opponent_active_special_count(self) -> int:
        return sum(
            reading.special_active
            and reading.inferred_cell is Cell.OPPONENT_SPECIAL
            for reading in self.readings
        )

    @property
    def empty_count(self) -> int:
        return max(
            0,
            self.playable_cells - self.self_count - self.opponent_count,
        )

    @property
    def initial_board_ready(self) -> bool:
        return (
            self.playable_cells > 0
            and self.self_count + self.opponent_count <= 4
        )


@dataclass(frozen=True, slots=True)
class ActiveSpecialObservation:
    self_cells: frozenset[tuple[int, int]]
    opponent_cells: frozenset[tuple[int, int]]
    self_special_cells: frozenset[tuple[int, int]] = frozenset()
    opponent_special_cells: frozenset[tuple[int, int]] = frozenset()

    @property
    def self_count(self) -> int:
        return len(self.self_cells)

    @property
    def opponent_count(self) -> int:
        return len(self.opponent_cells)

    @property
    def self_special_count(self) -> int:
        return len(self.self_special_cells)

    @property
    def opponent_special_count(self) -> int:
        return len(self.opponent_special_cells)


class SpecialActivityTracker:
    ENTER_SCORE = 20.0
    EXIT_MISSES = 30
    MISS_DECAY = 0.25

    def __init__(self) -> None:
        self._scores: dict[tuple[Cell, int, int], float] = {}
        self._active: set[tuple[Cell, int, int]] = set()
        self._misses: dict[tuple[Cell, int, int], int] = {}

    def reset(self) -> None:
        self._scores.clear()
        self._active.clear()
        self._misses.clear()

    def update(self, observation: BoardObservation) -> ActiveSpecialObservation:
        specials = {
            (reading.inferred_cell, reading.row, reading.col)
            for reading in observation.readings
            if reading.inferred_cell
            in {Cell.SELF_SPECIAL, Cell.OPPONENT_SPECIAL}
        }
        candidates = {
            (reading.inferred_cell, reading.row, reading.col)
            for reading in observation.readings
            if reading.special_active
        }
        for key in tuple(self._scores):
            if key not in specials:
                self._scores[key] = max(
                    0.0,
                    self._scores[key] - self.MISS_DECAY,
                )
                if key in self._active:
                    misses = self._misses.get(key, 0) + 1
                    self._misses[key] = misses
                    if misses >= self.EXIT_MISSES:
                        self._active.discard(key)
                        self._misses.pop(key, None)
                        del self._scores[key]
                else:
                    del self._scores[key]
        for key in specials:
            score = self._scores.get(key, 0.0)
            score = min(30.0, score + 1.0) if key in candidates else max(
                0.0,
                score - self.MISS_DECAY,
            )
            self._scores[key] = score
            if key not in self._active and score >= self.ENTER_SCORE:
                self._active.add(key)
                self._misses[key] = 0
            elif key in self._active:
                if key in candidates:
                    self._misses[key] = 0
                else:
                    misses = self._misses.get(key, 0) + 1
                    self._misses[key] = misses
                    if misses >= self.EXIT_MISSES:
                        self._active.discard(key)
        return ActiveSpecialObservation(
            self_cells=frozenset(
                (row, col)
                for owner, row, col in self._active
                if owner is Cell.SELF_SPECIAL
            ),
            opponent_cells=frozenset(
                (row, col)
                for owner, row, col in self._active
                if owner is Cell.OPPONENT_SPECIAL
            ),
            self_special_cells=frozenset(
                {
                    (row, col)
                    for owner, row, col in specials
                    if owner is Cell.SELF_SPECIAL
                }
                | {
                    (row, col)
                    for owner, row, col in self._active
                    if owner is Cell.SELF_SPECIAL
                }
            ),
            opponent_special_cells=frozenset(
                {
                    (row, col)
                    for owner, row, col in specials
                    if owner is Cell.OPPONENT_SPECIAL
                }
                | {
                    (row, col)
                    for owner, row, col in self._active
                    if owner is Cell.OPPONENT_SPECIAL
                }
            ),
        )


class BoardGridReader:
    CELL_SIZE = 32

    def __init__(
        self,
        catalog: GameCatalog,
        config: BoardColorConfig | None = None,
    ):
        self.catalog = catalog
        self.config = config or BoardColorConfig()
        self.special_tiles = _load_special_tile_templates(
            Path(__file__).resolve().parents[3]
            / "assets"
            / "reference"
            / "stage"
        )

    def read(
        self,
        frame: np.ndarray,
        region: VisionRegion,
        stage_id: str,
        reference_frame: np.ndarray | None = None,
        profile: VisionProfile | None = None,
    ) -> BoardObservation:
        stage = self.catalog.stages[stage_id]
        if profile is not None and profile.stage_id == stage_id:
            if (
                region.grid_rows > 0
                and region.grid_cols > 0
                and (
                    region.grid_rows != stage.rows
                    or region.grid_cols != stage.cols
                )
            ):
                rows = region.grid_rows
                cols = region.grid_cols
                cells = [
                    [Cell.EMPTY for _ in range(cols)]
                    for _ in range(rows)
                ]
            else:
                cells = [list(row) for row in stage.cells]
            for override in profile.cell_overrides:
                if (
                    0 <= override.row < len(cells)
                    and 0 <= override.col < len(cells[0])
                ):
                    cells[override.row][override.col] = Cell(override.cell)
            special_points = tuple(
                SpecialPoint(
                    row=row,
                    col=col,
                    owner=(
                        "A" if cell is Cell.SELF_SPECIAL else "B"
                    ),
                    kind="initial",
                )
                for row, cells_row in enumerate(cells)
                for col, cell in enumerate(cells_row)
                if cell in {Cell.SELF_SPECIAL, Cell.OPPONENT_SPECIAL}
            )
            stage = replace(
                stage,
                cells=tuple(tuple(row) for row in cells),
                special_points=special_points,
                source_max_cells=sum(
                    cell in PLAYABLE_CELLS
                    for row in cells
                    for cell in row
                ),
            )
        self._validate_region(region, stage)
        width = stage.cols * self.CELL_SIZE
        height = stage.rows * self.CELL_SIZE
        warped = warp_region(frame, region, width, height)
        reference = (
            warp_region(reference_frame, region, width, height)
            if reference_frame is not None
            else None
        )
        return self.read_warped(warped, stage, reference)

    @classmethod
    def from_profile(
        cls,
        catalog: GameCatalog,
        profile_path: Path,
        stage_id: str,
        config: BoardColorConfig | None = None,
    ) -> tuple["BoardGridReader", VisionProfile, VisionRegion, np.ndarray]:
        profile = VisionProfile.load(profile_path)
        stage = catalog.stages[stage_id]
        candidates = [
            item
            for item in profile.regions
            if item.grid_rows > 1 and item.grid_cols > 1
        ]
        region = next(
            (
                item
                for item in candidates
                if item.grid_rows == stage.rows
                and item.grid_cols == stage.cols
            ),
            candidates[0] if candidates and profile.stage_id == stage_id else None,
        )
        if region is None:
            raise ValueError(
                f"{profile_path.name}: no {stage.rows}x{stage.cols} board region"
            )
        source_text = profile.notes.removeprefix("source=").strip()
        reference = (
            _imread_unicode(Path(source_text), cv2.IMREAD_COLOR)
            if source_text
            else None
        )
        if reference is None:
            raise ValueError(f"{profile_path.name}: cannot read reference image")
        return cls(catalog, config), profile, region, reference

    def read_warped(
        self,
        warped: np.ndarray,
        stage: StageSpec,
        reference: np.ndarray | None = None,
    ) -> BoardObservation:
        expected_size = (
            stage.cols * self.CELL_SIZE,
            stage.rows * self.CELL_SIZE,
        )
        if (warped.shape[1], warped.shape[0]) != expected_size:
            raise ValueError(
                f"warped board must be {expected_size[0]}x{expected_size[1]}"
            )
        if reference is not None and reference.shape[:2] != warped.shape[:2]:
            raise ValueError("reference board dimensions must match the frame")
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        reference_bgr = (
            reference.astype(np.float32) if reference is not None else None
        )
        inferred: list[list[Cell]] = []
        readings = []
        confidence_values = []
        for row in range(stage.rows):
            inferred_row = []
            for col in range(stage.cols):
                bounds = self._sample_bounds(row, col)
                y0, y1 = bounds[0], bounds[1]
                x0, x1 = bounds[2], bounds[3]
                patch_bgr = warped[y0:y1, x0:x1]
                full_patch = warped[
                    row * self.CELL_SIZE : (row + 1) * self.CELL_SIZE,
                    col * self.CELL_SIZE : (col + 1) * self.CELL_SIZE,
                ]
                full_patch_gray = gray[
                    row * self.CELL_SIZE : (row + 1) * self.CELL_SIZE,
                    col * self.CELL_SIZE : (col + 1) * self.CELL_SIZE,
                ]
                patch_hsv = hsv[y0:y1, x0:x1]
                patch_gray = gray[y0:y1, x0:x1]
                mean_bgr = np.mean(patch_bgr.reshape(-1, 3), axis=0)
                median_hsv = np.median(
                    patch_hsv.reshape(-1, 3),
                    axis=0,
                )
                self_fraction = self._color_fraction(patch_hsv, self_ratio=True)
                opponent_fraction = self._color_fraction(
                    patch_hsv,
                    self_ratio=False,
                )
                self_special_fraction = self._range_fraction(
                    patch_hsv,
                    self.config.self_special_hue_range,
                )
                opponent_special_fraction = self._range_fraction(
                    patch_hsv,
                    self.config.opponent_special_hue_range,
                )
                pattern = self._special_pattern_scores(full_patch)
                self_pattern_delta = (
                    pattern["self_special"] - pattern["self_ink"]
                )
                opponent_pattern_delta = (
                    pattern["opponent_special"] - pattern["opponent_ink"]
                )
                full_hsv = cv2.cvtColor(full_patch, cv2.COLOR_BGR2HSV)
                center_brightness_contrast = (
                    float(
                        full_hsv[13:19, 13:19, 2].mean()
                        - full_hsv[2:30, 2:30, 2].mean()
                    )
                )
                center_saturation = float(
                    full_hsv[13:19, 13:19, 1].mean()
                )
                color_change = 0.0
                if reference_bgr is not None:
                    reference_mean = np.mean(
                        reference_bgr[y0:y1, x0:x1].reshape(-1, 3),
                        axis=0,
                    )
                    color_change = float(
                        np.linalg.norm((mean_bgr - reference_mean) / 255.0)
                    )
                edges = cv2.Canny(patch_gray, 40, 100)
                edge_density = float(np.count_nonzero(edges)) / float(edges.size)
                inferred_cell = self._infer_cell(
                    stage.cells[row][col],
                    self_fraction,
                    opponent_fraction,
                    self_special_fraction,
                    opponent_special_fraction,
                    self_pattern_delta,
                    opponent_pattern_delta,
                    color_change,
                    reference is not None,
                )
                inferred_row.append(inferred_cell)
                confidence_values.append(
                    max(self_fraction, opponent_fraction)
                    if inferred_cell
                    in {Cell.SELF_INK, Cell.OPPONENT_INK}
                    else 1.0 - max(self_fraction, opponent_fraction)
                )
                readings.append(
                    BoardCellReading(
                        row=row,
                        col=col,
                        stage_cell=stage.cells[row][col],
                        inferred_cell=inferred_cell,
                        mean_bgr=tuple(int(round(value)) for value in mean_bgr),
                        median_hsv=tuple(
                            int(round(value)) for value in median_hsv
                        ),
                        self_fraction=self_fraction,
                        opponent_fraction=opponent_fraction,
                        self_special_fraction=self_special_fraction,
                        opponent_special_fraction=opponent_special_fraction,
                        self_special_pattern_delta=self_pattern_delta,
                        opponent_special_pattern_delta=(
                            opponent_pattern_delta
                        ),
                        center_brightness_contrast=(
                            center_brightness_contrast
                        ),
                        center_saturation=center_saturation,
                        special_active=(
                            inferred_cell
                            is Cell.OPPONENT_SPECIAL
                            and center_saturation <= 100
                            or inferred_cell is Cell.SELF_SPECIAL
                            and center_brightness_contrast >= 6
                        ),
                        color_change=color_change,
                        edge_density=edge_density,
                    preview_diagonal_score=(
                        preview_diagonal_score(full_patch_gray)
                    ),
                    )
                )
            inferred.append(inferred_row)
        return BoardObservation(
            stage_id=stage.stage_id,
            rows=stage.rows,
            cols=stage.cols,
            cells=tuple(tuple(row) for row in inferred),
            readings=tuple(readings),
            confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
            **self._preview_evidence(
                readings,
                stage,
                reference_available=reference is not None,
            ),
            playable_cells=sum(
                stage.cells[row][col]
                in {
                    Cell.EMPTY,
                    Cell.SELF_INK,
                    Cell.SELF_SPECIAL,
                    Cell.OPPONENT_INK,
                    Cell.OPPONENT_SPECIAL,
                }
                for row in range(stage.rows)
                for col in range(stage.cols)
            ),
            reference_available=reference is not None,
        )

    def _sample_bounds(self, row: int, col: int) -> tuple[int, int, int, int]:
        margin = self.CELL_SIZE // 4
        return (
            row * self.CELL_SIZE + margin,
            (row + 1) * self.CELL_SIZE - margin,
            col * self.CELL_SIZE + margin,
            (col + 1) * self.CELL_SIZE - margin,
        )

    def _color_fraction(
        self,
        hsv_patch: np.ndarray,
        *,
        self_ratio: bool,
    ) -> float:
        ink_range, special_range = (
            (
                self.config.self_ink_hue_range,
                self.config.self_special_hue_range,
            )
            if self_ratio
            else (
                self.config.opponent_ink_hue_range,
                self.config.opponent_special_hue_range,
            )
        )
        return min(
            1.0,
            self._range_fraction(hsv_patch, ink_range)
            + self._range_fraction(hsv_patch, special_range),
        )

    def _range_fraction(
        self,
        hsv_patch: np.ndarray,
        hue_range: tuple[int, int],
    ) -> float:
        hue_min, hue_max = hue_range
        hue = hsv_patch[:, :, 0]
        saturation = hsv_patch[:, :, 1]
        value = hsv_patch[:, :, 2]
        mask = (
            (hue >= hue_min)
            & (hue <= hue_max)
            & (saturation >= self.config.minimum_saturation)
            & (value >= self.config.minimum_value)
        )
        return float(np.count_nonzero(mask)) / float(mask.size)

    def _infer_cell(
        self,
        stage_cell: Cell,
        self_fraction: float,
        opponent_fraction: float,
        self_special_fraction: float,
        opponent_special_fraction: float,
        self_pattern_delta: float,
        opponent_pattern_delta: float,
        color_change: float,
        has_reference: bool,
    ) -> Cell:
        if stage_cell in {Cell.VOID, Cell.WALL}:
            return stage_cell
        if stage_cell in {Cell.SELF_SPECIAL, Cell.OPPONENT_SPECIAL}:
            return stage_cell
        if (
            has_reference
            and color_change < self.config.change_threshold
            and max(self_fraction, opponent_fraction)
            < self.config.minimum_ink_fraction
            and max(self_pattern_delta, opponent_pattern_delta) < 0.15
        ):
            return Cell.EMPTY
        if (
            self_pattern_delta >= 0.15
            and self_fraction >= 0.20
        ):
            return Cell.SELF_SPECIAL
        if (
            opponent_pattern_delta >= 0.15
            and (
                opponent_fraction >= 0.20
                or opponent_special_fraction >= 0.05
            )
        ):
            return Cell.OPPONENT_SPECIAL
        if (
            self_fraction >= self.config.minimum_ink_fraction
            and self_fraction > opponent_fraction * 1.15
        ):
            return (
                Cell.SELF_SPECIAL
                if self_special_fraction > self_fraction * 0.55
                else Cell.SELF_INK
            )
        if (
            opponent_fraction >= self.config.minimum_ink_fraction
            and opponent_fraction > self_fraction * 1.15
        ):
            return (
                Cell.OPPONENT_SPECIAL
                if opponent_special_fraction
                > opponent_fraction * 0.55
                else Cell.OPPONENT_INK
            )
        return Cell.EMPTY

    def _special_pattern_scores(
        self,
        patch: np.ndarray,
    ) -> dict[str, float]:
        if not self.special_tiles:
            return {
                "self_ink": 0.0,
                "self_special": 0.0,
                "opponent_ink": 0.0,
                "opponent_special": 0.0,
            }
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return {
            name: _masked_correlation(gray, template, mask)
            for name, (template, mask) in self.special_tiles.items()
        }

    @staticmethod
    def _preview_evidence(
        readings: list[BoardCellReading],
        stage: StageSpec,
        *,
        reference_available: bool,
    ) -> dict[str, float | bool]:
        playable = [
            reading
            for reading in readings
            if stage.cells[reading.row][reading.col]
            in {
                Cell.EMPTY,
                Cell.SELF_INK,
                Cell.SELF_SPECIAL,
                Cell.OPPONENT_INK,
                Cell.OPPONENT_SPECIAL,
            }
        ]
        if not playable:
            return {
                "placement_preview_score": 0.0,
                "placement_preview_present": False,
            }
        preview_cell_count = sum(
            reading.preview_diagonal_score > PREVIEW_INK_RATIO
            and (
                not reference_available
                or reading.color_change >= 0.12
            )
            for reading in playable
        )
        score = preview_cell_count / len(playable)
        return {
            "placement_preview_score": score,
            "placement_preview_present": preview_cell_count >= 2,
        }

    @staticmethod
    def _validate_region(region: VisionRegion, stage: StageSpec) -> None:
        if region.grid_rows != stage.rows or region.grid_cols != stage.cols:
            raise ValueError(
                f"region grid {region.grid_rows}x{region.grid_cols} "
                f"does not match stage {stage.rows}x{stage.cols}"
            )


def _load_special_tile_templates(
    root: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    names = {
        "self_ink": "AreaPiece_00.png",
        "opponent_ink": "AreaPiece_01.png",
        "self_special": "AreaPiece_02.png",
        "opponent_special": "AreaPiece_03.png",
    }
    result = {}
    for key, filename in names.items():
        image = cv2.imread(str(root / filename), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            continue
        bgr = cv2.resize(
            image[:, :, :3],
            (BoardGridReader.CELL_SIZE, BoardGridReader.CELL_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        mask = cv2.resize(
            image[:, :, 3],
            (BoardGridReader.CELL_SIZE, BoardGridReader.CELL_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        result[key] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask)
    return result


def _imread_unicode(path: Path, flags: int):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _masked_correlation(
    patch: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = mask > 0
    if not np.any(selected):
        return 0.0
    first = patch.astype(np.float64)[selected]
    second = template.astype(np.float64)[selected]
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return (
        float(np.dot(first, second)) / denominator
        if denominator > 1e-9
        else 0.0
    )
