from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.card import CardCell
from tableturf_rebuild.domain.grid import PLAYABLE_CELLS
from tableturf_rebuild.vision.board_reader import (
    BoardCellReading,
    BoardObservation,
)
from tableturf_rebuild.vision.calibration import VisionRegion, warp_region
from tableturf_rebuild.vision.preview_features import (
    is_diagonal_preview_patch,
)


@dataclass(frozen=True, slots=True)
class PlacementPreviewObservation:
    cells: frozenset[tuple[int, int]]
    rotation: int
    origin: tuple[int, int]
    anchor: tuple[int, int]
    score: float
    valid: bool
    valid_fraction: float


class PlacementPreviewReader:
    EDGE_THRESHOLD = 0.18
    FALLBACK_MINIMUM_EDGE = 0.05
    COLOR_CHANGE_THRESHOLD = 0.12
    CELL_SIZE = 32

    def __init__(self, catalog: GameCatalog):
        self.catalog = catalog

    def read(
        self,
        board: BoardObservation,
        card_id: str,
    ) -> PlacementPreviewObservation | None:
        preview_cells = self._candidate_cells(
            board,
            self.EDGE_THRESHOLD,
        )
        if not preview_cells:
            return self._scan_for_preview(board, card_id)
        matched = self._match_cells(
            preview_cells,
            card_id,
            board,
        )
        if matched is None:
            return self._scan_for_preview(board, card_id)
        return matched

    def read_frame(
        self,
        frame: np.ndarray,
        region: VisionRegion,
        stage_id: str,
        card_id: str,
        board: BoardObservation | None = None,
    ) -> PlacementPreviewObservation | None:
        stage = self.catalog.stages[stage_id]
        warped = warp_region(
            frame,
            region,
            stage.cols * self.CELL_SIZE,
            stage.rows * self.CELL_SIZE,
        )
        preview_cells = frozenset(
            (row, col)
            for row in range(stage.rows)
            for col in range(stage.cols)
            if (
                stage.cells[row][col] in PLAYABLE_CELLS
                and self._is_diagonal_preview_cell(
                    warped[
                        row * self.CELL_SIZE : (row + 1) * self.CELL_SIZE,
                        col * self.CELL_SIZE : (col + 1) * self.CELL_SIZE,
                    ]
                )
            )
        )
        if not preview_cells:
            return None
        return self._match_cells(
            preview_cells,
            card_id,
            board,
        )

    def _match_cells(
        self,
        preview_cells: frozenset[tuple[int, int]],
        card_id: str,
        board: BoardObservation | None,
    ) -> PlacementPreviewObservation | None:
        card = self.catalog.cards[card_id]
        best: tuple[
            float,
            int,
            tuple[int, int],
            frozenset[tuple[int, int]],
        ] | None = None
        for component in self._connected_components(preview_cells):
            preview_top = min(row for row, _ in component)
            preview_left = min(col for _, col in component)
            normalized = frozenset(
                (row - preview_top, col - preview_left)
                for row, col in component
            )
            for rotation in range(4):
                occupied = card.shape.rotated(rotation).occupied
                shape_top = min(row for row, _ in occupied)
                shape_left = min(col for _, col in occupied)
                shape = frozenset(
                    (row - shape_top, col - shape_left)
                    for row, col in occupied
                )
                union = normalized | shape
                score = (
                    len(normalized & shape) / len(union)
                    if union
                    else 0.0
                )
                candidate = (
                    score,
                    rotation,
                    (preview_top, preview_left),
                    component,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None or best[0] < 0.72:
            return None
        score, rotation, origin, preview_cells = best
        valid, valid_fraction = (
            self._placement_validity(
                board,
                card_id,
                rotation,
                origin,
                preview_cells,
            )
            if board is not None
            else (False, 0.0)
        )
        return PlacementPreviewObservation(
            cells=preview_cells,
            rotation=rotation,
            origin=origin,
            anchor=min(preview_cells),
            score=score,
            valid=valid,
            valid_fraction=valid_fraction,
        )

    def _is_diagonal_preview_cell(self, patch: np.ndarray) -> bool:
        return is_diagonal_preview_patch(patch)

    @staticmethod
    def _connected_components(
        cells: frozenset[tuple[int, int]],
    ) -> tuple[frozenset[tuple[int, int]], ...]:
        remaining = set(cells)
        components = []
        while remaining:
            seed = remaining.pop()
            component = {seed}
            stack = [seed]
            while stack:
                row, col = stack.pop()
                for row_delta in (-1, 0, 1):
                    for col_delta in (-1, 0, 1):
                        if row_delta == 0 and col_delta == 0:
                            continue
                        neighbor = (
                            row + row_delta,
                            col + col_delta,
                        )
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component.add(neighbor)
                            stack.append(neighbor)
            components.append(frozenset(component))
        components.sort(key=len, reverse=True)
        return tuple(components)

    def _scan_for_preview(
        self,
        board: BoardObservation,
        card_id: str,
    ) -> PlacementPreviewObservation | None:
        edge_map = {
            (reading.row, reading.col): reading.edge_density
            for reading in board.readings
            if self._is_candidate(
                board,
                reading,
                self.FALLBACK_MINIMUM_EDGE,
            )
        }
        card = self.catalog.cards[card_id]
        best: tuple[float, int, tuple[int, int], frozenset[tuple[int, int]]] | None = None
        for rotation in range(4):
            occupied = card.shape.rotated(rotation).occupied
            shape_top = min(row for row, _ in occupied)
            shape_left = min(col for _, col in occupied)
            normalized = tuple(
                (row - shape_top, col - shape_left)
                for row, col in occupied
            )
            height = max(row for row, _ in normalized) + 1
            width = max(col for _, col in normalized) + 1
            for origin_row in range(0, board.rows - height + 1):
                for origin_col in range(0, board.cols - width + 1):
                    cells = frozenset(
                        (
                            origin_row + row,
                            origin_col + col,
                        )
                        for row, col in normalized
                    )
                    mean_edge = sum(
                        edge_map.get(cell, 0.0)
                        for cell in cells
                    ) / len(cells)
                    if mean_edge < self.FALLBACK_MINIMUM_EDGE:
                        continue
                    candidate = (
                        mean_edge,
                        rotation,
                        (origin_row, origin_col),
                        cells,
                    )
                    if best is None or candidate[0] > best[0]:
                        best = candidate
        if best is None:
            return None
        mean_edge, rotation, origin, cells = best
        valid, valid_fraction = self._placement_validity(
            board,
            card_id,
            rotation,
            origin,
            cells,
        )
        return PlacementPreviewObservation(
            cells=cells,
            rotation=rotation,
            origin=origin,
            anchor=min(cells),
            score=min(1.0, mean_edge / 0.20),
            valid=valid,
            valid_fraction=valid_fraction,
        )

    def _candidate_cells(
        self,
        board: BoardObservation,
        minimum_edge: float,
    ) -> frozenset[tuple[int, int]]:
        return frozenset(
            (reading.row, reading.col)
            for reading in board.readings
            if self._is_candidate(board, reading, minimum_edge)
        )

    def _is_candidate(
        self,
        board: BoardObservation,
        reading: BoardCellReading,
        minimum_edge: float,
    ) -> bool:
        return (
            reading.stage_cell in PLAYABLE_CELLS
            and reading.edge_density >= minimum_edge
            and (
                not board.reference_available
                or reading.color_change >= self.COLOR_CHANGE_THRESHOLD
            )
        )

    def _placement_validity(
        self,
        board: BoardObservation,
        card_id: str,
        rotation: int,
        origin: tuple[int, int],
        cells: frozenset[tuple[int, int]],
    ) -> tuple[bool, float]:
        shape = self.catalog.cards[card_id].shape.rotated(rotation)
        shape_top = min(row for row, _ in shape.occupied)
        shape_left = min(col for _, col in shape.occupied)
        special_cells = frozenset(
            (
                origin[0] + row - shape_top,
                origin[1] + col - shape_left,
            )
            for row, row_cells in enumerate(shape.cells)
            for col, cell in enumerate(row_cells)
            if cell is CardCell.SPECIAL
        )
        normal_cells = cells - special_cells
        readings = {
            (reading.row, reading.col): reading
            for reading in board.readings
        }
        colored = 0
        for cell in normal_cells:
            reading = readings.get(cell)
            if reading is None:
                continue
            hue, saturation, value = reading.median_hsv
            if (
                reading.self_fraction >= 0.20
                or 15 <= hue <= 55
                and saturation >= 80
                and value >= 60
            ):
                colored += 1
        fraction = colored / max(1, len(normal_cells))
        return fraction >= 0.55, fraction
