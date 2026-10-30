from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import time

import numpy as np

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.card import CardCell
from tableturf_rebuild.domain.grid import PLAYABLE_CELLS, Cell
from tableturf_rebuild.vision.legacy_preview import context_dot_score
from tableturf_rebuild.vision.board_reader import (
    BoardCellReading,
    BoardObservation,
)
from tableturf_rebuild.vision.calibration import VisionRegion, warp_region
from tableturf_rebuild.vision.preview_features import (
    PREVIEW_INK_RATIO,
    is_diagonal_preview_patch,
    preview_diagonal_score,
    preview_dot_score,
    contextual_preview_dot_score,
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
    COLOR_CHANGE_THRESHOLD = 0.12
    CELL_SIZE = 32

    def __init__(self, catalog: GameCatalog):
        self.catalog = catalog
        self.last_diagnostics: dict = {}
        self._history = deque(maxlen=3)
        self._history_key = None
        self._history_region = None

    def reset_history(self) -> None:
        self._history.clear()
        self._history_key = None
        self._history_region = None

    def read(
        self,
        board: BoardObservation,
        card_id: str,
        *,
        special_attack: bool = False,
    ) -> PlacementPreviewObservation | None:
        direct_dots = frozenset((r.row, r.col) for r in board.readings if r.preview_dot_score >= 0.6)
        preview_cells = self._candidate_cells(board) - direct_dots
        self.last_diagnostics = {
            "source": "board", "card_id": card_id,
            "candidate_count": len(preview_cells), "candidate_cells": sorted(preview_cells),
            "threshold": PREVIEW_INK_RATIO,
            "current_direct_dot_cells": sorted(direct_dots),
        }
        if not preview_cells:
            self.last_diagnostics["failure"] = "NO_DIAGONAL_CELLS"
            return None
        return self._match_cells(
            preview_cells,
            card_id,
            board,
            direct_dots,
            special_attack=special_attack,
            current_direct_special_cells=direct_dots,
        )

    def read_frame(
        self,
        frame: np.ndarray,
        region: VisionRegion,
        stage_id: str,
        card_id: str,
        board: BoardObservation | None = None,
        *,
        temporal: bool = False,
        background: BoardObservation | None = None,
        special_attack: bool = False,
    ) -> PlacementPreviewObservation | None:
        stage = self.catalog.stages[stage_id]
        rows = board.rows if board is not None else region.grid_rows or stage.rows
        cols = board.cols if board is not None else region.grid_cols or stage.cols
        # An invalid preview can extend beyond the board or cover a void tile.
        # Keep those stripes: clipping them changes the apparent shape/origin.
        shape = self.catalog.cards[card_id].shape
        margin = max(shape.height, shape.width) - 1
        if region.kind != "rect":
            warped = warp_region(
                frame, region, cols * self.CELL_SIZE, rows * self.CELL_SIZE,
                padding=margin * self.CELL_SIZE,
            )
            scores = {}
            dots = set()
            for row in range(-margin, rows + margin):
                for col in range(-margin, cols + margin):
                    patch = warped[
                        (row + margin) * self.CELL_SIZE : (row + margin + 1) * self.CELL_SIZE,
                        (col + margin) * self.CELL_SIZE : (col + margin + 1) * self.CELL_SIZE,
                    ]
                    scores[(row, col)] = preview_diagonal_score(patch)
                    if preview_dot_score(patch) >= 0.6:
                        dots.add((row, col))
            dot_cells = frozenset(dots)
            direct_dot_cells = dot_cells
            context_dot_cells = frozenset()
            preview_cells = frozenset(cell for cell, score in scores.items()
                                      if score > PREVIEW_INK_RATIO) - dot_cells
            source = "padded_frame"
        if region.kind == "rect":
            # Original AutoTableTurf samples an inset of each native cell.
            # Avoid resizing away thin moving lines and dot components.
            points = region.pixel_points(frame.shape[1], frame.shape[0])
            dx = (points[1, 0] - points[0, 0]) / cols
            dy = (points[1, 1] - points[0, 1]) / rows
            inset = 3 * frame.shape[0] / 1080.0
            scores = {}
            native_dots = set()
            native_direct_dots = set()
            native_context_dots = set()
            base = background or board
            active_cells = {(reading.row, reading.col) for reading in base.readings
                            if reading.special_active} if base is not None else set()
            for row in range(-margin, rows + margin):
                for col in range(-margin, cols + margin):
                    x1 = max(0, round(float(points[0, 0] + col * dx + inset)))
                    y1 = max(0, round(float(points[0, 1] + row * dy + inset)))
                    x2 = min(frame.shape[1], round(float(points[0, 0] + (col + 1) * dx - inset)))
                    y2 = min(frame.shape[0], round(float(points[0, 1] + (row + 1) * dy - inset)))
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    patch = frame[y1:y2, x1:x2]
                    scores[(row, col)] = preview_diagonal_score(patch)
                    cell = (base.cells[row][col]
                            if base and 0 <= row < base.rows and 0 <= col < base.cols else Cell.WALL)
                    dots = preview_dot_score(patch)
                    if dots >= 0.6:
                        native_direct_dots.add((row, col))
                        # Round dots can themselves form diagonal edge chains.
                        # Keep the texture type so two-cell cards do not flip
                        # by 180 degrees or borrow an existing orange tile.
                        scores[(row, col)] = 0.0
                    context_score = context_dot_score(patch, cell)
                    below = (base.cells[row + 1][col]
                             if base and 0 <= row + 1 < base.rows
                             and 0 <= col < base.cols and (row + 1, col) in active_cells
                             else None)
                    special_background = cell in {Cell.SELF_SPECIAL, Cell.OPPONENT_SPECIAL}
                    if below in {Cell.SELF_SPECIAL, Cell.OPPONENT_SPECIAL}:
                        # Merge the flame colors into each mask before finding
                        # components, as the original detector does. Counting
                        # ink colors alone can fragment a flame into fake dots.
                        context_score = contextual_preview_dot_score(
                            patch, cell, fire_below=below,
                        )
                    if special_background:
                        # A cursor can hide the activation marker, and its dot
                        # palette can match the fiery mask even on an unlit
                        # special square. Require repeated small components
                        # from either palette instead of trusting active=False
                        # to exclude the correct one. Stripes retain priority.
                        context_score = max(context_score, contextual_preview_dot_score(
                            patch, cell, fiery=True, fire_below=below,
                        ))
                    if max(dots, context_score) >= 0.6:
                        native_dots.add((row, col))
                    if context_score >= 0.6:
                        native_context_dots.add((row, col))
            preview_cells = frozenset(cell for cell, score in scores.items() if score > PREVIEW_INK_RATIO)
            dot_cells = frozenset(native_dots)
            direct_dot_cells = frozenset(native_direct_dots)
            context_dot_cells = frozenset(native_context_dots)
            source = "native_grid"
        # Context-only color components are secondary to clear stripe evidence.
        dot_cells = dot_cells - preview_cells
        direct_dot_cells = direct_dot_cells - preview_cells
        context_dot_cells = context_dot_cells - preview_cells
        historical_direct_dots = frozenset()
        historical_context_dots = frozenset()
        if temporal:
            key = (stage_id, card_id, region.kind, rows, cols, frame.shape[:2])
            points = np.asarray(region.points) * (frame.shape[1], frame.shape[0])
            now = time.perf_counter()
            # Analysis can take longer than 1.5 s, and the empty-cell fit can
            # drift by a fraction of a pixel while the animated cursor stays
            # still. Keep those phases together; controller pulses clear them.
            if (key != self._history_key
                or (self._history_region is not None
                    and np.max(np.abs(points - self._history_region)) > 2.5 * frame.shape[0] / 1080)
                or (self._history and now - self._history[-1][0] > 4.0)):
                self.reset_history()
            self._history_key = key
            if self._history_region is None:
                self._history_region = points
            self._history.append((now, preview_cells, dot_cells, direct_dot_cells, context_dot_cells))
            historical_direct_dots = frozenset().union(*(item[3] for item in tuple(self._history)[:-1]))
            historical_context_dots = frozenset().union(*(item[4] for item in tuple(self._history)[:-1]))
            preview_cells = frozenset().union(*(item[1] for item in self._history))
            dot_cells = frozenset().union(*(item[2] for item in self._history))
            preview_cells = preview_cells - dot_cells
        self.last_diagnostics = {
            "source": source, "card_id": card_id,
            "rows": rows, "cols": cols, "margin": margin,
            "region": region.points, "threshold": PREVIEW_INK_RATIO,
            "candidate_count": len(preview_cells), "candidate_cells": sorted(preview_cells),
            "dot_cells": sorted(dot_cells), "temporal_frames": len(self._history) if temporal else 1,
            "current_direct_dot_cells": sorted(direct_dot_cells),
            "current_context_dot_cells": sorted(context_dot_cells),
            "historical_direct_dot_cells": sorted(historical_direct_dots),
            "historical_context_dot_cells": sorted(historical_context_dots),
            "strongest_cells": [
                {"cell": cell, "score": score}
                for cell, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:20]
            ],
        }
        if not preview_cells:
            self.last_diagnostics["failure"] = "NO_DIAGONAL_CELLS"
            return None
        return self._match_cells(
            preview_cells,
            card_id,
            board,
            dot_cells,
            background=background,
            special_attack=special_attack,
            current_direct_special_cells=direct_dot_cells,
            historical_direct_special_cells=historical_direct_dots,
        )

    def _match_cells(
        self,
        preview_cells: frozenset[tuple[int, int]],
        card_id: str,
        board: BoardObservation | None,
        special_preview_cells: frozenset[tuple[int, int]] = frozenset(),
        *,
        background: BoardObservation | None = None,
        special_attack: bool = False,
        current_direct_special_cells: frozenset[tuple[int, int]] = frozenset(),
        historical_direct_special_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> PlacementPreviewObservation | None:
        card = self.catalog.cards[card_id]
        best: tuple[
            float,
            int,
            tuple[int, int],
            frozenset[tuple[int, int]],
        ] | None = None
        best_rank = (-1.0, -1, -1, -1, -1, -1)
        best_special_positions: set[frozenset[tuple[int, int]]] = set()
        observed_specials = {
            (reading.row, reading.col)
            for reading in board.readings
            if reading.self_special_fraction >= 0.4 and reading.median_hsv[0] < 27
            and not (background is not None
                     and 0 <= reading.row < background.rows and 0 <= reading.col < background.cols
                     and background.cells[reading.row][reading.col] is Cell.SELF_SPECIAL)
        } if board is not None else set()
        observed = preview_cells | special_preview_cells
        for first in sorted(observed):
            for rotation in range(4):
                shape = card.shape.rotated(rotation)
                # The first occupied square can be a special tile without
                # stripes. Try all offsets rather than shifting its origin.
                for offset_row, offset_col in shape.occupied:
                    origin = (first[0] - offset_row, first[1] - offset_col)
                    expected = frozenset(
                        (origin[0] + row, origin[1] + col)
                        for row, col in shape.occupied
                    )
                    expected_specials = {
                        (origin[0] + row, origin[1] + col)
                        for row, col in shape.occupied
                        if shape.cells[row][col] is CardCell.SPECIAL
                    }
                    special_tiles = expected_specials & (observed_specials | special_preview_cells)
                    # Ignore unrelated texture elsewhere in the frame, but
                    # penalize extra cells inside this candidate's bounding box.
                    component = frozenset(
                        (row, col) for row, col in observed
                        if origin[0] <= row < origin[0] + shape.height
                        and origin[1] <= col < origin[1] + shape.width
                    )
                    detected = component | special_tiles
                    score = len(detected & expected) / len(detected | expected)
                    special_matches = len(special_tiles)
                    wrong_dots = len((expected & special_preview_cells) - special_tiles)
                    score -= wrong_dots / max(1, len(expected))
                    dot_matches = len(expected_specials & special_preview_cells)
                    # Whole-frame temporal unions may contain an orange tile's
                    # transient context components at the opposite end of a
                    # symmetric shape. Current texture evidence is stronger
                    # than that accumulated hint; neither dictates rotation.
                    rank = (score,
                            len(expected_specials & current_direct_special_cells),
                            len(expected_specials & historical_direct_special_cells),
                            dot_matches, len(component & expected), special_matches)
                    if rank > best_rank:
                        best_rank = rank
                        best = (score, rotation, origin, component)
                        best_special_positions = {frozenset(expected_specials)}
                    elif rank == best_rank and best is not None:
                        best_special_positions.add(frozenset(expected_specials))
        self.last_diagnostics.update({
            "best_score": best[0] if best else 0.0,
            "best_rotation": best[1] if best else None,
            "best_origin": best[2] if best else None,
            "minimum_shape_score": 0.72,
            "matched_special_dots": max(0, best_rank[3]),
            "matched_current_direct_dots": max(0, best_rank[1]),
            "matched_historical_direct_dots": max(0, best_rank[2]),
        })
        if best is None or best[0] < 0.72:
            self.last_diagnostics["failure"] = "SHAPE_MISMATCH"
            return None
        if len(best_special_positions) > 1:
            self.last_diagnostics["failure"] = "AMBIGUOUS_SPECIAL_ORIENTATION"
            return None
        self.last_diagnostics["failure"] = None
        score, rotation, origin, preview_cells = best
        valid, valid_fraction = (
            self._placement_validity(
                board,
                card_id,
                rotation,
                origin,
                preview_cells,
                background=background,
                special_attack=special_attack,
            )
            if board is not None
            else (False, 0.0)
        )
        return PlacementPreviewObservation(
            cells=preview_cells,
            rotation=rotation,
            origin=origin,
            anchor=min(
                (origin[0] + row, origin[1] + col)
                for row, col in card.shape.rotated(rotation).occupied
            ),
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

    def _candidate_cells(
        self,
        board: BoardObservation,
    ) -> frozenset[tuple[int, int]]:
        return frozenset(
            (reading.row, reading.col)
            for reading in board.readings
            if self._is_candidate(board, reading)
        )

    def _is_candidate(
        self,
        board: BoardObservation,
        reading: BoardCellReading,
    ) -> bool:
        return (
            reading.preview_diagonal_score > PREVIEW_INK_RATIO
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
        *,
        background: BoardObservation | None = None,
        special_attack: bool = False,
    ) -> tuple[bool, float]:
        shape = self.catalog.cards[card_id].shape.rotated(rotation)
        if background is not None:
            expected = {(origin[0] + row, origin[1] + col) for row, col in shape.occupied}
            allowed = ({Cell.EMPTY, Cell.SELF_INK, Cell.OPPONENT_INK}
                       if special_attack else {Cell.EMPTY})
            if any(not (0 <= row < background.rows and 0 <= col < background.cols)
                   or background.cells[row][col] not in allowed for row, col in expected):
                return False, 0.0
            neighbors = {Cell.SELF_SPECIAL} if special_attack else {Cell.SELF_INK, Cell.SELF_SPECIAL}
            if not any(
                0 <= row + dr < background.rows and 0 <= col + dc < background.cols
                and background.cells[row + dr][col + dc] in neighbors
                for row, col in expected for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if dr or dc
            ):
                return False, 0.0
        readings = {(reading.row, reading.col): reading for reading in board.readings}
        if any(
            (origin[0] + row, origin[1] + col) not in readings
            or readings[(origin[0] + row, origin[1] + col)].stage_cell not in PLAYABLE_CELLS
            for row, col in shape.occupied
        ):
            return False, 0.0
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
