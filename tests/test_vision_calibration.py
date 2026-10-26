import pytest
import numpy as np

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.board_reader import BoardGridReader
from tableturf_rebuild.vision.calibration import (
    CellMaskOverride,
    VisionProfile,
    VisionRegion,
    drag_resize_region,
    region_pixel_bounds,
    region_pixel_size,
    region_grid_cells,
    resize_region_to_pixels,
    translate_region_pixels,
    warp_region,
)
from tableturf_rebuild.vision.calibration_ui import CalibrationApp


def test_vision_profile_round_trip_and_grid_generation(tmp_path) -> None:
    region = VisionRegion(
        name="hand.card.0.grid",
        kind="quad",
        points=((0.1, 0.1), (0.3, 0.12), (0.28, 0.4), (0.08, 0.38)),
        grid_rows=8,
        grid_cols=8,
    )
    profile = VisionProfile(
        profile_id="test",
        source_width=1920,
        source_height=1080,
        stage_id="Rectangle",
        regions=(region,),
        cell_overrides=(CellMaskOverride(3, 4, 32),),
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = VisionProfile.load(path)

    assert loaded == profile
    cells = region_grid_cells(region, 1920, 1080)
    assert len(cells) == 64
    assert cells[0][1] == "0,0"
    assert cells[-1][1] == "7,7"


def test_cost_region_supports_six_slots() -> None:
    region = VisionRegion(
        name="hand.card.0.cost",
        kind="rect",
        points=((0.2, 0.5), (0.4, 0.53)),
        grid_rows=1,
        grid_cols=6,
    )

    assert len(region_grid_cells(region, 1920, 1080)) == 6


def test_resize_region_keeps_center_and_shape() -> None:
    region = VisionRegion(
        name="hand.card.0.art",
        kind="quad",
        points=(
            (0.2, 0.2),
            (0.6, 0.25),
            (0.58, 0.7),
            (0.18, 0.65),
        ),
    )

    resized = resize_region_to_pixels(
        region,
        1001,
        1001,
        target_width=200,
        target_height=300,
    )

    assert region_pixel_size(resized, 1001, 1001) == (
        pytest.approx(200),
        pytest.approx(300),
    )
    old_left, old_top, old_right, old_bottom = region_pixel_bounds(
        region, 1001, 1001
    )
    new_left, new_top, new_right, new_bottom = region_pixel_bounds(
        resized, 1001, 1001
    )
    assert (new_left + new_right) / 2 == pytest.approx(
        (old_left + old_right) / 2
    )
    assert (new_top + new_bottom) / 2 == pytest.approx(
        (old_top + old_bottom) / 2
    )


def test_translate_region_is_clamped_without_resizing() -> None:
    region = VisionRegion(
        name="card",
        kind="rect",
        points=((0.8, 0.7), (0.9, 0.9)),
    )

    translated = translate_region_pixels(
        region,
        1001,
        1001,
        delta_x=1000,
        delta_y=1000,
    )

    assert region_pixel_bounds(translated, 1001, 1001) == (
        900.0,
        800.0,
        1000.0,
        1000.0,
    )


def test_drag_rect_corner_updates_only_dragged_edges() -> None:
    region = VisionRegion(
        name="cost",
        kind="rect",
        points=((0.2, 0.2), (0.6, 0.8)),
    )

    resized = drag_resize_region(
        region,
        1001,
        1001,
        "se",
        pointer_start=(600, 800),
        pointer_end=(700, 900),
    )

    assert region_pixel_bounds(resized, 1001, 1001) == (
        200.0,
        200.0,
        700.0,
        900.0,
    )


def test_drag_quad_vertex_preserves_other_vertices() -> None:
    region = VisionRegion(
        name="selected-card",
        kind="quad",
        points=(
            (0.1, 0.1),
            (0.5, 0.12),
            (0.52, 0.6),
            (0.12, 0.58),
        ),
    )

    resized = drag_resize_region(
        region,
        1001,
        1001,
        "vertex:1",
        pointer_start=(500, 120),
        pointer_end=(530, 100),
    )

    assert resized.points[0] == region.points[0]
    assert resized.points[2] == region.points[2]
    assert resized.points[3] == region.points[3]
    assert resized.points[1] == pytest.approx((0.53, 0.1))


def test_drag_quad_edge_moves_its_two_vertices_on_one_axis() -> None:
    region = VisionRegion(
        name="selected-card",
        kind="quad",
        points=(
            (0.1, 0.1),
            (0.5, 0.1),
            (0.5, 0.6),
            (0.1, 0.6),
        ),
    )

    resized = drag_resize_region(
        region,
        1001,
        1001,
        "edge:1",
        pointer_start=(500, 300),
        pointer_end=(530, 300),
    )

    assert resized.points[0] == region.points[0]
    assert resized.points[3] == region.points[3]
    assert resized.points[1] == pytest.approx((0.53, 0.1))
    assert resized.points[2] == pytest.approx((0.53, 0.6))


def test_copy_names_are_unique_and_numbered() -> None:
    app = object.__new__(CalibrationApp)
    app.regions = [
        VisionRegion(
            name="card-grid-copy",
            kind="rect",
            points=((0.1, 0.1), (0.2, 0.2)),
        ),
        VisionRegion(
            name="card-grid-copy-2",
            kind="rect",
            points=((0.1, 0.1), (0.2, 0.2)),
        ),
    ]

    assert app._unique_region_name("card-grid-copy") == "card-grid-copy-3"
    assert app._unique_region_name("card-grid") == "card-grid"


def test_warp_region_returns_requested_size() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[20:80, 20:100] = (20, 100, 220)
    region = VisionRegion(
        name="test",
        kind="rect",
        points=((0.1, 0.2), (0.5, 0.8)),
    )

    warped = warp_region(image, region, 64, 48)

    assert warped.shape == (48, 64, 3)
    assert np.median(warped[:, :, 2]) > 180


def test_quad_points_are_canonicalized_before_warping() -> None:
    region = VisionRegion(
        name="reversed",
        kind="quad",
        points=(
            (0.1, 0.1),
            (0.1, 0.8),
            (0.8, 0.8),
            (0.8, 0.1),
        ),
    )

    assert region.points == (
        (0.1, 0.1),
        (0.8, 0.1),
        (0.8, 0.8),
        (0.1, 0.8),
    )


def test_board_reader_uses_static_stage_and_color_evidence() -> None:
    catalog = GameCatalog.from_package()
    stage = catalog.stages["Rectangle"]
    reader = BoardGridReader(catalog)
    reference = np.full(
        (stage.rows * reader.CELL_SIZE, stage.cols * reader.CELL_SIZE, 3),
        (180, 180, 180),
        dtype=np.uint8,
    )
    current = reference.copy()
    empty_cells = [
        (row, col)
        for row in range(stage.rows)
        for col in range(stage.cols)
        if stage.cells[row][col] is Cell.EMPTY
    ]
    self_cell = empty_cells[0]
    opponent_cell = empty_cells[1]
    for row, col, color in (
        (*self_cell, (30, 230, 220)),
        (*opponent_cell, (230, 60, 170)),
    ):
        y0 = row * reader.CELL_SIZE
        y1 = y0 + reader.CELL_SIZE
        x0 = col * reader.CELL_SIZE
        x1 = x0 + reader.CELL_SIZE
        current[y0:y1, x0:x1] = color

    observation = reader.read_warped(current, stage, reference)

    assert observation.cells[self_cell[0]][self_cell[1]] is Cell.SELF_INK
    assert (
        observation.cells[opponent_cell[0]][opponent_cell[1]]
        is Cell.OPPONENT_INK
    )
    special = stage.special_points[0]
    assert observation.cells[special.row][special.col] is stage.cells[special.row][
        special.col
    ]
    assert observation.self_count == 2
    assert observation.opponent_count == 2
    assert observation.self_special_count == 1
    assert observation.opponent_special_count == 1


def test_board_reader_distinguishes_dynamic_special_colors() -> None:
    catalog = GameCatalog.from_package()
    stage = catalog.stages["Rectangle"]
    reader = BoardGridReader(catalog)
    reference = np.full(
        (stage.rows * reader.CELL_SIZE, stage.cols * reader.CELL_SIZE, 3),
        (180, 180, 180),
        dtype=np.uint8,
    )
    current = reference.copy()
    empty_cells = [
        (row, col)
        for row in range(stage.rows)
        for col in range(stage.cols)
        if stage.cells[row][col] is Cell.EMPTY
    ]
    self_special = empty_cells[2]
    opponent_special = empty_cells[3]
    for row, col, color in (
        (*self_special, (2, 163, 255)),
        (*opponent_special, (255, 239, 8)),
    ):
        y0 = row * reader.CELL_SIZE
        x0 = col * reader.CELL_SIZE
        current[
            y0 : y0 + reader.CELL_SIZE,
            x0 : x0 + reader.CELL_SIZE,
        ] = color

    observation = reader.read_warped(current, stage, reference)

    assert observation.cells[self_special[0]][self_special[1]] is Cell.SELF_SPECIAL
    assert (
        observation.cells[opponent_special[0]][opponent_special[1]]
        is Cell.OPPONENT_SPECIAL
    )
    assert observation.self_special_count == 2
    assert observation.opponent_special_count == 2
    assert observation.self_active_special_count == 0
    opponent_reading = next(
        reading
        for reading in observation.readings
        if (reading.row, reading.col) == opponent_special
    )
    assert opponent_reading.special_active is False


def test_board_reader_detects_active_special_highlights() -> None:
    catalog = GameCatalog.from_package()
    stage = catalog.stages["Rectangle"]
    reader = BoardGridReader(catalog)
    reference = np.full(
        (stage.rows * reader.CELL_SIZE, stage.cols * reader.CELL_SIZE, 3),
        (180, 180, 180),
        dtype=np.uint8,
    )
    current = reference.copy()
    empty_cells = [
        (row, col)
        for row in range(stage.rows)
        for col in range(stage.cols)
        if stage.cells[row][col] is Cell.EMPTY
    ]
    for row, col, color in (
        (*empty_cells[4], (12, 210, 240)),
        (*empty_cells[5], (244, 252, 165)),
    ):
        y0 = row * reader.CELL_SIZE
        x0 = col * reader.CELL_SIZE
        cell = current[
            y0 : y0 + reader.CELL_SIZE,
            x0 : x0 + reader.CELL_SIZE,
        ]
        cell[:] = color
        cell[13:19, 13:19] = (255, 255, 255)

    observation = reader.read_warped(current, stage, reference)

    assert observation.self_active_special_count == 1
    assert observation.opponent_active_special_count >= 1
    opponent_reading = next(
        reading
        for reading in observation.readings
        if (reading.row, reading.col) == empty_cells[5]
    )
    assert opponent_reading.special_active is True

