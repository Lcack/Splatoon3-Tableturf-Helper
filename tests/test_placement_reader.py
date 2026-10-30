from dataclasses import replace
from pathlib import Path

import cv2

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.board_reader import BoardGridReader
from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader


ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = (
    ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-221749"
    / "session-20261021-222704_annotations"
)


def test_placement_preview_recovers_rotation_and_origin() -> None:
    catalog = GameCatalog.from_package()
    board_reader, profile, region, reference = BoardGridReader.from_profile(
        catalog,
        ROOT / "profiles" / "vision" / "对战场地-正直大道.json",
        "Rectangle",
    )
    frame = cv2.imread(str(FRAME_DIR / "017_video_state.png"))
    assert frame is not None
    board = board_reader.read(
        frame,
        region,
        "Rectangle",
        reference,
        profile,
    )

    preview = PlacementPreviewReader(catalog).read(
        board,
        "MiniGame_Mother",
    )

    assert preview is not None
    # The special tile is detected through its animated dot texture.
    assert len(preview.cells) == 15
    assert preview.rotation == 2
    assert preview.origin == (20, 2)
    assert preview.anchor == (20, 4)
    assert preview.score >= 0.90
    assert preview.valid is False
    assert preview.valid_fraction < 0.20


def test_raw_preview_recovers_diagonal_grid_and_anchor() -> None:
    catalog = GameCatalog.from_package()
    board_reader, profile, region, reference = BoardGridReader.from_profile(
        catalog,
        ROOT / "profiles" / "vision" / "对战场地-正直大道.json",
        "Rectangle",
    )
    frame = cv2.imread(str(FRAME_DIR / "017_video_state.png"))
    assert frame is not None
    board = board_reader.read(
        frame,
        region,
        "Rectangle",
        reference,
        profile,
    )

    preview = PlacementPreviewReader(catalog).read_frame(
        frame,
        region,
        "Rectangle",
        "MiniGame_Mother",
        board,
    )

    assert preview is not None
    assert preview.rotation == 2
    assert preview.origin == (20, 2)
    assert preview.anchor == (20, 4)
    assert preview.score >= 0.90
    assert preview.valid is False


def test_placement_validity_distinguishes_gray_and_playable_preview() -> None:
    catalog = GameCatalog.from_package()
    board_reader, profile, region, reference = BoardGridReader.from_profile(
        catalog,
        ROOT / "profiles" / "vision" / "对战场地-正直大道.json",
        "Rectangle",
    )
    reader = PlacementPreviewReader(catalog)
    invalid_frame = cv2.imread(
        str(
            ROOT
            / "artifacts"
            / "state_capture"
            / "run-20261021-110258"
            / "007_state.png"
        )
    )
    valid_frame = cv2.imread(
        str(
            ROOT
            / "artifacts"
            / "state_capture"
            / "run-20261021-110258"
            / "008_state.png"
        )
    )
    assert invalid_frame is not None and valid_frame is not None

    invalid_board = board_reader.read(
        invalid_frame,
        region,
        "Rectangle",
        reference,
        profile,
    )
    valid_board = board_reader.read(
        valid_frame,
        region,
        "Rectangle",
        reference,
        profile,
    )
    invalid = reader.read(invalid_board, "MiniGame_Denchinamazu")
    valid = reader.read(valid_board, "MiniGame_Denchinamazu")

    assert invalid is not None and invalid.valid is False
    assert valid is not None and valid.valid is True
    assert valid.valid_fraction >= 0.55


def test_reference_only_edges_are_not_treated_as_a_preview() -> None:
    catalog = GameCatalog.from_package()
    board_reader, profile, region, reference = BoardGridReader.from_profile(
        catalog,
        ROOT / "profiles" / "vision" / "对战场地-正直大道.json",
        "Rectangle",
    )
    frame = cv2.imread(str(FRAME_DIR / "017_video_state.png"))
    assert frame is not None
    board = board_reader.read(
        frame,
        region,
        "Rectangle",
        reference,
        profile,
    )
    unchanged = replace(
        board,
        readings=tuple(
            replace(reading, color_change=0.0)
            for reading in board.readings
        ),
    )

    preview = PlacementPreviewReader(catalog).read(
        unchanged,
        "MiniGame_Mother",
    )

    assert preview is None
