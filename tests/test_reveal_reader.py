from pathlib import Path

import cv2

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.reveal_reader import RevealReader


ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = ROOT / "profiles" / "vision"
FRAME_DIR = (
    ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-221749"
    / "session-20261021-222704_annotations"
)


def test_reveal_reads_both_card_shapes_points_and_sp_costs() -> None:
    catalog = GameCatalog.from_package()
    reader = RevealReader(catalog, VISION_DIR)
    frame = cv2.imread(str(FRAME_DIR / "034_video_state.png"))
    assert frame is not None

    observation = reader.read(
        frame,
        own_card_ids=catalog.decks["MiniGame_Starter"].card_ids,
    )

    assert observation.own.card_id == "MiniGame_SaberLight00"
    assert observation.own.area == 5
    assert observation.own.sp_cost == 2
    assert observation.own.visible_cost_lights == 2
    assert observation.opponent.card_id == "MiniGame_Mother"
    assert observation.opponent.area == 15
    assert observation.opponent.sp_cost == 5
    assert observation.opponent.visible_cost_lights == 5


def test_reveal_matches_identical_shapes_on_both_sides() -> None:
    catalog = GameCatalog.from_package()
    reader = RevealReader(catalog, VISION_DIR)
    frame = cv2.imread(str(FRAME_DIR / "035_video_state.png"))
    assert frame is not None

    observation = reader.read(frame)

    assert observation.own.card_id == "MiniGame_TakoDozer"
    assert observation.own.area == 14
    assert observation.own.sp_cost == 5
    assert observation.opponent.card_id == "MiniGame_TakoDozer"
    assert observation.opponent.area == 14
    assert observation.opponent.sp_cost == 5
