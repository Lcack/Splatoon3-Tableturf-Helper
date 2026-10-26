from pathlib import Path

import cv2

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.hand_reader import HandReader


ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = ROOT / "profiles" / "vision"
FRAME_DIR = (
    ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-221749"
    / "session-20261021-222704_annotations"
)


def test_hand_reader_identifies_redraw_hand_from_known_deck() -> None:
    catalog = GameCatalog.from_package()
    reader = HandReader(catalog, VISION_DIR)
    frame = cv2.imread(str(FRAME_DIR / "016_video_state.png"))
    assert frame is not None

    observation = reader.read(
        frame,
        VISION_DIR / "battle_select_clean.json",
        catalog.decks["MiniGame_Starter"].card_ids,
    )

    assert [card.area for card in observation.cards] == [9, 5, 3, 8]
    assert [card.sp_cost for card in observation.cards] == [4, 2, 1, 3]
    assert observation.confidence == 1.0


def test_hand_reader_handles_selected_card_profile() -> None:
    catalog = GameCatalog.from_package()
    reader = HandReader(catalog, VISION_DIR)
    frame = cv2.imread(str(FRAME_DIR / "042_video_state.png"))
    assert frame is not None

    observation = reader.read(
        frame,
        VISION_DIR / "battle_select12.json",
        catalog.decks["MiniGame_Starter"].card_ids,
    )

    assert len(observation.cards) == 4
    assert [card.area for card in observation.cards] == [10, 8, 3, 5]
    assert observation.confidence >= 0.9
