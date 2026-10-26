from pathlib import Path

import cv2

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.calibration import VisionProfile
from tableturf_rebuild.vision.deck_reader import DeckGridReader


ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = ROOT / "profiles" / "vision"
FRAME_DIR = (
    ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-221749"
    / "session-20261021-222704_annotations"
)
CUSTOM_FRAME = (
    ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261024-205207"
    / "001_state.png"
)


def test_deck_reader_reports_all_card_points() -> None:
    profile = VisionProfile.load(VISION_DIR / "卡组选择.json")
    frame = cv2.imread(str(Path(profile.notes.removeprefix("source="))))
    assert frame is not None
    reader = DeckGridReader(GameCatalog.from_package(), VISION_DIR)

    observation = reader.read(frame)

    assert len(observation.cards) == 15
    assert observation.total_area == 131
    assert observation.total_sp_cost == 51
    assert [card.area for card in observation.cards] == [
        8,
        6,
        9,
        8,
        8,
        14,
        9,
        11,
        5,
        8,
        5,
        10,
        12,
        3,
        15,
    ]


def test_deck_fingerprint_detects_deck_change_and_ignores_same_frame() -> None:
    reader = DeckGridReader(GameCatalog.from_package(), VISION_DIR)
    first = cv2.imread(str(FRAME_DIR / "013_video_state.png"))
    other = cv2.imread(str(FRAME_DIR / "012_video_state.png"))
    assert first is not None and other is not None

    fingerprint = reader.fingerprint(first)
    repeated = reader.fingerprint(first)
    changed = reader.fingerprint(other)

    assert not fingerprint.materially_different(repeated)
    assert fingerprint.materially_different(changed)


def test_deck_reader_recognizes_custom_deck_without_npc_preset() -> None:
    frame = cv2.imread(str(CUSTOM_FRAME))
    assert frame is not None
    reader = DeckGridReader(GameCatalog.from_package(), VISION_DIR)

    observation = reader.read(frame)

    assert observation.deck_id == "CUSTOM_AUTO"
    assert [card.area for card in observation.cards] == [
        2,
        4,
        5,
        6,
        7,
        7,
        9,
        9,
        11,
        12,
        12,
        12,
        13,
        13,
        14,
    ]
    assert [card.sp_cost for card in observation.cards] == [
        1,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        4,
        3,
        5,
        3,
        5,
        5,
        5,
    ]
    assert observation.total_area == 136
    assert observation.total_sp_cost == 52
    assert all(card.match_score == 1.0 for card in observation.cards)
