from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.calibration import VisionProfile
from tableturf_rebuild.vision.npc_menu_reader import (
    NpcMenuReader,
    _expand_name_region,
)


ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = ROOT / "profiles" / "vision"
LIVE_DIR = ROOT / "artifacts" / "state_capture" / "run-20261023-101741"


def _source(profile_name: str):
    profile = VisionProfile.load(VISION_DIR / profile_name)
    image = cv2.imread(str(Path(profile.notes.removeprefix("source="))))
    assert image is not None
    return image


def test_npc_roster_reads_name_and_wins() -> None:
    observation = NpcMenuReader(VISION_DIR).read(
        _source("npc选择.json"),
        "NPC_ROSTER",
    )

    assert observation.npc_name == "武器子"
    assert observation.npc_id == "MiniGame_WeaponShop"
    assert observation.level_wins == (3, 0, 0)
    assert observation.name_confidence > 0.85


def test_difficulty_menu_reads_name_wins_and_selected_level() -> None:
    observation = NpcMenuReader(VISION_DIR).read(
        _source("难度选择.json"),
        "MATCH_SETUP",
    )

    assert observation.npc_name == "新手水母"
    assert observation.npc_id == "MiniGame_MiniGameKurage0"
    assert observation.level_wins == (3, 3, 3)
    assert observation.difficulty == 3
    assert observation.stage_id == "Rectangle"
    assert observation.stage_name == "正直大道"
    assert observation.binding_consistent is True
    assert observation.difficulty_confidence > 0.5
    assert observation.stage_confidence > 0.4


def test_difficulty_digit_font_distinguishes_levels_one_and_two() -> None:
    reader = NpcMenuReader(VISION_DIR)
    level_one = reader.read(
        cv2.imread(str(LIVE_DIR / "005_state.png")),
        "MATCH_SETUP",
    )
    level_two = reader.read(
        cv2.imread(str(LIVE_DIR / "006_state.png")),
        "MATCH_SETUP",
    )

    assert level_one.difficulty == 1
    assert level_one.difficulty_confidence > 0.8
    assert level_two.difficulty == 2
    assert level_two.difficulty_confidence > 0.8


def test_long_and_ambiguous_npc_names_are_recognized() -> None:
    reader = NpcMenuReader(VISION_DIR)
    xiaoji = reader.read(
        cv2.imread(str(LIVE_DIR / "001_state.png")),
        "NPC_ROSTER",
    )
    long_name = reader.read(
        cv2.imread(str(LIVE_DIR / "003_state.png")),
        "NPC_ROSTER",
    )

    assert xiaoji.npc_name == "小姬"
    assert long_name.npc_name == "独当一面水母"


def test_complex_npc_names_use_dual_font_and_stage_candidates() -> None:
    reader = NpcMenuReader(VISION_DIR)
    profile = reader.profiles["NPC_ROSTER"]
    region = _expand_name_region(profile.region_map()["npc名字"])
    points = np.array(
        [
            [x * (profile.source_width - 1), y * (profile.source_height - 1)]
            for x, y in region.points
        ]
    )
    x1, y1 = points.min(axis=0).astype(int)
    x2, y2 = points.max(axis=0).astype(int)
    font_path = (
        Path(r"E:\Splatoon\字体\华康新综艺体W7-P.ttf")
    )

    for text, stage_id in (("小葵", "SShape"), ("竹荚雄", "SmallSquare")):
        image = Image.new("RGB", (1920, 1080), (55, 175, 180))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(font_path), max(34, y2 - y1 - 18))
        box = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        width = box[2] - box[0]
        height = box[3] - box[1]
        draw.text(
            (
                (x1 + x2 - width) // 2,
                (y1 + y2 - height) // 2 - 4,
            ),
            text,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        frame = np.asarray(image)[:, :, ::-1]
        allowed = tuple(
            npc.name_zh
            for npc in GameCatalog.from_package().npcs.values()
            if stage_id in npc.map_ids
        )

        recognition = reader.name_recognizer.recognize(
            frame,
            region,
            names=allowed,
        )

        assert recognition.name == text
