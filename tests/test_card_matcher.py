from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild import GameCatalog
from tableturf_rebuild.vision.card_assets import CardAssetIndex
from tableturf_rebuild.vision.card_matcher import CardArtMatcher, rectify_card


CARD_DIR = Path(r"E:\Splatoon\splat3\images\minigame\card")
SLEEVE_DIR = Path(r"E:\Splatoon\splat3\images\minigame\sleeve")


def test_card_asset_index_maps_all_cards_and_sleeves() -> None:
    catalog = GameCatalog.from_package()
    assets = CardAssetIndex.load(catalog, CARD_DIR, SLEEVE_DIR)

    assert len(assets.faces) == 266
    assert assets.face("MiniGame_Ajio").name == "Ajio"
    assert (
        assets.sleeve("MngCardSleeve_Aori").path.name
        == "MngCardSleeve_Aori.png"
    )


def test_card_matcher_identifies_synthetic_perspective_card() -> None:
    catalog = GameCatalog.from_package()
    assets = CardAssetIndex.load(catalog, CARD_DIR, SLEEVE_DIR)
    source = cv2.imread(
        str(assets.face("MiniGame_Ajio").path),
        cv2.IMREAD_UNCHANGED,
    )
    assert source is not None
    canvas = np.full((1200, 900, 3), 28, dtype=np.uint8)
    corners = np.float32(
        [[180, 120], [690, 160], [620, 1080], [90, 1000]]
    )
    source_corners = np.float32(
        [
            [0, 0],
            [source.shape[1] - 1, 0],
            [source.shape[1] - 1, source.shape[0] - 1],
            [0, source.shape[0] - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source_corners, corners)
    color = cv2.warpPerspective(source[:, :, :3], matrix, (900, 1200))
    alpha = cv2.warpPerspective(source[:, :, 3], matrix, (900, 1200))
    mask = alpha > 0
    canvas[mask] = color[mask]
    image = rectify_card(canvas, corners)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    noise = np.random.default_rng(3).normal(0, 3, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    matcher = CardArtMatcher(assets, max_features=900)
    matches = matcher.match(
        image,
        ("MiniGame_Ajio", "MiniGame_Aori", "MiniGame_Hotaru"),
    )

    assert matches
    assert matches[0].card_id == "MiniGame_Ajio"
    assert matches[0].inliers >= 4
