from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild import GameCatalog
from tableturf_rebuild.vision.stage_assets import StageAssetIndex
from tableturf_rebuild.vision.stage_matcher import (
    StagePreviewMatcher,
    validate_expected_stage,
)


STAGE_DIR = Path(r"E:\Splatoon\tableturf\AutoTableTurf-rebuild\assets\reference\stage")


def test_stage_asset_index_and_preview_match() -> None:
    catalog = GameCatalog.from_package()
    assets = StageAssetIndex.load(catalog, STAGE_DIR)
    assert len(assets.stages) == 15

    source = cv2.imread(
        str(assets.stages["Rectangle"].path),
        cv2.IMREAD_UNCHANGED,
    )
    assert source is not None
    canvas = np.zeros((360, 360, 3), dtype=np.uint8)
    color = source[:, :, :3]
    alpha = source[:, :, 3]
    resized = cv2.resize(color, (300, 300), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (300, 300), interpolation=cv2.INTER_AREA)
    mask = alpha > 0
    canvas[30:330, 30:330][mask] = resized[mask]
    blurred = cv2.GaussianBlur(canvas, (3, 3), 0)

    matcher = StagePreviewMatcher(assets)
    matches = matcher.match(blurred)
    valid, message = validate_expected_stage("Rectangle", matches)

    assert matches[0].stage_id == "Rectangle"
    assert valid, message
