from pathlib import Path
import json
import runpy
import time

import cv2
import pytest

from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.legacy_grid import fit_empty_cell_grid
from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("sample, expected, rows, cols", [
    ("019_state.png", "Zigzag", 22, 16),
    ("016_state.png", "SShape", 25, 17),
])
def test_original_empty_cell_method_finds_actual_grid(sample, expected, rows, cols):
    frame = cv2.imread(str(ROOT / "artifacts/state_capture/run-20261023-101741" / sample))
    fit = fit_empty_cell_grid(frame, GameCatalog.from_package())
    assert fit is not None
    assert fit.stage_id == expected
    assert (fit.region.grid_rows, fit.region.grid_cols) == (rows, cols)
    assert 39 <= fit.step_x <= 41 and 39 <= fit.step_y <= 41


def test_grid_corrects_wrong_map_without_replacing_calibrated_board_roi():
    frame = cv2.imread(str(next((ROOT / "artifacts/task_debug/run-20261025-133539").rglob("*.png"))))
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("Zigzag")
    for _ in range(8):
        observation = runtime.update(frame)
    assert runtime.stage_id == "SShape"
    assert runtime.board_region.kind == "quad"
    assert runtime.preview_region.kind == "rect"
    assert observation.board.self_count == observation.board.opponent_count == 1
    preview = PlacementPreviewReader(runtime.catalog).read_frame(
        frame, runtime.preview_region, runtime.stage_id, "MiniGame_ShockSonar", observation.board,
    )
    assert preview is not None and preview.score == 1
    assert preview.origin == (20, -1)
    assert len(preview.cells) == 12


def test_reader_selects_active_fresh_session_over_newer_idle_window(tmp_path):
    root = tmp_path / "live_task"
    active = root / "run-active"
    active.mkdir(parents=True)
    idle_value = {"running": True, "updated_unix": time.time(), "mode": "idle", "pid": 1}
    (root / "latest.json").write_text(json.dumps(idle_value), encoding="utf-8")
    active_value = {**idle_value, "updated_unix": time.time() - 1, "mode": "battle_debug", "pid": 2}
    (active / "latest.json").write_text(json.dumps(active_value), encoding="utf-8")
    read = runpy.run_path(str(ROOT / "tools/read_live_task.py"))["read_snapshot"]
    result = read(root / "latest.json")
    assert result["pid"] == 2 and result["live"] is True
