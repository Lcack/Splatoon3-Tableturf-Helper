import json
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from tableturf_rebuild.application.live_diagnostics import LiveDiagnostics, explain_snapshot
from tableturf_rebuild.domain.actions import Action


def base_snapshot():
    return {
        "running": True, "mode": "battle_debug", "message": "等待预览",
        "capture": {"sequence": 1, "age_seconds": 0.1},
        "analysis": {"busy": False},
        "observation": {"state": "TARGET_PLACEMENT"},
        "executor": {"phase": "WAITING"},
    }


def test_snapshot_keeps_exact_analysis_frame_and_deduplicates_events(tmp_path):
    writer = LiveDiagnostics(tmp_path)
    data = base_snapshot()
    data["numpy_scalar"] = np.int64(7)
    data["executor"]["action"] = {"kind": Action.place("test", 0, (1, 2)).action_type}
    raw = np.zeros((16, 16, 3), dtype=np.uint8)
    result = writer.publish(data, frame=raw, analysis_frame=raw, sequence=1, analysis_sequence=1)
    assert Path(result["capture_frame"]).is_file()
    assert Path(result["analysis_frame"]).suffix == ".png"
    first_png = Path(result["analysis_frame"]).read_bytes()
    writer.publish(data, frame=raw + 200, analysis_frame=raw + 200, sequence=2, analysis_sequence=1)
    assert Path(result["analysis_frame"]).read_bytes() == first_png
    assert len((writer.session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["executor"]["action"]["kind"] == "place"
    assert latest["updated_unix"] <= time.time()


@pytest.mark.parametrize("failure, text", [
    ("NO_DIAGONAL_CELLS", "没有斜线分数超过阈值"),
    ("SHAPE_MISMATCH", "形状不匹配"),
])
def test_preview_failure_reasons_are_distinct(failure, text):
    data = base_snapshot()
    data["executor"]["preview_attempts"] = [{
        "source": "padded_frame", "failure": failure,
        "candidate_count": 4, "best_score": 0.4,
    }]
    assert text in explain_snapshot(data)


def test_wrong_state_is_not_reported_as_preview_match_failure():
    data = base_snapshot()
    data["observation"] = {
        "state": "TURN_REVEAL", "hand_visibility": {"selection_visible": False},
        "board": {"preview_present": False},
    }
    assert "没有运行卡牌预览定位" in explain_snapshot(data)


def test_stale_capture_and_busy_analysis_are_separate():
    data = base_snapshot()
    data["analysis"] = {"busy": True, "busy_seconds": 5}
    assert "分析线程尚未返回" in explain_snapshot(data)
    data["capture"]["age_seconds"] = 4
    assert "没有新帧" in explain_snapshot(data)


def test_error_evidence_is_preserved_and_shutdown_is_explicit(tmp_path):
    writer = LiveDiagnostics(tmp_path)
    data = base_snapshot()
    data["executor"]["phase"] = "ERROR"
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    result = writer.publish(data, analysis_frame=frame, analysis_sequence=1)
    assert Path(result["evidence_frame"]).is_file()
    assert len(list((writer.session_dir / "evidence").glob("*.json"))) == 1
    writer.publish(data, analysis_frame=frame, analysis_sequence=1)
    assert len(list((writer.session_dir / "evidence").glob("*.json"))) == 1
    data["running"] = False
    assert "已关闭" in writer.publish(data)["diagnosis"]


def test_unknown_and_confirmed_results_each_keep_an_evidence_frame(tmp_path):
    writer = LiveDiagnostics(tmp_path)
    data = base_snapshot()
    data["observation"] = {"state": "MATCH_RESULT", "match_result": {"outcome": "UNKNOWN"}}
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    writer.publish(data, analysis_frame=frame, analysis_sequence=1)
    writer.publish(data, analysis_frame=frame, analysis_sequence=1)
    assert len(list((writer.session_dir / "evidence").glob("*.json"))) == 1
    data["observation"]["match_result"]["outcome"] = "WIN"
    writer.publish(data, analysis_frame=frame, analysis_sequence=2)
    assert len(list((writer.session_dir / "evidence").glob("*.json"))) == 2


def test_fresh_publisher_keeps_the_existing_pre_reload_writer_session(tmp_path):
    writer = LiveDiagnostics(tmp_path)
    class PreviousWriter:
        _write_image = staticmethod(LiveDiagnostics._write_image)
    previous = PreviousWriter()
    previous.__dict__.update(writer.__dict__)
    del previous._result_evidence_keys
    data = base_snapshot()
    data["observation"] = {"state": "MATCH_RESULT", "match_result": {"outcome": "UNKNOWN"}}
    result = LiveDiagnostics.publish(previous, data,
        analysis_frame=np.zeros((16, 16, 3), dtype=np.uint8), analysis_sequence=1)
    assert result["session_dir"] == str(writer.session_dir)
    assert Path(result["evidence_frame"]).is_file()


def test_later_match_with_same_outcome_keeps_new_reward_evidence(tmp_path):
    writer = LiveDiagnostics(tmp_path)
    data = base_snapshot()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    for _ in range(2):
        data["observation"] = {"state": "TURN_SELECT"}
        writer.publish(data, analysis_frame=image)
        data["observation"] = {"state": "MATCH_REWARD"}
        writer.publish(data, analysis_frame=image)
        writer.publish(data, analysis_frame=image)
    assert len(list((writer.session_dir / "evidence").glob("*.json"))) == 2


def test_stalled_settlement_saves_one_later_frame_without_unbounded_duplicates(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("tableturf_rebuild.application.live_diagnostics.time.monotonic", lambda: clock[0])
    writer = LiveDiagnostics(tmp_path)
    data = base_snapshot()
    data.update(status="WAITING_RECOGNITION", message="等待奖励页")
    data["observation"] = {"state": "MATCH_REWARD"}
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    writer.publish(data, analysis_frame=image)
    clock[0] += 5
    saved = writer.publish(data, analysis_frame=image + 200)
    encoded = np.frombuffer(Path(saved["evidence_frame"]).read_bytes(), dtype=np.uint8)
    assert cv2.imdecode(encoded, cv2.IMREAD_COLOR).mean() == 200
    clock[0] += 600
    writer.publish(data, analysis_frame=image)
    assert len(list((writer.session_dir / "evidence").glob("*.json"))) == 2
