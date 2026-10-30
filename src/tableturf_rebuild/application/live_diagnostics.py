"""Local, read-only diagnostic files for a running task terminal."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import time
import uuid
import numbers


def _json_default(value):
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported diagnostic value: {type(value).__name__}")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def observation_details(observation) -> dict | None:
    if observation is None:
        return None
    board = observation.board
    return {
        "state": observation.state.state.value,
        "grid_fit": observation.grid_fit,
        "confidence": observation.state.confidence,
        "timings_ms": getattr(observation, "timings_ms", None),
        "sp": asdict(observation.sp) if getattr(observation, "sp", None) else None,
        "special_mode": getattr(observation, "special_mode", None),
        "mode_evidence": asdict(observation.mode_evidence) if getattr(observation, "mode_evidence", None) else None,
        "discard_mode": getattr(observation, "discard_mode", None),
        "discard_evidence": asdict(observation.discard_evidence) if getattr(observation, "discard_evidence", None) else None,
        "replay_choice_evidence": asdict(observation.replay_choice_evidence) if getattr(observation, "replay_choice_evidence", None) else None,
        "reward_page_evidence": asdict(observation.reward_page_evidence) if getattr(observation, "reward_page_evidence", None) else None,
        "reveal": asdict(observation.reveal) if getattr(observation, "reveal", None) else None,
        "candidates": [
            {"state": item.state.value, "score": item.score}
            for item in observation.state.candidates
        ],
        "hand_visibility": asdict(observation.hand_visibility) if observation.hand_visibility else None,
        "hand_confidence": observation.hand.confidence if observation.hand else None,
        "hand_deck_mismatch": getattr(observation.hand, "deck_mismatch", False),
        "hand_unknown_card_ids": getattr(observation.hand, "unknown_card_ids", ()),
        "hand": [asdict(card) for card in observation.hand.cards] if observation.hand else [],
        "focus": observation.selection_profile.focus_slot if observation.selection_profile else None,
        "board": {
            "stage_id": board.stage_id, "rows": board.rows, "cols": board.cols,
            "confidence": board.confidence,
            "self_count": board.self_count, "opponent_count": board.opponent_count,
            "preview_present": board.placement_preview_present,
            "cells": board.cells,
            "readings": [asdict(reading) for reading in board.readings],
        } if board else None,
        "score": asdict(observation.score) if observation.score else None,
        "match_result": asdict(observation.match_result) if observation.match_result else None,
    }


def explain_snapshot(data: dict) -> str:
    """Describe observed waits without inventing a causal diagnosis."""
    capture = data.get("capture", {})
    analysis = data.get("analysis", {})
    observation = data.get("observation") or {}
    executor = data.get("executor") or {}
    if not data.get("running", True):
        return "任务终端已关闭，保留的是最后一次现场。"
    if capture.get("error"):
        return "采集错误：" + capture["error"]
    if capture.get("sequence") is None:
        return "尚未收到采集画面，当前无法识别战斗。"
    if capture.get("age_seconds", 0) > 3:
        return f"采集已 {capture['age_seconds']:.1f} 秒没有新帧。"
    if analysis.get("busy") and analysis.get("busy_seconds", 0) >= 3:
        return f"分析线程尚未返回，已运行 {analysis['busy_seconds']:.1f} 秒；当前显示的是上次结果。"
    if data.get("error"):
        return "分析异常，完整错误已写入诊断文件。"
    if executor.get("phase") == "ERROR":
        return "执行器已停止：" + executor.get("message", "")
    if observation.get("hand_deck_mismatch"):
        return "实际手牌出现预设卡组之外的牌形，不能继续使用旧卡牌身份执行；需要核对本局卡组。"
    state = observation.get("state", "UNKNOWN")
    if state in {"MATCH_RESULT", "MATCH_REWARD", "REPLAY_CHOICE"}:
        return data.get("message") or "等待结算页面确认。"
    if data.get("mode") == "idle":
        return f"当前仅观察采集画面，没有自动对战任务运行；识别状态 {state}。"
    visibility = observation.get("hand_visibility") or {}
    board = observation.get("board") or {}
    if state in {"TURN_REVEAL", "TURN_RESOLVE"}:
        return (
            f"当前被识别为 {state}，程序只等待动画，没有运行卡牌预览定位。"
            + ("手牌区域灰暗；棋盘未检出斜线，需结合保存的原图核对。"
               if visibility.get("selection_visible") is False and not board.get("preview_present") else "")
        )
    attempts = executor.get("preview_attempts", [])
    if state == "TARGET_PLACEMENT" and attempts and not executor.get("preview"):
        raw = next((item for item in attempts if item.get("source") == "padded_frame"), attempts[0])
        if raw.get("failure") == "NO_DIAGONAL_CELLS":
            return "已进入摆放定位，但采样格中没有斜线分数超过阈值；原图、ROI和各格分数已保存。"
        if raw.get("failure") == "SHAPE_MISMATCH":
            return (
                f"检出 {raw.get('candidate_count', 0)} 个斜线格，但与选中卡牌形状不匹配；"
                f"最高匹配 {raw.get('best_score', 0):.2f} / 要求 {raw.get('minimum_shape_score', .72):.2f}。"
            )
    if state == "TURN_SELECT" and (observation.get("hand_confidence") or 0) < 0.55:
        return "正在等待四张手牌可靠识别，当前置信度不足。"
    return executor.get("message") or data.get("message") or "等待首次分析结果。"


class LiveDiagnostics:
    def __init__(self, root: Path):
        self.root = root.resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_dir = self.root / f"run-{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.session_dir.mkdir(parents=True)
        self._previous_key = None
        self._image_sequence = -1
        self._analysis_sequence = -1
        self._last_evidence_at = -float("inf")
        self._last_error_key = None
        self._result_evidence_keys = set()

    @staticmethod
    def _write_image(path: Path, frame, extension: str) -> None:
        import cv2

        ok, encoded = cv2.imencode(extension, frame)
        if not ok:
            raise OSError("诊断图片编码失败")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(path)

    def publish(self, data: dict, *, frame=None, analysis_frame=None, sequence=-1, analysis_sequence=-1) -> dict:
        data = dict(data)
        data.update({
            "schema_version": 1, "pid": os.getpid(), "session_dir": str(self.session_dir),
            "updated_at": datetime.now(timezone.utc).isoformat(), "updated_unix": time.time(),
        })
        if frame is not None and sequence != self._image_sequence:
            self._write_image(self.session_dir / "capture.jpg", frame, ".jpg")
            self._image_sequence = sequence
        if analysis_frame is not None and analysis_sequence != self._analysis_sequence:
            # Full-resolution PNG preserves diagonal texture for offline replay.
            self._write_image(self.session_dir / "analysis.png", analysis_frame, ".png")
            self._analysis_sequence = analysis_sequence
        data["capture_frame"] = str(self.session_dir / "capture.jpg") if self._image_sequence != -1 else None
        data["analysis_frame"] = str(self.session_dir / "analysis.png") if self._analysis_sequence != -1 else None
        data["diagnosis"] = explain_snapshot(data)
        key = (data.get("mode"), data.get("message"),
               (data.get("observation") or {}).get("state"),
               (data.get("executor") or {}).get("action"), data.get("error"))
        key = json.dumps(key, sort_keys=True, default=_json_default)
        if key != self._previous_key:
            event = {k: data.get(k) for k in ("updated_at", "mode", "message", "diagnosis", "error")}
            event["state"] = (data.get("observation") or {}).get("state")
            event["executor"] = {k: (data.get("executor") or {}).get(k)
                                 for k in ("phase", "action", "preview", "last_command")}
            with (self.session_dir / "events.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
            self._previous_key = key
        executor = data.get("executor") or {}
        issue = (bool(data.get("error")) or executor.get("phase") == "ERROR"
                 or executor.get("preview_missing_reads", 0) > 0
                 or (executor.get("rejection_recovery") or {}).get("pending", False))
        observation = data.get("observation") or {}
        settlement_states = {"MATCH_RESULT", "MATCH_REWARD", "REPLAY_CHOICE"}
        state = observation.get("state")
        # De-duplicate within one match, rather than discarding every later
        # match's reward images while the same analysis core stays loaded.
        if state not in settlement_states | {"UNKNOWN"}:
            getattr(self, "_result_evidence_keys", set()).clear()
        result_key = (observation.get("state"),
                      (observation.get("match_result") or {}).get("outcome"),
                      data.get("core_generation"),
                      (observation.get("reward_page_evidence") or {}).get("page_kind"))
        now = time.monotonic()
        waiting = state in settlement_states and data.get("status") == "WAITING_RECOGNITION"
        if not waiting:
            self._settlement_wait_key = None
        elif getattr(self, "_settlement_wait_key", None) != result_key:
            self._settlement_wait_key = result_key
            self._settlement_wait_since = now
            self._settlement_wait_saved = False
        stalled = (waiting and not getattr(self, "_settlement_wait_saved", False)
                   and now - self._settlement_wait_since >= 5 and analysis_frame is not None)
        # Keep the first unknown and the first definite result, rather than
        # losing the only evidence when a later menu/button advances the UI.
        result_evidence = (state in settlement_states
                           and result_key not in getattr(self, "_result_evidence_keys", set()))
        if result_evidence or stalled or (issue and key != self._last_error_key and now - self._last_evidence_at >= 10):
            evidence = self.session_dir / "evidence"
            evidence.mkdir(exist_ok=True)
            name = datetime.now(timezone.utc).strftime("%H%M%S-%f")
            if analysis_frame is not None:
                path = evidence / f"{name}.png"
                self._write_image(path, analysis_frame, ".png")
                data["evidence_frame"] = str(path)
            _write_json(evidence / f"{name}.json", data)
            self._last_error_key = key
            self._last_evidence_at = time.monotonic()
            if stalled:
                self._settlement_wait_saved = True
            if result_evidence:
                if not hasattr(self, "_result_evidence_keys"):
                    self._result_evidence_keys = set()
                self._result_evidence_keys.add(result_key)
        _write_json(self.session_dir / "latest.json", data)
        _write_json(self.root / "latest.json", data)
        return data
