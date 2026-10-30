"""Replay saved battle frames without connecting or sending controller input.

Example (run from the project root):
  python tools/diagnose_battle.py frame.png --stage SShape --card MiniGame_ShockSonar
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.deck_config import load_deck_config
from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader


def diagnose(runtime, frame, card_id=None):
    runtime.reset()
    # This is a stationary-frame check, not a video-transition benchmark.
    for _ in range(8):
        observation = runtime.update(frame, include_selection=True)
    board = observation.board
    raw = runtime.state_machine.recognizer.classify(frame)
    result = {
        "stage_id": runtime.stage_id,
        "raw_state": raw.state.value,
        "stable_state": observation.state.state.value,
        "state_candidates": [
            {"state": item.state.value, "score": item.score}
            for item in observation.state.candidates
        ],
        "hand_visibility": asdict(observation.hand_visibility) if observation.hand_visibility else None,
        "selection_focus": observation.selection_profile.focus_slot if observation.selection_profile else None,
        "hand_confidence": observation.hand.confidence if observation.hand else None,
        "hand": [
            {"slot": [card.slot_row, card.slot_col], "id": card.card_id,
             "name": card.name, "score": card.match_score}
            for card in observation.hand.cards
        ] if observation.hand else [],
        "hud_score": asdict(observation.score) if observation.score else None,
        "board": None,
        "preview": None,
    }
    if board is not None:
        result["board"] = {
            "rows": board.rows, "cols": board.cols,
            "self_count": board.self_count, "opponent_count": board.opponent_count,
            "preview_present": board.placement_preview_present,
            "diagonal_cells": [
                [item.row, item.col] for item in board.readings
                if item.preview_diagonal_score > 0.15
            ],
        }
        if card_id is not None:
            reader = PlacementPreviewReader(runtime.catalog)
            preview = reader.read_frame(
                frame, runtime.board_region, runtime.stage_id, card_id, board,
            )
            if preview is not None:
                result["preview"] = {
                    **asdict(preview), "cells": sorted(preview.cells),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description="离线检查战斗识别，不发送控制器按键")
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument("--stage", required=True, help="使用已标定的地图ID")
    parser.add_argument("--card", help="摆放预览对应的卡牌ID")
    parser.add_argument("--deck", type=Path, help="预设卡组JSON，可提高选牌识别精度")
    parser.add_argument("--output", type=Path, help="保存诊断JSON")
    args = parser.parse_args()
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage(args.stage)
    if args.card and args.card not in runtime.catalog.cards:
        parser.error(f"未知卡牌ID: {args.card}")
    if args.deck:
        deck = load_deck_config(args.deck, runtime.catalog)
        runtime.set_expected_deck(deck.name, deck.card_ids)
    results = []
    for path in args.frames:
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"无法读取图片: {path}")
        results.append({"frame": str(path.resolve()), **diagnose(runtime, frame, args.card)})
    value = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value + "\n", encoding="utf-8")
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
