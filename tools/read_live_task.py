"""Read the task terminal's local diagnostics without controlling the game."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time


def read_snapshot(path: Path) -> dict:
    paths = [path]
    if path.name == "latest.json" and path.parent.name == "live_task":
        paths.extend(path.parent.glob("run-*/latest.json"))
    candidates = []
    for candidate in paths:
        try:
            item = json.loads(candidate.read_text(encoding="utf-8"))
            age = max(0, time.time() - item["updated_unix"])
            fresh = item.get("running", False) and age <= 5
            active = item.get("mode") in {"task", "battle_debug"}
            candidates.append(((fresh, active, item["updated_unix"]), item))
        except (OSError, ValueError, KeyError):
            continue
    if not candidates:
        raise ValueError("no readable task session")
    value = max(candidates, key=lambda item: item[0])[1]
    value["read_age_seconds"] = max(0, time.time() - value["updated_unix"])
    value["live"] = value.get("running", False) and value["read_age_seconds"] <= 5
    return value


def main():
    parser = argparse.ArgumentParser(description="读取任务终端实时状态，不发送游戏按键")
    parser.add_argument("--path", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/live_task/latest.json")
    parser.add_argument("--json", action="store_true", help="输出完整JSON，包含逐格识别分数")
    parser.add_argument("--seconds", type=float, default=0, help="持续读取指定秒数，默认读取一次")
    parser.add_argument("--tail", type=int, default=0, help="同时显示最近若干行终端日志")
    args = parser.parse_args()
    if args.seconds < 0 or args.tail < 0:
        parser.error("seconds 和 tail 不能为负数")
    deadline = time.monotonic() + args.seconds
    while True:
        try:
            data = read_snapshot(args.path)
        except (OSError, ValueError, KeyError) as error:
            print(f"未读到有效的实时诊断：{error}。新版任务终端启动后会自动创建该文件。")
            return 1
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            observation = data.get("observation") or {}
            executor = data.get("executor") or {}
            freshness = "实时" if data["live"] else "历史快照/终端无心跳"
            print(f"[{freshness}] {data['updated_at']}，距更新 {data['read_age_seconds']:.1f}s")
            print(f"模式={data.get('mode')} 地图={data.get('stage_id')} 状态={observation.get('state')} 执行={executor.get('phase')}")
            print(data.get("diagnosis", ""))
            print(f"目标={executor.get('action')} 预览={executor.get('preview')} 最近按键={executor.get('last_command')}")
            print(f"当前采集={data.get('capture_frame')}\n分析原图={data.get('analysis_frame')}")
        if args.tail:
            log = Path(data["session_dir"]) / "terminal.log"
            if log.is_file():
                with log.open(encoding="utf-8") as file:
                    print("".join(deque(file, maxlen=args.tail)), end="")
        if time.monotonic() >= deadline:
            return 0
        time.sleep(min(1, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())
