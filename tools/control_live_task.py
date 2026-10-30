"""Read/control the running terminal through its loopback API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description="通过本机接口读取画面、暂停、重载和发送有时限的按键")
    parser.add_argument("--endpoint", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/live_task/control_endpoint.json")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--compact", action="store_true")
    frame = sub.add_parser("frame")
    frame.add_argument("--kind", choices=("capture", "analysis"), default="capture")
    frame.add_argument("--output", type=Path, required=True)
    press = sub.add_parser("press")
    press.add_argument("buttons", nargs="+")
    press.add_argument("--hold-ms", type=int, default=120)
    for name in ("pause", "resume", "reload", "release"):
        sub.add_parser(name)
    args = parser.parse_args()
    endpoint = json.loads(args.endpoint.read_text(encoding="utf-8"))
    if not endpoint.get("running"):
        parser.error("本机控制接口已关闭")

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(endpoint["url"] + path, data=data, headers={
            "X-Tableturf-Token": endpoint["token"], "Content-Type": "application/json",
        })
        with urlopen(req, timeout=8) as response:
            body = response.read()
            if path.startswith("/frame"):
                return body, dict(response.headers)
            return json.loads(body)

    if args.command == "frame":
        data, headers = request("/frame?kind=" + args.kind)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        print(json.dumps({"path": str(args.output.resolve()),
                          "age_seconds": headers.get("X-Frame-Age-Seconds"),
                          "sequence": headers.get("X-Frame-Sequence")}, ensure_ascii=False))
    elif args.command == "status":
        value = request("/status")
        if args.compact:
            observation, executor = value.get("observation") or {}, value.get("executor") or {}
            value = {"status": value.get("status"), "message": value.get("message"),
                     "state": observation.get("state"), "core_generation": value.get("core_generation"),
                     "snapshot_age": value.get("read_age_seconds"), "frames": value.get("frames"),
                     "action": executor.get("action"), "preview": executor.get("preview"),
                     "preview_attempts": executor.get("preview_attempts"), "control": value.get("control")}
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        payload = {"command": args.command}
        if args.command == "press":
            payload.update(buttons=args.buttons, hold_ms=args.hold_ms)
        receipt = request("/commands", payload)
        deadline = time.monotonic() + 15
        while receipt["status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.1)
            receipt = request("/commands/" + receipt["id"])
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        if receipt["status"] != "succeeded":
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, HTTPError, URLError) as error:
        print(f"本机控制接口调用失败: {error}")
        raise SystemExit(1)
