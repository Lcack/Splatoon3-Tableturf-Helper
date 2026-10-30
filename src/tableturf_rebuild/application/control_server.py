"""Loopback diagnostics and queued controls for the existing task terminal.

HTTP handlers only read files/cache and enqueue commands. The app's main thread
must call ``drain``; this module never calls Tk or a controller backend.
"""
from __future__ import annotations

from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import secrets
import threading
import time
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from tableturf_rebuild.control.buttons import Button


COMMAND_DEFERRED = object()
COMMAND_NAMES = frozenset({"pause", "resume", "reload", "press", "release"})
FRAME_FILES = {"analysis": ("analysis.png", "image/png"), "capture": ("capture.jpg", "image/jpeg")}
MAX_REQUEST_BYTES = 8192
MAX_RETAINED_COMMANDS = 256


class _ControlHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True
    block_on_close = False
    request_queue_size = 8


class LocalControlServer:
    def __init__(self, session_dir: Path, *, preferred_port: int = 8765, max_pending: int = 32):
        if type(preferred_port) is not int or not 0 <= preferred_port <= 65535:
            raise ValueError("preferred_port must be 0..65535")
        if type(max_pending) is not int or not 1 <= max_pending <= 256:
            raise ValueError("max_pending must be 1..256")
        self.session_dir = Path(session_dir).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.preferred_port = preferred_port
        self.max_pending = max_pending
        self.token = secrets.token_urlsafe(32)
        self._pending: queue.Queue[dict] = queue.Queue(max_pending)
        self._urgent: queue.Queue[dict] = queue.Queue(max_pending)
        self._commands: OrderedDict[str, dict] = OrderedDict()
        self._frames: dict[str, tuple[bytes, int | None, float]] = {}
        self._lock = threading.RLock()
        self._server: _ControlHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._endpoint: dict | None = None
        self._closed = False

    @property
    def endpoint(self) -> dict | None:
        """Local discovery metadata, including the access token; do not log it."""
        with self._lock:
            return dict(self._endpoint) if self._endpoint is not None else None

    def start(self) -> dict:
        with self._lock:
            if self._endpoint is not None:
                if self._closed:
                    raise RuntimeError("a closed control server cannot be restarted")
                return dict(self._endpoint)
            owner = self

            class Handler(BaseHTTPRequestHandler):
                server_version = "TableturfControl/1"
                sys_version = ""

                def setup(self):
                    super().setup()
                    self.connection.settimeout(5)

                def log_message(self, *_):
                    pass

                def send_payload(self, status, body, content_type="application/json; charset=utf-8", headers=None):
                    if not isinstance(body, bytes):
                        body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store, max-age=0")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Connection", "close")
                    for key, value in (headers or {}).items():
                        self.send_header(key, str(value))
                    self.end_headers()
                    self.close_connection = True
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                def error_payload(self, status, message):
                    self.send_payload(status, {"error": message})

                def authorized(self):
                    # Token auth also prevents another local website from issuing
                    # commands. Do not enable CORS or accept browser Origin requests.
                    try:
                        host = urlsplit("//" + self.headers.get("Host", "")).hostname
                    except ValueError:
                        host = None
                    if host not in {"127.0.0.1", "localhost"} or self.headers.get("Origin") is not None:
                        self.error_payload(403, "only local non-browser clients are allowed")
                        return False
                    supplied = self.headers.get("X-Tableturf-Token", "").encode("utf-8")
                    if not secrets.compare_digest(supplied, owner.token.encode("utf-8")):
                        self.error_payload(401, "missing or invalid access token")
                        return False
                    return True

                def do_GET(self):
                    route = urlsplit(self.path)
                    if route.path == "/health" and not route.query:
                        self.send_payload(200, {"service": "tableturf-control", "schema_version": 1, "running": not owner._closed})
                        return
                    if not self.authorized():
                        return
                    if route.path == "/status" and not route.query:
                        try:
                            self.send_payload(200, owner.status())
                        except (OSError, ValueError):
                            self.error_payload(503, "diagnostic snapshot is not available yet")
                    elif route.path == "/frame":
                        query = parse_qs(route.query, keep_blank_values=True)
                        if set(query) - {"kind"} or len(query.get("kind", ["capture"])) != 1:
                            self.error_payload(400, "frame only accepts one kind=analysis|capture")
                            return
                        kind = query.get("kind", ["capture"])[0]
                        if kind not in FRAME_FILES:
                            self.error_payload(400, "kind must be analysis or capture")
                            return
                        try:
                            data, metadata = owner.frame(kind)
                        except OSError:
                            self.error_payload(503, "requested frame is not available yet")
                            return
                        self.send_payload(200, data, FRAME_FILES[kind][1], {
                            "X-Frame-Age-Seconds": f"{metadata['age_seconds']:.3f}",
                            "X-Frame-Source": metadata["source"],
                            "X-Frame-Sequence": metadata["sequence"] if metadata["sequence"] is not None else "",
                        })
                    elif route.path.startswith("/commands/") and not route.query:
                        command = owner.command_result(route.path[len("/commands/"):])
                        if command is None:
                            self.error_payload(404, "unknown command")
                        else:
                            self.send_payload(200, command)
                    else:
                        self.error_payload(404, "unknown endpoint")

                def do_POST(self):
                    if not self.authorized():
                        return
                    if self.path != "/commands":
                        self.error_payload(404, "unknown endpoint")
                        return
                    if self.headers.get("Transfer-Encoding") is not None:
                        self.error_payload(400, "chunked command bodies are not supported")
                        return
                    if self.headers.get_content_type() != "application/json":
                        self.error_payload(415, "commands require application/json")
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                    except ValueError:
                        self.error_payload(400, "invalid Content-Length")
                        return
                    if not 0 < length <= MAX_REQUEST_BYTES:
                        self.error_payload(413, "command body must be 1..8192 bytes")
                        return
                    try:
                        payload = json.loads(self.rfile.read(length))
                        result = owner.submit(payload)
                    except (ValueError, UnicodeDecodeError) as error:
                        self.error_payload(400, str(error))
                    except queue.Full:
                        self.error_payload(429, "command queue is full; wait for existing commands")
                    except RuntimeError as error:
                        self.error_payload(503, str(error))
                    else:
                        self.send_payload(202, result)

            try:
                server = _ControlHTTPServer(("127.0.0.1", self.preferred_port), Handler)
            except OSError as error:
                if self.preferred_port == 0 or getattr(error, "errno", None) not in {13, 48, 98, 10013, 10048}:
                    raise
                server = _ControlHTTPServer(("127.0.0.1", 0), Handler)
            self._server = server
            self._endpoint = {
                "schema_version": 1, "service": "tableturf-control", "running": True,
                "pid": os.getpid(), "url": f"http://127.0.0.1:{server.server_port}",
                "token": self.token, "token_header": "X-Tableturf-Token",
                "session_dir": str(self.session_dir), "started_unix": time.time(),
            }
            self._thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1},
                                            name="tableturf-local-control", daemon=True)
            try:
                self._write_discovery()
                self._thread.start()
            except Exception:
                server.server_close()
                self._server = None
                self._endpoint = None
                raise
            return dict(self._endpoint)

    def _write_discovery(self):
        for path in (self.session_dir / "control_endpoint.json", self.session_dir.parent / "control_endpoint.json"):
            if self._closed and path.is_file():
                try:
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    if previous.get("token") != self.token:
                        continue  # A newer terminal owns the shared discovery file.
                except (OSError, ValueError):
                    continue
            temporary = path.with_name(path.name + f".{os.getpid()}.{self.token[:8]}.tmp")
            temporary.write_text(json.dumps(self._endpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)

    @staticmethod
    def _validated_command(payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("command body must be a JSON object")
        name = payload.get("command")
        if not isinstance(name, str) or name not in COMMAND_NAMES:
            raise ValueError("command must be pause, resume, reload, press or release")
        allowed = {"command", "buttons", "hold_ms"} if name == "press" else {"command"}
        if set(payload) - allowed:
            raise ValueError("unexpected command fields")
        result = {"command": name}
        if name == "press":
            buttons = payload.get("buttons")
            valid_buttons = {item.value for item in Button}
            if (not isinstance(buttons, list) or not 1 <= len(buttons) <= 4
                or any(not isinstance(item, str) or item not in valid_buttons for item in buttons)
                or len(set(buttons)) != len(buttons)):
                raise ValueError("buttons must contain 1..4 distinct controller Button names")
            hold = payload.get("hold_ms", 120)
            if type(hold) is not int or not 30 <= hold <= 1000:
                raise ValueError("hold_ms must be an integer in 30..1000")
            result.update(buttons=list(buttons), hold_ms=hold)
        return result

    def submit(self, payload: dict) -> dict:
        command = self._validated_command(payload)
        with self._lock:
            if self._closed:
                raise RuntimeError("control server is closed")
            outstanding = sum(item["status"] in {"queued", "running"} for item in self._commands.values())
            if outstanding >= self.max_pending:
                raise queue.Full
            identifier = secrets.token_hex(12)
            command.update(id=identifier, submitted_unix=time.time())
            record = {**command, "status": "queued"}
            (self._urgent if command["command"] == "release" else self._pending).put_nowait(command)
            self._commands[identifier] = record
            for key in list(self._commands):
                if len(self._commands) <= MAX_RETAINED_COMMANDS:
                    break
                if self._commands[key]["status"] not in {"queued", "running"}:
                    self._commands.pop(key)
            return dict(record)

    def command_result(self, identifier: str) -> dict | None:
        with self._lock:
            record = self._commands.get(identifier)
            return json.loads(json.dumps(record)) if record is not None else None

    def complete_command(self, identifier: str, *, result=None, error: str | None = None) -> None:
        # Freeze the caller's result before HTTP threads can serialize it.
        frozen_result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
        with self._lock:
            record = self._commands.get(identifier)
            if record is None or record["status"] not in {"queued", "running"}:
                return
            record.update(status="failed" if error is not None else "succeeded", finished_unix=time.time())
            if error is not None:
                record["error"] = str(error)
            else:
                record["result"] = frozen_result

    def drain(self, handler: Callable[[dict], object], *, max_commands: int = 4,
              urgent_only: bool = False) -> int:
        """Execute pending work on the calling (UI) thread, never an HTTP thread.

        Returning COMMAND_DEFERRED leaves the receipt running until a later
        complete_command call, e.g. after the timed release of a button pulse.
        """
        processed = 0
        while processed < max_commands and not self._closed:
            try:
                command = self._urgent.get_nowait()
            except queue.Empty:
                if urgent_only:
                    break
                try:
                    command = self._pending.get_nowait()
                except queue.Empty:
                    break
            with self._lock:
                self._commands[command["id"]].update(status="running", started_unix=time.time())
            try:
                result = handler(command)
                if result is not COMMAND_DEFERRED:
                    self.complete_command(command["id"], result=result)
            except Exception as error:
                self.complete_command(command["id"], error=str(error))
            processed += 1
        return processed

    def set_frame(self, kind: str, data: bytes, *, sequence: int | None = None, captured_at: float | None = None):
        """Publish an already encoded live frame; this method performs no encoding."""
        if kind not in FRAME_FILES:
            raise ValueError("kind must be analysis or capture")
        if not isinstance(data, bytes) or not data or len(data) > 24 * 1024 * 1024:
            raise ValueError("frame must contain 1..24MiB of encoded bytes")
        stamp = time.time() if captured_at is None else float(captured_at)
        with self._lock:
            self._frames[kind] = (data, sequence, stamp)

    def _frame_metadata(self, kind: str) -> dict:
        with self._lock:
            cached = self._frames.get(kind)
        if cached is not None:
            stamp, sequence, source = cached[2], cached[1], "memory"
        else:
            stamp = (self.session_dir / FRAME_FILES[kind][0]).stat().st_mtime
            sequence, source = None, "diagnostic_file"
        return {"available": True, "captured_unix": stamp, "age_seconds": max(0.0, time.time() - stamp),
                "sequence": sequence, "source": source}

    def frame(self, kind: str) -> tuple[bytes, dict]:
        if kind not in FRAME_FILES:
            raise ValueError("kind must be analysis or capture")
        with self._lock:
            cached = self._frames.get(kind)
        if cached is not None:
            return cached[0], {"available": True, "captured_unix": cached[2],
                "age_seconds": max(0.0, time.time() - cached[2]), "sequence": cached[1], "source": "memory"}
        path = self.session_dir / FRAME_FILES[kind][0]
        # Open once, so the bytes and timestamp refer to the same atomic version.
        with path.open("rb") as file:
            stamp = os.fstat(file.fileno()).st_mtime
            data = file.read()
        return data, {"available": True, "captured_unix": stamp, "age_seconds": max(0.0, time.time() - stamp),
                      "sequence": None, "source": "diagnostic_file"}

    def status(self) -> dict:
        value = json.loads((self.session_dir / "latest.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("diagnostic snapshot must be a JSON object")
        value["read_age_seconds"] = max(0.0, time.time() - value["updated_unix"]) if "updated_unix" in value else None
        frames = {}
        for kind in FRAME_FILES:
            try:
                frames[kind] = self._frame_metadata(kind)
            except OSError:
                frames[kind] = {"available": False, "age_seconds": None}
        with self._lock:
            outstanding = sum(item["status"] in {"queued", "running"} for item in self._commands.values())
        value["control"] = {"schema_version": 1, "running": not self._closed,
                            "pending_commands": outstanding, "command_names": sorted(COMMAND_NAMES)}
        value["frames"] = frames
        return value

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for record in self._commands.values():
                if record["status"] in {"queued", "running"}:
                    record.update(status="failed", error="task terminal closed before command completed", finished_unix=time.time())
            if self._endpoint is not None:
                self._endpoint.update(running=False, stopped_unix=time.time())
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._endpoint is not None:
            self._write_discovery()
