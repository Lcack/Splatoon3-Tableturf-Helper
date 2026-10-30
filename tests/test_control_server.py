"""Exercise the actual loopback HTTP boundary without Tk or a controller."""
import json
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from tableturf_rebuild.application.control_server import COMMAND_DEFERRED, LocalControlServer


def request(server, method, path, body=None, *, token=True, headers=None):
    endpoint = server.endpoint
    request_headers = {"X-Tableturf-Token": endpoint["token"]} if token else {}
    if body is not None:
        body = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    req = Request(endpoint["url"] + path, data=body, headers=request_headers, method=method)
    try:
        response = urlopen(req, timeout=2)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read()
        value = json.loads(raw) if response.headers.get_content_type() == "application/json" else raw
        return response.status, value, response.headers


@pytest.fixture
def server(tmp_path):
    session_dir = tmp_path / "run-test"
    session_dir.mkdir()
    (session_dir / "latest.json").write_text(json.dumps({
        "updated_unix": time.time() - 3, "status": "PAUSED", "observation": {"state": "TARGET_PLACEMENT"},
    }), encoding="utf-8")
    value = LocalControlServer(session_dir, preferred_port=0, max_pending=2)
    value.start()
    try:
        yield value
    finally:
        value.close()


def test_discovery_binds_only_loopback_and_does_not_expose_token_in_http_metadata(server):
    endpoint = server.endpoint
    assert endpoint["url"].startswith("http://127.0.0.1:")
    for path in (server.session_dir / "control_endpoint.json", server.session_dir.parent / "control_endpoint.json"):
        assert json.loads(path.read_text(encoding="utf-8")) == endpoint
    code, health, _ = request(server, "GET", "/health", token=False)
    assert code == 200
    assert health["service"] == "tableturf-control"
    code, status, _ = request(server, "GET", "/status")
    assert code == 200
    assert endpoint["token"] not in json.dumps(health) + json.dumps(status)


def test_authentication_and_browser_requests_cannot_enqueue_commands(server):
    assert request(server, "GET", "/status", token=False)[0] == 401
    assert request(server, "POST", "/commands", {"command": "press", "buttons": ["A"]}, token=False)[0] == 401
    assert request(server, "GET", "/status", headers={"X-Tableturf-Token": "wrong"})[0] == 401
    assert request(server, "POST", "/commands", {"command": "resume"}, headers={"Origin": "http://localhost"})[0] == 403
    assert request(server, "GET", "/status", headers={"Host": "attacker.example"})[0] == 403
    assert request(server, "GET", "/status")[1]["control"]["pending_commands"] == 0


def test_status_and_frame_are_uncached_and_include_actual_frame_freshness(server):
    frame_path = server.session_dir / "analysis.png"
    frame_path.write_bytes(b"png-file-1")
    code, first, headers = request(server, "GET", "/frame?kind=analysis")
    assert (code, first) == (200, b"png-file-1")
    assert headers.get_content_type() == "image/png"
    assert "no-store" in headers["Cache-Control"]
    assert headers["X-Frame-Source"] == "diagnostic_file"
    frame_path.write_bytes(b"png-file-2")
    assert request(server, "GET", "/frame?kind=analysis")[1] == b"png-file-2"
    server.set_frame("capture", b"new-live-jpeg", sequence=405, captured_at=time.time() - 0.2)
    code, live, headers = request(server, "GET", "/frame?kind=capture")
    assert (code, live) == (200, b"new-live-jpeg")
    assert headers.get_content_type() == "image/jpeg"
    assert headers["X-Frame-Source"] == "memory"
    assert headers["X-Frame-Sequence"] == "405"
    assert 0.2 <= float(headers["X-Frame-Age-Seconds"]) < 2
    code, status, headers = request(server, "GET", "/status")
    assert code == 200 and status["read_age_seconds"] >= 3
    assert status["frames"]["capture"]["sequence"] == 405
    assert status["frames"]["analysis"]["source"] == "diagnostic_file"
    assert "no-store" in headers["Cache-Control"]


def test_missing_diagnostics_and_frames_are_explicitly_unavailable(server):
    assert request(server, "GET", "/frame")[0] == 503
    status = request(server, "GET", "/status")[1]
    assert status["frames"]["capture"] == {"available": False, "age_seconds": None}
    (server.session_dir / "latest.json").write_text("{bad", encoding="utf-8")
    assert request(server, "GET", "/status")[0] == 503


def test_commands_wait_for_ui_drain_and_async_completion(server):
    code, command, _ = request(server, "POST", "/commands", {"command": "press", "buttons": ["L", "R"]})
    assert code == 202 and command["status"] == "queued"
    assert command["hold_ms"] == 120
    assert request(server, "GET", f"/commands/{command['id']}")[1]["status"] == "queued"
    handler_threads = []
    seen = []

    def handler(item):
        handler_threads.append(threading.get_ident())
        seen.append(item)
        return COMMAND_DEFERRED

    assert server.drain(handler) == 1
    assert handler_threads == [threading.get_ident()]
    assert seen[0]["buttons"] == ["L", "R"]
    assert request(server, "GET", f"/commands/{command['id']}")[1]["status"] == "running"
    server.complete_command(command["id"], result={"released": True})
    done = request(server, "GET", f"/commands/{command['id']}")[1]
    assert done["status"] == "succeeded" and done["result"] == {"released": True}


def test_synchronous_results_errors_and_queue_limits_include_running_commands(server):
    first = request(server, "POST", "/commands", {"command": "pause"})[1]
    second = request(server, "POST", "/commands", {"command": "resume"})[1]
    assert request(server, "POST", "/commands", {"command": "reload"})[0] == 429
    assert server.drain(lambda item: COMMAND_DEFERRED, max_commands=1) == 1
    assert request(server, "POST", "/commands", {"command": "reload"})[0] == 429
    server.complete_command(first["id"], result={"paused": True})
    assert server.drain(lambda item: {"resumed": True}) == 1
    assert request(server, "GET", f"/commands/{second['id']}")[1]["result"] == {"resumed": True}
    third = request(server, "POST", "/commands", {"command": "reload"})[1]

    def fail(_):
        raise RuntimeError("analysis still running")

    assert server.drain(fail) == 1
    failure = request(server, "GET", f"/commands/{third['id']}")[1]
    assert failure["status"] == "failed" and failure["error"] == "analysis still running"
    assert request(server, "GET", "/commands/not-a-real-id")[0] == 404


@pytest.mark.parametrize("body", [
    [], {}, {"command": "exec", "code": "print('bad')"},
    {"command": "pause", "file": "other"},
    {"command": "press", "buttons": []},
    {"command": "press", "buttons": ["unknown"]},
    {"command": "press", "buttons": ["A", "A"]},
    {"command": "press", "buttons": ["A"], "hold_ms": True},
    {"command": "press", "buttons": ["A"], "hold_ms": 29},
    {"command": "press", "buttons": ["A"], "hold_ms": 1001},
    {"command": "press", "buttons": ["A"], "hold_ms": 31.5},
])
def test_invalid_commands_are_rejected_before_ui_queue(server, body):
    assert request(server, "POST", "/commands", body)[0] == 400
    assert server.drain(lambda _: pytest.fail("invalid command reached UI")) == 0


@pytest.mark.parametrize("path, expected", [
    ("/frame?kind=../latest.json", 400), ("/frame?path=../secret.txt", 400),
    ("/frame?kind=capture&kind=analysis", 400), ("/../secret.txt", 404),
    ("/status?path=../secret.txt", 404), ("/pause", 404),
])
def test_fixed_routes_cannot_read_arbitrary_files_or_mutate_via_get(server, path, expected):
    assert request(server, "GET", path)[0] == expected
    assert server.drain(lambda _: pytest.fail("GET enqueued a mutation")) == 0


def test_oversized_requests_are_rejected(server):
    assert request(server, "POST", "/commands", {"command": "press", "padding": "x" * 8192})[0] == 413


def test_close_cancels_queued_work_and_updates_only_its_own_discovery_file(server):
    command = server.submit({"command": "press", "buttons": ["A"], "hold_ms": 1000})
    endpoint = server.endpoint
    shared_path = server.session_dir.parent / "control_endpoint.json"
    shared_path.write_text(json.dumps({"token": "newer-server", "running": True}), encoding="utf-8")
    server.close()
    assert server.command_result(command["id"])["status"] == "failed"
    assert server.drain(lambda _: pytest.fail("closed server ran queued work")) == 0
    assert json.loads((server.session_dir / "control_endpoint.json").read_text(encoding="utf-8"))["running"] is False
    assert json.loads(shared_path.read_text(encoding="utf-8"))["token"] == "newer-server"
    with pytest.raises(RuntimeError, match="closed"):
        server.submit({"command": "resume"})
    with pytest.raises(URLError):
        urlopen(endpoint["url"] + "/health", timeout=0.5)


def test_port_collision_falls_back_without_taking_existing_server(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        fallback = LocalControlServer(tmp_path / "run-fallback", preferred_port=port)
        try:
            endpoint = fallback.start()
            assert int(endpoint["url"].rsplit(":", 1)[1]) != port
            assert request(fallback, "GET", "/health", token=False)[0] == 200
        finally:
            fallback.close()
