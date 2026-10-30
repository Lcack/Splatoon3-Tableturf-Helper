from types import SimpleNamespace
import threading

from tableturf_rebuild.application.control_server import LocalControlServer
from tableturf_rebuild.application.terminal_control import TerminalControlBridge
from tableturf_rebuild.application.task_runner import TaskStatus
from tableturf_rebuild.control.buttons import Button


def make_bridge(tmp_path):
    sent, now = [], [1.0]
    controller = SimpleNamespace(connected=True, last_error="", send_state=lambda state: sent.append(state) or True)
    runner = SimpleNamespace(status=TaskStatus.RUNNING, task=object(), battle_executor=None)
    runner.pause = lambda: setattr(runner, "status", TaskStatus.PAUSED)
    runner.resume = lambda: setattr(runner, "status", TaskStatus.RUNNING)
    runner.snapshot = lambda: runner.status
    app = SimpleNamespace(controller=controller, runner=runner, battle_debug_active=False,
        battle_debug_executor=None, _task_thread=None, _pending_task_result=None,
        _collect_task_result=lambda: None, _update_snapshot=lambda _: None, log_message=lambda _: None,
        worker=object(), _reload_context=None, _core_generation=0,
        status_var=SimpleNamespace(get=lambda: "CORE_RELOADED_PAUSED"))
    def reload():
        app._core_generation += 1
    app.reload_analysis_core = reload
    server = LocalControlServer(tmp_path, preferred_port=0)
    bridge = TerminalControlBridge(app, server=server, clock=lambda: now[0], stream_frames=False)
    return bridge, app, sent, now


def test_manual_pulse_waits_for_analysis_and_releases_before_success(tmp_path):
    bridge, app, sent, now = make_bridge(tmp_path)
    try:
        active = [True]
        app._task_thread = SimpleNamespace(is_alive=lambda: active[0])
        receipt = bridge.server.submit({"command": "press", "buttons": ["Y"], "hold_ms": 120})
        bridge.poll()
        assert app.runner.status is TaskStatus.PAUSED and app._local_control_manual_busy
        now[0] += 1
        bridge.poll()
        assert not sent  # Existing vision work cannot race a manual press.
        app.runner.status = TaskStatus.RUNNING  # Simulate a tick finishing after the initial pause.
        active[0] = False
        bridge.poll()
        assert app.runner.status is TaskStatus.PAUSED
        assert sent[-1].buttons == frozenset({Button.Y})
        assert bridge.server.command_result(receipt["id"])["status"] == "running"
        now[0] += 0.13
        bridge.poll()
        assert not sent[-1].buttons and not app._local_control_manual_busy
        assert bridge.server.command_result(receipt["id"])["status"] == "succeeded"
    finally:
        bridge.close()


def test_release_interrupts_held_button_and_does_not_resume_automation(tmp_path):
    bridge, app, sent, now = make_bridge(tmp_path)
    try:
        first = bridge.server.submit({"command": "press", "buttons": ["A"], "hold_ms": 1000})
        bridge.poll()
        now[0] += 0.1
        bridge.poll()
        release = bridge.server.submit({"command": "release"})
        bridge.poll()
        assert not sent[-1].buttons and app.runner.status is TaskStatus.PAUSED
        assert bridge.server.command_result(first["id"])["status"] == "failed"
        assert bridge.server.command_result(release["id"])["status"] == "succeeded"
    finally:
        bridge.close()


def test_reload_preserves_bridge_and_resume_needs_configured_task(tmp_path):
    bridge, app, _, now = make_bridge(tmp_path)
    try:
        reload = bridge.server.submit({"command": "reload"})
        bridge.poll()
        now[0] += 0.1
        bridge.poll()
        assert app._core_generation == 1
        assert bridge.server.command_result(reload["id"])["result"]["paused"]
        app.runner.task = None
        resume = bridge.server.submit({"command": "resume"})
        bridge.poll()
        now[0] += 0.1
        bridge.poll()
        assert bridge.server.command_result(resume["id"])["status"] == "failed"
        assert app.runner.status is TaskStatus.PAUSED
    finally:
        bridge.close()


def test_closing_terminal_releases_pending_input(tmp_path):
    bridge, app, sent, now = make_bridge(tmp_path)
    receipt = bridge.server.submit({"command": "press", "buttons": ["DPAD_UP"]})
    bridge.poll()
    now[0] += 0.1
    bridge.poll()
    bridge.close()
    bridge.close()
    assert not sent[-1].buttons and not app._local_control_manual_busy
    assert bridge.server.command_result(receipt["id"])["status"] == "failed"


def test_multiple_pulses_stay_queued_until_previous_release(tmp_path):
    bridge, _, sent, now = make_bridge(tmp_path)
    try:
        first = bridge.server.submit({"command": "press", "buttons": ["Y"]})
        second = bridge.server.submit({"command": "press", "buttons": ["X"]})
        bridge.poll()
        now[0] += 0.1
        bridge.poll()
        assert sent[-1].buttons == frozenset({Button.Y})
        assert bridge.server.command_result(second["id"])["status"] == "queued"
        now[0] += 0.13
        bridge.poll()
        assert bridge.server.command_result(first["id"])["status"] == "succeeded"
        now[0] += 0.1
        bridge.poll()
        assert sent[-1].buttons == frozenset({Button.X})
    finally:
        bridge.close()


def test_watchdog_releases_even_without_another_ui_tick(tmp_path):
    bridge, app, sent, now = make_bridge(tmp_path)
    released = threading.Event()
    original = app.controller.send_state
    def send(state):
        result = original(state)
        if not state.buttons:
            released.set()
        return result
    app.controller.send_state = send
    try:
        bridge.server.submit({"command": "press", "buttons": ["Y"], "hold_ms": 30})
        bridge.poll()
        now[0] += 0.1
        bridge.poll()
        assert sent[-1].buttons == frozenset({Button.Y})
        assert released.wait(1)
        assert not sent[-1].buttons
    finally:
        bridge.close()
