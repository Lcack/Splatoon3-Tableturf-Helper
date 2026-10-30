from pathlib import Path
from types import ModuleType, SimpleNamespace
import os
import subprocess
import sys
import threading

import pytest

from tableturf_rebuild.application import core_reload
from tableturf_rebuild.application import task_terminal_ui as ui


class MainThreadValue:
    def __init__(self, value=None):
        self.value = value
        self.owner = threading.get_ident()

    def set(self, value):
        assert threading.get_ident() == self.owner
        self.value = value


def app_without_tk():
    app = object.__new__(ui.TaskTerminalApp)
    app._task_thread = None
    app._task_thread_lock = threading.Lock()
    app._task_result_lock = threading.Lock()
    app._pending_task_result = None
    app._last_task_dispatch = app._last_idle_analysis = app._last_preview_at = 0
    app._task_started_at = 0
    app._analysis_counter = app._analysis_sequence = 0
    app._last_analysis_at = app._last_analysis_elapsed = 0
    app.last_analysis = None
    app.last_sequence = -1
    app.battle_debug_active = False
    app.battle_debug_executor = None
    app.runner = SimpleNamespace(status=ui.TaskStatus.PAUSED, battle_executor=None)
    app.status_var = MainThreadValue("PAUSED")
    app.message_var = MainThreadValue()
    app.logs = []
    app.log_message = app.logs.append
    app.observed = []
    app._update_battle_debug_live = app.observed.append
    app.shown = []
    app._show_frame = app.shown.append
    app.debug_refresh_threads = []
    app.debug_window = SimpleNamespace(refresh_from_parent=lambda: app.debug_refresh_threads.append(threading.get_ident()))
    app.packet = SimpleNamespace(sequence=1, frame="frame-1")
    app.worker = SimpleNamespace(latest=lambda: app.packet, error=None)
    return app


def test_collector_restores_progress_after_cached_reload_callback_installs_runner():
    app = app_without_tk()
    previous = object()
    checkpoint = {"confirmed_win": True}
    restored, snapshots = [], []
    app.runner = SimpleNamespace(restore_progress=restored.append, snapshot=lambda: "restored")
    app._pending_progress_restore = (previous, checkpoint)
    app._update_snapshot = snapshots.append
    app._collect_task_result()
    app._collect_task_result()
    assert restored == [checkpoint] and snapshots == ["restored"]


def test_idle_analysis_runs_off_tk_and_drops_old_frames_without_parallel_runtime():
    app = app_without_tk()
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    calls = []
    def update(frame, **kwargs):
        assert threading.get_ident() != main_thread
        calls.append(frame)
        entered.set()
        assert release.wait(5)
        return frame + "-observation"
    app.runtime = SimpleNamespace(update=update)
    try:
        app._refresh_frame()
        assert entered.wait(1)
        first_worker = app._task_thread
        assert first_worker.is_alive()
        app.packet = SimpleNamespace(sequence=2, frame="frame-2")
        app._last_preview_at = 0
        app._refresh_frame()
        assert app.shown == ["frame-1", "frame-2"]
        assert calls == ["frame-1"]
        assert app._start_task_worker("task-frame", mode="task") is False
        assert app._task_thread is first_worker
        release.set()
        first_worker.join(2)
        assert not first_worker.is_alive()
        app._collect_task_result()
        assert app.last_analysis == "frame-1-observation"
        assert app.status_var.value == "PAUSED"
        app.packet = SimpleNamespace(sequence=3, frame="frame-3")
        app._last_preview_at = app._last_idle_analysis = 0
        app._refresh_frame()
        app._task_thread.join(2)
        app._collect_task_result()
        assert calls == ["frame-1", "frame-3"]  # no queue for the stale second frame
        assert app.last_analysis == "frame-3-observation"
        assert set(app.debug_refresh_threads) == {main_thread}
    finally:
        release.set()
        if app._task_thread:
            app._task_thread.join(2)


def test_result_waiting_to_be_collected_cannot_be_overwritten():
    app = app_without_tk()
    app._pending_task_result = ("idle", "completed", None, "old-frame")
    app.runtime = SimpleNamespace(update=lambda *args, **kwargs: pytest.fail("must collect first"))
    assert app._start_task_worker("new-frame", mode="idle") is False
    app._collect_task_result()
    assert app.last_analysis == "completed"


def test_reload_keeps_displaying_capture_without_starting_analysis():
    app = app_without_tk()
    app._reload_context = {}
    app.runtime = SimpleNamespace(update=lambda *args, **kwargs: pytest.fail("reload is paused"))
    app._refresh_frame()
    assert app.shown == ["frame-1"]
    assert app._task_thread is None
    assert app._start_task_worker("frame", mode="idle") is False


def test_manual_ownership_overrides_late_running_result_before_redispatch():
    app = app_without_tk()
    app._local_control_manual_busy = True
    app.runner.status = ui.TaskStatus.RUNNING
    app.runner.pause = lambda: setattr(app.runner, "status", ui.TaskStatus.PAUSED)
    app.runner.snapshot = lambda: app.runner.status
    app._update_snapshot = lambda snapshot: app.status_var.set(snapshot.value)
    app.runtime = SimpleNamespace(update=lambda *args, **kwargs: pytest.fail("manual control owns the controller"))
    app._refresh_frame()
    assert app.runner.status is ui.TaskStatus.PAUSED
    assert app._task_thread is None


def test_manual_release_reobserves_orientation_without_resetting_match(monkeypatch):
    from tableturf_rebuild.application import terminal_control
    app = app_without_tk()
    app.worker = None
    resets = []
    app.runner.battle_executor = SimpleNamespace(reset_preview_feedback=lambda: resets.append(True))
    app._local_control_bridge = SimpleNamespace(pending={"id": "manual-Y", "command": "press"})
    def complete(current):
        current._local_control_bridge.pending = None
    monkeypatch.setattr(terminal_control, "service_terminal_control", complete)
    app._refresh_frame()
    app._refresh_frame()
    assert resets == [True]


def test_idle_exception_is_reported_on_main_thread_without_changing_task_status():
    app = app_without_tk()
    app._task_started_at = 0
    app._pending_task_result = ("idle", None, ValueError("test observer error"), "frame")
    app._collect_task_result()
    assert app.status_var.value == "PAUSED"
    assert "observer error" in app.message_var.value
    assert "ValueError" in app._last_error


def test_configuration_cannot_continue_if_worker_wait_times_out():
    app = app_without_tk()
    app._task_thread = SimpleNamespace(join=lambda **kwargs: None, is_alive=lambda: True)
    with pytest.raises(RuntimeError, match="configuration was not changed"):
        app._wait_for_task_worker()
    app._reload_context = {}
    app._wait_for_task_worker()  # the reload callback will schedule its retry


def test_debug_pause_during_analysis_prevents_late_controller_action():
    app = app_without_tk()
    app.battle_debug_active = True
    app.battle_debug_executor = SimpleNamespace(step=lambda *args: pytest.fail("paused executor must not send keys"))
    def update(*args, **kwargs):
        app.battle_debug_active = False
        app._reload_context = {}
        return "observation"
    app.runtime = SimpleNamespace(update=update)
    app._run_battle_debug_tick("frame")
    kind, payload, error, frame = app._pending_task_result
    assert kind == "debug" and error is None
    assert payload[0] == "observation" and payload[2] == "WAITING"


def patch_fixture(monkeypatch, metaclass=type):
    terminal = metaclass("ExistingTerminal", (ui.TaskTerminalApp,), {})
    fake_module = ModuleType(core_reload.TERMINAL_UI_MODULE)
    fake_module.__dict__.update(ui.__dict__)
    fake_module.TaskTerminalApp = terminal
    monkeypatch.setitem(sys.modules, core_reload.TERMINAL_UI_MODULE, fake_module)
    app = object.__new__(terminal)
    app._reload_context = {}
    app._task_thread = None
    app._pending_task_result = None
    app.battle_debug_active = False
    app.runner = SimpleNamespace(status="PAUSED")
    return terminal, app


def test_whitelist_hotpatch_never_constructs_tk_or_replaces_existing_class(monkeypatch):
    terminal, app = patch_fixture(monkeypatch)
    monkeypatch.setattr(ui.tk, "Tk", lambda: pytest.fail("must not create a Tk window"))
    constructor, builder = terminal.__init__, terminal._build
    source = Path(ui.__file__).read_text(encoding="utf-8-sig") + "\ntk.Tk()\n"
    token = core_reload.patch_loaded_terminal_ui(app=app, source=source)
    assert type(app) is terminal
    assert terminal.__init__ is constructor and terminal._build is builder
    assert terminal._refresh_patch_version
    assert "_run_idle_tick" in terminal.__dict__
    core_reload.rollback_terminal_refresh_patch(token)
    assert "_run_idle_tick" not in terminal.__dict__


def test_hotpatch_validation_failure_leaves_class_untouched(monkeypatch):
    terminal, app = patch_fixture(monkeypatch)
    before = dict(terminal.__dict__)
    with pytest.raises(RuntimeError, match="missing methods"):
        core_reload.patch_loaded_terminal_ui(app=app, source="class TaskTerminalApp:\n    pass\n")
    assert dict(terminal.__dict__) == before
    app._task_thread = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(RuntimeError, match="paused and drained"):
        core_reload.patch_loaded_terminal_ui(app=app)
    assert dict(terminal.__dict__) == before


def test_hotpatch_partial_assignment_rolls_back(monkeypatch):
    switch = {"fail": True}
    class FailingMeta(type):
        def __setattr__(cls, name, value):
            if name == "_run_idle_tick" and switch["fail"]:
                switch["fail"] = False
                raise RuntimeError("injected assignment failure")
            super().__setattr__(name, value)
    terminal, app = patch_fixture(monkeypatch, FailingMeta)
    before = dict(terminal.__dict__)
    with pytest.raises(RuntimeError, match="injected assignment"):
        core_reload.patch_loaded_terminal_ui(app=app)
    assert dict(terminal.__dict__) == before


@pytest.mark.parametrize("fail_runtime", [False, True])
def test_cached_older_reload_bootstraps_existing_ui_in_isolated_process(fail_runtime):
    code = r'''
import sys
from types import SimpleNamespace
import tableturf_rebuild.application.task_terminal_ui as ui
import tableturf_rebuild.application.core_reload as support
from tableturf_rebuild.data.catalog import GameCatalog
ui.tk.Tk = lambda: (_ for _ in ()).throw(AssertionError("new GUI forbidden"))
terminal = ui.TaskTerminalApp
app = object.__new__(terminal)
app._reload_context = {}
app._task_thread = None
app._pending_task_result = None
app.battle_debug_active = False
app.runner = SimpleNamespace(status="PAUSED")
def old_synchronous_refresh(self):
    raise AssertionError("legacy refresh must be replaced")
terminal._refresh_frame = old_synchronous_refresh
before = {name: terminal.__dict__.get(name) for name in support.TERMINAL_REFRESH_METHODS}
cached_old_reload = support.reload_core
# Emulate the feature surface of the already imported generation-8 module.
for name in ("UI_REFRESH_PATCH_API", "patch_loaded_terminal_ui", "bootstrap_terminal_refresh"):
    delattr(support, name)
if FAIL_RUNTIME:
    GameCatalog.from_package = classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("injected runtime failure")))
def reload_analysis_core(self):
    bindings, versions = cached_old_reload()
    ui.__dict__.update(bindings)
    return bindings["BattleVisionRuntime"].default()
try:
    runtime = reload_analysis_core(app)
except RuntimeError as error:
    assert FAIL_RUNTIME and "injected runtime failure" in str(error)
    assert all(terminal.__dict__.get(name) is method for name, method in before.items())
else:
    assert not FAIL_RUNTIME
    assert type(app) is terminal
    assert terminal._refresh_frame is not old_synchronous_refresh
    assert terminal._refresh_patch_version
    assert support.UI_REFRESH_PATCH_API == 1
print("legacy bootstrap and rollback OK")
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", "FAIL_RUNTIME=" + repr(fail_runtime) + "\n" + code],
                            cwd=root, text=True, capture_output=True, timeout=40,
                            env={**os.environ, "PYTHONPATH": str(root / "src")})
    assert result.returncode == 0, result.stdout + result.stderr
