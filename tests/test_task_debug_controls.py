"""Debug controls share terminal state and remain valid across callback reloads."""
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest

from tableturf_rebuild.application.task_debug_ui import TaskDebugWindow


@pytest.fixture
def debug_controls():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"native Tk is unavailable: {error}")
    root.withdraw()
    window = tk.Toplevel(root)
    window.withdraw()
    calls = []
    app = window.app = SimpleNamespace(
        debug_stage_var=tk.StringVar(root, value="Cross"),
        debug_stage_ids=("Cross", "Zigzag"),
        _battle_debug_button_text=tk.StringVar(root, value="打牌调试"),
        toggle_battle_debug=lambda: calls.append("old-debug"),
        reload_analysis_core=lambda: calls.append("old-reload"),
    )
    try:
        yield window, app, calls
    finally:
        root.destroy()


def test_debug_map_and_running_label_share_the_terminal_variables(debug_controls):
    window, app, _ = debug_controls
    TaskDebugWindow._build_debug_controls(window)
    app.debug_stage_var.set("Zigzag")
    assert window.debug_stage_box.get() == "Zigzag"
    window.debug_stage_box.current(0)
    assert app.debug_stage_var.get() == "Cross"
    app._battle_debug_button_text.set("停止打牌调试")
    assert window.battle_debug_button.cget("text") == "停止打牌调试"


def test_buttons_use_current_callbacks_and_reload_can_close_the_window(debug_controls):
    window, app, calls = debug_controls
    TaskDebugWindow._build_debug_controls(window)
    app.toggle_battle_debug = lambda: calls.append("fresh-debug")
    def reload_and_close():
        calls.append("fresh-reload")
        window.destroy()
    app.reload_analysis_core = reload_and_close
    window.battle_debug_button.invoke()
    window.reload_core_button.invoke()
    assert calls == ["fresh-debug", "fresh-reload"]
    assert not window.winfo_exists()


def test_upgrading_an_open_debug_window_inserts_controls_once(debug_controls):
    window, _, _ = debug_controls
    old_controls = ttk.Frame(window)
    old_controls.pack(fill="x")
    panel = TaskDebugWindow._build_debug_controls(window)
    assert window.pack_slaves()[0] is panel
    assert TaskDebugWindow._build_debug_controls(window) is panel
    assert len(window.winfo_children()) == 2
