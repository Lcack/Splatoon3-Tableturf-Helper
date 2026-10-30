"""Native Tk queue editing and handoff without capture or device operations."""
from dataclasses import replace
import tkinter as tk

import pytest

from tableturf_rebuild.application.task_terminal_ui import TaskTerminalApp
from tableturf_rebuild.application.task_plan import NpcTaskDefinition, build_npc_plan
from tableturf_rebuild.application.task_queue import TaskQueue
from tableturf_rebuild.application.task_runner import TaskStatus
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from test_task_queue_routing import safe_menu
from test_win_plan_feedback import observation


NPC = "MiniGame_WeaponShop"
OTHER = "MiniGame_MiniGameKurage0"


@pytest.fixture
def app(tmp_path):
    root = tk.Tk()
    root.withdraw()
    app = TaskTerminalApp(root, auto_start=False, queue_path=tmp_path / "queue.json")
    backend = DummyBackend()
    backend.connect()
    app.controller = app.runner.controller = backend
    try:
        yield app
    finally:
        app.worker = None
        if not app._closing:
            app.close()


def task(app, npc=NPC, goal=30):
    return NpcTaskDefinition(npc, app.catalog.npcs[npc].name_zh, 3, goal)


def activate(app):
    entry = app.task_queue.append(task(app))
    app.task_queue.start_next()
    app.runner.configure(entry.task, entry.deck)
    app.runner.start()
    app._queue_enabled = True
    return entry


def select_input(app, npc, wins):
    app.npc_var.set(next(label for label, npc_id in app._npc_labels.items() if npc_id == npc))
    app.difficulty_var.set("3")
    app.target_wins_var.set(str(wins))


def test_adding_during_a_running_task_never_reconfigures_it(app):
    current = activate(app)
    runner, original_task = app.runner, app.runner.task
    select_input(app, OTHER, 7)
    label = next(iter(app._presets))
    app.deck_var.set(label)
    new = app.add_queue_task()
    app.target_wins_var.set("99")
    app.deck_var.set("未指定（在卡组界面等待）")
    assert app.runner is runner and runner.task is original_task
    assert app.task_queue.current.entry_id == current.entry_id
    assert new.task.npc_id == OTHER and new.task.target_wins == 7
    assert len(new.deck.card_ids) == 15
    loaded = TaskQueue.load(app.task_queue.path, app.catalog)
    assert loaded.pending[0].task.target_wins == 7 and loaded.pending[0].deck == new.deck
    assert not any(state.buttons for state in app.controller.history)


def test_pending_order_and_removal_leave_current_task_untouched(app):
    current = activate(app)
    first = app.task_queue.append(task(app, OTHER, 5))
    second = app.task_queue.append(task(app, OTHER, 9))
    app._refresh_queue_view()
    app.queue_tree.selection_set(second.entry_id)
    app.move_queue_task(-1)
    assert [x.entry_id for x in app.task_queue.pending] == [second.entry_id, first.entry_id]
    app.remove_queue_task()
    assert [x.entry_id for x in app.task_queue.pending] == [first.entry_id]
    app.queue_tree.selection_set(current.entry_id)
    app.remove_queue_task()
    assert app.task_queue.current.entry_id == current.entry_id
    assert "只能" in app.message_var.get()


def test_handoff_requires_menu_freshness_and_a_drained_worker(app):
    old = activate(app)
    new = app.task_queue.append(task(app, OTHER, 7))
    app.runner._stable_level_wins = (3, 3, 30)
    app.runner.plan = build_npc_plan(old.task, (3, 3, 30))
    app.runner.status = TaskStatus.COMPLETED
    app.last_analysis = observation(GameState.REPLAY_CHOICE)
    app._maybe_advance_task_queue()
    assert app.task_queue.current.entry_id == old.entry_id
    menu = safe_menu(wins=(3, 3, 30))
    app.last_analysis = menu
    app._maybe_advance_task_queue()
    app._maybe_advance_task_queue()
    assert app.task_queue.current.entry_id == old.entry_id
    app._task_thread = object()
    app.last_analysis = replace(menu)
    app._maybe_advance_task_queue()
    assert app.task_queue.current.entry_id == old.entry_id
    app._task_thread = None
    app._maybe_advance_task_queue()
    assert app.task_queue.current.entry_id == new.entry_id
    assert app.runner.task.npc_id == OTHER and app.runner.status is TaskStatus.RUNNING
    assert app.task_queue.entries[0].status.value == "completed"
    assert app.task_queue.entries[0].observed_wins == (3, 3, 30)
    assert not any(state.buttons for state in app.controller.history)


def test_pause_stop_and_pending_edit_never_start_the_next_task(app):
    current = activate(app)
    app.task_queue.append(task(app, OTHER, 7))
    app.toggle_pause()
    assert not app._queue_enabled and app.runner.status is TaskStatus.PAUSED
    app.last_analysis = safe_menu(wins=(3, 3, 30))
    app.runner.status = TaskStatus.COMPLETED
    app._maybe_advance_task_queue()
    app.last_analysis = replace(app.last_analysis)
    app._maybe_advance_task_queue()
    assert app.task_queue.current.entry_id == current.entry_id
    app.stop_task()
    assert not app._queue_enabled and len(app.task_queue.pending) == 1


def test_start_from_an_empty_queue_snapshots_input_but_waits_for_npc_menu(app):
    select_input(app, NPC, 30)
    app.worker = object()  # Suppress capture startup; this test never dispatches frames.
    app.start_task()
    assert app._queue_start_requested and app.runner.task is None
    app.toggle_pause()
    assert not app._queue_enabled and app.runner.status is TaskStatus.IDLE
    app.toggle_pause()
    assert app._queue_enabled and app.runner.task is None and app.runner.status is TaskStatus.IDLE
    app.last_analysis = observation(GameState.TURN_SELECT)
    app._maybe_advance_task_queue()
    assert app.runner.task is None
    app.last_analysis = safe_menu(wins=(3, 3, 4))
    app._maybe_advance_task_queue()
    app.last_analysis = replace(app.last_analysis)
    app._maybe_advance_task_queue()
    assert app.runner.task.npc_id == NPC and app.runner.task.target_wins == 30
    assert not app._queue_start_requested


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def test_debug_controls_only_exist_inside_the_debug_window_and_reload_keeps_it(app, monkeypatch):
    from tableturf_rebuild.application import core_reload
    labels = {widget.cget("text") for widget in descendants(app.root) if "text" in widget.keys()}
    assert {"追加任务", "开始队列"} <= labels
    assert not {"调试地图", "打牌调试", "重载核心（暂停）"} & labels
    app.task_queue.append(task(app))
    app._save_task_queue()
    queue = app.task_queue
    app.open_debug_window()
    debug = app.debug_window
    assert debug.debug_stage_box.cget("textvariable") == str(app.debug_stage_var)
    assert debug.reload_core_button.cget("text") == "重载核心（暂停）"
    monkeypatch.setattr(core_reload, "reload_core", lambda: ({}, {"test": "same-core"}))
    app.reload_analysis_core()
    assert app.debug_window is debug and debug.winfo_exists()
    assert app.task_queue is queue and len(queue.pending) == 1
    assert not app._queue_enabled and app.status_var.get() == "CORE_RELOADED_PAUSED"


def test_loading_a_running_queue_does_not_start_a_task(tmp_path):
    catalog = GameCatalog.from_package()
    path = tmp_path / "running.json"
    queue = TaskQueue(catalog, path)
    queue.append(NpcTaskDefinition(NPC, catalog.npcs[NPC].name_zh, 3, 30))
    queue.start_next()
    queue.save()
    root = tk.Tk()
    root.withdraw()
    app = TaskTerminalApp(root, auto_start=False, queue_path=path)
    try:
        assert app.task_queue.current is not None
        assert app.runner.task is None and app.runner.status is TaskStatus.IDLE
        assert not app._queue_enabled
    finally:
        app.close()
