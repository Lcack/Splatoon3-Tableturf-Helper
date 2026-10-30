"""Queue controls and main-thread task handoff for the Tk terminal."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from tableturf_rebuild.application.task_plan import NpcTaskDefinition
from tableturf_rebuild.application.task_queue import TaskQueue


class TaskQueueControls:
    def _ensure_task_queue(self):
        if getattr(self, "_queue_initialized", False):
            return getattr(self, "task_queue", None)
        self._queue_initialized = True
        self._queue_enabled = False
        self._queue_start_requested = False
        self._queue_menu_key = None
        self._queue_menu_reads = 0
        self._queue_last_observation = None
        self._queue_error = None
        path = getattr(self, "_queue_path", None) or self.project_root / "profiles/task_queue.json"
        try:
            self.task_queue = TaskQueue.load(path, self.catalog)
            task = getattr(self.runner, "task", None)
            if task is not None and getattr(self.runner.status, "value", "IDLE") != "IDLE":
                self.task_queue.adopt_current(task, self.runner.deck,
                    getattr(self.runner, "_stable_level_wins", None))
                self._queue_enabled = self.runner.status.value not in {"PAUSED", "ERROR", "COMPLETED"}
                self.task_queue.save()
        except (OSError, ValueError) as error:
            self.task_queue = None
            self._queue_error = str(error)
        return self.task_queue

    def _build_task_controls(self, parent):
        npcs = sorted(self.catalog.npcs.values(), key=lambda npc: npc.name_zh or npc.name)
        self._npc_labels = {f"{npc.name_zh or npc.name} | {npc.npc_id}": npc.npc_id for npc in npcs}
        for name, default in (("npc_var", next(iter(self._npc_labels), "")),
                              ("difficulty_var", "3"), ("target_wins_var", "30"),
                              ("deck_var", "未指定（在卡组界面等待）")):
            if not hasattr(self, name):
                setattr(self, name, tk.StringVar(master=self.root, value=default))
        if not hasattr(self, "debug_stage_var"):
            self.debug_stage_var = tk.StringVar(master=self.root, value=(
                "Rectangle" if "Rectangle" in self.debug_stage_ids else next(iter(self.debug_stage_ids), "")))
        for widget in parent.winfo_children():
            widget.destroy()
        ttk.Label(parent, text="NPC").grid(row=0, column=0, padx=4, pady=4)
        ttk.Combobox(parent, textvariable=self.npc_var, values=tuple(self._npc_labels),
                     state="readonly", width=36).grid(row=0, column=1, padx=4)
        ttk.Label(parent, text="目标难度").grid(row=0, column=2, padx=4)
        ttk.Combobox(parent, textvariable=self.difficulty_var, values=("1", "2", "3"),
                     state="readonly", width=5).grid(row=0, column=3, padx=4)
        ttk.Label(parent, text="目标累计胜场").grid(row=0, column=4, padx=4)
        ttk.Spinbox(parent, from_=1, to=999, textvariable=self.target_wins_var,
                    width=7).grid(row=0, column=5, padx=4)
        ttk.Label(parent, text="预设卡组").grid(row=1, column=0, padx=4, pady=4)
        self.deck_box = ttk.Combobox(parent, textvariable=self.deck_var, state="readonly", width=36,
            values=("未指定（在卡组界面等待）", *getattr(self, "_presets", {})))
        self.deck_box.grid(row=1, column=1, padx=4, pady=4, sticky="w")
        buttons = (("管理预设卡组", self.open_deck_editor), ("刷新预设", self.refresh_presets),
                   ("追加任务", self.add_queue_task), ("开始队列", self.start_task),
                   ("暂停/继续", self.toggle_pause), ("停止", self.stop_task),
                   ("调试", self.open_debug_window))
        for column, (label, command) in enumerate(buttons, 2):
            ttk.Button(parent, text=label, command=command).grid(row=1, column=column, padx=4)

    def _install_queue_controls(self):
        self._ensure_task_queue()
        panel = ttk.LabelFrame(self.progress.master, text="任务队列（按顺序执行）")
        panel.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.queue_summary_var = tk.StringVar(master=self.root)
        ttk.Label(panel, textvariable=self.queue_summary_var).pack(anchor="w", padx=5, pady=3)
        listing = ttk.Frame(panel)
        listing.pack(fill="both", expand=True, padx=4)
        columns = ("order", "npc", "goal", "deck", "status")
        self.queue_tree = ttk.Treeview(listing, columns=columns, show="headings", height=6, selectmode="browse")
        for column, label, width in zip(columns, ("顺序", "NPC", "目标累计", "卡组", "状态"),
                                        (38, 90, 110, 125, 80), strict=True):
            self.queue_tree.heading(column, text=label)
            self.queue_tree.column(column, width=width, anchor="center")
        scrollbar = ttk.Scrollbar(listing, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.queue_tree.pack(side="left", fill="both", expand=True)
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="上移", command=lambda: self.move_queue_task(-1)).pack(side="left", padx=4)
        ttk.Button(row, text="下移", command=lambda: self.move_queue_task(1)).pack(side="left", padx=4)
        ttk.Button(row, text="移除待办", command=self.remove_queue_task).pack(side="left", padx=4)
        self._refresh_queue_view()

    def _save_task_queue(self):
        try:
            self.task_queue.save()
            self._queue_error = None
            return True
        except OSError as error:
            self._queue_enabled = False
            self._queue_error = f"队列保存失败：{error}"
            self.message_var.set(self._queue_error)
            self.log_message(self._queue_error)
            return False

    def _queue_runner_matches(self, entry):
        task = getattr(self.runner, "task", None)
        return (task is not None and (task.npc_id, task.target_level, task.target_wins)
                == (entry.task.npc_id, entry.task.target_level, entry.task.target_wins)
                and self.runner.deck == entry.deck)

    def _refresh_queue_view(self):
        if not hasattr(self, "queue_tree"):
            return
        queue = getattr(self, "task_queue", None)
        if queue is None:
            self.queue_summary_var.set("队列无法载入：" + (self._queue_error or "未知错误"))
            return
        rows = []
        for index, entry in enumerate(queue.entries, 1):
            label = {"pending": "待执行", "running": "待恢复", "completed": "已完成"}[entry.status.value]
            if entry.status.value == "running" and self._queue_runner_matches(entry):
                label = {"PAUSED": "已暂停", "IDLE": "已停止", "ERROR": "发生错误",
                         "COMPLETED": "等待交接"}.get(self.runner.status.value, "执行中")
            wins = entry.observed_wins
            current = wins[entry.task.target_level - 1] if wins else None
            target = f"{entry.task.target_level}级 {current}/{entry.task.target_wins}胜" if current is not None else f"{entry.task.target_level}级 {entry.task.target_wins}胜"
            rows.append((entry.entry_id, (index, entry.task.npc_name, target,
                                         entry.deck.name if entry.deck else "未指定", label)))
        wanted = {entry_id for entry_id, _ in rows}
        for entry_id in self.queue_tree.get_children():
            if entry_id not in wanted:
                self.queue_tree.delete(entry_id)
        for index, (entry_id, values) in enumerate(rows):
            if self.queue_tree.exists(entry_id):
                if tuple(map(str, self.queue_tree.item(entry_id, "values"))) != tuple(map(str, values)):
                    self.queue_tree.item(entry_id, values=values)
                self.queue_tree.move(entry_id, "", index)
            else:
                self.queue_tree.insert("", "end", iid=entry_id, values=values)
        completed = sum(entry.status.value == "completed" for entry in queue.entries)
        note = " · 等待NPC菜单后开始" if self._queue_start_requested else ""
        if self._queue_error:
            note += " · " + self._queue_error
        self.queue_summary_var.set(f"当前 {int(queue.current is not None)} 项 · 待办 {len(queue.pending)} 项 · 已完成 {completed} 项{note}")

    def add_queue_task(self):
        queue = self._ensure_task_queue()
        if queue is None:
            messagebox.showerror("任务队列", self._queue_error)
            return None
        try:
            npc_id = self._npc_labels.get(self.npc_var.get())
            if npc_id is None:
                raise ValueError("请选择NPC")
            preset = self._presets.get(self.deck_var.get())
            deck = preset.config if preset else None
            task = NpcTaskDefinition(npc_id, self.catalog.npcs[npc_id].name_zh,
                int(self.difficulty_var.get()), int(self.target_wins_var.get()), deck.name if deck else None)
            entry = queue.append(task, deck)
        except (ValueError, TypeError) as error:
            messagebox.showerror("任务队列", str(error))
            return None
        self._save_task_queue()
        self._refresh_queue_view()
        self.log_message(f"追加待办：{task.npc_name} 难度{task.target_level} 累计{task.target_wins}胜")
        return entry

    def remove_queue_task(self):
        selection = self.queue_tree.selection()
        if not selection:
            return
        try:
            self.task_queue.remove(selection[0])
            self._save_task_queue()
        except ValueError as error:
            self.message_var.set(str(error))
        self._refresh_queue_view()

    def move_queue_task(self, offset):
        selection = self.queue_tree.selection()
        if not selection:
            return
        try:
            self.task_queue.move(selection[0], offset)
            self._save_task_queue()
        except ValueError as error:
            self.message_var.set(str(error))
        self._refresh_queue_view()

    def start_task_queue(self):
        from tableturf_rebuild.application.task_runner import TaskStatus
        queue = self._ensure_task_queue()
        if queue is None:
            messagebox.showerror("任务队列", self._queue_error)
            return
        if not self.controller.connected:
            messagebox.showwarning("任务队列", "请先连接EasyCon控制器")
            return
        self._wait_for_task_worker()
        if self.battle_debug_active:
            self._stop_battle_debug()
        if self.worker is None:
            self.start_capture()
        if queue.current is None and not queue.pending and self.add_queue_task() is None:
            return
        self._queue_enabled = True
        self._queue_menu_reads = 0
        self._queue_menu_key = self._queue_last_observation = None
        current = queue.current
        if current is not None and self._queue_runner_matches(current):
            if self.runner.status is TaskStatus.PAUSED:
                self.runner.resume()
            elif self.runner.status is TaskStatus.IDLE:
                self.runner.start()
            self._update_snapshot(self.runner.snapshot())
        else:
            self._queue_start_requested = True
            self.message_var.set("队列已就绪，等待NPC选择或难度菜单后开始")
        self._refresh_queue_view()

    def _start_queue_entry(self, entry):
        self.runner.configure(entry.task, entry.deck)
        self.runner.start()
        self._queue_start_requested = False
        self._queue_menu_reads = 0
        self._queue_last_observation = None
        self._last_ai_log_key = None
        self._last_task_dispatch = 0.0
        self.log_message(f"队列启动：{entry.task.npc_name} 难度{entry.task.target_level} 累计{entry.task.target_wins}胜")
        self._update_snapshot(self.runner.snapshot())

    def _maybe_advance_task_queue(self):
        from tableturf_rebuild.application.task_runner import TaskRunner, TaskStatus
        queue = getattr(self, "task_queue", None)
        if queue is None:
            return
        current = queue.current
        if current is not None and self._queue_runner_matches(current):
            wins = getattr(self.runner, "_stable_level_wins", None)
            if wins is not None and tuple(wins) != current.observed_wins:
                queue.update_current_wins(wins)
                self._save_task_queue()
        self._refresh_queue_view()
        if (not self._queue_enabled or self.battle_debug_active
            or getattr(self, "_reload_context", None) is not None
            or getattr(self, "_local_control_manual_busy", False)
            or getattr(self, "_task_thread", None) is not None
            or getattr(self, "_pending_task_result", None) is not None):
            return
        if not self._queue_start_requested and not (
            current is not None and self._queue_runner_matches(current)
            and self.runner.status is TaskStatus.COMPLETED
        ):
            return
        observation = self.last_analysis
        if observation is None or not TaskRunner.safe_menu_visible(observation):
            self._queue_menu_reads = 0
            self._queue_last_observation = None
            self._queue_menu_key = None
            return
        key = observation.state.state.value, observation.menu.npc_id
        if key != self._queue_menu_key:
            self._queue_menu_key, self._queue_menu_reads = key, 0
            self._queue_last_observation = None
        if observation is not self._queue_last_observation:
            self._queue_menu_reads += 1
            self._queue_last_observation = observation
        if self._queue_menu_reads < 2:
            return
        if current is not None and not self._queue_start_requested:
            queue.mark_current_completed()
            if not self._save_task_queue():
                return
        entry = queue.start_next()
        if entry is None:
            self._queue_enabled = False
            self._queue_start_requested = False
            self.message_var.set("队列中的所有任务已完成")
            self._refresh_queue_view()
            return
        if not self._save_task_queue():
            return
        try:
            self._start_queue_entry(entry)
        except (ValueError, RuntimeError) as error:
            self._queue_enabled = False
            self.message_var.set(f"队列启动失败：{error}")
            self.log_message(f"队列启动失败：{error}")
        self._refresh_queue_view()
