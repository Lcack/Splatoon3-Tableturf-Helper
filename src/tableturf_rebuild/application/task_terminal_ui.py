from __future__ import annotations

import time
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.battle_executor import (
    VisionMctsBattleExecutor,
)
from tableturf_rebuild.application.deck_config import (
    DeckConfig,
    DeckPreset,
    DeckPresetStore,
)
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.application.task_plan import NpcTaskDefinition
from tableturf_rebuild.application.task_runner import (
    TaskRunner,
    TaskSnapshot,
    TaskStatus,
)
from tableturf_rebuild.capture.models import CaptureConfig
from tableturf_rebuild.capture.worker import CaptureWorker
from tableturf_rebuild.control.easycon import EasyConBackend
from tableturf_rebuild.control.buttons import ControllerState
from tableturf_rebuild.control.sequences import pair_virtual_controller
from tableturf_rebuild.data.catalog import GameCatalog


class TaskTerminalApp:
    PREVIEW_WIDTH = 800
    PREVIEW_HEIGHT = 450

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AutoTableTurf 任务终端")
        self.root.geometry("1450x900")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.catalog = GameCatalog.from_package()
        self.project_root = Path(__file__).resolve().parents[3]
        self.preset_store = DeckPresetStore(
            self.project_root / "profiles" / "decks",
            self.catalog,
        )
        self.runtime = BattleVisionRuntime.default()
        self.controller = EasyConBackend()
        self.runner = TaskRunner(self.runtime, self.controller)
        self.worker: CaptureWorker | None = None
        self.last_sequence = -1
        self._photo: ImageTk.PhotoImage | None = None
        self.last_analysis = None
        self._analysis_counter = 0
        self.debug_window = None
        self._task_thread: threading.Thread | None = None
        self._task_thread_lock = threading.Lock()
        self._task_result_lock = threading.Lock()
        self._pending_task_result = None
        self._last_task_dispatch = 0.0
        self._last_idle_analysis = 0.0
        self._last_preview_at = 0.0
        self._last_ai_log_key = None
        self._presets: dict[str, DeckPreset] = {}
        self._npc_labels: dict[str, str] = {}
        self._port_devices: dict[str, str] = {}
        self.debug_stage_ids = tuple(
            sorted(self.runtime.board_profile_names)
        )
        self.battle_debug_active = False
        self.battle_debug_executor: VisionMctsBattleExecutor | None = None
        self._battle_debug_button_text = tk.StringVar(value="打牌调试")
        self._build()
        self.refresh_ports()
        self.refresh_presets()
        self.root.after(700, self.auto_connect_startup)
        self.root.after(40, self._refresh)

    def _build(self) -> None:
        connection = ttk.LabelFrame(self.root, text="采集卡与控制器")
        connection.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Label(connection, text="采集后端").grid(row=0, column=0, padx=3)
        self.backend_var = tk.StringVar(value="DSHOW")
        ttk.Combobox(
            connection,
            textvariable=self.backend_var,
            values=("DSHOW", "MSMF", "ANY"),
            state="readonly",
            width=8,
        ).grid(row=0, column=1, padx=3)
        ttk.Label(connection, text="设备").grid(row=0, column=2, padx=3)
        self.device_var = tk.StringVar(value="0")
        ttk.Spinbox(
            connection,
            from_=0,
            to=10,
            textvariable=self.device_var,
            width=5,
        ).grid(row=0, column=3, padx=3)
        self.resolution_var = tk.StringVar(value="1920x1080")
        ttk.Combobox(
            connection,
            textvariable=self.resolution_var,
            values=("1920x1080", "1280x720", "960x540"),
            state="readonly",
            width=11,
        ).grid(row=0, column=4, padx=3)
        self.fps_var = tk.StringVar(value="60")
        ttk.Combobox(
            connection,
            textvariable=self.fps_var,
            values=("60", "30"),
            state="readonly",
            width=5,
        ).grid(row=0, column=5, padx=3)
        ttk.Button(
            connection,
            text="启动采集",
            command=self.start_capture,
        ).grid(row=0, column=6, padx=4)

        ttk.Label(connection, text="EasyCon串口").grid(
            row=1,
            column=0,
            padx=3,
            pady=(5, 3),
        )
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(
            connection,
            textvariable=self.port_var,
            state="normal",
            width=28,
        )
        self.port_box.grid(
            row=1,
            column=1,
            columnspan=3,
            sticky="w",
            pady=(5, 3),
        )
        ttk.Button(
            connection,
            text="刷新串口",
            command=self.refresh_ports,
        ).grid(row=1, column=4, padx=3, pady=(5, 3))
        ttk.Button(
            connection,
            text="连接控制器",
            command=self.connect_controller,
        ).grid(row=1, column=5, padx=3, pady=(5, 3))
        ttk.Button(
            connection,
            text="连接虚拟手柄",
            command=self.connect_virtual_gamepad,
        ).grid(row=1, column=6, padx=3, pady=(5, 3))
        self.controller_var = tk.StringVar(value="控制器：未连接")
        ttk.Label(connection, textvariable=self.controller_var).grid(
            row=1,
            column=7,
            sticky="w",
            padx=4,
            pady=(5, 3),
        )

        task_frame = ttk.LabelFrame(self.root, text="任务")
        task_frame.pack(fill="x", padx=8, pady=4)
        npc_labels = tuple(
            f"{npc.name_zh or npc.name} | {npc.npc_id}"
            for npc in sorted(
                self.catalog.npcs.values(),
                key=lambda item: item.name_zh or item.name,
            )
        )
        self._npc_labels = {
            label: npc.npc_id
            for label, npc in zip(
                npc_labels,
                sorted(
                    self.catalog.npcs.values(),
                    key=lambda item: item.name_zh or item.name,
                ),
                strict=True,
            )
        }
        ttk.Label(task_frame, text="NPC").grid(row=0, column=0, padx=4, pady=4)
        self.npc_var = tk.StringVar(value=npc_labels[0] if npc_labels else "")
        ttk.Combobox(
            task_frame,
            textvariable=self.npc_var,
            values=npc_labels,
            state="readonly",
            width=36,
        ).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(task_frame, text="目标难度").grid(row=0, column=2, padx=4)
        self.difficulty_var = tk.StringVar(value="3")
        ttk.Combobox(
            task_frame,
            textvariable=self.difficulty_var,
            values=("1", "2", "3"),
            state="readonly",
            width=5,
        ).grid(row=0, column=3, padx=4)
        ttk.Label(task_frame, text="目标累计胜场").grid(row=0, column=4, padx=4)
        self.target_wins_var = tk.StringVar(value="30")
        ttk.Spinbox(
            task_frame,
            from_=1,
            to=999,
            textvariable=self.target_wins_var,
            width=7,
        ).grid(row=0, column=5, padx=4)

        ttk.Label(task_frame, text="预设卡组").grid(
            row=1,
            column=0,
            padx=4,
            pady=4,
        )
        self.deck_var = tk.StringVar(value="未指定（在卡组界面等待）")
        self.deck_box = ttk.Combobox(
            task_frame,
            textvariable=self.deck_var,
            state="readonly",
            width=36,
        )
        self.deck_box.grid(
            row=1,
            column=1,
            padx=4,
            pady=4,
            sticky="w",
        )
        ttk.Button(
            task_frame,
            text="管理预设卡组",
            command=self.open_deck_editor,
        ).grid(row=1, column=2, padx=4)
        ttk.Button(
            task_frame,
            text="刷新预设",
            command=self.refresh_presets,
        ).grid(row=1, column=3, padx=4)
        ttk.Button(
            task_frame,
            text="开始任务",
            command=self.start_task,
        ).grid(row=1, column=4, padx=4)
        ttk.Button(
            task_frame,
            text="暂停/继续",
            command=self.toggle_pause,
        ).grid(row=1, column=5, padx=4)
        ttk.Button(
            task_frame,
            text="停止",
            command=self.stop_task,
        ).grid(row=1, column=6, padx=4)
        ttk.Button(
            task_frame,
            text="调试",
            command=self.open_debug_window,
        ).grid(row=1, column=7, padx=4)
        ttk.Label(task_frame, text="调试地图").grid(
            row=2,
            column=0,
            padx=4,
            pady=4,
        )
        default_stage = (
            "Rectangle"
            if "Rectangle" in self.debug_stage_ids
            else self.debug_stage_ids[0]
            if self.debug_stage_ids
            else ""
        )
        self.debug_stage_var = tk.StringVar(value=default_stage)
        ttk.Combobox(
            task_frame,
            textvariable=self.debug_stage_var,
            values=self.debug_stage_ids,
            state="readonly",
            width=20,
        ).grid(row=2, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(
            task_frame,
            textvariable=self._battle_debug_button_text,
            command=self.toggle_battle_debug,
        ).grid(row=2, column=2, padx=4)
        ttk.Label(
            task_frame,
            text="直接读取当前战斗画面；NPC数据缺失时用Starter牌组模拟对手",
        ).grid(
            row=2,
            column=3,
            columnspan=5,
            sticky="w",
            padx=4,
        )

        content = ttk.Frame(self.root)
        content.pack(fill="both", expand=True, padx=8, pady=4)
        preview_panel = ttk.LabelFrame(content, text="采集画面")
        preview_panel.pack(side="left", fill="both", padx=(0, 5))
        self.preview = tk.Label(
            preview_panel,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            background="black",
            foreground="white",
            text="等待采集画面",
        )
        self.preview.pack(fill="both", expand=True, padx=4, pady=4)

        side = ttk.LabelFrame(content, text="任务状态与进度")
        side.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.status_var = tk.StringVar(value="IDLE")
        ttk.Label(
            side,
            textvariable=self.status_var,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=8, pady=(8, 3))
        self.message_var = tk.StringVar(value="任务未开始")
        ttk.Label(
            side,
            textvariable=self.message_var,
            wraplength=510,
        ).pack(anchor="w", padx=8, pady=2)
        self.live_var = tk.StringVar(value="实时识别：-")
        ttk.Label(
            side,
            textvariable=self.live_var,
            wraplength=510,
        ).pack(anchor="w", padx=8, pady=2)
        self.plan_var = tk.StringVar(value="计划：等待NPC胜场识别")
        ttk.Label(
            side,
            textvariable=self.plan_var,
            wraplength=510,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=8, pady=(6, 3))

        columns = ("level", "goal", "current", "remaining", "status")
        self.progress = ttk.Treeview(
            side,
            columns=columns,
            show="headings",
            height=5,
        )
        labels = {
            "level": "难度",
            "goal": "目标",
            "current": "当前",
            "remaining": "剩余",
            "status": "状态",
        }
        for column in columns:
            self.progress.heading(column, text=labels[column])
            self.progress.column(column, width=84, anchor="center")
        self.progress.pack(fill="x", padx=8, pady=4)

        self.log = tk.Text(self.root, height=8, state="disabled")
        self.log.pack(fill="both", expand=False, padx=8, pady=(4, 8))

    def refresh_ports(self) -> None:
        ports = EasyConBackend.list_ports()
        self._port_devices = {
            f"{port.device} | {port.description or '串口'}": port.device
            for port in ports
        }
        values = tuple(self._port_devices)
        self.port_box.configure(values=values)
        if values and not self.port_var.get():
            self.port_var.set(values[0])

    def connect_controller(self, *, show_errors: bool = True) -> None:
        device = self._port_devices.get(self.port_var.get(), self.port_var.get())
        if not device:
            if show_errors:
                messagebox.showwarning("控制器", "没有可用串口")
            return
        if self.controller.connect(device):
            self.controller_var.set(f"控制器：已连接 {device}")
            self.log_message(f"EasyCon 已连接: {device}")
        else:
            self.controller_var.set("控制器：连接失败")
            if show_errors:
                messagebox.showerror(
                    "控制器",
                    self.controller.last_error or "连接失败",
                )

    def auto_connect_startup(self) -> None:
        if self._presets and self.deck_var.get() not in self._presets:
            label = next(iter(self._presets))
            self.deck_var.set(label)
            self.log_message(
                f"自动选择卡组: {self._presets[label].config.name}"
            )
        if not self.controller.connected:
            device = self._port_devices.get(
                self.port_var.get(),
                self.port_var.get(),
            )
            if not device:
                self.log_message("自动连接虚拟手柄跳过：没有可用串口")
                return
            self.connect_controller(show_errors=False)
        if self.controller.connected:
            try:
                pair_virtual_controller(self.controller)
            except RuntimeError as error:
                self.log_message(f"自动虚拟手柄配对失败: {error}")
                return
            self.controller_var.set("控制器：已自动连接虚拟手柄")
            self.log_message(
                "自动发送虚拟手柄配对：L+R 两次 → A，"
                "按键50ms，间隔350ms"
            )

    def connect_virtual_gamepad(self) -> None:
        if not self.controller.connected:
            messagebox.showwarning(
                "连接虚拟手柄",
                "请先连接EasyCon控制器",
            )
            return
        if self.debug_window is not None:
            self.debug_window.release_all()
            time.sleep(0.05)
        if self.runner.status not in {
            TaskStatus.IDLE,
            TaskStatus.PAUSED,
            TaskStatus.COMPLETED,
        }:
            self.runner.pause()
            self._update_snapshot(self.runner.snapshot())
            self.log_message("发送虚拟手柄配对序列，任务已暂停")
        try:
            pair_virtual_controller(self.controller)
        except RuntimeError as error:
            messagebox.showerror("连接虚拟手柄", str(error))
            self.log_message(f"虚拟手柄配对失败: {error}")
            return
        self.controller_var.set("控制器：已发送虚拟手柄配对")
        self.log_message(
            "已发送：L+R 两次 → A，按键保持50ms，操作间隔350ms"
        )

    def start_capture(self) -> None:
        self.stop_capture()
        try:
            width, height = (
                int(value)
                for value in self.resolution_var.get().split("x", 1)
            )
            config = CaptureConfig(
                device_index=int(self.device_var.get()),
                backend=self.backend_var.get(),
                width=width,
                height=height,
                fps=float(self.fps_var.get()),
            )
        except ValueError as error:
            messagebox.showerror("采集设置", str(error))
            return
        self.worker = CaptureWorker(config)
        self.worker.start()
        self.last_sequence = -1
        self.log_message(
            f"启动采集 {config.backend} index={config.device_index}"
        )

    def stop_capture(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def refresh_presets(self, selected: str | None = None) -> None:
        presets = self.preset_store.list()
        labels = ["未指定（在卡组界面等待）"]
        self._presets = {}
        for preset in presets:
            label = f"{preset.config.name} | {preset.path.stem}"
            labels.append(label)
            self._presets[label] = preset
        self.deck_box.configure(values=tuple(labels))
        if selected and selected in labels:
            self.deck_var.set(selected)
        elif self.deck_var.get() not in labels:
            self.deck_var.set(labels[0])

    def open_deck_editor(self) -> None:
        preset = self._presets.get(self.deck_var.get())
        DeckPresetDialog(
            self.root,
            self.catalog,
            self.preset_store,
            preset,
            on_saved=self._preset_saved,
        )

    def _preset_saved(self, preset: DeckPreset | None) -> None:
        if preset is None:
            self.refresh_presets()
            return
        label = f"{preset.config.name} | {preset.path.stem}"
        self.refresh_presets(selected=label)

    def start_task(self) -> None:
        self._wait_for_task_worker()
        if self.battle_debug_active:
            self._stop_battle_debug()
        if not self.controller.connected:
            messagebox.showwarning("任务", "请先连接EasyCon控制器")
            return
        if self.worker is None:
            self.start_capture()
        npc_id = self._npc_labels.get(self.npc_var.get())
        if npc_id is None:
            messagebox.showerror("任务", "请选择NPC")
            return
        try:
            task = NpcTaskDefinition(
                npc_id=npc_id,
                npc_name=self.catalog.npcs[npc_id].name_zh,
                target_level=int(self.difficulty_var.get()),
                target_wins=int(self.target_wins_var.get()),
                deck_name=(
                    self._presets[self.deck_var.get()].config.name
                    if self.deck_var.get() in self._presets
                    else None
                ),
            )
        except ValueError as error:
            messagebox.showerror("任务", str(error))
            return
        preset = self._presets.get(self.deck_var.get())
        deck = preset.config if preset is not None else None
        self.runner.configure(task, deck)
        self._last_ai_log_key = None
        try:
            self.runner.start()
        except RuntimeError as error:
            messagebox.showerror("任务", str(error))
            return
        self.log_message(
            f"任务启动: {task.npc_name} 难度{task.target_level} "
            f"累计{task.target_wins}胜 卡组={deck.name if deck else '未指定'}"
        )
        self._update_snapshot(self.runner.snapshot())
        self._last_task_dispatch = 0.0

    def toggle_pause(self) -> None:
        self._wait_for_task_worker()
        if self.runner.status is TaskStatus.PAUSED:
            self.runner.resume()
        else:
            self.runner.pause()
        self._update_snapshot(self.runner.snapshot())

    def stop_task(self) -> None:
        self._wait_for_task_worker()
        if self.battle_debug_active:
            self._stop_battle_debug()
        self.runner.stop()
        self.log_message("任务停止")
        self._update_snapshot(self.runner.snapshot())

    def open_debug_window(self) -> None:
        if self.debug_window is not None:
            self.debug_window.lift()
            return
        self._wait_for_task_worker()
        if self.battle_debug_active:
            self._stop_battle_debug()
        if self.runner.status not in {
            TaskStatus.IDLE,
            TaskStatus.PAUSED,
            TaskStatus.COMPLETED,
        }:
            self.runner.pause()
            self._update_snapshot(self.runner.snapshot())
            self.log_message("打开调试窗，任务已暂停")
        from tableturf_rebuild.application.task_debug_ui import (
            TaskDebugWindow,
        )

        self.debug_window = TaskDebugWindow(
            self.root,
            self,
            on_close=self._debug_closed,
        )

    def toggle_battle_debug(self) -> None:
        self._wait_for_task_worker()
        if self.battle_debug_active:
            self._stop_battle_debug(log=True)
            return
        if not self.controller.connected:
            messagebox.showwarning(
                "打牌调试",
                "请先连接EasyCon控制器",
            )
            return
        preset = self._presets.get(self.deck_var.get())
        if preset is None:
            messagebox.showwarning(
                "打牌调试",
                "请先选择或创建预设卡组",
            )
            return
        if self.worker is None:
            self.start_capture()
        if self.runner.status not in {
            TaskStatus.IDLE,
            TaskStatus.PAUSED,
            TaskStatus.COMPLETED,
        }:
            self.runner.pause()
        stage_id = self.debug_stage_var.get()
        try:
            self.runtime.reset()
            self.runtime.configure_stage(stage_id)
            self.runtime.set_expected_deck(
                preset.config.name,
                preset.config.card_ids,
            )
        except (KeyError, ValueError) as error:
            messagebox.showerror("打牌调试", str(error))
            return
        fallback = (
            "MiniGame_Starter"
            if "MiniGame_Starter" in self.catalog.decks
            else next(iter(self.catalog.decks))
        )
        self.battle_debug_executor = VisionMctsBattleExecutor(
            self.runtime,
            self.controller,
            preset.config.card_ids,
            fallback_opponent_deck_id=fallback,
        )
        self.battle_debug_active = True
        self._last_ai_log_key = None
        self._battle_debug_button_text.set("停止打牌调试")
        self.status_var.set("BATTLE_DEBUG")
        self.message_var.set("打牌调试已启动，直接分析当前战斗画面")
        self.plan_var.set(
            f"调试地图：{stage_id} | 卡组：{preset.config.name} | "
            f"对手模型：{fallback}"
        )
        self._last_task_dispatch = 0.0
        self.log_message(
            f"打牌调试启动: stage={stage_id} deck={preset.config.name}"
        )

    def _stop_battle_debug(self, *, log: bool = False) -> None:
        self._wait_for_task_worker()
        self.battle_debug_active = False
        self.battle_debug_executor = None
        self._battle_debug_button_text.set("打牌调试")
        self.controller.send_state(ControllerState.neutral())
        self.status_var.set("BATTLE_DEBUG_STOPPED")
        self.message_var.set("打牌调试已停止，正式任务保持暂停")
        if log:
            self.log_message("打牌调试停止")

    def _debug_closed(self) -> None:
        self.debug_window = None

    def _refresh(self) -> None:
        self._collect_task_result()
        if self.worker is not None:
            packet = self.worker.latest()
            if packet is not None and packet.sequence != self.last_sequence:
                self.last_sequence = packet.sequence
                now = time.perf_counter()
                if now - self._last_preview_at >= 0.05:
                    self._show_frame(packet.frame)
                    self._last_preview_at = now
                if self.battle_debug_active:
                    if now - self._last_task_dispatch >= 0.12:
                        self._start_task_worker(packet.frame)
                elif self.runner.status not in {
                    TaskStatus.IDLE,
                    TaskStatus.PAUSED,
                    TaskStatus.COMPLETED,
                }:
                    if now - self._last_task_dispatch >= 0.12:
                        self._start_task_worker(packet.frame)
                else:
                    if now - self._last_idle_analysis >= 0.1:
                        self._analysis_counter += 1
                        self.last_analysis = self.runtime.update(
                            packet.frame,
                            include_selection=self._analysis_counter % 5 == 0,
                        )
                        self._last_idle_analysis = now
                if (
                    self.debug_window is not None
                    and now - self._last_preview_at <= 0.06
                ):
                    self.debug_window.refresh_from_parent()
            error = self.worker.error
            if error:
                self.message_var.set(f"采集错误：{error}")
        self.root.after(40, self._refresh)

    def _start_task_worker(self, frame) -> None:
        with self._task_thread_lock:
            if self._task_thread is not None:
                return
            self._last_task_dispatch = time.perf_counter()
            thread = threading.Thread(
                target=(
                    self._run_battle_debug_tick
                    if self.battle_debug_active
                    else self._run_task_tick
                ),
                args=(frame,),
                daemon=True,
            )
            self._task_thread = thread
            thread.start()

    def _run_task_tick(self, frame) -> None:
        result = None
        error = None
        try:
            result = self.runner.tick(frame)
        except Exception as exception:
            error = exception
        with self._task_result_lock:
            self._pending_task_result = ("task", result, error)

    def _run_battle_debug_tick(self, frame) -> None:
        payload = None
        error = None
        try:
            executor = self.battle_debug_executor
            if executor is None:
                raise RuntimeError("打牌调试执行器未初始化")
            observation = self.runtime.update(
                frame,
                include_selection=True,
                force_selection=bool(
                    getattr(executor, "requires_selection_data", False)
                ),
            )
            executor.step(observation, frame)
            payload = (
                observation,
                executor.message,
                executor.phase.value,
            )
        except Exception as exception:
            error = exception
        with self._task_result_lock:
            self._pending_task_result = ("debug", payload, error)

    def _collect_task_result(self) -> None:
        with self._task_result_lock:
            pending = self._pending_task_result
            self._pending_task_result = None
        with self._task_thread_lock:
            thread = self._task_thread
            if thread is not None and not thread.is_alive():
                self._task_thread = None
        if pending is None:
            return
        kind, payload, error = pending
        if error is not None:
            self.status_var.set(TaskStatus.ERROR.value)
            self.message_var.set(f"任务线程错误：{error}")
            self.log_message(f"任务线程错误: {error}")
            return
        if kind == "debug":
            if payload is None:
                return
            observation, message, phase = payload
            self.last_analysis = observation
            self.status_var.set(f"BATTLE_DEBUG:{phase}")
            self.message_var.set(message)
            self._update_battle_debug_live(observation)
            self._log_ai_action(
                ("BATTLE_DEBUG", phase, message),
                f"AI操作 [{phase}] {message}",
            )
            if self.debug_window is not None:
                self.debug_window.refresh_from_parent()
            return
        snapshot = payload
        if snapshot is not None:
            self._update_snapshot(snapshot)
            self.last_analysis = snapshot.observation
            if self.debug_window is not None:
                self.debug_window.refresh_from_parent()

    def _update_battle_debug_live(self, observation) -> None:
        hand = observation.hand
        hand_text = "手牌=-"
        if hand is not None:
            hand_text = "手牌=" + ",".join(
                f"{card.name}({card.area}/{card.sp_cost})"
                for card in hand.cards
            )
        self.live_var.set(
            f"状态={observation.state.state.value} | {hand_text}"
        )

    def _wait_for_task_worker(self) -> None:
        with self._task_thread_lock:
            thread = self._task_thread
        if thread is not None:
            thread.join(timeout=3)
        self._collect_task_result()

    def _show_frame(self, frame) -> None:
        image = Image.fromarray(frame[:, :, ::-1])
        image = ImageOps.contain(
            image,
            (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT),
            Image.Resampling.LANCZOS,
        )
        self._photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self._photo, text="")

    def _update_snapshot(self, snapshot: TaskSnapshot) -> None:
        self.status_var.set(snapshot.status.value)
        self.message_var.set(snapshot.message)
        wins = snapshot.observed_level_wins or (None, None, None)
        self.live_var.set(
            f"状态={snapshot.state.value} | "
            f"NPC={snapshot.current_npc_name or '?'} | "
            f"难度={snapshot.current_difficulty or '?'} | "
            f"胜场={wins[0] or 0}/{wins[1] or 0}/{wins[2] or 0}"
        )
        self._log_ai_action(
            (snapshot.status.value, snapshot.message),
            f"AI操作 [{snapshot.status.value}] {snapshot.message}",
        )
        plan = snapshot.plan
        for item in self.progress.get_children():
            self.progress.delete(item)
        if plan is None:
            self.plan_var.set("计划：等待三档胜利数稳定识别")
            return
        self.plan_var.set(
            f"计划：{plan.task.npc_name} → 难度{plan.task.target_level} "
            f"累计{plan.task.target_wins}胜 | "
            f"剩余{plan.total_remaining_wins}胜"
        )
        for stage in plan.stages:
            self.progress.insert(
                "",
                "end",
                values=(
                    f"难度{stage.level}",
                    stage.goal_wins,
                    stage.current_wins,
                    stage.remaining_wins,
                    "完成" if stage.complete else "进行中",
                ),
            )
    def _log_ai_action(self, key, message: str) -> None:
        if key == self._last_ai_log_key:
            return
        self._last_ai_log_key = key
        self.log_message(message)

    def log_message(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(
            "end",
            time.strftime("%H:%M:%S ") + message + "\n",
        )
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self) -> None:
        self._wait_for_task_worker()
        if self.battle_debug_active:
            self._stop_battle_debug()
        if self.debug_window is not None:
            self.debug_window.close()
            self.debug_window = None
        self.runner.stop()
        self.stop_capture()
        self.controller.disconnect()
        self.root.destroy()


class DeckPresetDialog(tk.Toplevel):
    def __init__(
        self,
        parent,
        catalog: GameCatalog,
        store: DeckPresetStore,
        preset: DeckPreset | None,
        *,
        on_saved,
    ):
        super().__init__(parent)
        self.title("预设卡组管理")
        self.geometry("1080x720")
        self.catalog = catalog
        self.store = store
        self.preset = preset
        self.on_saved = on_saved
        self.selected_ids: list[str] = (
            list(preset.config.card_ids) if preset is not None else []
        )
        self._filtered_ids: list[str] = []
        self.name_var = tk.StringVar(
            value=preset.config.name if preset is not None else ""
        )
        self.search_var = tk.StringVar()
        self.count_var = tk.StringVar()
        self._build()
        self._refresh_search()
        self._refresh_selected()

    def _build(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="卡组名称").pack(side="left")
        ttk.Entry(top, textvariable=self.name_var, width=30).pack(
            side="left",
            padx=6,
        )
        ttk.Label(top, textvariable=self.count_var).pack(
            side="left",
            padx=12,
        )
        ttk.Button(top, text="保存", command=self.save).pack(side="right")
        if self.preset is not None:
            ttk.Button(top, text="删除", command=self.delete).pack(
                side="right",
                padx=4,
            )

        body = ttk.Frame(self, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)
        left = ttk.LabelFrame(body, text="全部卡牌")
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        search = ttk.Entry(left, textvariable=self.search_var)
        search.pack(fill="x", padx=5, pady=5)
        search.bind("<KeyRelease>", lambda event: self._refresh_search())
        self.results = tk.Listbox(left, exportselection=False)
        self.results.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.results.bind("<Double-Button-1>", lambda event: self.add_selected())
        ttk.Button(left, text="加入牌组", command=self.add_selected).pack(
            fill="x",
            padx=5,
            pady=(0, 5),
        )

        right = ttk.LabelFrame(body, text="已选择15张")
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.chosen = tk.Listbox(right, exportselection=False)
        self.chosen.pack(fill="both", expand=True, padx=5, pady=5)
        self.chosen.bind(
            "<Double-Button-1>",
            lambda event: self.remove_selected(),
        )
        ttk.Button(right, text="移除", command=self.remove_selected).pack(
            fill="x",
            padx=5,
            pady=(0, 5),
        )

    def _refresh_search(self) -> None:
        query = self.search_var.get().strip().casefold()
        rows = []
        for card in self.catalog.cards.values():
            haystack = " ".join(
                (card.card_id, card.name, card.name_zh)
            ).casefold()
            if query and query not in haystack:
                continue
            rows.append(
                (
                    card.card_id,
                    f"{card.name_zh or card.name} | {card.name} | "
                    f"{card.area}点 SP{card.sp_cost} | {card.card_id}",
                )
            )
        rows.sort(key=lambda item: item[1])
        self._filtered_ids = [row[0] for row in rows[:200]]
        self.results.delete(0, "end")
        for _, label in rows[:200]:
            self.results.insert("end", label)

    def add_selected(self) -> None:
        selection = self.results.curselection()
        if not selection or len(self.selected_ids) >= 15:
            return
        card_id = self._filtered_ids[selection[0]]
        if card_id in self.selected_ids:
            return
        self.selected_ids.append(card_id)
        self._refresh_selected()

    def remove_selected(self) -> None:
        selection = self.chosen.curselection()
        if not selection:
            return
        del self.selected_ids[selection[0]]
        self._refresh_selected()

    def _refresh_selected(self) -> None:
        self.chosen.delete(0, "end")
        for index, card_id in enumerate(self.selected_ids, start=1):
            card = self.catalog.cards[card_id]
            self.chosen.insert(
                "end",
                f"{index:02d}. {card.name_zh or card.name} | "
                f"{card.area}点 SP{card.sp_cost} | {card_id}",
            )
        self.count_var.set(f"{len(self.selected_ids)}/15")

    def save(self) -> None:
        try:
            preset = self.store.save(
                self.name_var.get(),
                tuple(self.selected_ids),
            )
        except Exception as error:
            messagebox.showerror("保存卡组", str(error), parent=self)
            return
        self.preset = preset
        self.on_saved(preset)
        messagebox.showinfo("保存卡组", "预设卡组已保存", parent=self)

    def delete(self) -> None:
        if self.preset is None:
            return
        if not messagebox.askyesno(
            "删除卡组",
            f"删除预设“{self.preset.config.name}”？",
            parent=self,
        ):
            return
        self.store.delete(self.preset)
        self.preset = None
        self.selected_ids.clear()
        self.name_var.set("")
        self._refresh_selected()
        self.on_saved(None)


def run_task_terminal() -> int:
    root = tk.Tk()
    TaskTerminalApp(root)
    root.mainloop()
    return 0
