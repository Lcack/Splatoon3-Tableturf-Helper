from __future__ import annotations

import os
import queue
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

import cv2
from PIL import ImageTk

from tableturf_rebuild.application.battle_vision import BattleVisionObservation
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.capture.recorder import VideoRecorder
from tableturf_rebuild.control.buttons import (
    Button,
    ControllerState,
    StickVector,
)
from tableturf_rebuild.control.debug_ui import (
    BUTTON_LABELS,
    StateSender,
    VirtualStick,
)


class TaskDebugWindow(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        app,
        *,
        on_close: Callable[[], None],
    ):
        super().__init__(parent)
        self.app = app
        self._on_close = on_close
        self.title("AutoTableTurf 任务调试")
        self.geometry("1220x860")
        self.minsize(1120, 800)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.sender = StateSender(app.controller, self._report_error)
        self.pressed_buttons: set[Button] = set()
        self.sticky_mode = False
        self.button_widgets: dict[Button, tk.Button] = {}
        self.combo_widgets: dict[tuple[Button, ...], tk.Button] = {}
        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self.recorder = VideoRecorder()
        self.session_dir = (
            Path("artifacts/task_debug")
            / datetime.now().strftime("run-%Y%m%d-%H%M%S")
        )
        self._last_recorded_at = 0.0
        self._last_photo: ImageTk.PhotoImage | None = None
        self._build()
        self.after(50, self._drain_ui_queue)

    def _build(self) -> None:
        capture_row = ttk.LabelFrame(self, text="截图与录制")
        capture_row.pack(fill="x", padx=6, pady=(6, 3))
        ttk.Button(
            capture_row,
            text="截图当前画面",
            command=self.take_screenshot,
        ).pack(side="left", padx=4, pady=4)
        ttk.Button(
            capture_row,
            text="开始录制",
            command=self.start_recording,
        ).pack(side="left", padx=4)
        ttk.Button(
            capture_row,
            text="停止录制",
            command=self.stop_recording,
        ).pack(side="left", padx=4)
        ttk.Button(
            capture_row,
            text="打开保存目录",
            command=self.open_output_dir,
        ).pack(side="left", padx=4)
        self.record_var = tk.StringVar(value="录制：未开始")
        ttk.Label(capture_row, textvariable=self.record_var).pack(
            side="left",
            padx=12,
        )

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True, padx=6, pady=3)
        controls = ttk.Frame(content)
        controls.pack(side="left", fill="both", expand=True)
        left_stick_panel = self._panel(controls, "左摇杆", 0, 0)
        right_stick_panel = self._panel(controls, "右摇杆", 0, 2)
        dpad_panel = self._panel(controls, "十字键", 1, 0)
        face_panel = self._panel(controls, "ABXY", 1, 2)
        center_panel = self._panel(
            controls,
            "肩键 / 系统键",
            0,
            1,
            rowspan=2,
        )
        self.left_stick = VirtualStick(
            left_stick_panel,
            "左摇杆",
            lambda value: self._stick_changed("left", value),
        )
        self.left_stick.place(relx=0.5, rely=0.5, anchor="center")
        self.right_stick = VirtualStick(
            right_stick_panel,
            "右摇杆",
            lambda value: self._stick_changed("right", value),
        )
        self.right_stick.place(relx=0.5, rely=0.5, anchor="center")
        self._square_button(
            left_stick_panel,
            "L3",
            184,
            184,
            34,
            button=Button.L_STICK_PRESS,
        )
        self._square_button(
            right_stick_panel,
            "R3",
            184,
            184,
            34,
            button=Button.R_STICK_PRESS,
        )
        self._place_dpad(dpad_panel)
        self._place_face_buttons(face_panel)
        self._place_shoulder_buttons(center_panel)

        status_panel = ttk.LabelFrame(content, text="状态机实时识别")
        status_panel.pack(side="right", fill="both", padx=(6, 0))
        status_panel.configure(width=350)
        status_panel.pack_propagate(False)
        self.task_var = tk.StringVar(value="任务：-")
        self.state_var = tk.StringVar(value="状态：等待采集")
        self.confidence_var = tk.StringVar(value="置信度：-")
        self.menu_var = tk.StringVar(value="NPC信息：-")
        self.deck_var = tk.StringVar(value="牌组：-")
        self.hand_var = tk.StringVar(value="手牌：-")
        self.board_var = tk.StringVar(value="棋盘：-")
        self.special_var = tk.StringVar(value="已激活特殊：-")
        self.selection_var = tk.StringVar(value="选牌几何：-")
        self.reveal_var = tk.StringVar(value="亮牌：-")
        self.result_var = tk.StringVar(value="比赛结果：-")
        ttk.Label(
            status_panel,
            textvariable=self.task_var,
            wraplength=320,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Label(
            status_panel,
            textvariable=self.state_var,
            wraplength=320,
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=8, pady=2)
        for variable in (
            self.confidence_var,
            self.menu_var,
            self.deck_var,
            self.hand_var,
            self.board_var,
            self.special_var,
            self.selection_var,
            self.reveal_var,
            self.result_var,
        ):
            ttk.Label(
                status_panel,
                textvariable=variable,
                wraplength=320,
            ).pack(anchor="w", padx=8, pady=1)

        self.log = tk.Text(self, height=6, state="disabled")
        self.log.pack(fill="x", padx=6, pady=(3, 6))
        self.log_message("调试窗已打开，任务自动控制保持暂停")

    def _panel(
        self,
        parent,
        title: str,
        row: int,
        column: int,
        rowspan: int = 1,
    ) -> tk.Frame:
        width = 310 if column == 1 else 230
        height = 466 if rowspan == 2 else 230
        panel = tk.Frame(
            parent,
            width=width,
            height=height,
            borderwidth=1,
            relief="solid",
        )
        panel.grid(
            row=row,
            column=column,
            rowspan=rowspan,
            padx=3,
            pady=3,
            sticky="nsew",
        )
        panel.grid_propagate(False)
        ttk.Label(panel, text=title).place(relx=0.5, y=4, anchor="n")
        return panel

    def _square_button(
        self,
        parent,
        label: str,
        x: int,
        y: int,
        size: int,
        button: Button | None = None,
        combo: tuple[Button, ...] | None = None,
        box_width: int | None = None,
    ) -> tk.Button:
        widget = tk.Button(
            parent,
            text=label,
            width=0,
            height=0,
            font=("Segoe UI", 9, "bold"),
            takefocus=0,
            relief="raised",
            borderwidth=2,
            wraplength=max(20, (box_width or size) - 8),
            justify="center",
        )
        widget.place(x=x, y=y, width=box_width or size, height=size)
        if button is not None:
            widget.bind(
                "<ButtonPress-1>",
                lambda event, value=button: self._button_down(value),
            )
            widget.bind(
                "<ButtonRelease-1>",
                lambda event, value=button: self._button_up(value),
            )
            self.button_widgets[button] = widget
        if combo is not None:
            key = tuple(combo)
            widget.bind(
                "<ButtonPress-1>",
                lambda event, values=key: self._combo_down(values),
            )
            widget.bind(
                "<ButtonRelease-1>",
                lambda event, values=key: self._combo_up(values),
            )
            self.combo_widgets[key] = widget
        return widget

    def _place_dpad(self, panel: tk.Frame) -> None:
        size = 44
        self._square_button(
            panel, "↑", 93, 50, size, button=Button.DPAD_UP
        )
        self._square_button(
            panel, "←", 43, 98, size, button=Button.DPAD_LEFT
        )
        self._square_button(
            panel, "→", 143, 98, size, button=Button.DPAD_RIGHT
        )
        self._square_button(
            panel, "↓", 93, 146, size, button=Button.DPAD_DOWN
        )

    def _place_face_buttons(self, panel: tk.Frame) -> None:
        size = 44
        self._square_button(panel, "X", 93, 50, size, button=Button.X)
        self._square_button(panel, "Y", 43, 98, size, button=Button.Y)
        self._square_button(panel, "A", 143, 98, size, button=Button.A)
        self._square_button(panel, "B", 93, 146, size, button=Button.B)

    def _place_shoulder_buttons(self, panel: tk.Frame) -> None:
        size = 52
        combo_width = 90
        x_positions = (50, 110, 208)
        y_positions = (86, 150, 214, 278, 342)
        buttons = (
            ("ZL", x_positions[0], 0, Button.ZL, None, None),
            (
                "ZL+ZR",
                x_positions[1],
                0,
                None,
                (Button.ZL, Button.ZR),
                combo_width,
            ),
            ("ZR", x_positions[2], 0, Button.ZR, None, None),
            ("L", x_positions[0], 1, Button.L, None, None),
            (
                "L+R",
                x_positions[1],
                1,
                None,
                (Button.L, Button.R),
                combo_width,
            ),
            ("R", x_positions[2], 1, Button.R, None, None),
            ("截图", 97, 2, Button.CAPTURE, None, None),
            ("HOME", 161, 2, Button.HOME, None, None),
            ("−", 97, 3, Button.MINUS, None, None),
            ("+", 161, 3, Button.PLUS, None, None),
        )
        for label, x, row, button, combo, width in buttons:
            self._square_button(
                panel,
                label,
                x,
                y_positions[row],
                size,
                button=button,
                combo=combo,
                box_width=width,
            )
        ttk.Button(
            panel,
            text="回中",
            command=self.release_all,
        ).place(x=97, y=y_positions[4], width=size, height=size)
        self.sticky_button = tk.Button(
            panel,
            text="粘滞键",
            font=("Segoe UI", 9, "bold"),
            takefocus=0,
            relief="raised",
            borderwidth=1,
            command=self._toggle_sticky_mode,
        )
        self.sticky_button.place(
            x=161,
            y=y_positions[4],
            width=size,
            height=size,
        )

    def _button_down(self, button: Button) -> None:
        if button in self.pressed_buttons:
            if not self.sticky_mode:
                return
            self.pressed_buttons.remove(button)
        else:
            self.pressed_buttons.add(button)
        self._refresh_button_colors()
        self._push_state()
        self.log_message(
            f"{BUTTON_LABELS[button]} "
            f"{'粘滞' if self.sticky_mode else '按下'}"
        )

    def _button_up(self, button: Button) -> None:
        if self.sticky_mode or button not in self.pressed_buttons:
            return
        self.pressed_buttons.remove(button)
        self._refresh_button_colors()
        self._push_state()
        self.log_message(f"{BUTTON_LABELS[button]} 释放")

    def _combo_down(self, buttons: tuple[Button, ...]) -> None:
        active = all(button in self.pressed_buttons for button in buttons)
        if self.sticky_mode and active:
            self.pressed_buttons.difference_update(buttons)
        else:
            self.pressed_buttons.update(buttons)
        self._refresh_button_colors()
        self._push_state()
        self.log_message(
            "组合" + " + ".join(button.value for button in buttons)
        )

    def _combo_up(self, buttons: tuple[Button, ...]) -> None:
        if self.sticky_mode:
            return
        self.pressed_buttons.difference_update(buttons)
        self._refresh_button_colors()
        self._push_state()

    def _toggle_sticky_mode(self) -> None:
        self.sticky_mode = not self.sticky_mode
        self.sticky_button.configure(
            background="#3778c2" if self.sticky_mode else "SystemButtonFace",
            foreground="white" if self.sticky_mode else "black",
            relief="sunken" if self.sticky_mode else "raised",
        )
        if not self.sticky_mode:
            self.release_all()

    def _refresh_button_colors(self) -> None:
        for button, widget in self.button_widgets.items():
            active = button in self.pressed_buttons
            widget.configure(
                background="#3778c2" if active else "SystemButtonFace",
                foreground="white" if active else "black",
            )
        for buttons, widget in self.combo_widgets.items():
            active = all(button in self.pressed_buttons for button in buttons)
            widget.configure(
                background="#3778c2" if active else "SystemButtonFace",
                foreground="white" if active else "black",
            )

    def _stick_changed(self, side: str, value: StickVector) -> None:
        self._push_state(side=side, value=value)

    def _push_state(
        self,
        side: str | None = None,
        value: StickVector | None = None,
    ) -> None:
        left = self.left_stick.value
        right = self.right_stick.value
        if side == "left" and value is not None:
            left = value
        if side == "right" and value is not None:
            right = value
        self.sender.send(
            ControllerState(
                buttons=frozenset(self.pressed_buttons),
                left_stick=left,
                right_stick=right,
            )
        )

    def release_all(self) -> None:
        self.pressed_buttons.clear()
        self._refresh_button_colors()
        self.left_stick.reset()
        self.right_stick.reset()
        self._push_state()
        self.log_message("释放全部按键，摇杆归中")

    def take_screenshot(self) -> None:
        frame = self._current_frame()
        if frame is None:
            messagebox.showwarning("截图", "当前没有采集画面", parent=self)
            return
        directory = self.session_dir / "screenshots"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = directory / f"state-{stamp}.png"
        success, encoded = cv2.imencode(".png", frame)
        if not success:
            messagebox.showerror("截图", "PNG编码失败", parent=self)
            return
        path.write_bytes(encoded.tobytes())
        self.record_var.set(f"截图：{path}")
        self.log_message(f"截图已保存: {path.resolve()}")

    def start_recording(self) -> None:
        if self.recorder.recording:
            return
        frame = self._current_frame()
        if frame is None:
            messagebox.showwarning(
                "录制",
                "当前没有采集画面",
                parent=self,
            )
            return
        try:
            path = self.recorder.start(
                self.session_dir / "videos",
                frame.shape,
                fps=30.0,
            )
        except Exception as error:
            messagebox.showerror("录制", str(error), parent=self)
            return
        self._last_recorded_at = 0.0
        self.record_var.set(f"录制中：{path}")
        self.log_message(f"开始录制: {path.resolve()}")

    def stop_recording(self) -> None:
        if not self.recorder.recording:
            return
        path = self.recorder.stop()
        self.record_var.set(
            f"录制完成：{path}" if path is not None else "录制：未开始"
        )
        if path is not None:
            self.log_message(f"录像已保存: {path.resolve()}")

    def open_output_dir(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(self.session_dir.resolve())
        except OSError as error:
            messagebox.showerror("打开目录", str(error), parent=self)

    def _current_frame(self):
        packet = self.app.worker.latest() if self.app.worker else None
        return packet.frame.copy() if packet is not None else None

    def refresh_from_parent(self) -> None:
        if self.recorder.recording:
            packet = self.app.worker.latest() if self.app.worker else None
            if (
                packet is not None
                and packet.timestamp - self._last_recorded_at >= 1 / 30
            ):
                self.recorder.write(packet.frame)
                self._last_recorded_at = packet.timestamp
        snapshot = self.app.runner.snapshot()
        self.task_var.set(
            f"任务：{snapshot.status.value} | "
            f"剩余{snapshot.plan.total_remaining_wins if snapshot.plan else '?'}胜"
        )
        self._update_analysis(self.app.last_analysis)

    def _update_analysis(
        self,
        analysis: BattleVisionObservation | None,
    ) -> None:
        if analysis is None:
            return
        observation = analysis.state
        self.state_var.set(f"状态：{observation.state.value}")
        self.confidence_var.set(f"置信度：{observation.confidence:.1%}")
        if analysis.menu is not None:
            wins = "/".join(
                str(value) if value is not None else "?"
                for value in analysis.menu.level_wins
            )
            self.menu_var.set(
                f"NPC：{analysis.menu.npc_name or '未知'} | "
                f"胜场 {wins} | 难度 {analysis.menu.difficulty or '?'}"
            )
        elif analysis.opponent_context is not None:
            context = analysis.opponent_context
            self.menu_var.set(
                f"对手：{context.npc_name} | 难度 {context.difficulty}"
            )
        else:
            self.menu_var.set("NPC信息：-")
        if analysis.deck is not None:
            self.deck_var.set(
                f"牌组：{analysis.deck.deck_id} | "
                f"{analysis.deck.total_area}点 | "
                f"SP{analysis.deck.total_sp_cost}"
            )
        else:
            self.deck_var.set("牌组：-")
        if analysis.hand is not None:
            cards = " | ".join(
                f"{card.name} {card.area}/{card.sp_cost}"
                for card in analysis.hand.cards
            )
            self.hand_var.set(f"手牌：{cards}")
        else:
            self.hand_var.set("手牌：当前阶段不读取")
        if analysis.board is not None:
            self.board_var.set(
                f"我方{analysis.board.self_count}："
                f"敌方{analysis.board.opponent_count}"
            )
            active = analysis.active_specials
            if active is not None:
                self.special_var.set(
                    f"特殊：{active.self_special_count}/"
                    f"{active.opponent_special_count} | "
                    f"激活 {active.self_count}/{active.opponent_count}"
                )
            else:
                self.special_var.set("特殊：等待识别")
        else:
            self.board_var.set("棋盘：当前阶段不读取")
            self.special_var.set("特殊：-")
        if analysis.selection_profile is not None:
            self.selection_var.set(
                f"选牌几何：{analysis.selection_profile.variant}"
            )
        elif observation.state in {
            GameState.TURN_SELECT,
            GameState.SKIP_DISCARD,
            GameState.SPECIAL_SELECT,
        }:
            self.selection_var.set("选牌几何：等待稳定识别")
        else:
            self.selection_var.set("选牌几何：-")
        if analysis.reveal is not None:
            self.reveal_var.set(
                f"亮牌：我方 {analysis.reveal.own.name or '?'} / "
                f"对方 {analysis.reveal.opponent.name or '?'}"
            )
        else:
            self.reveal_var.set("亮牌：-")
        if analysis.match_result is not None:
            self.result_var.set(
                f"比赛结果：{analysis.match_result.outcome.value} "
                f"{analysis.match_result.confidence:.0%}"
            )
        else:
            self.result_var.set("比赛结果：-")

    def _report_error(self, message: str) -> None:
        self._ui_queue.put(lambda: self.log_message(f"控制器错误: {message}"))

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            callback()
        self.after(50, self._drain_ui_queue)

    def log_message(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(
            "end",
            time.strftime("%H:%M:%S ") + message + "\n",
        )
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self) -> None:
        if self.recorder.recording:
            path = self.recorder.stop()
            if path is not None:
                self.log_message(f"录像已保存: {path.resolve()}")
        try:
            self.release_all()
        finally:
            self.sender.stop()
            self.app.controller.send_state(ControllerState.neutral())
            self._on_close()
            self.destroy()
