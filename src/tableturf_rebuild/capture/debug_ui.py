from __future__ import annotations

import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from tableturf_rebuild.application.battle_vision import (
    BattleVisionObservation,
    BattleVisionRuntime,
    CARD_SELECTION_STATES,
)
from tableturf_rebuild.application.state_machine import GameState, STATE_DISPLAY_NAMES
from tableturf_rebuild.application.deck_config import load_deck_config
from tableturf_rebuild.capture.annotation_ui import AnnotationDialog
from tableturf_rebuild.capture.annotations import AnnotationShape, save_annotation_bundle
from tableturf_rebuild.capture.models import CaptureConfig
from tableturf_rebuild.capture.recorder import VideoRecorder
from tableturf_rebuild.capture.worker import CaptureWorker
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.calibration import region_grid_cells
from tableturf_rebuild.vision.profile_semantics import (
    ProfileSemantics,
    RegionRole,
)


class CaptureDebugApp:
    PREVIEW_WIDTH = 960
    PREVIEW_HEIGHT = 540

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AutoTableTurf 采集卡实时视觉分析")
        self.root.geometry("1360x820")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker: CaptureWorker | None = None
        self.frozen_frame = None
        self.last_sequence = -1
        self.last_frame_time = 0.0
        self.photo: ImageTk.PhotoImage | None = None
        self.session_dir = Path("artifacts/state_capture") / datetime.now().strftime(
            "run-%Y%m%d-%H%M%S"
        )
        self.annotation_sequence = 1
        self.annotation_dialog: AnnotationDialog | None = None
        self.recorder = VideoRecorder()
        self.last_recorded_at = 0.0
        self.vision_runtime = BattleVisionRuntime.default()
        self.stage_ids = tuple(sorted(self.vision_runtime.board_profile_names))
        self.stage_var = tk.StringVar(
            value=self.stage_ids[0] if self.stage_ids else ""
        )
        if self.stage_ids:
            self.vision_runtime.configure_stage(self.stage_ids[0])
        self.state_machine_enabled = tk.BooleanVar(value=True)
        self.overlay_enabled = tk.BooleanVar(value=True)
        self.state_var = tk.StringVar(value="状态：-")
        self.confidence_var = tk.StringVar(value="置信度：-")
        self.menu_var = tk.StringVar(value="NPC信息：-")
        self.deck_var = tk.StringVar(value="牌组：-")
        self.board_var = tk.StringVar(value="棋盘：-")
        self.active_special_var = tk.StringVar(value="已激活特殊：-")
        self.result_var = tk.StringVar(value="比赛结果：-")
        self.selection_var = tk.StringVar(value="选牌几何：-")
        self.reveal_var = tk.StringVar(value="亮牌：-")
        self._analysis_counter = 0
        self._last_analysis: BattleVisionObservation | None = None
        self._selection_semantics_cache: dict[Path, ProfileSemantics] = {}
        self._build()
        self.root.after(30, self._refresh)

    def _build(self) -> None:
        toolbar = ttk.LabelFrame(self.root, text="采集设置")
        toolbar.pack(fill="x", padx=8, pady=6)

        self.backend_var = tk.StringVar(value="DSHOW")
        ttk.Combobox(
            toolbar,
            textvariable=self.backend_var,
            values=("DSHOW", "MSMF", "ANY"),
            state="readonly",
            width=10,
        ).grid(row=0, column=0, padx=4, pady=3)

        self.index_var = tk.StringVar(value="0")
        ttk.Spinbox(
            toolbar,
            from_=0,
            to=10,
            textvariable=self.index_var,
            width=5,
        ).grid(row=0, column=1, padx=4)

        self.resolution_var = tk.StringVar(value="1920x1080")
        ttk.Combobox(
            toolbar,
            textvariable=self.resolution_var,
            values=("1920x1080", "1280x720", "960x540", "640x480"),
            state="readonly",
            width=12,
        ).grid(row=0, column=2, padx=4)

        self.fps_var = tk.StringVar(value="60")
        ttk.Combobox(
            toolbar,
            textvariable=self.fps_var,
            values=("60", "30", "0"),
            state="readonly",
            width=6,
        ).grid(row=0, column=3, padx=4)

        ttk.Button(toolbar, text="开始采集", command=self.start).grid(
            row=0, column=4, padx=4
        )
        ttk.Button(toolbar, text="停止", command=self.stop).grid(
            row=0, column=5, padx=4
        )
        ttk.Button(toolbar, text="冻结/恢复", command=self.toggle_freeze).grid(
            row=0, column=6, padx=4
        )
        ttk.Button(toolbar, text="标注截图", command=self.annotate_current).grid(
            row=0, column=7, padx=4
        )
        ttk.Button(toolbar, text="开始录像", command=self.start_recording).grid(
            row=0, column=8, padx=4
        )
        ttk.Button(toolbar, text="停止录像", command=self.stop_recording).grid(
            row=0, column=9, padx=4
        )

        self.status_var = tk.StringVar(value="未启动")
        ttk.Label(toolbar, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=10, sticky="w", padx=4, pady=(2, 4)
        )

        analysis_options = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        analysis_options.pack(fill="x")
        ttk.Label(analysis_options, text="棋盘场地").pack(side="left")
        stage_box = ttk.Combobox(
            analysis_options,
            textvariable=self.stage_var,
            values=self.stage_ids,
            state="readonly",
            width=16,
        )
        stage_box.pack(side="left", padx=(3, 10))
        stage_box.bind("<<ComboboxSelected>>", self._stage_changed)
        ttk.Checkbutton(
            analysis_options,
            text="状态机识别",
            variable=self.state_machine_enabled,
        ).pack(side="left")
        ttk.Checkbutton(
            analysis_options,
            text="分析覆盖",
            variable=self.overlay_enabled,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            analysis_options,
            text="加载牌组配置",
            command=self.load_deck_config,
        ).pack(side="left", padx=(10, 2))
        ttk.Button(
            analysis_options,
            text="恢复自动牌组",
            command=self.clear_deck_config,
        ).pack(side="left")

        content = ttk.Frame(self.root, padding=(8, 0, 8, 3))
        content.pack(fill="both", expand=True)
        self.preview = tk.Label(
            content,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            background="black",
            foreground="white",
            text="等待采集卡画面",
        )
        self.preview.pack(side="left")

        side = ttk.LabelFrame(content, text="实时视觉分析")
        side.pack(side="right", fill="both", padx=(6, 0))
        side.configure(width=340)
        side.pack_propagate(False)
        ttk.Label(
            side,
            textvariable=self.state_var,
            font=("Segoe UI", 12, "bold"),
            wraplength=310,
        ).pack(anchor="w", padx=8, pady=(8, 2))
        for variable in (
            self.confidence_var,
            self.menu_var,
            self.deck_var,
            self.board_var,
            self.active_special_var,
            self.result_var,
            self.selection_var,
            self.reveal_var,
        ):
            ttk.Label(
                side,
                textvariable=variable,
                wraplength=310,
            ).pack(anchor="w", padx=8, pady=1)

        self.log = tk.Text(self.root, height=5, state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=(3, 8))

    def _stage_changed(self, event=None) -> None:
        del event
        try:
            self.vision_runtime.configure_stage(self.stage_var.get())
        except (KeyError, ValueError) as error:
            self.status_var.set(f"棋盘配置失败: {error}")
            return
        self.board_var.set(f"棋盘：{self.stage_var.get()} 等待分析")
        self.status_var.set(f"已切换棋盘 Profile: {self.stage_var.get()}")

    def load_deck_config(self) -> None:
        path = filedialog.askopenfilename(
            title="加载自定义牌组",
            filetypes=(("JSON配置", "*.json"), ("所有文件", "*.*")),
        )
        if not path:
            return
        try:
            config = load_deck_config(Path(path), self.vision_runtime.catalog)
            self.vision_runtime.set_expected_deck(
                config.name,
                config.card_ids,
            )
        except Exception as error:
            messagebox.showerror("加载牌组配置", str(error))
            return
        self.status_var.set(f"已指定牌组: {config.name}")

    def clear_deck_config(self) -> None:
        self.vision_runtime.clear_expected_deck()
        self.deck_var.set("牌组：恢复自动识别")
        self.status_var.set("已恢复自动牌组识别")

    def start(self) -> None:
        self.stop()
        try:
            width, height = (
                int(value) for value in self.resolution_var.get().split("x", 1)
            )
            config = CaptureConfig(
                device_index=int(self.index_var.get()),
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
        self.vision_runtime.reset()
        self._analysis_counter = 0
        self._last_analysis = None
        self.frozen_frame = None
        self.last_sequence = -1
        self.status_var.set("正在打开采集设备")
        self.log_message(
            f"启动 {config.backend} index={config.device_index} "
            f"{config.width}x{config.height}@{config.fps:g}"
        )

    def stop(self) -> None:
        self.stop_recording()
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.status_var.set("已停止")

    def start_recording(self) -> None:
        if self.recorder.recording:
            return
        packet = self.worker.latest() if self.worker else None
        if packet is None:
            messagebox.showwarning("开始录像", "请先开始采集并等待画面")
            return
        try:
            path = self.recorder.start(
                self.session_dir,
                packet.frame.shape,
                fps=30.0,
            )
        except Exception as error:
            messagebox.showerror("开始录像", str(error))
            return
        self.last_recorded_at = 0.0
        self.log_message(f"开始录像: {path.resolve()}")

    def stop_recording(self) -> None:
        if not self.recorder.recording:
            return
        path = self.recorder.stop()
        if path is not None:
            self.log_message(f"录像结束: {path.resolve()}")

    def toggle_freeze(self) -> None:
        if self.frozen_frame is None:
            packet = self.worker.latest() if self.worker else None
            if packet is not None:
                self.frozen_frame = packet.frame.copy()
                self.status_var.set("画面已冻结")
        else:
            self.frozen_frame = None
            self.status_var.set("画面已恢复")

    def annotate_current(self) -> None:
        if self.annotation_dialog is not None:
            self.annotation_dialog.lift()
            return
        packet = self.worker.latest() if self.worker else None
        frame = self.frozen_frame if self.frozen_frame is not None else (
            packet.frame if packet else None
        )
        if frame is None:
            messagebox.showwarning("标注截图", "当前没有可标注的画面")
            return
        image = Image.fromarray(frame[:, :, ::-1].copy())
        default_name = f"state_{self.annotation_sequence:03d}"
        self.annotation_dialog = AnnotationDialog(
            self.root,
            image,
            default_name,
            lambda name, description, attention, actions, next_state, shapes: (
                self._save_annotation(
                    frame,
                    name,
                    description,
                    attention,
                    actions,
                    next_state,
                    shapes,
                )
            ),
        )
        self.annotation_dialog.bind(
            "<Destroy>",
            lambda event: self._annotation_closed(event.widget),
        )

    def _annotation_closed(self, widget) -> None:
        if widget is self.annotation_dialog:
            self.annotation_dialog = None

    def _save_annotation(
        self,
        frame,
        name: str,
        description: str,
        attention: str,
        actions: str,
        next_state: str,
        shapes: list[AnnotationShape],
    ) -> None:
        capture_info = {}
        if self.worker is not None:
            config = self.worker.config
            capture_info = {
                "backend": config.backend,
                "device_index": config.device_index,
                "requested_width": config.width,
                "requested_height": config.height,
                "requested_fps": config.fps,
                "actual_width": int(frame.shape[1]),
                "actual_height": int(frame.shape[0]),
            }
        image = Image.fromarray(frame[:, :, ::-1].copy())
        metadata = save_annotation_bundle(
            image,
            self.session_dir,
            self.annotation_sequence,
            name,
            description,
            attention,
            actions,
            next_state,
            shapes,
            capture_info,
        )
        self.annotation_sequence += 1
        self.log_message(
            f"已保存标注: {self.session_dir.resolve() / metadata['annotated_image']}"
        )

    def _refresh(self) -> None:
        if self.worker is not None:
            error = self.worker.error
            if error:
                self.status_var.set(f"错误: {error}")
            packet = self.worker.latest()
            if packet is not None and packet.sequence != self.last_sequence:
                self.last_sequence = packet.sequence
                self.last_frame_time = packet.timestamp
                frame = self.frozen_frame if self.frozen_frame is not None else packet.frame
                if self.recorder.recording:
                    if packet.timestamp - self.last_recorded_at >= 1 / 30:
                        self.recorder.write(packet.frame)
                        self.last_recorded_at = packet.timestamp
                display_frame = frame
                if self.frozen_frame is None and self.state_machine_enabled.get():
                    self._analysis_counter += 1
                    analysis = self.vision_runtime.update(
                        packet.frame,
                        include_selection=self._analysis_counter % 5 == 0,
                    )
                    self._last_analysis = analysis
                    self._update_analysis_panel(analysis)
                elif not self.state_machine_enabled.get():
                    self._last_analysis = None
                    self._clear_analysis_panel()
                if (
                    self.overlay_enabled.get()
                    and self._last_analysis is not None
                ):
                    display_frame = self._render_analysis_overlay(
                        frame,
                        self._last_analysis,
                    )
                self._show_frame(display_frame)
                stale = time.perf_counter() - packet.timestamp
                self.status_var.set(
                    f"{packet.frame.shape[1]}x{packet.frame.shape[0]} "
                    f"采集 FPS={packet.measured_fps:.1f} 延迟={stale * 1000:.0f}ms"
                )
            elif (
                packet is not None
                and time.perf_counter() - packet.timestamp > 2
                and self.frozen_frame is None
                and not error
            ):
                self.status_var.set("超过 2 秒没有新帧")
        self.root.after(30, self._refresh)

    def _update_analysis_panel(
        self,
        analysis: BattleVisionObservation,
    ) -> None:
        observation = analysis.state
        if (
            self.vision_runtime.stage_id is not None
            and self.stage_var.get() != self.vision_runtime.stage_id
        ):
            self.stage_var.set(self.vision_runtime.stage_id)
        self.state_var.set(STATE_DISPLAY_NAMES[observation.state])
        self.confidence_var.set(f"置信度：{observation.confidence:.1%}")
        if analysis.menu is not None:
            wins = "/".join(
                str(value) if value is not None else "?"
                for value in analysis.menu.level_wins
            )
            text = (
                f"NPC：{analysis.menu.npc_name or '未知'} | "
                f"胜场 {wins}"
            )
            if analysis.menu.difficulty is not None:
                text += (
                    f" | 难度 {analysis.menu.difficulty}"
                    f" ({analysis.menu.difficulty_confidence:.0%})"
                )
            if analysis.menu.stage_name is not None:
                text += f" | 地图 {analysis.menu.stage_name}"
            elif analysis.menu.binding_consistent is False:
                text += " | 地图与NPC绑定冲突"
            self.menu_var.set(text)
        elif analysis.opponent_context is not None:
            context = analysis.opponent_context
            self.menu_var.set(
                f"对手：{context.npc_name} | 难度 {context.difficulty} "
                f"| 地图 {context.stage_name}"
            )
        else:
            self.menu_var.set("NPC信息：-")

        if analysis.deck is not None:
            deck = analysis.deck
            areas = ",".join(str(card.area) for card in deck.cards)
            costs = ",".join(str(card.sp_cost) for card in deck.cards)
            rows = []
            same_shape = []
            for row in range(1, 6):
                names = [
                    card.name
                    for card in deck.cards
                    if card.slot_row == row
                ]
                rows.append(f"第{row}行：" + "、".join(names))
            for card in deck.cards:
                if (
                    len(card.candidates) > 1
                    and card.candidates[0].score
                    - card.candidates[1].score
                    < 0.02
                ):
                    same_shape.append(
                        f"{card.slot_row},{card.slot_col} "
                        f"{card.candidates[0].name}/"
                        f"{card.candidates[1].name}"
                    )
            self.deck_var.set(
                f"牌组：{deck.deck_id} | "
                f"{'指定' if deck.source == 'configured' else '自动'} "
                f"| 总计{deck.total_area}点 "
                f"| SP{deck.total_sp_cost} | {deck.confidence:.0%}\n"
                + "\n".join(rows)
                + "\n"
                f"卡点：{areas}\n"
                f"SP：{costs}"
                + (
                    "\n同形候选：" + "；".join(same_shape)
                    if same_shape
                    else ""
                )
            )
        elif observation.state is GameState.DECK_SELECT:
            self.deck_var.set("牌组：等待15张卡识别")
        elif analysis.opponent_context is not None:
            self.deck_var.set(
                f"牌组：{analysis.opponent_context.deck_id}"
            )
        else:
            self.deck_var.set("牌组：-")

        board = analysis.board
        if board is not None:
            self_score = board.self_count
            opponent_score = board.opponent_count
            estimated = True
            if (
                analysis.score is not None
                and analysis.score.self_value is not None
                and analysis.score.opponent_value is not None
            ):
                self_score = analysis.score.self_value
                opponent_score = analysis.score.opponent_value
                estimated = False
            prefix = "~" if estimated else ""
            total = analysis.active_specials
            self_total = (
                total.self_special_count
                if total is not None
                else board.self_special_count
            )
            opponent_total = (
                total.opponent_special_count
                if total is not None
                else board.opponent_special_count
            )
            self_active = (
                total.self_count
                if total is not None
                else board.self_active_special_count
            )
            opponent_active = (
                total.opponent_count
                if total is not None
                else board.opponent_active_special_count
            )
            self.board_var.set(
                f"我方{prefix}{self_score} 特殊{self_total}："
                f"敌方{prefix}{opponent_score} 特殊{opponent_total}"
            )
            self.active_special_var.set(
                f"已激活特殊：我方{self_active} / 敌方{opponent_active}"
            )
        else:
            self.board_var.set("棋盘：当前阶段不读取")
            self.active_special_var.set("已激活特殊：-")

        if analysis.selection_profile is not None:
            self.selection_var.set(
                f"选牌几何：{analysis.selection_profile.variant}"
            )
        elif observation.state in CARD_SELECTION_STATES:
            self.selection_var.set("选牌几何：等待稳定识别")
        else:
            self.selection_var.set("选牌几何：-")

        if analysis.reveal is not None:
            self.reveal_var.set(
                "亮牌我方："
                + _revealed_card_text(analysis.reveal.own)
                + "\n亮牌敌方："
                + _revealed_card_text(analysis.reveal.opponent)
            )
        elif observation.state is GameState.TURN_REVEAL:
            self.reveal_var.set("亮牌：等待双方卡牌识别")
        else:
            self.reveal_var.set("亮牌：-")

        if analysis.match_result is not None:
            labels = {
                "WIN": "我方胜利",
                "LOSS": "我方失败",
                "DRAW": "待确认",
                "UNKNOWN": "无法判断",
            }
            self.result_var.set(
                f"比赛结果：{labels[analysis.match_result.outcome.value]} "
                f"({analysis.match_result.confidence:.1%})"
            )
        elif observation.state is GameState.MATCH_RESULT:
            self.result_var.set("比赛结果：等待稳定识别")
        else:
            self.result_var.set("比赛结果：-")

    def _clear_analysis_panel(self) -> None:
        self.state_var.set("状态：识别关闭")
        self.confidence_var.set("置信度：-")
        self.menu_var.set("NPC信息：-")
        self.deck_var.set("牌组：-")
        self.board_var.set("棋盘：-")
        self.active_special_var.set("已激活特殊：-")
        self.result_var.set("比赛结果：-")
        self.selection_var.set("选牌几何：-")
        self.reveal_var.set("亮牌：-")

    def _render_analysis_overlay(
        self,
        frame,
        analysis: BattleVisionObservation,
    ):
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        for x1, y1, x2, y2 in analysis.state.boxes:
            cv2.rectangle(
                canvas,
                (round(x1 * (width - 1)), round(y1 * (height - 1))),
                (round(x2 * (width - 1)), round(y2 * (height - 1))),
                (255, 212, 0),
                4,
            )
        for x1, y1, x2, y2 in analysis.state.focus_boxes:
            cv2.rectangle(
                canvas,
                (round(x1 * (width - 1)), round(y1 * (height - 1))),
                (round(x2 * (width - 1)), round(y2 * (height - 1))),
                (0, 204, 255),
                5,
            )
        canvas = self._draw_board_overlay(canvas, analysis)
        canvas = self._draw_selection_overlay(canvas, analysis)
        canvas = self._draw_result_overlay(canvas, analysis)
        return canvas

    def _draw_board_overlay(
        self,
        canvas,
        analysis: BattleVisionObservation,
    ):
        board = analysis.board
        region = self.vision_runtime.board_region
        if board is None or region is None:
            return canvas
        height, width = canvas.shape[:2]
        overlay = canvas.copy()
        active = analysis.active_specials
        active_self = active.self_cells if active is not None else frozenset()
        active_opponent = (
            active.opponent_cells if active is not None else frozenset()
        )
        colors = {
            Cell.SELF_INK: (0, 255, 120),
            Cell.SELF_SPECIAL: (0, 230, 255),
            Cell.OPPONENT_INK: (255, 80, 180),
            Cell.OPPONENT_SPECIAL: (255, 220, 0),
            Cell.WALL: (130, 130, 130),
        }
        readings = {
            (reading.row, reading.col): reading
            for reading in board.readings
        }
        for polygon, label in region_grid_cells(region, width, height):
            row, col = (int(value) for value in label.split(",", 1))
            cell = board.cells[row][col]
            points = polygon.astype("int32")
            fill = colors.get(cell)
            if fill is not None:
                cv2.fillPoly(overlay, [points], fill)
            reading = readings[(row, col)]
            if reading.edge_density >= 0.12:
                cv2.polylines(
                    canvas,
                    [points],
                    True,
                    (0, 255, 255),
                    2,
                )
            if cell is Cell.VOID:
                continue
            if (row, col) in active_self or (row, col) in active_opponent:
                cv2.polylines(
                    canvas,
                    [points],
                    True,
                    (255, 255, 255),
                    4,
                )
        canvas = cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0)
        return canvas

    def _draw_selection_overlay(
        self,
        canvas,
        analysis: BattleVisionObservation,
    ):
        match = analysis.selection_profile
        if match is None:
            return canvas
        semantics = self._selection_semantics_cache.get(match.profile_path)
        if semantics is None:
            semantics = ProfileSemantics.load(match.profile_path)
            self._selection_semantics_cache[match.profile_path] = semantics
        height, width = canvas.shape[:2]
        for item in semantics.regions:
            color = (
                (0, 255, 255)
                if item.role is RegionRole.HAND_GRID
                else (255, 212, 0)
                if item.role is RegionRole.HAND_CARD
                else None
            )
            if color is None:
                continue
            points = [
                (round(x * (width - 1)), round(y * (height - 1)))
                for x, y in item.region.points
            ]
            if item.region.kind == "rect":
                cv2.rectangle(canvas, points[0], points[1], color, 2)
            else:
                cv2.polylines(
                    canvas,
                    [np.int32(points)],
                    True,
                    color,
                    2,
                )
        return canvas

    def _draw_result_overlay(
        self,
        canvas,
        analysis: BattleVisionObservation,
    ):
        reader = self.vision_runtime.match_result_reader
        result = analysis.match_result
        if reader is None or result is None:
            return canvas
        height, width = canvas.shape[:2]
        for region, player_result in (
            (reader.opponent_region, result.opponent_result),
            (reader.self_region, result.self_result),
        ):
            x1, y1 = region.points[0]
            x2, y2 = region.points[1]
            color = (
                (80, 80, 255)
                if player_result.value == "WIN"
                else (120, 255, 80)
                if player_result.value == "LOSE"
                else (0, 212, 255)
            )
            cv2.rectangle(
                canvas,
                (round(x1 * (width - 1)), round(y1 * (height - 1))),
                (round(x2 * (width - 1)), round(y2 * (height - 1))),
                color,
                4,
            )
        return canvas

    def _show_frame(self, frame) -> None:
        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        image = ImageOps.contain(
            image,
            (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text="")

    def log_message(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self) -> None:
        if self.annotation_dialog is not None:
            self.annotation_dialog.destroy()
        self.stop_recording()
        self.stop()
        self.root.destroy()


def run_capture_debug() -> int:
    root = tk.Tk()
    CaptureDebugApp(root)
    root.mainloop()
    return 0


def _revealed_card_text(card) -> str:
    if card.name is None:
        if card.candidates:
            candidate = card.candidates[0]
            return (
                f"{candidate.name}? | {candidate.area}点 | "
                f"SP{candidate.sp_cost} | {candidate.score:.0%}"
            )
        return (
            f"未知 | 格子{card.detected_cells} | "
            f"SP{card.visible_cost_lights}"
        )
    return (
        f"{card.name} | {card.area}点 | SP{card.sp_cost} | "
        f"{card.confidence:.0%}"
    )

