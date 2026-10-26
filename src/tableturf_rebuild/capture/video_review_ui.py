from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from tableturf_rebuild.application.battle_vision import (
    BattleVisionObservation,
    BattleVisionRuntime,
    CARD_SELECTION_STATES,
)
from tableturf_rebuild.capture.annotation_ui import AnnotationDialog
from tableturf_rebuild.capture.annotations import AnnotationShape, save_annotation_bundle
from tableturf_rebuild.capture.video_source import OpenCvVideoSource
from tableturf_rebuild.application.deck_config import load_deck_config
from tableturf_rebuild.application.state_machine import (
    GameState,
    STATE_DISPLAY_NAMES,
    StateMachine,
    save_manual_state_template,
)
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.calibration import region_grid_cells
from tableturf_rebuild.vision.profile_semantics import (
    ProfileSemantics,
    RegionRole,
)


class VideoReviewApp:
    PREVIEW_WIDTH = 960
    PREVIEW_HEIGHT = 540

    def __init__(self, root: tk.Tk, initial_path: Path | None = None):
        self.root = root
        self.root.title("AutoTableTurf 视频视觉分析（比分/特殊点/胜负）")
        self.root.geometry("1360x820")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.source: OpenCvVideoSource | None = None
        self.current_frame = None
        self.current_index = 0
        self.playing = False
        self._after_id = None
        self._slider_updating = False
        self._photo: ImageTk.PhotoImage | None = None
        self._display_geometry = (0, 0, 0, 0)
        self.annotation_dialog: AnnotationDialog | None = None
        self.annotation_sequence = 1
        self.state_boxes: tuple[tuple[float, float, float, float], ...] = ()
        self.focus_boxes: tuple[tuple[float, float, float, float], ...] = ()
        self.vision_runtime = BattleVisionRuntime.default()
        self.state_machine = self.vision_runtime.state_machine
        self.stage_ids = tuple(sorted(self.vision_runtime.board_profile_names))
        self.stage_var = tk.StringVar(
            value=self.stage_ids[0] if self.stage_ids else ""
        )
        self.board_var = tk.StringVar(value="棋盘：未配置")
        self.preview_var = tk.StringVar(value="摆牌预览：-")
        self.active_special_var = tk.StringVar(value="已激活特殊：-")
        self.selection_var = tk.StringVar(value="选牌几何：-")
        self.menu_var = tk.StringVar(value="NPC信息：-")
        self.deck_var = tk.StringVar(value="牌组：-")
        self.reveal_var = tk.StringVar(value="亮牌：-")
        self.result_var = tk.StringVar(value="比赛结果：-")
        self.analysis_overlay_enabled = tk.BooleanVar(value=True)
        self._analysis_frame_counter = 0
        self._last_board = None
        self._last_selection = None
        self._selection_semantics_cache: dict[Path, ProfileSemantics] = {}
        if self.stage_ids:
            try:
                self.vision_runtime.configure_stage(self.stage_ids[0])
            except (KeyError, ValueError) as error:
                self.board_var.set(f"棋盘：配置失败 {error}")
        self._build()
        if initial_path is not None:
            self.open_video(initial_path)

    def _build(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="打开视频", command=self.choose_video).pack(
            side="left"
        )
        self.play_button = ttk.Button(
            toolbar, text="播放", command=self.toggle_play
        )
        self.play_button.pack(side="left", padx=4)
        ttk.Button(toolbar, text="上一帧", command=lambda: self.step(-1)).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="下一帧", command=lambda: self.step(1)).pack(
            side="left", padx=2
        )
        self.speed_var = tk.StringVar(value="1x")
        ttk.Combobox(
            toolbar,
            textvariable=self.speed_var,
            values=("0.25x", "0.5x", "1x", "2x", "4x"),
            state="readonly",
            width=6,
        ).pack(side="left", padx=6)
        ttk.Button(
            toolbar, text="标注当前帧", command=self.annotate_current
        ).pack(side="left", padx=6)
        self.state_machine_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar,
            text="状态机识别",
            variable=self.state_machine_enabled,
        ).pack(side="left", padx=4)
        ttk.Label(toolbar, text="场地").pack(side="left", padx=(6, 2))
        stage_box = ttk.Combobox(
            toolbar,
            textvariable=self.stage_var,
            values=self.stage_ids,
            state="readonly",
            width=14,
        )
        stage_box.pack(side="left")
        stage_box.bind("<<ComboboxSelected>>", self._stage_changed)
        ttk.Checkbutton(
            toolbar,
            text="分析覆盖",
            variable=self.analysis_overlay_enabled,
        ).pack(side="left", padx=4)
        ttk.Button(
            toolbar,
            text="加载牌组配置",
            command=self.load_deck_config,
        ).pack(side="left", padx=(6, 2))
        ttk.Button(
            toolbar,
            text="自动牌组",
            command=self.clear_deck_config,
        ).pack(side="left", padx=(0, 4))
        self.time_var = tk.StringVar(value="0.000 / 0.000 秒")
        ttk.Label(toolbar, textvariable=self.time_var).pack(side="right")

        content = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        content.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            content,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            background="black",
        )
        self.canvas.pack(side="left")

        side = ttk.LabelFrame(content, text="状态机观察")
        side.pack(side="right", fill="both", padx=(6, 0))
        side.configure(width=340)
        side.pack_propagate(False)
        self.state_var = tk.StringVar(value="尚未接入状态机")
        ttk.Label(
            side,
            textvariable=self.state_var,
            font=("Segoe UI", 12, "bold"),
            wraplength=280,
        ).pack(anchor="w", padx=8, pady=(10, 4))
        self.confidence_var = tk.StringVar(value="置信度：-")
        ttk.Label(side, textvariable=self.confidence_var).pack(
            anchor="w", padx=8
        )
        ttk.Label(
            side,
            textvariable=self.board_var,
            wraplength=310,
        ).pack(anchor="w", padx=8, pady=(8, 0))
        ttk.Label(
            side,
            textvariable=self.preview_var,
            wraplength=310,
        ).pack(anchor="w", padx=8)
        ttk.Label(
            side,
            textvariable=self.active_special_var,
            wraplength=310,
        ).pack(anchor="w", padx=8)
        ttk.Label(
            side,
            textvariable=self.selection_var,
            wraplength=310,
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(
            side,
            textvariable=self.menu_var,
            wraplength=310,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(
            side,
            textvariable=self.deck_var,
            wraplength=310,
            font=("Consolas", 9),
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(
            side,
            textvariable=self.reveal_var,
            wraplength=310,
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(
            side,
            textvariable=self.result_var,
            wraplength=310,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(side, text="可能操作").pack(anchor="w", padx=8, pady=(10, 2))
        self.actions_text = tk.Text(side, height=8, width=34, state="disabled")
        self.actions_text.pack(fill="x", padx=8)
        ttk.Label(side, text="观察说明").pack(anchor="w", padx=8, pady=(10, 2))
        self.notes_text = tk.Text(side, height=10, width=34, state="disabled")
        self.notes_text.pack(fill="x", padx=8)
        correction = ttk.LabelFrame(side, text="修正状态")
        correction.pack(fill="x", padx=8, pady=(10, 4))
        self.correction_var = tk.StringVar(
            value=STATE_DISPLAY_NAMES[GameState.NPC_ROSTER]
        )
        ttk.Combobox(
            correction,
            textvariable=self.correction_var,
            values=tuple(
                STATE_DISPLAY_NAMES[state]
                for state in GameState
                if state is not GameState.UNKNOWN
            ),
            state="readonly",
        ).pack(fill="x", padx=4, pady=4)
        ttk.Button(
            correction,
            text="以当前帧修正并保存模板",
            command=self.correct_state,
        ).pack(fill="x", padx=4, pady=(0, 4))

        self.slider = ttk.Scale(
            self.root,
            from_=0,
            to=1,
            orient="horizontal",
            command=self.seek_from_slider,
        )
        self.slider.pack(fill="x", padx=8, pady=(2, 8))

    def _stage_changed(self, event=None) -> None:
        del event
        try:
            self.vision_runtime.configure_stage(self.stage_var.get())
        except (KeyError, ValueError) as error:
            self.board_var.set(f"棋盘：配置失败 {error}")
            return
        self._last_board = None
        self.board_var.set(f"棋盘：{self.stage_var.get()} 等待分析")

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
        self.deck_var.set(f"牌组：已指定 {config.name}")

    def clear_deck_config(self) -> None:
        self.vision_runtime.clear_expected_deck()
        self.deck_var.set("牌组：恢复自动识别")

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="打开录制视频",
            filetypes=(
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*"),
            ),
        )
        if path:
            self.open_video(Path(path))

    def open_video(self, path: Path) -> None:
        self.stop_playback()
        if self.source is not None:
            self.source.close()
        source = OpenCvVideoSource(path)
        try:
            source.open()
        except Exception as error:
            messagebox.showerror("打开视频", str(error))
            return
        self.source = source
        self.vision_runtime.reset()
        self._analysis_frame_counter = 0
        self._last_board = None
        self._last_selection = None
        self.annotation_sequence = 1
        self.slider.configure(to=max(1, source.frame_count - 1))
        self.seek(0)
        self.root.title(
            f"AutoTableTurf 视频视觉分析（比分/特殊点/胜负） - {path.name}"
        )

    def toggle_play(self) -> None:
        if self.source is None:
            return
        if self.playing:
            self.stop_playback()
        else:
            self.playing = True
            self.play_button.configure(text="暂停")
            self._schedule_next()

    def stop_playback(self) -> None:
        self.playing = False
        self.play_button.configure(text="播放")
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _schedule_next(self) -> None:
        if not self.playing or self.source is None:
            return
        delay = max(1, round(1000 / (self.source.fps * self.speed())))
        self._after_id = self.root.after(delay, self._play_next)

    def _play_next(self) -> None:
        if not self.playing or self.source is None:
            return
        frame = self.source.read()
        if frame is None:
            self.stop_playback()
            return
        self.current_frame = frame
        self.current_index = max(0, self.source.position - 1)
        self._display_frame()
        self._schedule_next()

    def step(self, delta: int) -> None:
        self.stop_playback()
        self.seek(
            self.current_index + delta,
            reset_state=delta < 0 or abs(delta) > 1,
        )

    def seek(self, index: int, reset_state: bool = True) -> None:
        if self.source is None:
            return
        actual = self.source.seek(index)
        frame = self.source.read()
        if frame is None:
            return
        self.current_index = actual
        self.current_frame = frame
        if reset_state:
            self.vision_runtime.reset()
            self._analysis_frame_counter = 0
            self._last_board = None
            self._last_selection = None
        self._display_frame()

    def seek_from_slider(self, raw_value: str) -> None:
        if self.source is None or self._slider_updating:
            return
        self.stop_playback()
        self.seek(int(float(raw_value)))

    def speed(self) -> float:
        return float(self.speed_var.get().rstrip("x"))

    def _display_frame(self, classify: bool = True) -> None:
        if self.current_frame is None:
            return
        rgb = self.current_frame[:, :, ::-1].copy()
        image = Image.fromarray(rgb)
        image = ImageOps.contain(
            image,
            (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT),
            Image.Resampling.LANCZOS,
        )
        display_width, display_height = image.size
        offset_x = (self.PREVIEW_WIDTH - display_width) // 2
        offset_y = (self.PREVIEW_HEIGHT - display_height) // 2
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(
            offset_x, offset_y, anchor="nw", image=self._photo
        )
        self._display_geometry = (
            offset_x,
            offset_y,
            display_width,
            display_height,
        )
        analysis: BattleVisionObservation | None = None
        observation = None
        if classify and self.state_machine_enabled.get():
            self._analysis_frame_counter += 1
            include_selection = (
                self._analysis_frame_counter % 5 == 0
                or (
                    self.state_machine.current
                    in CARD_SELECTION_STATES
                    and self._last_selection is None
                )
            )
            analysis = self.vision_runtime.update(
                self.current_frame,
                include_selection=include_selection,
            )
            if (
                self.vision_runtime.stage_id is not None
                and self.stage_var.get() != self.vision_runtime.stage_id
            ):
                self.stage_var.set(self.vision_runtime.stage_id)
            observation = analysis.state
            self._last_board = analysis.board
            if observation.state in CARD_SELECTION_STATES:
                if include_selection:
                    self._last_selection = analysis.selection_profile
            else:
                self._last_selection = None
            self.state_var.set(STATE_DISPLAY_NAMES[observation.state])
            self.correction_var.set(STATE_DISPLAY_NAMES[observation.state])
            self.confidence_var.set(f"置信度：{observation.confidence:.1%}")
            self._set_readonly_text(
                self.actions_text,
                "\n".join(observation.actions) or "无可执行操作，等待画面稳定",
            )
            alternatives = ", ".join(
                f"{STATE_DISPLAY_NAMES[candidate.state]}={candidate.score:.2f}"
                for candidate in observation.candidates[1:4]
            )
            profile_evidence = ", ".join(
                f"{candidate.state}={candidate.score:.2f}"
                for candidate in observation.profile_candidates[:3]
            )
            detail_lines = [
                f"候选：{alternatives}" if alternatives else "无其他候选"
            ]
            if profile_evidence:
                detail_lines.append(f"ROI证据：{profile_evidence}")
            if analysis.board is not None:
                board = analysis.board
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
                    detail_lines.append(
                        f"HUD比分置信度：{analysis.score.confidence:.1%}"
                    )
                prefix = "~" if estimated else ""
                active_specials = analysis.active_specials
                self_active_specials = (
                    active_specials.self_count
                    if active_specials is not None
                    else board.self_active_special_count
                )
                self_total_specials = (
                    active_specials.self_special_count
                    if active_specials is not None
                    else board.self_special_count
                )
                opponent_total_specials = (
                    active_specials.opponent_special_count
                    if active_specials is not None
                    else board.opponent_special_count
                )
                opponent_active_specials = (
                    active_specials.opponent_count
                    if active_specials is not None
                    else board.opponent_active_special_count
                )
                self.board_var.set(
                    f"我方{prefix}{self_score} 特殊"
                    f"{self_total_specials}："
                    f"敌方{prefix}{opponent_score} 特殊"
                    f"{opponent_total_specials}"
                )
                self.active_special_var.set(
                    f"已激活特殊：我方{self_active_specials} / "
                    f"敌方{opponent_active_specials}"
                )
                self.preview_var.set(
                    f"{board.stage_id} | 摆牌预览："
                    f"{'有' if board.placement_preview_present else '无'}"
                    f" | 分数 {board.placement_preview_score:.3f}"
                )
                detail_lines.append(
                    f"棋盘置信度：{board.confidence:.1%}"
                )
            elif observation.state in {
                GameState.REDRAW,
                GameState.TURN_SELECT,
                GameState.TARGET_PLACEMENT,
                GameState.SPECIAL_PLACEMENT,
                GameState.TURN_REVEAL,
                GameState.TURN_RESOLVE,
            }:
                self.board_var.set("棋盘：当前场地未配置或不适用")
                self.preview_var.set("摆牌预览：-")
                self.active_special_var.set("已激活特殊：-")
            else:
                self.board_var.set("棋盘：当前阶段不读取")
                self.preview_var.set("摆牌预览：-")
                self.active_special_var.set("已激活特殊：-")
            if observation.state in CARD_SELECTION_STATES:
                if self._last_selection is not None:
                    self.selection_var.set(
                        f"选牌几何：{self._last_selection.variant} "
                        f"(匹配分 {self._last_selection.score:.2f}, "
                        f"亮度差 {self._last_selection.brightness_margin:+.1f})"
                    )
                elif analysis.selection_candidates:
                    first = analysis.selection_candidates[0]
                    self.selection_var.set(
                        f"选牌几何：不确定（{first.variant} "
                        f"匹配分 {first.score:.2f}）"
                    )
                    detail_lines.append(
                        "选牌Profile候选："
                        + ", ".join(
                            f"{item.variant}={item.score:.2f}"
                            for item in analysis.selection_candidates[:3]
                        )
                    )
                else:
                    self.selection_var.set("选牌几何：等待分析")
            else:
                self.selection_var.set("选牌几何：-")
            if analysis.menu is not None:
                wins = "/".join(
                    str(value) if value is not None else "?"
                    for value in analysis.menu.level_wins
                )
                menu_text = (
                    f"NPC：{analysis.menu.npc_name or '未知'} | "
                    f"胜场 {wins}"
                )
                if analysis.menu.difficulty is not None:
                    menu_text += (
                        f" | 难度 {analysis.menu.difficulty}"
                        f" ({analysis.menu.difficulty_confidence:.0%})"
                    )
                if analysis.menu.stage_name is not None:
                    menu_text += f" | 地图 {analysis.menu.stage_name}"
                elif analysis.menu.binding_consistent is False:
                    menu_text += " | 地图与NPC绑定冲突"
                self.menu_var.set(menu_text)
                detail_lines.append(
                    f"NPC识别置信度：名称 "
                    f"{analysis.menu.name_confidence:.1%} / "
                    f"胜场 {analysis.menu.wins_confidence:.1%} / "
                    f"难度 {analysis.menu.difficulty_confidence:.1%} / "
                    f"地图 {analysis.menu.stage_confidence:.1%}"
                )
            elif observation.state in {
                GameState.NPC_ROSTER,
                GameState.MATCH_SETUP,
            }:
                self.menu_var.set("NPC信息：等待稳定识别")
            elif analysis.opponent_context is not None:
                context = analysis.opponent_context
                self.menu_var.set(
                    f"对手上下文：{context.npc_name} | "
                    f"难度 {context.difficulty} | "
                    f"地图 {context.stage_name}"
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
                    f"卡点(1,1→5,3)：{areas}\n"
                    f"SP成本：{costs}"
                    + (
                        "\n同形候选：" + "；".join(same_shape)
                        if same_shape
                        else ""
                    )
                )
                detail_lines.append(
                    f"牌组置信度：{deck.confidence:.1%}"
                )
            elif observation.state is GameState.DECK_SELECT:
                self.deck_var.set("牌组：等待15张卡识别")
            elif analysis.opponent_context is not None:
                self.deck_var.set(
                    f"牌组：{analysis.opponent_context.deck_id}"
                )
            else:
                self.deck_var.set("牌组：-")
            if analysis.reveal is not None:
                self.reveal_var.set(
                    "亮牌我方："
                    + _revealed_card_text(analysis.reveal.own)
                    + "\n亮牌敌方："
                    + _revealed_card_text(analysis.reveal.opponent)
                )
                detail_lines.append(self.reveal_var.get())
            elif observation.state is GameState.TURN_REVEAL:
                self.reveal_var.set("亮牌：等待双方卡牌识别")
            else:
                self.reveal_var.set("亮牌：-")
            if analysis.match_result is not None:
                result = analysis.match_result
                outcome_label = {
                    "WIN": "我方胜利",
                    "LOSS": "我方失败",
                    "DRAW": "同结果/待确认",
                    "UNKNOWN": "无法判断",
                }[result.outcome.value]
                self.result_var.set(
                    f"比赛结果：{outcome_label}"
                    f" | 对方 {result.opponent_result.value}"
                    f" / 我方 {result.self_result.value}"
                    f" ({result.confidence:.1%})"
                )
                detail_lines.append(f"结果识别：{self.result_var.get()}")
            elif observation.state is GameState.MATCH_RESULT:
                self.result_var.set("比赛结果：等待稳定识别")
            else:
                self.result_var.set("比赛结果：-")
            self._set_readonly_text(
                self.notes_text,
                "\n".join(detail_lines),
            )
            self.state_boxes = observation.boxes
            self.focus_boxes = observation.focus_boxes
        elif not self.state_machine_enabled.get():
            self.state_var.set("状态机识别已关闭")
            self.confidence_var.set("置信度：-")
            self.board_var.set("棋盘：未分析")
            self.preview_var.set("摆牌预览：-")
            self.active_special_var.set("已激活特殊：-")
            self.selection_var.set("选牌几何：-")
            self.menu_var.set("NPC信息：-")
            self.deck_var.set("牌组：-")
            self.reveal_var.set("亮牌：-")
            self.result_var.set("比赛结果：-")
            self.state_boxes = ()
            self.focus_boxes = ()
            self._last_board = None
            self._last_selection = None
        for x1, y1, x2, y2 in self.state_boxes:
            self.canvas.create_rectangle(
                offset_x + x1 * display_width,
                offset_y + y1 * display_height,
                offset_x + x2 * display_width,
                offset_y + y2 * display_height,
                outline="#00d4ff",
                width=3,
                tags="state-overlay",
            )
        for x1, y1, x2, y2 in self.focus_boxes:
            self.canvas.create_rectangle(
                offset_x + x1 * display_width,
                offset_y + y1 * display_height,
                offset_x + x2 * display_width,
                offset_y + y2 * display_height,
                outline="#ffcc00",
                width=4,
                tags="focus-overlay",
            )
        if analysis is not None and self.analysis_overlay_enabled.get():
            self._draw_board_overlay(
                analysis,
                offset_x,
                offset_y,
                display_width,
                display_height,
            )
            self._draw_selection_overlay(
                offset_x,
                offset_y,
                display_width,
                display_height,
            )
            self._draw_match_result_overlay(
                analysis,
                offset_x,
                offset_y,
                display_width,
                display_height,
            )
        if self.source is not None:
            seconds = self.current_index / self.source.fps
            duration = self.source.frame_count / self.source.fps
            self.time_var.set(f"{seconds:.3f} / {duration:.3f} 秒")
            self._slider_updating = True
            self.slider.set(self.current_index)
            self._slider_updating = False

    def _draw_board_overlay(
        self,
        analysis: BattleVisionObservation,
        offset_x: int,
        offset_y: int,
        display_width: int,
        display_height: int,
    ) -> None:
        if (
            analysis.board is None
            or self.current_frame is None
            or self.vision_runtime.board_region is None
        ):
            return
        frame_height, frame_width = self.current_frame.shape[:2]
        scale_x = display_width / max(1, frame_width - 1)
        scale_y = display_height / max(1, frame_height - 1)
        reading_map = {
            (reading.row, reading.col): reading
            for reading in analysis.board.readings
        }
        fill_colors = {
            Cell.SELF_INK: "#a8ff00",
            Cell.SELF_SPECIAL: "#ffe600",
            Cell.OPPONENT_INK: "#8060ff",
            Cell.OPPONENT_SPECIAL: "#00e5ff",
            Cell.WALL: "#777777",
        }
        for polygon, label in region_grid_cells(
            self.vision_runtime.board_region,
            frame_width,
            frame_height,
        ):
            row, col = (int(value) for value in label.split(",", 1))
            cell = analysis.board.cells[row][col]
            reading = reading_map[(row, col)]
            points = [
                (
                    offset_x + float(x) * scale_x,
                    offset_y + float(y) * scale_y,
                )
                for x, y in polygon
            ]
            fill = fill_colors.get(cell, "")
            outline = "#ffcc00" if reading.edge_density >= 0.12 else "#4a4a4a"
            width = 2 if reading.edge_density >= 0.12 else 1
            active_cells = frozenset()
            if analysis.active_specials is not None:
                if cell is Cell.OPPONENT_SPECIAL:
                    active_cells = analysis.active_specials.opponent_cells
                elif cell is Cell.SELF_SPECIAL:
                    active_cells = analysis.active_specials.self_cells
            if (row, col) in active_cells:
                outline = "#ffffff"
                width = 3
            if cell is Cell.VOID:
                outline = "#222222"
            options = {
                "fill": fill,
                "outline": outline,
                "width": width,
            }
            if fill:
                options["stipple"] = "gray25"
            self.canvas.create_polygon(points, **options)
        self.canvas.create_text(
            offset_x + display_width - 8,
            offset_y + 8,
            anchor="ne",
            text=(
                f"A {analysis.board.self_count}  "
                f"B {analysis.board.opponent_count}  "
                f"预览 {analysis.board.placement_preview_score:.2f}"
            ),
            fill="#ffffff",
            font=("Segoe UI", 11, "bold"),
        )

    def _draw_selection_overlay(
        self,
        offset_x: int,
        offset_y: int,
        display_width: int,
        display_height: int,
    ) -> None:
        if self._last_selection is None:
            return
        path = self._last_selection.profile_path
        semantics = self._selection_semantics_cache.get(path)
        if semantics is None:
            try:
                semantics = ProfileSemantics.load(path)
            except (OSError, ValueError):
                return
            self._selection_semantics_cache[path] = semantics
        roles = {
            RegionRole.HAND_CARD: "#00d4ff",
            RegionRole.HAND_GRID: "#ffcc00",
        }
        for item in semantics.regions:
            color = roles.get(item.role)
            if color is None:
                continue
            points = [
                (
                    offset_x + x * display_width,
                    offset_y + y * display_height,
                )
                for x, y in item.region.points
            ]
            if item.region.kind == "rect":
                self.canvas.create_rectangle(
                    *points[0],
                    *points[1],
                    outline=color,
                    width=2,
                )
            else:
                self.canvas.create_polygon(
                    points,
                    fill="",
                    outline=color,
                    width=2,
                )

    def _draw_match_result_overlay(
        self,
        analysis: BattleVisionObservation,
        offset_x: int,
        offset_y: int,
        display_width: int,
        display_height: int,
    ) -> None:
        reader = self.vision_runtime.match_result_reader
        result = analysis.match_result
        if reader is None or result is None:
            return
        values = (
            (
                reader.opponent_region,
                f"对方 {result.opponent_result.value}",
                result.opponent_result.value,
            ),
            (
                reader.self_region,
                f"我方 {result.self_result.value}",
                result.self_result.value,
            ),
        )
        for region, label, state in values:
            x1, y1 = region.points[0]
            x2, y2 = region.points[1]
            left = offset_x + min(x1, x2) * display_width
            right = offset_x + max(x1, x2) * display_width
            top = offset_y + min(y1, y2) * display_height
            bottom = offset_y + max(y1, y2) * display_height
            color = "#ff4d4f" if state == "WIN" else "#54ff9f"
            if state == "UNKNOWN":
                color = "#ffcc00"
            self.canvas.create_rectangle(
                left,
                top,
                right,
                bottom,
                outline=color,
                width=3,
            )
            self.canvas.create_text(
                left,
                top - 2,
                anchor="sw",
                text=label,
                fill=color,
                font=("Segoe UI", 10, "bold"),
            )

    def correct_state(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("修正状态", "当前没有视频帧")
            return
        try:
            state = GameState(self.correction_var.get().split("|", 1)[0].strip())
            path = save_manual_state_template(self.current_frame, state)
        except Exception as error:
            messagebox.showerror("修正状态", str(error))
            return
        self.stop_playback()
        self.state_machine.recognizer.reload()
        self.state_machine.force_state(state)
        self.set_state_observation(
            STATE_DISPLAY_NAMES[state],
            1.0,
            self.actions_for_state(state),
            f"已追加修正模板：{path}",
            self.boxes_for_state(state),
        )

    @staticmethod
    def actions_for_state(state: GameState) -> tuple[str, ...]:
        from tableturf_rebuild.application.state_machine import STATE_ACTIONS

        return STATE_ACTIONS.get(state, ())

    @staticmethod
    def boxes_for_state(state: GameState) -> tuple[tuple[float, float, float, float], ...]:
        from tableturf_rebuild.application.state_machine import STATE_BOXES

        return STATE_BOXES.get(state, ())

    def set_state_observation(
        self,
        state_name: str,
        confidence: float,
        actions: tuple[str, ...] | list[str],
        notes: str = "",
        boxes: tuple[tuple[float, float, float, float], ...] = (),
        focus_boxes: tuple[tuple[float, float, float, float], ...] = (),
    ) -> None:
        self.state_var.set(state_name)
        self.confidence_var.set(f"置信度：{confidence:.1%}")
        self._set_readonly_text(self.actions_text, "\n".join(actions))
        self._set_readonly_text(self.notes_text, notes)
        self.state_boxes = tuple(boxes)
        self.focus_boxes = tuple(focus_boxes)
        self._display_frame(classify=False)

    @staticmethod
    def _set_readonly_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def annotate_current(self) -> None:
        if self.current_frame is None or self.source is None:
            messagebox.showwarning("标注", "请先打开视频并定位到一帧")
            return
        if self.annotation_dialog is not None:
            self.annotation_dialog.lift()
            return
        self.stop_playback()
        image = Image.fromarray(self.current_frame[:, :, ::-1].copy())
        self.annotation_dialog = AnnotationDialog(
            self.root,
            image,
            f"video_state_{self.annotation_sequence:03d}",
            self._save_annotation,
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
        name: str,
        description: str,
        attention: str,
        actions: str,
        next_state: str,
        shapes: list[AnnotationShape],
    ) -> None:
        if self.source is None or self.current_frame is None:
            raise RuntimeError("没有可保存的视频帧")
        output_dir = (
            self.source.path.parent / f"{self.source.path.stem}_annotations"
        )
        metadata = save_annotation_bundle(
            Image.fromarray(self.current_frame[:, :, ::-1].copy()),
            output_dir,
            self.annotation_sequence,
            name,
            description,
            attention,
            actions,
            next_state,
            shapes,
            {
                "source_video": str(self.source.path.resolve()),
                "frame_index": self.current_index,
                "timestamp_seconds": self.current_index / self.source.fps,
                "width": self.source.width,
                "height": self.source.height,
                "fps": self.source.fps,
            },
        )
        self.annotation_sequence += 1
        self.state_var.set(f"已保存: {name}")
        self.notes_text.configure(state="normal")
        self.notes_text.delete("1.0", "end")
        self.notes_text.insert("1.0", str(output_dir / metadata["annotated_image"]))
        self.notes_text.configure(state="disabled")

    def close(self) -> None:
        self.stop_playback()
        if self.annotation_dialog is not None:
            self.annotation_dialog.destroy()
        if self.source is not None:
            self.source.close()
            self.source = None
        self.root.destroy()


def run_video_review(initial_path: Path | None = None) -> int:
    root = tk.Tk()
    app = VideoReviewApp(root, initial_path)
    del app
    root.mainloop()
    return 0


def run_default_video_review() -> int:
    project_root = Path(__file__).resolve().parents[3]
    video = (
        project_root
        / "artifacts"
        / "state_capture"
        / "run-20261021-221749"
        / "session-20261021-222704.mp4"
    )
    return run_video_review(video if video.exists() else None)


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

