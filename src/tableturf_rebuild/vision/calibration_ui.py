from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageTk

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.calibration import (
    CellMaskOverride,
    RegionHandle,
    VisionProfile,
    VisionRegion,
    drag_resize_region,
    region_pixel_bounds,
    region_pixel_size,
    region_grid_cells,
    resize_region_to_pixels,
    translate_region_pixels,
)


class CalibrationApp:
    def __init__(
        self,
        root: tk.Tk,
        initial_image: Path | None = None,
        profile_path: Path | None = None,
    ):
        self.root = root
        self.root.title("AutoTableTurf 高精度视觉标定")
        self.root.geometry("1440x900")
        self.image_path: Path | None = None
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self._display_key: tuple[Path | None, int, int, float] | None = None
        self._display_size = (1, 1)
        self.scale = 1.0
        self.pending: list[tuple[float, float]] = []
        self.regions: list[VisionRegion] = []
        self.catalog = GameCatalog.from_package()
        self.cell_overrides: dict[tuple[int, int], int] = {}
        self.selected_index: int | None = None
        self.drawing_new = False
        self.drag: dict[str, object] | None = None
        self._updating_list = False
        self._list_signature: tuple[str, ...] | None = None
        self.mode_var = tk.StringVar(value="rect")
        self.name_var = tk.StringVar(value="board")
        self.rows_var = tk.IntVar(value=0)
        self.cols_var = tk.IntVar(value=0)
        self.stage_options = tuple(
            f"{stage.stage_id}|{stage.name_zh}"
            for stage in self.catalog.stages.values()
        )
        self.stage_var = tk.StringVar(value="不套用")
        self.mask_edit_var = tk.BooleanVar(value=False)
        self.mask_cell_types = {
            "循环": None,
            "可玩": Cell.EMPTY,
            "棋盘外 (VOID)": Cell.VOID,
            "灰色不可放置 (WALL)": Cell.WALL,
            "我方初始墨": Cell.SELF_INK,
            "我方特殊点": Cell.SELF_SPECIAL,
            "敌方初始墨": Cell.OPPONENT_INK,
            "敌方特殊点": Cell.OPPONENT_SPECIAL,
        }
        self.mask_type_var = tk.StringVar(value="循环")
        self.size_width_var = tk.StringVar(value="")
        self.size_height_var = tk.StringVar(value="")
        self.profile_path_var = tk.StringVar(
            value=str(
                profile_path
                or Path("profiles/vision/splat3-final-1920x1080.json")
            )
        )
        self.coord_var = tk.StringVar(value="x=- y=-")
        self.status_var = tk.StringVar(
            value="画框时右键撤一个点；拖框内移动；拖控制点改尺寸"
        )
        self.profile_path_var.trace_add("write", self._profile_path_changed)
        self._build()
        self._update_window_title()
        if initial_image is not None:
            self.open_image(initial_image)
            self.load_profile(silent=True)

    def _build(self) -> None:
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="打开截图", command=self.choose_image).pack(
            side="left"
        )
        ttk.Button(toolbar, text="放大", command=lambda: self.zoom(1.25)).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="缩小", command=lambda: self.zoom(0.8)).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="适应窗口", command=self.fit_image).pack(
            side="left", padx=3
        )
        ttk.Label(toolbar, textvariable=self.coord_var).pack(side="right")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            padding=(8, 0, 8, 5),
        ).pack(fill="x")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        canvas_frame = ttk.Frame(body)
        canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, background="#202020")
        xscroll = ttk.Scrollbar(
            canvas_frame, orient="horizontal", command=self.canvas.xview
        )
        yscroll = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(
            xscrollcommand=xscroll.set,
            yscrollcommand=yscroll.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self._canvas_click)
        self.canvas.bind("<ButtonPress-3>", self._canvas_right_click)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Motion>", self._mouse_move)

        side = ttk.LabelFrame(body, text="区域")
        side.pack(side="right", fill="y", padx=6, pady=6)
        side.configure(width=390)
        side.pack_propagate(False)

        ttk.Label(side, text="区域名称").pack(anchor="w", padx=6)
        ttk.Entry(side, textvariable=self.name_var).pack(
            fill="x", padx=6, pady=3
        )
        ttk.Label(side, text="工具").pack(anchor="w", padx=6)
        tools = ttk.Frame(side)
        tools.pack(fill="x", padx=6)
        ttk.Radiobutton(
            tools, text="矩形（2点）", value="rect", variable=self.mode_var
        ).pack(anchor="w")
        ttk.Radiobutton(
            tools,
            text="透视四边形（4点，顺序：左上右上右下左下）",
            value="quad",
            variable=self.mode_var,
        ).pack(anchor="w")
        ttk.Button(
            tools,
            text="开始绘制新区域",
            command=self.begin_new_region,
        ).pack(anchor="w", pady=(4, 0))
        grid = ttk.Frame(side)
        grid.pack(fill="x", padx=6, pady=5)
        ttk.Label(grid, text="网格行").pack(side="left")
        ttk.Spinbox(grid, from_=0, to=40, textvariable=self.rows_var, width=5).pack(
            side="left", padx=3
        )
        ttk.Label(grid, text="列").pack(side="left")
        ttk.Spinbox(grid, from_=0, to=40, textvariable=self.cols_var, width=5).pack(
            side="left", padx=3
        )
        ttk.Button(
            grid,
            text="更新名称/网格",
            command=self.apply_selected_properties,
        ).pack(side="left", padx=(8, 0))
        stage_row = ttk.Frame(side)
        stage_row.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(stage_row, text="关卡掩码").pack(side="left")
        stage_box = ttk.Combobox(
            stage_row,
            textvariable=self.stage_var,
            values=("不套用",) + self.stage_options,
            state="readonly",
            width=22,
        )
        stage_box.pack(side="left", fill="x", expand=True, padx=4)
        stage_box.bind("<<ComboboxSelected>>", lambda event: self.redraw())
        ttk.Button(
            stage_row,
            text="套用尺寸",
            command=self.apply_stage_size,
        ).pack(side="left")
        mask_actions = ttk.Frame(side)
        mask_actions.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Checkbutton(
            mask_actions,
            text="点击格子编辑掩码（右键反向）",
            variable=self.mask_edit_var,
        ).pack(side="left")
        ttk.Button(
            mask_actions,
            text="清空覆盖",
            command=self.clear_cell_overrides,
        ).pack(side="right")
        mask_type_row = ttk.Frame(side)
        mask_type_row.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(mask_type_row, text="格类型").pack(side="left")
        ttk.Combobox(
            mask_type_row,
            textvariable=self.mask_type_var,
            values=tuple(self.mask_cell_types),
            state="readonly",
            width=22,
        ).pack(side="left", fill="x", expand=True, padx=4)

        size = ttk.LabelFrame(side, text="选中区域尺寸（像素，可从中心缩放）")
        size.pack(fill="x", padx=6, pady=4)
        ttk.Label(size, text="拖框内移动；拖角点改形状；拖边中点改宽高").grid(
            row=0,
            column=0,
            columnspan=6,
            sticky="w",
            padx=5,
            pady=(2, 4),
        )
        ttk.Label(size, text="宽").grid(row=1, column=0, padx=(5, 2))
        ttk.Spinbox(
            size,
            from_=1,
            to=5000,
            textvariable=self.size_width_var,
            width=7,
        ).grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(size, text="高").grid(row=2, column=0, padx=(5, 2))
        ttk.Spinbox(
            size,
            from_=1,
            to=5000,
            textvariable=self.size_height_var,
            width=7,
        ).grid(row=2, column=1, padx=2, pady=2)
        for column, delta in enumerate((-10, -1, 1, 10), start=2):
            ttk.Button(
                size,
                text=f"{delta:+d}",
                width=4,
                command=lambda value=delta: self.nudge_selected_size(
                    delta_width=value
                ),
            ).grid(row=1, column=column, padx=1, pady=2)
            ttk.Button(
                size,
                text=f"{delta:+d}",
                width=4,
                command=lambda value=delta: self.nudge_selected_size(
                    delta_height=value
                ),
            ).grid(row=2, column=column, padx=1, pady=2)
        ttk.Button(
            size,
            text="应用宽高",
            command=self.apply_selected_size,
        ).grid(row=3, column=0, columnspan=6, sticky="ew", padx=5, pady=(3, 5))
        buttons = ttk.Frame(side)
        buttons.pack(fill="x", padx=6, pady=4)
        for column in range(3):
            buttons.grid_columnconfigure(column, weight=1)
        ttk.Button(
            buttons, text="添加区域", command=self.add_region
        ).grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=2)
        ttk.Button(
            buttons, text="清除当前绘制", command=self.clear_pending
        ).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(2, 0), pady=2)
        ttk.Button(
            buttons, text="复制选中", command=self.copy_selected
        ).grid(row=1, column=0, sticky="ew", padx=(0, 2), pady=2)
        ttk.Button(
            buttons, text="重命名选中", command=self.rename_selected
        ).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(
            buttons, text="删除选中", command=self.delete_selected
        ).grid(row=1, column=2, sticky="ew", padx=(2, 0), pady=2)

        self.region_list = tk.Listbox(side, exportselection=False)
        self.region_list.pack(fill="both", expand=True, padx=6, pady=5)
        self.region_list.bind("<<ListboxSelect>>", self._on_region_selected)
        self.region_list.bind("<Double-Button-1>", self.rename_selected)
        self.root.bind("<F2>", self.rename_selected)

        ttk.Label(side, text="Profile 路径").pack(anchor="w", padx=6)
        profile_entry = ttk.Frame(side)
        profile_entry.pack(fill="x", padx=6, pady=3)
        ttk.Entry(profile_entry, textvariable=self.profile_path_var).pack(
            side="left",
            fill="x",
            expand=True,
        )
        ttk.Button(
            profile_entry,
            text="选择现有",
            width=8,
            command=self.choose_existing_profile,
        ).pack(side="left", padx=(4, 0))
        profile_actions = ttk.Frame(side)
        profile_actions.pack(fill="x", padx=6, pady=(0, 8))
        profile_actions.grid_columnconfigure(0, weight=1)
        profile_actions.grid_columnconfigure(1, weight=1)
        ttk.Button(
            profile_actions,
            text="另存为新路径",
            command=self.choose_new_profile_path,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=2)
        ttk.Button(
            profile_actions,
            text="加载 Profile",
            command=self.load_profile,
        ).grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=2)
        ttk.Button(
            profile_actions,
            text="保存 Profile",
            command=self.save_profile,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=(("图片", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*"))
        )
        if path:
            self.open_image(Path(path))

    def choose_existing_profile(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择现有视觉 Profile",
            initialdir=self._profile_dialog_directory(),
            filetypes=(("视觉 Profile", "*.json"), ("所有文件", "*.*")),
        )
        if path:
            self.profile_path_var.set(path)
            self.status_var.set("已选择 Profile 路径；点击“加载 Profile”读取")

    def choose_new_profile_path(self) -> None:
        current = Path(self.profile_path_var.get().strip())
        filename = current.name if current.suffix.lower() == ".json" else "vision.json"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="选择新 Profile 保存路径",
            initialdir=self._profile_dialog_directory(),
            initialfile=filename,
            defaultextension=".json",
            filetypes=(("视觉 Profile", "*.json"), ("所有文件", "*.*")),
        )
        if path:
            self.profile_path_var.set(path)
            self.status_var.set("已选择新 Profile 路径；标定后点击“保存 Profile”")

    def _profile_dialog_directory(self) -> str:
        current = Path(self.profile_path_var.get().strip())
        directory = current.parent
        if not current.is_absolute() or str(directory) in {"", "."}:
            directory = Path.cwd() / directory
        if not directory.exists():
            directory = Path.cwd() / "profiles" / "vision"
        return str(directory)

    def open_image(self, path: Path) -> None:
        image = Image.open(path).convert("RGB")
        self.image_path = path
        self.image = image
        self._display_key = None
        self.pending.clear()
        self.drawing_new = False
        self.drag = None
        self._update_window_title()
        self.fit_image()

    def _profile_path_changed(self, *args) -> None:
        del args
        self._update_window_title()

    def _update_window_title(self) -> None:
        profile_name = Path(self.profile_path_var.get().strip()).stem or "未命名"
        image_name = self.image_path.name if self.image_path is not None else "未打开截图"
        self.root.title(
            f"AutoTableTurf 高精度视觉标定 | {profile_name} | {image_name}"
        )

    def fit_image(self) -> None:
        if self.image is None:
            return
        self.root.update_idletasks()
        available_width = max(400, self.canvas.winfo_width() - 20)
        available_height = max(300, self.canvas.winfo_height() - 20)
        self.scale = min(
            available_width / self.image.width,
            available_height / self.image.height,
            1.0,
        )
        self.redraw()

    def zoom(self, factor: float) -> None:
        if self.image is None:
            return
        self.scale = max(0.1, min(5.0, self.scale * factor))
        self.redraw()

    def _canvas_click(self, event) -> None:
        if self.image is None:
            return
        display_x = self.canvas.canvasx(event.x)
        display_y = self.canvas.canvasy(event.y)
        if not (
            0 <= display_x <= self._display_size[0]
            and 0 <= display_y <= self._display_size[1]
        ):
            return
        point = self._event_position(event)
        if self.mask_edit_var.get() and not self.pending:
            if self.selected_index is None:
                region_index = self._hit_region(point)
                if region_index is not None:
                    self._select_region(region_index)
            if self._cycle_cell_mask(point):
                return
        if self.selected_index is not None:
            handle = self._hit_selected_handle(event)
            if handle is not None:
                self.drag = {
                    "action": "resize",
                    "handle": handle,
                    "index": self.selected_index,
                    "region": self.regions[self.selected_index],
                    "start": point,
                }
                self.canvas.configure(cursor="sizing")
                self.status_var.set("正在调整区域尺寸；松开鼠标完成")
                return
        if not self.pending and not self.drawing_new:
            region_index = self._hit_region(point)
            if region_index is not None:
                self._select_region(region_index)
                self.drag = {
                    "action": "move",
                    "handle": None,
                    "index": region_index,
                    "region": self.regions[region_index],
                    "start": point,
                }
                self.canvas.configure(cursor="fleur")
                self.status_var.set("拖动中；松开鼠标完成平移")
                return
        if not self.pending:
            self.drawing_new = True
            self.status_var.set(
                f"正在画新区域：还需要 "
                f"{(2 if self.mode_var.get() == 'rect' else 4)} 个点"
            )
        self.pending.append(point)
        required = 2 if self.mode_var.get() == "rect" else 4
        if len(self.pending) > required:
            self.pending = self.pending[-required:]
        remaining = required - len(self.pending)
        if remaining > 0:
            self.status_var.set(f"正在画新区域：还需要 {remaining} 个点")
        else:
            self.status_var.set("区域已画完；检查名称和网格后点击“添加区域”")
        self.redraw()

    def _canvas_right_click(self, event) -> None:
        if self.image is not None and self.mask_edit_var.get() and not self.pending:
            point = self._event_position(event)
            if self._cycle_cell_mask(point, reverse=True):
                return
        self.undo_last_point()

    def undo_last_point(self) -> None:
        if self.pending:
            self.pending.pop()
            self.drawing_new = True
            required = 2 if self.mode_var.get() == "rect" else 4
            remaining = required - len(self.pending)
            self.status_var.set(
                f"已撤销一个点；还需要 {remaining} 个点"
                if remaining > 0
                else "区域已画完；检查名称和网格后点击“添加区域”"
            )
        elif self.drawing_new:
            self.drawing_new = False
            self.status_var.set("已退出本次新区域绘制")
        else:
            self.status_var.set("当前没有可撤销的绘制点")
        self.redraw()

    def _canvas_drag(self, event) -> None:
        if self.image is None or self.drag is None:
            return
        point = self._event_position(event)
        index = int(self.drag["index"])
        region = self.drag["region"]
        if not isinstance(region, VisionRegion):
            return
        start = self.drag["start"]
        if not isinstance(start, tuple):
            return
        action = self.drag["action"]
        if action == "resize":
            handle = self.drag["handle"]
            assert isinstance(handle, str)
            start_pixels = (
                start[0] * (self.image.width - 1),
                start[1] * (self.image.height - 1),
            )
            end_pixels = (
                point[0] * (self.image.width - 1),
                point[1] * (self.image.height - 1),
            )
            updated = drag_resize_region(
                region,
                self.image.width,
                self.image.height,
                handle,  # type: ignore[arg-type]
                start_pixels,
                end_pixels,
            )
        else:
            delta_x = (point[0] - start[0]) * (self.image.width - 1)
            delta_y = (point[1] - start[1]) * (self.image.height - 1)
            updated = translate_region_pixels(
                region,
                self.image.width,
                self.image.height,
                delta_x,
                delta_y,
            )
        self.regions[index] = updated
        self._sync_size_fields()
        self.redraw()

    def _canvas_release(self, event) -> None:
        del event
        if self.drag is None:
            return
        self.drag = None
        self._list_signature = None
        self.status_var.set("编辑完成；可直接保存 Profile")
        self.canvas.configure(cursor="crosshair")
        self.redraw()

    def _mouse_move(self, event) -> None:
        if self.image is None:
            return
        point = self._event_position(event)
        x = point[0] * (self.image.width - 1)
        y = point[1] * (self.image.height - 1)
        self.coord_var.set(
            f"x={x:.1f} y={y:.1f} scale={self.scale:.2f}"
        )
        if self.drag is not None:
            return
        if (
            self.selected_index is not None
            and self._hit_selected_handle(event) is not None
        ):
            self.canvas.configure(cursor="sizing")
        elif self._hit_region(point) is not None:
            self.canvas.configure(cursor="fleur")
        elif self.pending or self.drawing_new:
            self.canvas.configure(cursor="crosshair")
        else:
            self.canvas.configure(cursor="tcross")

    def redraw(self) -> None:
        if self.image is None:
            return
        self._ensure_display_image()
        self.canvas.delete("all")
        width, height = self._display_size
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        need = 2 if self.mode_var.get() == "rect" else 4
        if len(self.pending) == need:
            self._draw_region(
                VisionRegion(
                    name=self.name_var.get() or "pending",
                    kind=self.mode_var.get(),
                    points=tuple(self.pending),
                    grid_rows=int(self.rows_var.get()),
                    grid_cols=int(self.cols_var.get()),
                ),
                color="#00d4ff",
            )
        elif self.pending:
            points = [
                self._point_to_display(x, y) for x, y in self.pending
            ]
            if len(points) > 1:
                self.canvas.create_line(
                    points,
                    fill="#00d4ff",
                    width=2,
                    dash=(5, 3),
                )
            for point in points:
                self.canvas.create_oval(
                    point[0] - 3,
                    point[1] - 3,
                    point[0] + 3,
                    point[1] + 3,
                    fill="#ffcc00",
                )
        for index, region in enumerate(self.regions):
            if index == self.selected_index:
                continue
            self._draw_region(
                region,
                color="#ff3b30" if index % 2 == 0 else "#ffcc00",
            )
        if (
            self.selected_index is not None
            and 0 <= self.selected_index < len(self.regions)
        ):
            self._draw_region(
                self.regions[self.selected_index],
                color="#00ff88",
                selected=True,
            )
        self._refresh_list()

    def _ensure_display_image(self) -> None:
        if self.image is None:
            return
        key = (
            self.image_path,
            self.image.width,
            self.image.height,
            self.scale,
        )
        if self.photo is not None and self._display_key == key:
            return
        display = self.image.resize(
            (
                max(1, round(self.image.width * self.scale)),
                max(1, round(self.image.height * self.scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(display)
        self._display_key = key
        self._display_size = (display.width, display.height)

    def _draw_region(
        self,
        region: VisionRegion,
        color: str,
        selected: bool = False,
    ) -> None:
        if self.image is None:
            return
        points = [self._point_to_display(x, y) for x, y in region.points]
        if region.kind == "rect":
            x1, y1 = points[0]
            x2, y2 = points[1]
            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=color,
                width=3 if selected else 2,
            )
        else:
            self.canvas.create_polygon(
                points,
                outline=color,
                fill="",
                width=3 if selected else 2,
            )
        if region.grid_rows > 0 and region.grid_cols > 0:
            stage = self._selected_stage()
            use_stage_mask = self._region_uses_stage_mask(region)
            for cell, label in region_grid_cells(
                region,
                self.image.width,
                self.image.height,
            ):
                scaled = [
                    self._pixel_to_display(float(x), float(y))
                    for x, y in cell
                ]
                cell_fill = ""
                cell_outline = color
                cell_stipple = ""
                if use_stage_mask:
                    stage_row, stage_col = (
                        int(value) for value in label.split(",", 1)
                    )
                    stage_cell = self._effective_stage_cell(
                        stage_row,
                        stage_col,
                    )
                    if stage_cell is Cell.VOID:
                        cell_fill = "#111111"
                        cell_outline = "#666666"
                    elif stage_cell is Cell.WALL:
                        cell_fill = "#777777"
                        cell_outline = "#dddddd"
                    elif stage_cell is Cell.SELF_INK:
                        cell_fill = "#aaff00"
                        cell_stipple = "gray25"
                    elif stage_cell is Cell.OPPONENT_INK:
                        cell_fill = "#8060ff"
                        cell_stipple = "gray25"
                    elif stage_cell is Cell.SELF_SPECIAL:
                        cell_fill = "#aaff00"
                        cell_stipple = "gray50"
                    elif stage_cell is Cell.OPPONENT_SPECIAL:
                        cell_fill = "#8060ff"
                        cell_stipple = "gray50"
                options = {
                    "outline": cell_outline,
                    "fill": cell_fill,
                    "width": 1,
                }
                if cell_stipple:
                    options["stipple"] = cell_stipple
                self.canvas.create_polygon(
                    scaled,
                    **options,
                )
        if selected:
            first = points[0]
            self.canvas.create_text(
                first[0] + 5,
                first[1] + 5,
                anchor="nw",
                text=region.name,
                fill=color,
                font=("Microsoft YaHei UI", 10, "bold"),
            )
            for handle, position in self._handle_positions(region):
                x, y = self._point_to_display(*position)
                radius = 5
                if handle.startswith("vertex:") or len(handle) == 2:
                    self.canvas.create_oval(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        fill="#ffffff",
                        outline=color,
                        width=2,
                    )
                else:
                    self.canvas.create_rectangle(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        fill="#ffffff",
                        outline=color,
                        width=2,
                    )

    def begin_new_region(self) -> None:
        self.pending.clear()
        self.selected_index = None
        self.drag = None
        self.drawing_new = True
        self.region_list.selection_clear(0, "end")
        self.status_var.set(
            f"点击 {2 if self.mode_var.get() == 'rect' else 4} 个点绘制新区域"
        )
        self.canvas.configure(cursor="crosshair")
        self.redraw()

    def add_region(self) -> None:
        required = 2 if self.mode_var.get() == "rect" else 4
        if len(self.pending) != required:
            messagebox.showwarning("区域未完成", f"需要 {required} 个点")
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("区域名称", "请输入区域名称")
            return
        region = VisionRegion(
            name=name,
            kind=self.mode_var.get(),
            points=tuple(self.pending),
            grid_rows=int(self.rows_var.get()),
            grid_cols=int(self.cols_var.get()),
        )
        existing_index = next(
            (
                index
                for index, item in enumerate(self.regions)
                if item.name == name
            ),
            None,
        )
        if existing_index is None:
            self.regions.append(region)
            self.selected_index = len(self.regions) - 1
        else:
            self.regions[existing_index] = region
            self.selected_index = existing_index
        self.pending.clear()
        self.drawing_new = False
        self._list_signature = None
        self._sync_selected_fields()
        self.status_var.set("区域已添加；可拖拽调整，完成后保存 Profile")
        self.redraw()

    def clear_pending(self) -> None:
        self.pending.clear()
        self.drawing_new = False
        self.status_var.set(
            "已取消当前绘制；拖拽已有区域可直接编辑"
        )
        self.redraw()

    def copy_selected(self) -> None:
        if self.selected_index is None or self.image is None:
            self.status_var.set("请先选中要复制的区域")
            return
        if self.pending:
            messagebox.showwarning(
                "复制区域",
                "请先完成当前绘制，或右键撤销绘制点",
            )
            return
        source = self.regions[self.selected_index]
        copy_name = self._unique_region_name(f"{source.name}-copy")
        copied = translate_region_pixels(
            source,
            self.image.width,
            self.image.height,
            delta_x=12,
            delta_y=12,
        )
        copied = VisionRegion(
            name=copy_name,
            kind=copied.kind,
            points=copied.points,
            grid_rows=copied.grid_rows,
            grid_cols=copied.grid_cols,
        )
        self.regions.append(copied)
        self.selected_index = len(self.regions) - 1
        self.drawing_new = False
        self._list_signature = None
        self._sync_selected_fields()
        self.status_var.set(
            f"已复制为“{copy_name}”；可直接拖动到目标位置"
        )
        self.redraw()

    def rename_selected(self, event=None) -> None:
        del event
        if self.selected_index is None:
            self.status_var.set("请先选中要重命名的区域")
            return
        region = self.regions[self.selected_index]
        name = simpledialog.askstring(
            "重命名区域",
            "新名称：",
            initialvalue=region.name,
            parent=self.root,
        )
        if name is None:
            return
        name = name.strip()
        if not name or name == region.name:
            return
        if any(
            index != self.selected_index and item.name == name
            for index, item in enumerate(self.regions)
        ):
            messagebox.showwarning("重命名区域", "该名称已被其他区域使用")
            return
        self.regions[self.selected_index] = VisionRegion(
            name=name,
            kind=region.kind,
            points=region.points,
            grid_rows=region.grid_rows,
            grid_cols=region.grid_cols,
        )
        self._list_signature = None
        self._sync_selected_fields()
        self.status_var.set(f"已重命名为“{name}”")
        self.redraw()

    def _unique_region_name(self, preferred: str) -> str:
        existing = {region.name for region in self.regions}
        if preferred not in existing:
            return preferred
        index = 2
        while f"{preferred}-{index}" in existing:
            index += 1
        return f"{preferred}-{index}"

    def delete_selected(self) -> None:
        if self.selected_index is None:
            return
        index = self.selected_index
        if 0 <= index < len(self.regions):
            self.regions.pop(index)
            if self.regions:
                self.selected_index = min(index, len(self.regions) - 1)
                self._sync_selected_fields()
            else:
                self.selected_index = None
                self.name_var.set("board")
                self.rows_var.set(0)
                self.cols_var.set(0)
                self.size_width_var.set("")
                self.size_height_var.set("")
            self._list_signature = None
            self.status_var.set("已删除选中区域")
            self.redraw()

    def _selected_stage(self):
        value = self.stage_var.get()
        if value == "不套用" or "|" not in value:
            return None
        return self.catalog.stages.get(value.split("|", 1)[0])

    def apply_stage_size(self) -> None:
        stage = self._selected_stage()
        if stage is None:
            return
        region = (
            self.regions[self.selected_index]
            if self.selected_index is not None
            else None
        )
        if region is not None and region.grid_rows > 0 and region.grid_cols > 0:
            self.rows_var.set(region.grid_rows)
            self.cols_var.set(region.grid_cols)
            mismatch = (
                region.grid_rows != stage.rows
                or region.grid_cols != stage.cols
            )
            self.status_var.set(
                (
                    f"保留 Profile 网格 {region.grid_rows}x{region.grid_cols}，"
                    f"解包数据为 {stage.rows}x{stage.cols}；使用手动掩码修正"
                )
                if mismatch
                else f"已套用 {stage.stage_id}: {stage.rows}x{stage.cols}"
            )
        else:
            self.rows_var.set(stage.rows)
            self.cols_var.set(stage.cols)
            if region is not None:
                self.regions[self.selected_index] = VisionRegion(
                    name=region.name,
                    kind=region.kind,
                    points=region.points,
                    grid_rows=stage.rows,
                    grid_cols=stage.cols,
                )
                self._list_signature = None
            self.status_var.set(
                f"已套用 {stage.stage_id}: {stage.rows}x{stage.cols}"
            )
        self.redraw()

    def _stage_dimensions_match(self, region: VisionRegion) -> bool:
        stage = self._selected_stage()
        return (
            stage is not None
            and region.grid_rows == stage.rows
            and region.grid_cols == stage.cols
        )

    def _region_uses_stage_mask(self, region: VisionRegion) -> bool:
        return (
            self._selected_stage() is not None
            and region.grid_rows > 0
            and region.grid_cols > 0
        )

    def clear_cell_overrides(self) -> None:
        self.cell_overrides.clear()
        self.status_var.set("已清空格子掩码覆盖")
        self.redraw()

    def _effective_stage_cell(self, row: int, col: int):
        stage = self._selected_stage()
        if stage is None:
            return None
        override = self.cell_overrides.get((row, col))
        if override is not None:
            return Cell(override)
        if (
            0 <= row < stage.rows
            and 0 <= col < stage.cols
            and self.selected_index is not None
            and self._stage_dimensions_match(
                self.regions[self.selected_index]
            )
        ):
            return stage.cells[row][col]
        return Cell.EMPTY

    def _cell_at_point(
        self,
        point: tuple[float, float],
    ) -> tuple[int, int] | None:
        if self.image is None or self.selected_index is None:
            return None
        stage = self._selected_stage()
        region = self.regions[self.selected_index]
        if (
            stage is None
            or region.grid_rows <= 0
            or region.grid_cols <= 0
        ):
            return None
        x = point[0] * (self.image.width - 1)
        y = point[1] * (self.image.height - 1)
        for polygon, label in region_grid_cells(
            region,
            self.image.width,
            self.image.height,
        ):
            if self._point_in_polygon(x, y, polygon):
                row, col = (int(value) for value in label.split(",", 1))
                return row, col
        return None

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon) -> bool:
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = previous
            x2, y2 = current
            if (y1 > y) != (y2 > y):
                boundary_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < boundary_x:
                    inside = not inside
            previous = current
        return inside

    def _cycle_cell_mask(
        self,
        point: tuple[float, float],
        reverse: bool = False,
    ) -> bool:
        cell = self._cell_at_point(point)
        if cell is None:
            return False
        row, col = cell
        current = self._effective_stage_cell(row, col)
        selected_type = self.mask_cell_types[self.mask_type_var.get()]
        if selected_type is None:
            cycle = (
                Cell.EMPTY,
                Cell.SELF_INK,
                Cell.SELF_SPECIAL,
                Cell.OPPONENT_INK,
                Cell.OPPONENT_SPECIAL,
                Cell.VOID,
                Cell.WALL,
            )
            try:
                index = cycle.index(current)
            except ValueError:
                index = 0
            next_cell = cycle[
                (index + (-1 if reverse else 1)) % len(cycle)
            ]
        elif reverse:
            self.cell_overrides.pop((row, col), None)
            self.status_var.set(f"格 ({row},{col}) 已恢复关卡原始值")
            self.redraw()
            return True
        else:
            next_cell = selected_type
        base_stage = self._selected_stage()
        if base_stage is not None and next_cell is base_stage.cells[row][col]:
            self.cell_overrides.pop((row, col), None)
        else:
            self.cell_overrides[(row, col)] = int(next_cell)
        labels = {
            Cell.EMPTY: "可玩",
            Cell.SELF_INK: "我方初始墨",
            Cell.SELF_SPECIAL: "我方特殊点",
            Cell.OPPONENT_INK: "敌方初始墨",
            Cell.OPPONENT_SPECIAL: "敌方特殊点",
            Cell.VOID: "棋盘外",
            Cell.WALL: "灰色不可放置",
        }
        self.status_var.set(
            f"格 ({row},{col}) -> {labels[next_cell]}"
        )
        self.redraw()
        return True

    def apply_selected_properties(self) -> None:
        if self.selected_index is None:
            messagebox.showwarning("更新区域", "请先选择一个区域")
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("区域名称", "请输入区域名称")
            return
        if any(
            index != self.selected_index and region.name == name
            for index, region in enumerate(self.regions)
        ):
            messagebox.showwarning("区域名称", "该名称已被其他区域使用")
            return
        try:
            rows = max(0, int(self.rows_var.get()))
            cols = max(0, int(self.cols_var.get()))
        except (tk.TclError, ValueError):
            messagebox.showwarning("网格", "网格行列必须是整数")
            return
        region = self.regions[self.selected_index]
        self.regions[self.selected_index] = VisionRegion(
            name=name,
            kind=region.kind,
            points=region.points,
            grid_rows=rows,
            grid_cols=cols,
        )
        self._list_signature = None
        self.status_var.set("名称和网格已更新")
        self.redraw()

    def apply_selected_size(self) -> None:
        if self.selected_index is None or self.image is None:
            messagebox.showwarning("调整尺寸", "请先选择一个区域")
            return
        try:
            width = float(self.size_width_var.get())
            height = float(self.size_height_var.get())
            if width <= 0 or height <= 0:
                raise ValueError
        except (tk.TclError, ValueError):
            messagebox.showwarning("调整尺寸", "宽度和高度必须是正数")
            return
        self.regions[self.selected_index] = resize_region_to_pixels(
            self.regions[self.selected_index],
            self.image.width,
            self.image.height,
            width,
            height,
        )
        self._list_signature = None
        self._sync_size_fields()
        self.status_var.set("已按输入宽高从中心缩放")
        self.redraw()

    def nudge_selected_size(
        self,
        delta_width: float = 0,
        delta_height: float = 0,
    ) -> None:
        if self.selected_index is None or self.image is None:
            return
        width, height = region_pixel_size(
            self.regions[self.selected_index],
            self.image.width,
            self.image.height,
        )
        self.size_width_var.set(f"{max(1.0, width + delta_width):.0f}")
        self.size_height_var.set(f"{max(1.0, height + delta_height):.0f}")
        self.apply_selected_size()

    def _select_region(self, index: int) -> None:
        if not 0 <= index < len(self.regions):
            return
        self.selected_index = index
        self._updating_list = True
        try:
            self.region_list.selection_clear(0, "end")
            self.region_list.selection_set(index)
            self.region_list.see(index)
        finally:
            self._updating_list = False
        self._sync_selected_fields()

    def _on_region_selected(self, event) -> None:
        del event
        if self._updating_list:
            return
        selection = self.region_list.curselection()
        if not selection:
            return
        self._select_region(int(selection[0]))
        self.status_var.set("已选中；拖框内移动，拖白色控制点改尺寸")
        self.redraw()

    def _sync_selected_fields(self) -> None:
        if (
            self.selected_index is None
            or not 0 <= self.selected_index < len(self.regions)
        ):
            return
        region = self.regions[self.selected_index]
        self.name_var.set(region.name)
        self.rows_var.set(region.grid_rows)
        self.cols_var.set(region.grid_cols)
        self._sync_size_fields()

    def _sync_size_fields(self) -> None:
        if self.selected_index is None or self.image is None:
            return
        width, height = region_pixel_size(
            self.regions[self.selected_index],
            self.image.width,
            self.image.height,
        )
        self.size_width_var.set(f"{width:.0f}")
        self.size_height_var.set(f"{height:.0f}")

    def _refresh_list(self) -> None:
        if self.image is None:
            return
        if self.drag is not None:
            return
        entries = []
        for region in self.regions:
            grid = (
                f" grid={region.grid_rows}x{region.grid_cols}"
                if region.grid_rows and region.grid_cols
                else ""
            )
            width, height = region_pixel_size(
                region,
                self.image.width,
                self.image.height,
            )
            entries.append(
                f"{region.name} [{region.kind}] {width:.0f}x{height:.0f}{grid}"
            )
        signature = tuple(entries)
        if signature != self._list_signature:
            self._updating_list = True
            try:
                self.region_list.delete(0, "end")
                for entry in entries:
                    self.region_list.insert("end", entry)
                self._list_signature = signature
                if (
                    self.selected_index is not None
                    and self.selected_index < self.region_list.size()
                ):
                    self.region_list.selection_set(self.selected_index)
                    self.region_list.see(self.selected_index)
            finally:
                self._updating_list = False

    def _event_position(self, event) -> tuple[float, float]:
        x = self.canvas.canvasx(event.x) / max(1, self._display_size[0])
        y = self.canvas.canvasy(event.y) / max(1, self._display_size[1])
        return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))

    def _point_to_display(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        return (
            x * max(1, self._display_size[0] - 1),
            y * max(1, self._display_size[1] - 1),
        )

    def _pixel_to_display(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        assert self.image is not None
        return (
            x * max(1, self._display_size[0] - 1)
            / max(1, self.image.width - 1),
            y * max(1, self._display_size[1] - 1)
            / max(1, self.image.height - 1),
        )

    @staticmethod
    def _point_in_region(
        region: VisionRegion,
        point: tuple[float, float],
    ) -> bool:
        x, y = point
        if region.kind == "rect":
            x1, y1 = region.points[0]
            x2, y2 = region.points[1]
            return (
                min(x1, x2) <= x <= max(x1, x2)
                and min(y1, y2) <= y <= max(y1, y2)
            )
        inside = False
        previous = region.points[-1]
        for current in region.points:
            x1, y1 = previous
            x2, y2 = current
            if (y1 > y) != (y2 > y):
                boundary_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < boundary_x:
                    inside = not inside
            previous = current
        return inside

    def _hit_region(self, point: tuple[float, float]) -> int | None:
        if self.image is None:
            return None
        hits = [
            index
            for index, region in enumerate(self.regions)
            if self._point_in_region(region, point)
        ]
        if not hits:
            return None
        return min(
            hits,
            key=lambda index: (
                region_pixel_size(
                    self.regions[index],
                    self.image.width,
                    self.image.height,
                )[0]
                * region_pixel_size(
                    self.regions[index],
                    self.image.width,
                    self.image.height,
                )[1],
                -index,
            ),
        )

    @staticmethod
    def _handle_positions(
        region: VisionRegion,
    ) -> tuple[tuple[str, tuple[float, float]], ...]:
        if region.kind == "rect":
            xs = sorted(point[0] for point in region.points)
            ys = sorted(point[1] for point in region.points)
            left, right = xs[0], xs[-1]
            top, bottom = ys[0], ys[-1]
            mid_x = (left + right) / 2.0
            mid_y = (top + bottom) / 2.0
            return (
                ("nw", (left, top)),
                ("n", (mid_x, top)),
                ("ne", (right, top)),
                ("e", (right, mid_y)),
                ("se", (right, bottom)),
                ("s", (mid_x, bottom)),
                ("sw", (left, bottom)),
                ("w", (left, mid_y)),
            )
        points = tuple(region.points)
        handles: list[tuple[str, tuple[float, float]]] = [
            (f"vertex:{index}", point) for index, point in enumerate(points)
        ]
        for index, first in enumerate(points):
            second = points[(index + 1) % len(points)]
            midpoint = (
                (first[0] + second[0]) / 2.0,
                (first[1] + second[1]) / 2.0,
            )
            handles.append((f"edge:{index}", midpoint))
        return tuple(handles)

    def _hit_selected_handle(self, event) -> str | None:
        if (
            self.selected_index is None
            or not 0 <= self.selected_index < len(self.regions)
        ):
            return None
        event_norm = self._event_position(event)
        best: tuple[float, str] | None = None
        for handle, position in self._handle_positions(
            self.regions[self.selected_index]
        ):
            dx = (position[0] - event_norm[0]) * self._display_size[0]
            dy = (position[1] - event_norm[1]) * self._display_size[1]
            distance = math.hypot(dx, dy)
            if distance <= 10 and (best is None or distance < best[0]):
                best = (distance, handle)
        return best[1] if best is not None else None

    def save_profile(self) -> None:
        if self.image is None:
            messagebox.showwarning("保存", "请先打开截图")
            return
        path = Path(self.profile_path_var.get())
        stage = self._selected_stage()
        profile = VisionProfile(
            profile_id=path.stem,
            source_width=self.image.width,
            source_height=self.image.height,
            stage_id=stage.stage_id if stage is not None else None,
            regions=tuple(self.regions),
            notes=f"source={self.image_path}",
            cell_overrides=tuple(
                CellMaskOverride(row, col, cell)
                for (row, col), cell in sorted(self.cell_overrides.items())
            ),
        )
        try:
            profile.save(path)
            self._export_overlay(path)
        except Exception as error:
            messagebox.showerror("保存失败", str(error))
            return
        messagebox.showinfo("保存成功", str(path.resolve()))

    def load_profile(self, silent: bool = False) -> None:
        path = Path(self.profile_path_var.get())
        if not path.exists():
            if not silent:
                messagebox.showwarning("加载 Profile", f"文件不存在: {path}")
            return
        try:
            profile = VisionProfile.load(path)
        except Exception as error:
            if not silent:
                messagebox.showerror("加载 Profile", str(error))
            return
        if self.image is None:
            source = ""
            if profile.notes.startswith("source="):
                source = profile.notes.removeprefix("source=")
            source_path = Path(source)
            if source and source_path.exists():
                self.open_image(source_path)
        if self.image is None:
            if not silent:
                messagebox.showwarning("加载 Profile", "请先打开对应截图")
            return
        if (
            profile.source_width != self.image.width
            or profile.source_height != self.image.height
        ):
            messagebox.showwarning(
                "尺寸不一致",
                f"Profile={profile.source_width}x{profile.source_height}, "
                f"图片={self.image.width}x{self.image.height}",
            )
            return
        self.regions = list(profile.regions)
        self.cell_overrides = {
            (override.row, override.col): override.cell
            for override in profile.cell_overrides
        }
        if profile.stage_id is not None:
            self.stage_var.set(next(
                (
                    option
                    for option in self.stage_options
                    if option.split("|", 1)[0] == profile.stage_id
                ),
                "不套用",
            ))
        else:
            self.stage_var.set("不套用")
        self.pending.clear()
        self.drawing_new = False
        self.drag = None
        self.selected_index = None
        self._list_signature = None
        self.redraw()

    def _export_overlay(self, profile_path: Path) -> None:
        if self.image is None:
            return
        from PIL import ImageDraw

        canvas = self.image.copy()
        draw = ImageDraw.Draw(canvas)
        colors = ("#ff3b30", "#ffcc00", "#00d4ff")
        for index, region in enumerate(self.regions):
            points = [
                (x * (canvas.width - 1), y * (canvas.height - 1))
                for x, y in region.points
            ]
            color = colors[index % len(colors)]
            if region.kind == "rect":
                draw.rectangle(points, outline=color, width=4)
            else:
                draw.line(points + [points[0]], fill=color, width=4)
            draw.text(
                (points[0][0] + 4, points[0][1] + 4),
                region.name,
                fill=color,
            )
            for cell, label in region_grid_cells(
                region,
                canvas.width,
                canvas.height,
            ):
                cell_points = [(float(x), float(y)) for x, y in cell]
                draw.line(cell_points + [cell_points[0]], fill=color, width=1)
        target = profile_path.with_name(f"{profile_path.stem}-overlay.png")
        canvas.save(target)


def run_calibration_ui(
    initial_image: Path | None = None,
    profile_path: Path | None = None,
) -> int:
    root = tk.Tk()
    app = CalibrationApp(root, initial_image, profile_path)
    del app
    root.mainloop()
    return 0

