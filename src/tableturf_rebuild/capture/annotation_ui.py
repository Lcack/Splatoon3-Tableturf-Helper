from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from tableturf_rebuild.capture.annotations import AnnotationShape


class AnnotationDialog(tk.Toplevel):
    CANVAS_WIDTH = 900
    CANVAS_HEIGHT = 506

    def __init__(
        self,
        master: tk.Misc,
        image: Image.Image,
        default_name: str,
        on_save: Callable[
            [str, str, str, str, str, list[AnnotationShape]], None
        ],
    ):
        super().__init__(master)
        self.title("标注当前画面")
        self.transient(master)
        self.grab_set()
        self.image = image
        self.on_save = on_save
        self.shapes: list[AnnotationShape] = []
        self._drag_start: tuple[float, float] | None = None
        self._preview_id: int | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._offset = (0.0, 0.0)
        self._display_size = (0, 0)
        self._build(default_name)
        self._render_image()

    def _build(self, default_name: str) -> None:
        body = ttk.Frame(self, padding=8)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body, width=320)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        tools = ttk.Frame(left)
        tools.pack(fill="x", pady=(0, 4))
        self.tool_var = tk.StringVar(value="ellipse")
        ttk.Radiobutton(
            tools, text="椭圆", value="ellipse", variable=self.tool_var
        ).pack(side="left")
        ttk.Radiobutton(
            tools, text="矩形", value="rectangle", variable=self.tool_var
        ).pack(side="left", padx=4)
        ttk.Radiobutton(
            tools, text="箭头", value="arrow", variable=self.tool_var
        ).pack(side="left", padx=4)
        self.color_var = tk.StringVar(value="#ff3b30")
        ttk.Combobox(
            tools,
            textvariable=self.color_var,
            values=("#ff3b30", "#ffcc00", "#00d4ff"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=8)
        ttk.Button(tools, text="撤销", command=self.undo).pack(side="right")
        ttk.Button(tools, text="清空", command=self.clear).pack(
            side="right", padx=4
        )

        self.canvas = tk.Canvas(
            left,
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
            background="black",
            cursor="crosshair",
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._start_shape)
        self.canvas.bind("<B1-Motion>", self._drag_shape)
        self.canvas.bind("<ButtonRelease-1>", self._finish_shape)

        ttk.Label(right, text="画面名称").pack(anchor="w")
        self.name_var = tk.StringVar(value=default_name)
        ttk.Entry(right, textvariable=self.name_var).pack(fill="x", pady=(0, 8))
        self.description = self._text_field(right, "画面说明")
        self.attention = self._text_field(right, "关注点")
        self.actions = self._text_field(right, "可以执行的操作")
        ttk.Label(right, text="下一个状态").pack(anchor="w")
        self.next_state_var = tk.StringVar()
        ttk.Entry(right, textvariable=self.next_state_var).pack(
            fill="x", pady=(0, 8)
        )
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", side="bottom")
        ttk.Button(buttons, text="取消", command=self.destroy).pack(
            side="right", padx=4
        )
        ttk.Button(buttons, text="保存标注", command=self.save).pack(
            side="right"
        )

    @staticmethod
    def _text_field(parent, label: str) -> tk.Text:
        ttk.Label(parent, text=label).pack(anchor="w")
        widget = tk.Text(parent, height=4, wrap="word")
        widget.pack(fill="x", pady=(0, 8))
        return widget

    def _render_image(self) -> None:
        width, height = self.image.size
        scale = min(
            self.CANVAS_WIDTH / width,
            self.CANVAS_HEIGHT / height,
        )
        display_width = max(1, round(width * scale))
        display_height = max(1, round(height * scale))
        offset_x = (self.CANVAS_WIDTH - display_width) / 2
        offset_y = (self.CANVAS_HEIGHT - display_height) / 2
        self._scale = scale
        self._offset = (offset_x, offset_y)
        self._display_size = (display_width, display_height)
        shown = self.image.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS,
        )
        self._photo = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image(
            offset_x,
            offset_y,
            anchor="nw",
            image=self._photo,
        )
        self._draw_shapes()

    def _canvas_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        offset_x, offset_y = self._offset
        width, height = self._display_size
        return (
            max(0.0, min(1.0, (x - offset_x) / width)),
            max(0.0, min(1.0, (y - offset_y) / height)),
        )

    def _normalized_to_canvas(
        self, point: tuple[float, float]
    ) -> tuple[float, float]:
        offset_x, offset_y = self._offset
        width, height = self._display_size
        return offset_x + point[0] * width, offset_y + point[1] * height

    def _start_shape(self, event) -> None:
        self._drag_start = self._canvas_to_normalized(event.x, event.y)
        if self._preview_id is not None:
            self.canvas.delete(self._preview_id)
        self._preview_id = None

    def _drag_shape(self, event) -> None:
        if self._drag_start is None:
            return
        start = self._normalized_to_canvas(self._drag_start)
        end = (event.x, event.y)
        if self._preview_id is not None:
            self.canvas.delete(self._preview_id)
        self._preview_id = self._draw_shape_canvas(
            self.tool_var.get(),
            start,
            end,
            self.color_var.get(),
        )

    def _finish_shape(self, event) -> None:
        if self._drag_start is None:
            return
        end = self._canvas_to_normalized(event.x, event.y)
        start = self._drag_start
        self._drag_start = None
        if self._preview_id is not None:
            self.canvas.delete(self._preview_id)
            self._preview_id = None
        if abs(start[0] - end[0]) < 0.002 and abs(start[1] - end[1]) < 0.002:
            return
        self.shapes.append(
            AnnotationShape(
                kind=self.tool_var.get(),
                start=start,
                end=end,
                color=self.color_var.get(),
            )
        )
        self._draw_shapes()

    def _draw_shapes(self) -> None:
        for shape in self.shapes:
            self._draw_shape_canvas(
                shape.kind,
                self._normalized_to_canvas(shape.start),
                self._normalized_to_canvas(shape.end),
                shape.color,
            )

    def _draw_shape_canvas(self, kind, start, end, color) -> int | None:
        x1, y1 = start
        x2, y2 = end
        if kind == "rectangle":
            return self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=color, width=3
            )
        if kind == "ellipse":
            return self.canvas.create_oval(
                x1, y1, x2, y2, outline=color, width=3
            )
        if kind == "arrow":
            return self.canvas.create_line(
                x1, y1, x2, y2, fill=color, width=3, arrow=tk.LAST
            )
        return None

    def undo(self) -> None:
        if self.shapes:
            self.shapes.pop()
            self._render_image()

    def clear(self) -> None:
        self.shapes.clear()
        self._render_image()

    def save(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("缺少名称", "请填写画面名称", parent=self)
            return
        try:
            self.on_save(
                name,
                self.description.get("1.0", "end-1c"),
                self.attention.get("1.0", "end-1c"),
                self.actions.get("1.0", "end-1c"),
                self.next_state_var.get(),
                list(self.shapes),
            )
        except Exception as error:
            messagebox.showerror("保存失败", str(error), parent=self)
            return
        self.destroy()

