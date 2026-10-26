from __future__ import annotations

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk


class DeckSearchMonitorApp:
    def __init__(self, root: tk.Tk, output_dir: Path):
        self.root = root
        self.output_dir = output_dir.resolve()
        self.root.title("AutoTableTurf 牌组搜索进度")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)
        self._leaderboard: list[dict] = []
        self._build()
        self.root.after(100, self.refresh)

    def _build(self) -> None:
        top = ttk.LabelFrame(self.root, text="运行状态")
        top.pack(fill="x", padx=8, pady=8)
        self.status_var = tk.StringVar(value="等待搜索进程...")
        self.time_var = tk.StringVar(value="本轮运行：0秒")
        self.progress_var = tk.StringVar(value="牌组：0/0  对局：0/0")
        self.current_var = tk.StringVar(value="当前：-")
        ttk.Label(
            top,
            textvariable=self.status_var,
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(top, textvariable=self.time_var).grid(
            row=0, column=1, sticky="e", padx=6
        )
        top.grid_columnconfigure(1, weight=1)
        ttk.Label(top, textvariable=self.progress_var).grid(
            row=1, column=0, sticky="w", padx=6
        )
        ttk.Label(top, textvariable=self.current_var).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4)
        )
        self.progress = ttk.Progressbar(top, maximum=1, value=0)
        self.progress.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6)
        )

        table_frame = ttk.LabelFrame(self.root, text="实时排行榜（前 20）")
        table_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        columns = ("rank", "deck", "games", "wins", "losses", "draws", "win_rate", "margin")
        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=18,
        )
        headings = {
            "rank": "#",
            "deck": "候选牌组",
            "games": "局数",
            "wins": "胜",
            "losses": "负",
            "draws": "平",
            "win_rate": "胜率",
            "margin": "平均净胜墨迹",
        }
        widths = {
            "rank": 45,
            "deck": 150,
            "games": 70,
            "wins": 60,
            "losses": 60,
            "draws": 60,
            "win_rate": 90,
            "margin": 130,
        }
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor="center")
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self._show_cards)

        details = ttk.LabelFrame(self.root, text="候选卡牌")
        details.pack(fill="x", padx=8, pady=(0, 8))
        self.cards_text = tk.Text(details, height=5, wrap="word", state="disabled")
        self.cards_text.pack(fill="x")

    def refresh(self) -> None:
        progress_path = self.output_dir / "progress.json"
        if progress_path.exists():
            try:
                data = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = None
            if data:
                self._apply(data)
        self.root.after(1000, self.refresh)

    def _apply(self, data: dict) -> None:
        status = data.get("status", "running")
        status_labels = {
            "running": "正在运行",
            "complete": "全部完成",
            "time_limit": "本轮达到时间上限",
            "interrupted": "已中断",
        }
        self.status_var.set(status_labels.get(status, status))
        elapsed = float(data.get("invocation_elapsed_seconds", 0))
        self.time_var.set(f"本轮运行：{format_duration(elapsed)}")
        decks_completed = int(data.get("decks_completed", 0))
        decks_total = int(data.get("decks_total", 0))
        games_completed = int(data.get("games_completed", 0))
        games_total = int(data.get("games_total", 0))
        self.progress_var.set(
            f"牌组：{decks_completed}/{decks_total}  "
            f"对局：{games_completed}/{games_total}"
        )
        self.current_var.set(
            f"当前：{data.get('current_deck', '-') or '-'} "
            f"{data.get('current_scenario', '') or ''} "
            f"第 {data.get('current_game', 0)} 场"
        )
        self.progress.configure(maximum=max(1, games_total), value=games_completed)
        leaderboard = data.get("leaderboard", [])
        self._leaderboard = list(leaderboard)
        current_selection = self.table.selection()
        for item in self.table.get_children():
            self.table.delete(item)
        for rank, row in enumerate(leaderboard, start=1):
            item = self.table.insert(
                "",
                "end",
                values=(
                    rank,
                    row["candidate_id"],
                    row["games"],
                    row["wins"],
                    row["losses"],
                    row["draws"],
                    f"{row['win_rate']:.1%}",
                    f"{row['average_score_margin']:.2f}",
                ),
            )
            if current_selection and item == current_selection[0]:
                self.table.selection_set(item)

    def _show_cards(self, event=None) -> None:
        del event
        selection = self.table.selection()
        if not selection:
            return
        index = self.table.index(selection[0])
        if not (0 <= index < len(self._leaderboard)):
            return
        cards = self._leaderboard[index]["card_ids"]
        self.cards_text.configure(state="normal")
        self.cards_text.delete("1.0", "end")
        self.cards_text.insert("1.0", "\n".join(cards))
        self.cards_text.configure(state="disabled")


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_deck_search_monitor(output_dir: Path) -> int:
    root = tk.Tk()
    app = DeckSearchMonitorApp(root, output_dir)
    del app
    root.mainloop()
    return 0

