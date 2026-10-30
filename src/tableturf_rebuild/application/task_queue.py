"""Persistent ordered task definitions; device control belongs to the caller."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import json
from pathlib import Path
from uuid import uuid4

from tableturf_rebuild.application.deck_config import DeckConfig
from tableturf_rebuild.application.task_plan import NpcTaskDefinition
from tableturf_rebuild.data.catalog import GameCatalog


class TaskQueueError(ValueError):
    pass


class TaskQueueStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class TaskQueueEntry:
    entry_id: str
    task: NpcTaskDefinition
    deck: DeckConfig | None = None
    status: TaskQueueStatus = TaskQueueStatus.PENDING
    observed_wins: tuple[int | None, int | None, int | None] | None = None


class TaskQueue:
    VERSION = 1

    def __init__(self, catalog: GameCatalog, path: Path | None = None):
        self.catalog = catalog
        self.path = Path(path) if path is not None else None
        self._entries: list[TaskQueueEntry] = []

    @property
    def entries(self) -> tuple[TaskQueueEntry, ...]:
        return tuple(self._entries)

    @property
    def pending(self) -> tuple[TaskQueueEntry, ...]:
        return tuple(entry for entry in self._entries if entry.status is TaskQueueStatus.PENDING)

    @property
    def current(self) -> TaskQueueEntry | None:
        return next((entry for entry in self._entries if entry.status is TaskQueueStatus.RUNNING), None)

    @property
    def active(self) -> TaskQueueEntry | None:
        return self.current

    def append(self, task: NpcTaskDefinition, deck: DeckConfig | None = None) -> TaskQueueEntry:
        task, deck = self._validate_definition(task, deck)
        entry = TaskQueueEntry(uuid4().hex, task, deck)
        self._entries.append(entry)
        return entry

    def adopt_current(
        self, task: NpcTaskDefinition, deck: DeckConfig | None = None,
        observed_wins: tuple[int | None, int | None, int | None] | None = None,
    ) -> TaskQueueEntry:
        """Attach an already configured runner without changing or starting it."""
        task, deck = self._validate_definition(task, deck)
        wins = _validate_wins(observed_wins)
        current = self.current
        if current is not None:
            if current.task != task or current.deck != deck:
                raise TaskQueueError("已有其他进行中的任务，不能覆盖当前任务")
            return self.update_current_wins(wins) if wins is not None else current
        existing = next((entry for entry in self.pending if entry.task == task and entry.deck == deck), None)
        if existing is not None:
            self._entries.remove(existing)
            entry = replace(existing, status=TaskQueueStatus.RUNNING,
                            observed_wins=_merge_wins(existing.observed_wins, wins))
        else:
            entry = TaskQueueEntry(uuid4().hex, task, deck, TaskQueueStatus.RUNNING, wins)
        # Completed history stays first; the current task precedes all waiting tasks.
        index = next((i for i, item in enumerate(self._entries)
                      if item.status is TaskQueueStatus.PENDING), len(self._entries))
        self._entries.insert(index, entry)
        return entry

    def start_next(self) -> TaskQueueEntry | None:
        """Mark the next task active; this never sends input or starts a runner.

        An already active entry is returned unchanged. A loaded running entry
        therefore remains available for an explicit resume by the application.
        """
        if self.current is not None:
            return self.current
        if not self.pending:
            return None
        return self._replace(replace(self.pending[0], status=TaskQueueStatus.RUNNING))

    def mark_current_completed(self) -> TaskQueueEntry:
        """The caller must confirm the runner finished before invoking this."""
        current = self.current
        if current is None:
            raise TaskQueueError("没有进行中的任务可标记完成")
        return self._replace(replace(current, status=TaskQueueStatus.COMPLETED))

    def update_current_wins(
        self, wins: tuple[int | None, int | None, int | None],
    ) -> TaskQueueEntry:
        current = self.current
        if current is None:
            raise TaskQueueError("没有进行中的任务可更新进度")
        checked = _validate_wins(wins)
        return self._replace(replace(current, observed_wins=_merge_wins(current.observed_wins, checked)))

    def remove(self, entry_id: str) -> TaskQueueEntry:
        entry = self._pending_entry(entry_id)
        self._entries.remove(entry)
        return entry

    def move(self, entry_id: str, offset: int) -> TaskQueueEntry:
        if type(offset) is not int or offset not in {-1, 1}:
            raise TaskQueueError("待办移动方向必须为上移或下移一项")
        entry = self._pending_entry(entry_id)
        pending = self.pending
        position = pending.index(entry)
        target = position + offset
        if 0 <= target < len(pending):
            first, second = self._entries.index(entry), self._entries.index(pending[target])
            self._entries[first], self._entries[second] = self._entries[second], self._entries[first]
        return entry

    def save(self, path: Path | None = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise TaskQueueError("未指定任务队列保存路径")
        value = {
            "version": self.VERSION,
            "entries": [{"entry_id": entry.entry_id, "task": asdict(entry.task),
                         "deck": asdict(entry.deck) if entry.deck is not None else None,
                         "status": entry.status.value, "observed_wins": entry.observed_wins}
                        for entry in self._entries],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        self.path = target

    @classmethod
    def load(cls, path: Path, catalog: GameCatalog) -> TaskQueue:
        """Load definitions and progress without activating external operations.

        Missing files mean an empty queue. Invalid files raise rather than
        silently deleting user tasks or rewriting the source document.
        """
        queue = cls(catalog, path)
        if not queue.path.exists():
            return queue
        try:
            value = json.loads(queue.path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, UnicodeError) as error:
            raise TaskQueueError("任务队列文件不是有效的JSON") from error
        if (not isinstance(value, dict) or type(value.get("version")) is not int
            or value["version"] != cls.VERSION or not isinstance(value.get("entries"), list)):
            raise TaskQueueError("任务队列文件的版本或结构无效")
        entries = [queue._parse_entry(item) for item in value["entries"]]
        if len({entry.entry_id for entry in entries}) != len(entries):
            raise TaskQueueError("任务队列中存在重复任务ID")
        if sum(entry.status is TaskQueueStatus.RUNNING for entry in entries) > 1:
            raise TaskQueueError("任务队列只能包含一个进行中的任务")
        order = {TaskQueueStatus.COMPLETED: 0, TaskQueueStatus.RUNNING: 1, TaskQueueStatus.PENDING: 2}
        if [order[entry.status] for entry in entries] != sorted(order[entry.status] for entry in entries):
            raise TaskQueueError("任务队列顺序无效：已完成、进行中和待办任务不能交错")
        queue._entries = entries
        return queue

    def _pending_entry(self, entry_id: str) -> TaskQueueEntry:
        entry = next((entry for entry in self._entries if entry.entry_id == entry_id), None)
        if entry is None:
            raise TaskQueueError("所选任务不存在")
        if entry.status is not TaskQueueStatus.PENDING:
            raise TaskQueueError("只能调整或删除待办任务，不能修改进行中或已完成的任务")
        return entry

    def _replace(self, entry: TaskQueueEntry) -> TaskQueueEntry:
        index = next(i for i, previous in enumerate(self._entries) if previous.entry_id == entry.entry_id)
        self._entries[index] = entry
        return entry

    def _validate_definition(
        self, task: NpcTaskDefinition, deck: DeckConfig | None,
    ) -> tuple[NpcTaskDefinition, DeckConfig | None]:
        if not isinstance(task, NpcTaskDefinition):
            raise TaskQueueError("任务定义无效")
        if not isinstance(task.npc_id, str) or task.npc_id not in self.catalog.npcs:
            raise TaskQueueError(f"未知NPC：{task.npc_id!r}")
        if not isinstance(task.npc_name, str) or not task.npc_name.strip():
            raise TaskQueueError("NPC名称不能为空")
        if type(task.target_level) is not int or task.target_level not in {1, 2, 3}:
            raise TaskQueueError("目标难度必须是1、2或3")
        if type(task.target_wins) is not int or task.target_wins <= 0:
            raise TaskQueueError("目标累计胜场必须是正整数")
        if task.deck_name is not None and (not isinstance(task.deck_name, str) or not task.deck_name.strip()):
            raise TaskQueueError("任务牌组名称无效")
        if deck is None:
            if task.deck_name is not None:
                raise TaskQueueError("指定牌组的任务缺少15张卡牌快照")
            return task, None
        if not isinstance(deck, DeckConfig) or not isinstance(deck.name, str) or not deck.name.strip():
            raise TaskQueueError("预设牌组名称不能为空")
        if not isinstance(deck.card_ids, (tuple, list)):
            raise TaskQueueError("任务牌组必须包含15张不重复卡牌")
        ids = tuple(deck.card_ids)
        if len(ids) != 15 or any(not isinstance(card, str) for card in ids) or len(set(ids)) != 15:
            raise TaskQueueError("任务牌组必须包含15张不重复卡牌")
        if any(card not in self.catalog.cards for card in ids):
            raise TaskQueueError("任务牌组包含未知卡牌")
        name = deck.name.strip()
        if task.deck_name is not None and task.deck_name.strip() != name:
            raise TaskQueueError("任务牌组名称与卡牌快照不一致")
        return replace(task, deck_name=name), DeckConfig(name, ids)

    def _parse_entry(self, item) -> TaskQueueEntry:
        if not isinstance(item, dict) or not isinstance(item.get("entry_id"), str) or not item["entry_id"].strip():
            raise TaskQueueError("任务队列项缺少有效ID")
        source = item.get("task")
        if not isinstance(source, dict):
            raise TaskQueueError("任务队列项缺少任务定义")
        try:
            task = NpcTaskDefinition(**source)
        except (TypeError, ValueError) as error:
            raise TaskQueueError("任务队列项中的NPC目标无效") from error
        deck = item.get("deck")
        if deck is not None:
            if (not isinstance(deck, dict) or not isinstance(deck.get("name"), str)
                or not isinstance(deck.get("card_ids"), list)):
                raise TaskQueueError("任务牌组快照结构无效")
            deck = DeckConfig(deck["name"], tuple(deck["card_ids"]))
        task, deck = self._validate_definition(task, deck)
        try:
            status = TaskQueueStatus(item["status"])
        except (KeyError, TypeError, ValueError) as error:
            raise TaskQueueError("任务队列状态无效") from error
        wins = _validate_wins(item.get("observed_wins"))
        return TaskQueueEntry(item["entry_id"], task, deck, status, wins)


def _validate_wins(value) -> tuple[int | None, int | None, int | None] | None:
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 3
        or any(win is not None and (type(win) is not int or win < 0) for win in value)):
        raise TaskQueueError("累计胜场必须是三个非负整数或未知值")
    return tuple(value)


def _merge_wins(old, new):
    if new is None:
        return old
    if old is None:
        return new
    return tuple(max(a, b) if a is not None and b is not None else a if b is None else b
                 for a, b in zip(old, new, strict=True))
