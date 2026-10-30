"""Queue edits cannot replace an active runner's immutable task definition."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from tableturf_rebuild.application.deck_config import DeckConfig
from tableturf_rebuild.application.task_plan import NpcTaskDefinition
from tableturf_rebuild.application.task_queue import TaskQueue, TaskQueueError, TaskQueueStatus
from tableturf_rebuild.data.catalog import GameCatalog


@pytest.fixture(scope="module")
def catalog():
    return GameCatalog.from_package()


def definition(catalog, *, other=False, wins=30, deck_name=None):
    npc_id = next(key for key in catalog.npcs if key != "MiniGame_WeaponShop") if other else "MiniGame_WeaponShop"
    npc = catalog.npcs[npc_id]
    return NpcTaskDefinition(npc_id, npc.name_zh or npc.name, 3, wins, deck_name)


def deck(catalog):
    return DeckConfig("共享卡组", tuple(catalog.cards)[:15])


def saved_queue(tmp_path, catalog):
    queue = TaskQueue(catalog, tmp_path / "queue.json")
    queue.append(definition(catalog), deck(catalog))
    queue.save()
    return queue


def test_append_while_running_preserves_current_and_executes_in_order(catalog):
    queue = TaskQueue(catalog)
    first = queue.append(definition(catalog), deck(catalog))
    active = queue.start_next()
    second = queue.append(definition(catalog, other=True, wins=12), deck(catalog))
    third = queue.append(definition(catalog, wins=60))
    assert queue.current == active and active.entry_id == first.entry_id
    assert queue.current.task.target_wins == 30
    assert queue.pending == (second, third)
    assert queue.start_next() is active  # Repeated start never skips the active task.
    queue.update_current_wins((3, 3, 30))
    finished = queue.mark_current_completed()
    assert finished.status is TaskQueueStatus.COMPLETED and queue.current is None
    assert queue.start_next().entry_id == second.entry_id
    assert queue.pending == (third,)


def test_adopt_current_keeps_an_existing_running_task_and_promotes_matching_pending(catalog):
    queue = TaskQueue(catalog)
    waiting = queue.append(definition(catalog, other=True))
    matching = queue.append(definition(catalog))
    active = queue.adopt_current(definition(catalog), observed_wins=(3, 3, 8))
    assert active.entry_id == matching.entry_id
    assert queue.entries == (active, waiting) and queue.current == active
    again = queue.adopt_current(definition(catalog), observed_wins=(3, None, 7))
    assert again.entry_id == active.entry_id and again.observed_wins == (3, 3, 8)
    with pytest.raises(TaskQueueError, match="不能覆盖"):
        queue.adopt_current(definition(catalog, other=True))
    assert queue.current == again


def test_pending_reordering_and_delete_cannot_change_running_or_completed_entries(catalog):
    queue = TaskQueue(catalog)
    first = queue.append(definition(catalog))
    queue.start_next()
    second = queue.append(definition(catalog, other=True))
    third = queue.append(definition(catalog, wins=60))
    queue.move(third.entry_id, -1)
    assert [entry.entry_id for entry in queue.pending] == [third.entry_id, second.entry_id]
    queue.move(third.entry_id, -1)
    assert queue.entries[0].entry_id == first.entry_id
    queue.remove(second.entry_id)
    with pytest.raises(TaskQueueError, match="只能"):
        queue.remove(first.entry_id)
    with pytest.raises(TaskQueueError, match="只能"):
        queue.move(first.entry_id, 1)
    queue.mark_current_completed()
    with pytest.raises(TaskQueueError, match="只能"):
        queue.remove(first.entry_id)


def test_round_trip_keeps_ids_active_status_progress_and_bound_deck(tmp_path, catalog):
    mutable_cards = list(deck(catalog).card_ids)
    queue = TaskQueue(catalog, tmp_path / "队列.json")
    first = queue.append(definition(catalog), DeckConfig("共享卡组", mutable_cards))
    queue.start_next()
    queue.update_current_wins((3, 3, 9))
    queue.update_current_wins((None, 2, 8))  # Stale/unknown readings do not erase confirmed wins.
    second = queue.append(definition(catalog, other=True))
    mutable_cards[0] = "an edited preset does not alter the snapshot"
    queue.save()
    loaded = TaskQueue.load(queue.path, catalog)
    assert loaded.entries == queue.entries
    assert loaded.current.entry_id == first.entry_id
    assert loaded.current.observed_wins == (3, 3, 9)
    assert loaded.current.deck == deck(catalog)
    assert loaded.pending[0].entry_id == second.entry_id
    assert loaded.current.status is TaskQueueStatus.RUNNING
    assert loaded.start_next().entry_id == first.entry_id


def test_missing_file_stays_empty_and_does_not_write_until_save(tmp_path, catalog):
    path = tmp_path / "nested/queue.json"
    queue = TaskQueue.load(path, catalog)
    assert not queue.entries and queue.current is None and queue.start_next() is None
    assert not path.exists()
    queue.save()
    assert path.is_file()


@pytest.mark.parametrize("mutation", (
    lambda value: value.update(version=2),
    lambda value: value.update(entries={}),
    lambda value: value["entries"][0]["task"].update(npc_id="missing"),
    lambda value: value["entries"][0]["task"].update(target_level=True),
    lambda value: value["entries"][0]["task"].update(target_wins=1.5),
    lambda value: value["entries"][0]["task"].update(target_wins=0),
    lambda value: value["entries"][0].update(status="unknown"),
    lambda value: value["entries"][0].update(observed_wins=[3, True, 1]),
    lambda value: value["entries"][0].update(observed_wins=[3, -1, 1]),
    lambda value: value["entries"][0].update(observed_wins=[3, 1]),
    lambda value: value["entries"][0]["deck"]["card_ids"].__setitem__(0, "missing"),
    lambda value: value["entries"][0]["deck"].update(card_ids=["MiniGame_TakoPod"] * 15),
    lambda value: value["entries"][0]["deck"].update(card_ids=[]),
    lambda value: value["entries"][0]["deck"].update(name="other preset"),
    lambda value: value["entries"][0].update(deck=None),
))
def test_invalid_persisted_tasks_are_rejected_without_overwriting_the_file(tmp_path, catalog, mutation):
    queue = saved_queue(tmp_path, catalog)
    value = json.loads(queue.path.read_text(encoding="utf-8"))
    mutation(value)
    content = json.dumps(value, ensure_ascii=False)
    queue.path.write_text(content, encoding="utf-8")
    with pytest.raises(TaskQueueError):
        TaskQueue.load(queue.path, catalog)
    assert queue.path.read_text(encoding="utf-8") == content


def test_duplicate_ids_multiple_running_and_interleaved_statuses_are_rejected(tmp_path, catalog):
    queue = saved_queue(tmp_path, catalog)
    value = json.loads(queue.path.read_text(encoding="utf-8"))
    duplicate = json.loads(json.dumps(value["entries"][0]))
    value["entries"].append(duplicate)
    queue.path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TaskQueueError, match="重复"):
        TaskQueue.load(queue.path, catalog)
    duplicate["entry_id"] = "distinct"
    for item in value["entries"]:
        item["status"] = "running"
    queue.path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TaskQueueError, match="一个进行"):
        TaskQueue.load(queue.path, catalog)
    value["entries"][0]["status"] = "pending"
    value["entries"][1]["status"] = "completed"
    queue.path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TaskQueueError, match="交错"):
        TaskQueue.load(queue.path, catalog)


def test_invalid_json_remains_available_for_recovery(tmp_path, catalog):
    path = tmp_path / "queue.json"
    path.write_text("{unfinished", encoding="utf-8")
    with pytest.raises(TaskQueueError, match="JSON"):
        TaskQueue.load(path, catalog)
    assert path.read_text(encoding="utf-8") == "{unfinished"


def test_failed_atomic_replace_keeps_previous_file_and_removes_temporary_file(tmp_path, catalog, monkeypatch):
    queue = saved_queue(tmp_path, catalog)
    previous = queue.path.read_bytes()
    queue.append(definition(catalog, other=True))
    def denied(source, target):
        raise OSError("simulated replacement failure")
    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(OSError, match="simulated"):
        queue.save()
    assert queue.path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))


def test_invalid_new_task_is_rejected_before_mutating_queue(catalog):
    queue = TaskQueue(catalog)
    with pytest.raises(TaskQueueError, match="未知NPC"):
        queue.append(replace(definition(catalog), npc_id="missing"))
    with pytest.raises(TaskQueueError, match="正整数"):
        queue.append(replace(definition(catalog), target_wins=True))
    with pytest.raises(TaskQueueError, match="15张"):
        queue.append(definition(catalog), DeckConfig("short", ("MiniGame_TakoPod",)))
    with pytest.raises(TaskQueueError, match="15张"):
        queue.append(definition(catalog), DeckConfig("invalid", None))
    assert not queue.entries
