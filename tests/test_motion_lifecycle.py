from types import SimpleNamespace

from tableturf_rebuild.application.task_runner import TaskRunner, TaskStatus
from tableturf_rebuild.control.dummy import DummyBackend


def test_runner_lifecycle_cancels_batches_and_explicitly_resumes_them():
    runner = object.__new__(TaskRunner)
    runner.controller = DummyBackend()
    runner.controller.connect()
    runner.task = object()
    calls = []
    runner.battle_executor = SimpleNamespace(
        cancel_pending_motion=lambda: calls.append("cancel"),
        resume_pending_motion=lambda: calls.append("resume"),
    )
    runner.start()
    runner.pause()
    runner.resume()
    runner.stop()
    assert calls == ["resume", "cancel", "resume", "cancel"]
    assert runner.status is TaskStatus.IDLE
    assert not runner.controller.history[-1].buttons


def test_pause_does_not_reopen_a_completed_task():
    runner = object.__new__(TaskRunner)
    runner.status = TaskStatus.COMPLETED
    calls = []
    runner.battle_executor = SimpleNamespace(cancel_pending_motion=lambda: calls.append("cancel"))
    runner.pause()
    runner.resume()
    assert calls == ["cancel"]
    assert runner.status is TaskStatus.COMPLETED
