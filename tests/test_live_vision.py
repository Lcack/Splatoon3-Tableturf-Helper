import time

import numpy as np

from tableturf_rebuild.application.battle_vision import BattleVisionObservation
from tableturf_rebuild.application.state_machine import (
    GameState,
    StateObservation,
    StateScore,
)
from tableturf_rebuild.capture.live_vision import run_live_vision
from tableturf_rebuild.capture.models import CaptureConfig
from tableturf_rebuild.capture.worker import CaptureWorker


class FakeSource:
    def __init__(self, config):
        self.config = config

    def open(self):
        pass

    def read(self):
        time.sleep(0.003)
        return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

    def close(self):
        pass


class FakeRuntime:
    def update(self, frame, include_selection=False, force_selection=False):
        del frame, include_selection, force_selection
        return BattleVisionObservation(
            state=StateObservation(
                state=GameState.TURN_SELECT,
                confidence=0.9,
                candidates=(StateScore(GameState.TURN_SELECT, 0.9),),
                actions=(),
                boxes=(),
            ),
            board=None,
            selection_profile=None,
            selection_candidates=(),
        )


def test_live_vision_runs_headless_with_limited_frames() -> None:
    records = []
    config = CaptureConfig(device_index=0, width=64, height=48, fps=30)

    summary = run_live_vision(
        config,
        FakeRuntime(),
        max_frames=3,
        max_seconds=2,
        on_record=records.append,
        worker_factory=lambda value: CaptureWorker(
            value,
            source_factory=FakeSource,
        ),
    )

    assert summary.frames == 3
    assert summary.stopped_reason == "max_frames"
    assert len(records) == 3
    assert all(record.state == "TURN_SELECT" for record in records)
