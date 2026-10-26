import time
import json

import numpy as np
from PIL import Image

from tableturf_rebuild.capture.annotations import (
    AnnotationShape,
    render_annotations,
    sanitize_filename,
    save_annotation_bundle,
)
from tableturf_rebuild.capture.models import CaptureConfig
from tableturf_rebuild.capture.recorder import VideoRecorder
from tableturf_rebuild.capture.video_source import OpenCvVideoSource
from tableturf_rebuild.capture.worker import CaptureWorker


class FakeSource:
    def __init__(self, config):
        self.config = config
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def read(self):
        time.sleep(0.005)
        return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_capture_worker_publishes_latest_frame():
    source = None

    def factory(config):
        nonlocal source
        source = FakeSource(config)
        return source

    worker = CaptureWorker(
        CaptureConfig(device_index=0, width=64, height=48, fps=30),
        source_factory=factory,
    )
    worker.start()
    deadline = time.perf_counter() + 2
    packet = None
    while time.perf_counter() < deadline:
        packet = worker.latest()
        if packet is not None and packet.sequence >= 2:
            break
        time.sleep(0.01)
    worker.stop()

    assert packet is not None
    assert packet.frame.shape == (48, 64, 3)
    assert packet.measured_fps > 0
    assert worker.error == ""
    assert source is not None and source.opened and source.closed


def test_annotation_rendering_and_bundle(tmp_path):
    image = Image.new("RGB", (200, 100), "black")
    shapes = [
        AnnotationShape("ellipse", (0.1, 0.1), (0.4, 0.8), "#ffcc00"),
        AnnotationShape("rectangle", (0.5, 0.2), (0.8, 0.7), "#00d4ff"),
        AnnotationShape("arrow", (0.2, 0.5), (0.6, 0.5), "#ff3b30"),
    ]
    rendered = render_annotations(image, shapes)
    metadata = save_annotation_bundle(
        image,
        tmp_path,
        sequence=1,
        state_name="battle: initial",
        description="棋盘和手牌完整可见",
        attention="关注手牌和 SP",
        actions="选择卡牌，旋转，移动，确认",
        next_state="battle_card_preview",
        shapes=shapes,
        capture_info={"device_index": 0},
    )

    assert rendered.size == image.size
    assert rendered.getbbox() is not None
    assert sanitize_filename("battle: initial") == "battle_initial"
    assert (tmp_path / metadata["original_image"]).exists()
    assert (tmp_path / metadata["annotated_image"]).exists()
    records = (tmp_path / "annotations.jsonl").read_text(encoding="utf-8")
    assert json.loads(records.splitlines()[0])["state_name"] == "battle: initial"

    second = save_annotation_bundle(
        image,
        tmp_path,
        sequence=1,
        state_name="battle: initial",
        description="",
        attention="",
        actions="",
        next_state="",
        shapes=[],
        capture_info={},
    )
    assert second["sequence"] == 2


def test_recording_and_video_reading(tmp_path):
    recorder = VideoRecorder()
    path = recorder.start(tmp_path, (48, 64, 3), fps=10)
    try:
        for index in range(3):
            frame = np.full((48, 64, 3), index * 40, dtype=np.uint8)
            recorder.write(frame)
    finally:
        recorded_path = recorder.stop()

    assert recorded_path is not None and recorded_path.exists()
    source = OpenCvVideoSource(path)
    source.open()
    try:
        frame = source.read()
        assert frame is not None
        assert frame.shape == (48, 64, 3)
        assert source.frame_count >= 3
    finally:
        source.close()

