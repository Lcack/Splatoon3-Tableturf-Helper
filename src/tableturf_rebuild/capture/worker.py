from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

from tableturf_rebuild.capture.models import CaptureConfig, FramePacket


class FrameSource(Protocol):
    def open(self) -> None: ...

    def read(self): ...

    def close(self) -> None: ...


class CaptureWorker:
    def __init__(
        self,
        config: CaptureConfig,
        source_factory: Callable[[CaptureConfig], FrameSource] | None = None,
    ):
        self.config = config
        if source_factory is None:
            from tableturf_rebuild.capture.opencv_source import OpenCvFrameSource

            source_factory = OpenCvFrameSource
        self._source_factory = source_factory
        self._source: FrameSource | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._packet: FramePacket | None = None
        self._error = ""
        self._times: deque[float] = deque(maxlen=120)
        self._sequence = 0

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        if self._source is not None:
            self._source.close()
            self._source = None

    def latest(self) -> FramePacket | None:
        with self._lock:
            return self._packet

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def _run(self) -> None:
        try:
            self._source = self._source_factory(self.config)
            self._source.open()
            while not self._stop.is_set():
                try:
                    frame = self._source.read()
                except Exception as error:
                    self._set_error(str(error))
                    time.sleep(0.1)
                    continue
                now = time.perf_counter()
                self._times.append(now)
                fps = 0.0
                if len(self._times) >= 2:
                    elapsed = self._times[-1] - self._times[0]
                    if elapsed > 0:
                        fps = (len(self._times) - 1) / elapsed
                self._sequence += 1
                with self._lock:
                    self._packet = FramePacket(
                        frame=frame,
                        timestamp=now,
                        sequence=self._sequence,
                        measured_fps=fps,
                    )
                    self._error = ""
        except Exception as error:
            self._set_error(str(error))

