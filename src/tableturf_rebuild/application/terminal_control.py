"""Main-thread bridge between the local HTTP queue and the task terminal."""
from __future__ import annotations

import time
import threading

from tableturf_rebuild.control.buttons import Button, ControllerState
from tableturf_rebuild.application.control_server import COMMAND_DEFERRED, LocalControlServer


class TerminalControlBridge:
    def __init__(self, app, *, server=None, clock=time.monotonic, stream_frames=True):
        self.app = app
        self.clock = clock
        self.server = server or LocalControlServer(app.diagnostics.session_dir)
        self.pending = None
        self.deadline = 0.0
        self.started_at = 0.0
        self.holding = False
        self._release_timer = None
        self._pulse_lock = threading.Lock()
        self._pulse_released = threading.Event()
        self._pulse_release_success = True
        self.resume_debug = False
        self.closed = False
        self._frame_stop = threading.Event()
        self._frame_thread = None
        self.server.start()
        if stream_frames:
            self._frame_thread = threading.Thread(target=self._stream_frames, daemon=True)
            self._frame_thread.start()

    def _stream_frames(self):
        # Encoding stays off Tk and off the vision worker. Packet images are
        # published by CaptureWorker and are never modified by this reader.
        import cv2
        last = None
        while not self._frame_stop.is_set():
            worker = getattr(self.app, "worker", None)
            packet = worker.latest() if worker is not None else None
            if packet is not None and (worker, packet.sequence) != last:
                try:
                    ok, encoded = cv2.imencode(".jpg", packet.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        captured_at = time.time() - max(0.0, time.perf_counter() - packet.timestamp)
                        self.server.set_frame("capture", encoded.tobytes(), sequence=packet.sequence,
                                              captured_at=captured_at)
                        last = (worker, packet.sequence)
                except Exception:
                    pass  # The file-backed frame remains available with its age.
            self._frame_stop.wait(0.2)

    def _pause(self):
        app = self.app
        self.resume_debug = self.resume_debug or app.battle_debug_active
        app.battle_debug_active = False
        app.runner.pause()
        app._update_snapshot(app.runner.snapshot())

    def handle(self, command):
        name = command["command"]
        if name == "release":
            success = self.app.controller.send_state(ControllerState.neutral())
            if self.pending is not None:
                self._finish(error="操作被释放命令中止")
            if not success:
                raise RuntimeError(self.app.controller.last_error or "控制器释放失败")
            return {"released": True}
        if self.pending is not None:
            raise RuntimeError("上一条控制命令尚未完成")
        if name not in {"pause", "resume", "reload", "press"}:
            raise ValueError("unsupported command")
        if name == "press":
            # Validate again at the application boundary, before pausing or sending.
            buttons = frozenset(Button(value) for value in command["buttons"])
            if not buttons or not 30 <= command["hold_ms"] <= 1000:
                raise ValueError("invalid button pulse")
            if not self.app.controller.connected:
                raise RuntimeError("控制器未连接")
        self._pause()
        self.pending = command
        self.started_at = self.clock()
        self.deadline = self.started_at + 0.06
        self.app._local_control_manual_busy = True
        return COMMAND_DEFERRED

    def _finish(self, *, result=None, error=None):
        pending, self.pending = self.pending, None
        self.holding = False
        if self._release_timer is not None:
            self._release_timer.cancel()
            self._release_timer = None
        self.app._local_control_manual_busy = False
        if pending is not None:
            self.server.complete_command(pending["id"], result=result, error=error)

    def _release_pulse(self, pulse=None):
        # A watchdog releases even if Tk is temporarily blocked by another UI
        # operation. It touches only the controller's thread-safe transport.
        with self._pulse_lock:
            if pulse is not None and pulse is not self._pulse_released:
                return True
            if not self._pulse_released.is_set():
                self._pulse_release_success = self.app.controller.send_state(ControllerState.neutral())
                self._pulse_released.set()
            return self._pulse_release_success

    def poll(self):
        if self.closed:
            return
        app = self.app
        if self.pending is not None:
            now = self.clock()
            if self.holding and (self._pulse_released.is_set() or now >= self.deadline):
                success = self._release_pulse()
                self._finish(result={"sent": True, "released": True},
                             error=None if success else "按键释放失败")
            elif now - self.started_at > 10:
                app.controller.send_state(ControllerState.neutral())
                self._finish(error="等待分析线程或重载完成超时，自动操作保持暂停")
            elif not self.holding and now >= self.deadline:
                thread = getattr(app, "_task_thread", None)
                if not (thread is not None and thread.is_alive()):
                    try:
                        app._collect_task_result()
                        self._pause()  # An in-flight tick may have published RUNNING after the first pause.
                        command = self.pending
                        name = command["command"]
                        if name == "press":
                            state = ControllerState(buttons=frozenset(Button(x) for x in command["buttons"]))
                            with self._pulse_lock:
                                self._pulse_released = threading.Event()
                                if not app.controller.send_state(state):
                                    raise RuntimeError(app.controller.last_error or "控制器按键发送失败")
                            self.holding = True
                            self.deadline = now + command["hold_ms"] / 1000.0
                            self._release_timer = threading.Timer(command["hold_ms"] / 1000.0,
                                                                  self._release_pulse,
                                                                  args=(self._pulse_released,))
                            self._release_timer.daemon = True
                            self._release_timer.start()
                            app.log_message(f"本机接口按键: {'+'.join(command['buttons'])} {command['hold_ms']}ms")
                        elif name == "reload":
                            app.reload_analysis_core()
                            self.resume_debug = False
                            if getattr(app, "_reload_context", None) is None:
                                if app.status_var.get() == "CORE_RELOAD_ERROR":
                                    raise RuntimeError(app.message_var.get())
                                self._finish(result={"core_generation": app._core_generation, "paused": True})
                        elif name == "resume":
                            if not app.controller.connected or app.worker is None:
                                raise RuntimeError("继续任务需要已连接的控制器和采集")
                            executor = app.battle_debug_executor if self.resume_debug else app.runner.battle_executor
                            phase = getattr(getattr(executor, "phase", None), "value", None)
                            if phase == "ERROR":
                                raise RuntimeError("执行器已停止，需要先重载核心再继续任务")
                            if self.resume_debug and app.battle_debug_executor is not None:
                                app.battle_debug_active = True
                                self.resume_debug = False
                            else:
                                if app.runner.task is None:
                                    raise RuntimeError("未配置任务，不能继续自动操作")
                                app.runner.resume()
                                if getattr(app.runner.status, "value", None) != "RUNNING":
                                    raise RuntimeError("当前任务无法继续")
                                app._update_snapshot(app.runner.snapshot())
                            self._finish(result={"resumed": True})
                        else:
                            self._finish(result={"paused": True})
                    except Exception as error:
                        app.controller.send_state(ControllerState.neutral())
                        self._finish(error=str(error))
        self.server.drain(self.handle, max_commands=1, urgent_only=self.pending is not None)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._frame_stop.set()
        if self.pending is not None:
            self.app.controller.send_state(ControllerState.neutral())
            self._finish(error="任务终端已关闭")
        self.server.close()
        if self._frame_thread is not None:
            self._frame_thread.join(timeout=1)


def service_terminal_control(app):
    if getattr(app, "diagnostics", None) is None:
        return
    bridge = getattr(app, "_local_control_bridge", None)
    if bridge is None:
        if getattr(app, "_local_control_start_failed", False):
            return
        try:
            bridge = TerminalControlBridge(app)
            app._local_control_bridge = bridge
            app.root.bind("<Destroy>", lambda event: bridge.close() if event.widget is app.root else None, add="+")
            app.log_message("本机实时控制接口已启动；连接信息见 artifacts/live_task/control_endpoint.json")
        except Exception as error:
            app._local_control_start_failed = True
            app.log_message(f"本机实时控制接口启动失败: {error}")
            return
    bridge.poll()
