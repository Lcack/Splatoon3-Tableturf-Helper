from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from tableturf_rebuild.control.backend import ControllerBackend
from tableturf_rebuild.control.buttons import Button, ControllerState


DEFAULT_PRESS_SECONDS = 0.05
DEFAULT_ACTION_GAP_SECONDS = 0.35


def pair_virtual_controller(
    backend: ControllerBackend,
    *,
    press_seconds: float = DEFAULT_PRESS_SECONDS,
    gap_seconds: float = 0.35,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if press_seconds <= 0 or gap_seconds <= 0:
        raise ValueError("pairing intervals must be positive")
    if not backend.connected:
        raise RuntimeError("controller is not connected")

    for _ in range(2):
        pulse_buttons(
            backend,
            (Button.L, Button.R),
            press_seconds,
            gap_seconds,
            sleep,
        )
    pulse_buttons(
        backend,
        (Button.A,),
        press_seconds,
        gap_seconds,
        sleep,
    )
    if not backend.send_state(ControllerState.neutral()):
        raise RuntimeError("failed to release controller after pairing")


def pulse_buttons(
    backend: ControllerBackend,
    buttons: Iterable[Button],
    press_seconds: float = DEFAULT_PRESS_SECONDS,
    gap_seconds: float = DEFAULT_ACTION_GAP_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if press_seconds <= 0 or gap_seconds <= 0:
        raise ValueError("button timing must be positive")
    if not backend.send_state(
        ControllerState(buttons=frozenset(buttons))
    ):
        raise RuntimeError("failed to press controller buttons")
    sleep(press_seconds)
    if not backend.send_state(ControllerState.neutral()):
        raise RuntimeError("failed to release controller buttons")
    sleep(gap_seconds)
