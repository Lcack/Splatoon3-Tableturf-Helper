from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.control.sequences import (
    pair_virtual_controller,
    pulse_buttons,
)


def test_virtual_controller_pairing_sequence() -> None:
    backend = DummyBackend()
    assert backend.connect()
    sleeps = []

    pair_virtual_controller(
        backend,
        press_seconds=0.05,
        gap_seconds=0.35,
        sleep=sleeps.append,
    )

    pressed = [
        state.buttons
        for state in backend.history
        if state.buttons
    ]
    assert pressed == [
        frozenset({Button.L, Button.R}),
        frozenset({Button.L, Button.R}),
        frozenset({Button.A}),
    ]
    assert sleeps == [
        0.05,
        0.35,
        0.05,
        0.35,
        0.05,
        0.35,
    ]
    assert backend.state.buttons == frozenset()


def test_automation_button_uses_50ms_hold_and_350ms_gap() -> None:
    backend = DummyBackend()
    assert backend.connect()
    sleeps = []

    pulse_buttons(
        backend,
        (Button.A,),
        sleep=sleeps.append,
    )

    assert sleeps == [0.05, 0.35]
    assert backend.state.buttons == frozenset()
