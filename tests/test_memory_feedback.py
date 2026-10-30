from types import SimpleNamespace

from tableturf_rebuild.application.battle_executor import VisionMctsBattleExecutor
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog


def executor():
    catalog = GameCatalog.from_package()
    return VisionMctsBattleExecutor(SimpleNamespace(catalog=catalog, stage_id="Square"),
        DummyBackend(), catalog.decks["MiniGame_Starter"].card_ids)


def revealed(card_id, confidence=1.0, margin=0.3):
    return SimpleNamespace(card_id=card_id, confidence=confidence,
        candidates=(SimpleNamespace(score=confidence), SimpleNamespace(score=confidence-margin)))


def test_single_or_ambiguous_reveal_cannot_poison_remaining_deck():
    bot = executor()
    own, opponent = "MiniGame_TakoPod", "MiniGame_BombCurling"
    bot._selected_card_id = own
    reading = SimpleNamespace(reveal=SimpleNamespace(own=revealed(own), opponent=revealed(opponent)))
    bot._observe_revealed_cards(reading)
    bot._observe_revealed_cards(reading)
    assert not bot._revealed_opponent_cards  # No committed live action.
    bot._place_sent = True
    bot._observe_revealed_cards(reading)
    assert not bot._revealed_opponent_cards
    bot._observe_revealed_cards(reading)
    assert not bot._revealed_opponent_cards  # Cached reader object is not a second sample.
    reading = SimpleNamespace(reveal=SimpleNamespace(own=revealed(own), opponent=revealed(opponent)))
    bot._observe_revealed_cards(reading)
    assert bot._revealed_own_cards == [own]
    assert bot._revealed_opponent_cards == [opponent]
    bot._observe_revealed_cards(reading)
    assert bot._revealed_opponent_cards == [opponent]


def test_alternating_low_confidence_and_tied_ids_stay_unknown():
    bot = executor()
    bot._place_sent = True
    bot._selected_card_id = "MiniGame_TakoPod"
    for opponent in (revealed("MiniGame_BombCurling", 0.7),
                     revealed("MiniGame_BombCurling", 1.0, 0.0),
                     revealed("MiniGame_BombCurling"), revealed("MiniGame_Tamaire")):
        bot._observe_revealed_cards(SimpleNamespace(reveal=SimpleNamespace(
            own=revealed("MiniGame_Tamaire"), opponent=opponent)))
    assert not bot._revealed_opponent_cards
    assert not bot._revealed_own_cards  # Contradicts the selected own card.


def test_reload_keeps_used_cards_and_turns_without_reviving_an_unsubmitted_action():
    from tableturf_rebuild.application.task_runner import TaskRunner
    from tableturf_rebuild.domain.actions import Action
    old = executor()
    own, opponent = "MiniGame_TakoPod", "MiniGame_BombCurling"
    old._completed_turns = 6
    old._revealed_own_cards = [own]
    old._revealed_opponent_cards = [opponent]
    old._selected_action = Action.special("MiniGame_SpGreatBarrier", 0, (11, 5))
    runner = object.__new__(TaskRunner)
    runner.task, runner.battle_executor = object(), old
    saved = runner.progress_checkpoint()
    fresh = executor()
    runner.battle_executor = fresh
    runner.restore_progress(saved)
    assert fresh._completed_turns == 6
    assert fresh._revealed_own_cards == [own] and fresh._revealed_opponent_cards == [opponent]
    assert fresh._selected_action is None and not fresh._place_sent
    fresh._completed_turns = 7
    runner.restore_progress(saved)
    assert fresh._completed_turns == 7


def test_live_runtime_keeps_special_cell_through_actual_flame_frames():
    import cv2
    from pathlib import Path
    from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
    from tableturf_rebuild.domain.grid import Cell
    runtime = BattleVisionRuntime.default()
    runtime.configure_stage("Cross")
    directory = Path(__file__).resolve().parents[1] / "artifacts/diagnostics/current-special-stall"
    observed = []
    for index in (0, 1, 2):
        image = cv2.imread(str(directory / f"clear-{index}.png"))
        assert image is not None
        obs = runtime.update(image, include_selection=True)
        assert obs.board is not None
        observed.append(obs.board.cells[15][6])
    assert observed[-2:] == [Cell.SELF_SPECIAL, Cell.SELF_SPECIAL]
    runtime.reset()
    assert not runtime.special_cell_memory._cells
