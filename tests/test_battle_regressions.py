from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import pytest

from tableturf_rebuild.application.battle_executor import BattleExecutorPhase, VisionMctsBattleExecutor
from tableturf_rebuild.application.battle_vision import BattleVisionObservation, BattleVisionRuntime
from tableturf_rebuild.application.state_machine import GameState, StateMachine, StateObservation
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action, ActionType
from tableturf_rebuild.domain.card import CardShape, CardSpec
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.domain.rules import BattleRules
from tableturf_rebuild.domain.state import PlayerState, Side
from tableturf_rebuild.vision.board_reader import BoardGridReader
from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader
from tableturf_rebuild.vision.hand_reader import HandReader
from tableturf_rebuild.vision.selection_profiles import SelectionProfileResolver


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "profiles" / "vision"


@pytest.fixture(scope="module")
def catalog():
    return GameCatalog.from_package()


def observation(state, **kwargs):
    return BattleVisionObservation(
        state=StateObservation(state, 0.9, (), (), ()),
        board=None, selection_profile=None, selection_candidates=(), **kwargs,
    )


def executor_for(catalog):
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        SimpleNamespace(catalog=catalog, stage_id="Rectangle"), backend,
        catalog.decks["MiniGame_Starter"].card_ids,
        fallback_opponent_deck_id="MiniGame_Starter",
    )
    return executor, backend


def test_turn_count_advances_all_twelve_turns_and_resets_for_next_match(catalog):
    executor, _ = executor_for(catalog)
    for turn in range(12):
        executor.step(observation(GameState.TURN_SELECT), None)
        executor._place_sent = True
        executor.step(observation(GameState.TURN_REVEAL), None)
        executor.step(observation(GameState.TURN_REVEAL), None)
        executor.step(observation(GameState.TURN_RESOLVE), None)
        assert executor._completed_turns == turn + 1
    executor._revealed_own_cards.append("used")
    executor.step(observation(GameState.REDRAW), None)
    assert executor._completed_turns == 0
    assert executor._revealed_own_cards == []


def test_reveal_like_frames_cannot_consume_turns_without_a_committed_move(catalog):
    executor, _ = executor_for(catalog)
    for _ in range(15):
        executor.step(observation(GameState.TURN_REVEAL), None)
        executor.step(observation(GameState.TURN_SELECT), None)
    assert executor._completed_turns == 0


def test_skip_button_confirms_even_when_gray_hand_loses_its_arrow(catalog):
    from tableturf_rebuild.vision.selection_profiles import HandVisibilityObservation
    executor, backend = executor_for(catalog)
    executor._selected_action = Action.skip("MiniGame_SpGreatBarrier")
    executor._selected_card_id = executor._selected_action.card_id
    executor._skip_down_sent = True
    executor.step(observation(GameState.TARGET_PLACEMENT,
                              hand_visibility=HandVisibilityObservation(True, 135, 0, (3, 1))), None)
    assert executor._skip_activate_sent is True
    assert [s.buttons for s in backend.history if s.buttons] == [frozenset({Button.A})]
    assert executor._completed_turns == 0


def test_discard_template_before_skip_activation_cannot_discard_gray_card(catalog):
    executor, backend = executor_for(catalog)
    executor._last_state = GameState.TURN_SELECT
    executor._selected_action = Action.skip("MiniGame_StringerShort00")
    executor._selected_card_id = executor._selected_action.card_id
    obs = replace(selection_observation(executor, focus=(2, 1), available=False),
                  state=StateObservation(GameState.SKIP_DISCARD, 0.9, (), (), ()))
    executor.step(obs, None)
    assert executor._discard_confirm_sent is False
    assert executor._skip_down_sent is True
    assert [s.buttons for s in backend.history if s.buttons] == [frozenset({Button.DPAD_DOWN})]


def test_skip_flow_survives_gray_button_and_transient_selection_state(catalog):
    from tableturf_rebuild.vision.selection_profiles import HandVisibilityObservation
    executor, backend = executor_for(catalog)
    executor._last_state = GameState.TURN_SELECT
    executor._selected_action = Action.skip("MiniGame_StringerShort00")
    executor._selected_card_id = executor._selected_action.card_id
    executor.step(selection_observation(executor, focus=(2, 1), available=False), None)
    executor.step(observation(GameState.TARGET_PLACEMENT,
                              hand_visibility=HandVisibilityObservation(True, 135, 0, (3, 1))), None)
    executor.step(replace(selection_observation(executor), discard_mode=True), None)
    assert executor._discard_confirm_sent is True
    assert executor._completed_turns == 0
    executor.step(observation(GameState.TURN_REVEAL), None)
    assert executor._completed_turns == 1
    assert [s.buttons for s in backend.history if s.buttons] == [
        frozenset({Button.DPAD_DOWN}), frozenset({Button.A}), frozenset({Button.A})]


@pytest.mark.parametrize("focus, expected", [
    ((2, 1), Button.DPAD_DOWN),
    ((3, 2), Button.DPAD_LEFT),
    (None, None),
])
def test_skip_never_confirms_from_a_stale_down_marker(catalog, focus, expected):
    from tableturf_rebuild.vision.selection_profiles import HandVisibilityObservation
    executor, backend = executor_for(catalog)
    executor._selected_action = Action.skip("MiniGame_SpGreatBarrier")
    executor._skip_down_sent = True
    executor.step(observation(GameState.SKIP_DISCARD,
        hand_visibility=HandVisibilityObservation(True, 255, 1, focus)), None)
    presses = [s.buttons for s in backend.history if s.buttons]
    assert presses == ([frozenset({expected})] if expected is not None else [])
    assert executor._skip_activate_sent is False


def test_skip_uses_actual_button_focus_without_requiring_special_mode(catalog):
    from tableturf_rebuild.vision.selection_profiles import HandVisibilityObservation
    executor, backend = executor_for(catalog)
    executor._selected_action = Action.skip("MiniGame_SpGreatBarrier")
    assert executor._skip_down_sent is False
    executor.step(observation(GameState.SKIP_DISCARD, special_mode=None,
        hand_visibility=HandVisibilityObservation(True, 135, 0, (3, 1))), None)
    assert executor._skip_activate_sent is True
    assert [s.buttons for s in backend.history if s.buttons] == [frozenset({Button.A})]


@pytest.mark.parametrize("discard_mode", [False, None])
def test_skip_does_not_treat_bright_cards_as_discard_acknowledgement(catalog, discard_mode):
    executor, backend = executor_for(catalog)
    executor._selected_action = Action.skip("MiniGame_StringerShort00")
    executor._skip_activate_sent = True
    executor._skip_activate_at = time.perf_counter()
    executor.step(replace(selection_observation(executor), discard_mode=discard_mode), None)
    assert executor._discard_confirm_sent is False
    assert not backend.history


@pytest.mark.parametrize("focus, expected", [
    ((1, 2), Button.A), ((3, 1), Button.DPAD_UP),
])
def test_already_open_discard_dialog_does_not_activate_return_button(catalog, focus, expected):
    executor, backend = executor_for(catalog)
    executor._selected_action = Action.skip("MiniGame_StringerShort00")
    executor._selected_card_id = "MiniGame_StringerShort00"
    executor.step(replace(selection_observation(executor, focus=focus), discard_mode=True), None)
    assert executor._skip_activate_sent is True
    assert executor._discard_confirm_sent is (expected is Button.A)
    assert [s.buttons for s in backend.history if s.buttons] == [frozenset({expected})]


@pytest.mark.parametrize("discard_mode, expected", [
    (True, GameState.SKIP_DISCARD), (False, GameState.TURN_SELECT),
])
def test_skip_button_text_distinguishes_discard_from_button_focus(discard_mode, expected):
    from tableturf_rebuild.application.state_machine import StateEvidence, StateScore
    machine = StateMachine(stable_frames=4)
    machine.force_state(GameState.TURN_SELECT)
    raw = StateObservation(GameState.SKIP_DISCARD, .84, (
        StateScore(GameState.SKIP_DISCARD, .84),
        StateScore(GameState.TURN_SELECT, .77),
    ), (), ())
    evidence = StateEvidence(hand_cursor_visible=True, hand_selection_visible=True,
        battle_controls_visible=True, discard_mode=discard_mode)
    machine.update_observation(raw, evidence)
    assert machine.update_observation(raw, evidence).state is expected


def test_committed_discard_waits_through_placement_like_animation(catalog):
    executor, backend = executor_for(catalog)
    executor._selected_action = Action.skip("MiniGame_SpGreatBarrier")
    executor._discard_confirm_sent = True
    executor.step(observation(GameState.TARGET_PLACEMENT), None)
    assert executor.phase is BattleExecutorPhase.WAITING_RESULT
    assert executor._preview_missing_reads == 0
    assert not backend.history


def test_match_result_replaces_stale_preview_diagnostics(catalog):
    executor, backend = executor_for(catalog)
    executor.message = "未识别到可信斜线预览"
    executor._last_preview_attempts = [{"failure": "SHAPE_MISMATCH"}]
    executor.step(observation(GameState.MATCH_RESULT), None)
    assert executor.phase is BattleExecutorPhase.WAITING_RESULT
    assert "结算" in executor.message
    assert executor._last_preview_attempts == []
    assert not backend.history


def test_edges_without_diagonal_stripes_never_create_a_preview(catalog):
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog, VISION / "对战场地-正直大道.json", "Rectangle",
    )
    board = reader.read(reference, region, "Rectangle", reference, profile)
    board = replace(
        board, placement_preview_present=True,
        readings=tuple(replace(
            reading, edge_density=0.4, color_change=0.8,
            preview_diagonal_score=0.0,
        ) for reading in board.readings),
    )
    assert PlacementPreviewReader(catalog).read(board, "MiniGame_Mother") is None


def test_striped_preview_outside_board_keeps_its_negative_origin(catalog):
    frame = cv2.imread(str(next(
        (ROOT / "artifacts/task_debug/run-20261025-133539").rglob("*.png")
    )))
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog, VISION / "对战场地-扭转河.json", "SShape",
    )
    board = reader.read(frame, region, "SShape", reference, profile)
    preview = PlacementPreviewReader(catalog).read_frame(
        frame, region, "SShape", "MiniGame_ShockSonar", board,
    )
    assert preview is not None
    assert preview.origin == (20, -1)
    assert len(preview.cells) == 12
    assert preview.valid is False


@pytest.mark.parametrize("sample, visible", [
    ("006_state.png", True), ("007_state.png", False), ("008_state.png", False),
])
def test_hand_visibility_does_not_depend_on_preview_being_yellow(sample, visible):
    frame = cv2.imread(str(ROOT / "artifacts/state_capture/run-20261021-110258" / sample))
    assert frame is not None
    assert SelectionProfileResolver.default().hand_visibility(frame).selection_visible is visible


def test_bright_hand_corrects_placement_even_after_card_confirmation(catalog):
    runtime = BattleVisionRuntime(catalog, VISION, state_machine=StateMachine(stable_frames=1))
    runtime.configure_stage("SShape")
    runtime.state_machine.force_state(GameState.TARGET_PLACEMENT)
    frame = cv2.imread(str(next(
        (ROOT / "artifacts/task_debug/run-20261025-103808").rglob("*.png")
    )))
    result = runtime.update(frame, include_selection=True, force_selection=False)
    assert result.hand_visibility.selection_visible is True
    assert result.state.state is GameState.TURN_SELECT
    assert runtime.state_machine.current is GameState.TURN_SELECT


def test_battle_state_uses_selection_snapshot_and_cached_hand(catalog):
    executor, _ = executor_for(catalog)
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog, VISION / "对战场地-正直大道.json", "Rectangle",
    )
    clean = reader.read(reference, region, "Rectangle", reference, profile)
    executor._selection_board = clean
    hand = catalog.decks["MiniGame_Starter"].card_ids[:4]
    executor._current_hand = SimpleNamespace(card_ids=hand)
    contaminated = replace(clean, cells=tuple(
        tuple(Cell.SELF_INK if cell is Cell.EMPTY else cell for cell in row)
        for row in clean.cells
    ))
    obs = replace(observation(GameState.TARGET_PLACEMENT), board=contaminated,
                  hand=SimpleNamespace(card_ids=("wrong",)))
    state = executor._build_battle_state(obs)
    assert state.board == clean.cells
    assert state.player_a.hand == hand


def test_retry_confirmation_rechecks_gray_preview_and_latches_error(catalog):
    executor, backend = executor_for(catalog)
    executor._selected_action = Action.place("MiniGame_ShockSonar", 0, (10, 1))
    executor._place_sent = True
    executor._place_sent_at = -10.0
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog, VISION / "对战场地-正直大道.json", "Rectangle",
    )
    board = reader.read(reference, region, "Rectangle", reference, profile)
    board = replace(board, placement_preview_present=True)
    preview = SimpleNamespace(rotation=0, origin=(10, 1), anchor=(10, 1),
                              valid=False, score=1.0)
    executor.placement_reader = SimpleNamespace(read=lambda *args: preview)
    obs = replace(observation(GameState.TARGET_PLACEMENT), board=board)
    for _ in range(6):
        executor.step(obs, None)
    assert executor.phase is BattleExecutorPhase.ERROR
    assert not any(Button.A in state.buttons for state in backend.history)


def test_pending_placement_is_cleared_when_next_selection_arrives(catalog):
    executor, backend = executor_for(catalog)
    executor._last_state = GameState.TARGET_PLACEMENT
    executor._selected_action = Action.place("MiniGame_ShockSonar", 0, (10, 1))
    executor._place_sent = True
    executor._place_sent_at = time.perf_counter() - 2.0
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog, VISION / "对战场地-正直大道.json", "Rectangle",
    )
    board = replace(reader.read(reference, region, "Rectangle", reference, profile),
                    placement_preview_present=True)
    preview = SimpleNamespace(rotation=0, origin=(10, 1), anchor=(10, 1),
                              valid=True, score=1.0)
    executor.placement_reader = SimpleNamespace(read=lambda *args: preview)
    executor.step(replace(observation(GameState.TARGET_PLACEMENT), board=board), None)
    assert executor._place_sent is True  # retry still waiting for stable frames
    assert not any(Button.A in state.buttons for state in backend.history)
    executor.step(observation(GameState.TURN_SELECT), None)
    assert executor._selected_action is None
    assert executor._place_sent is False
    assert executor._completed_turns == 1  # reveal animation can be missed


def test_normal_placement_requires_empty_cells_and_own_eight_neighbor(catalog):
    # A one-cell card isolates the rule from rotation and shape matching.
    card = CardSpec("probe", "probe", "测试", 1, CardShape.from_matrix([[1]]), 1)
    catalog = replace(catalog, cards={**catalog.cards, card.card_id: card})
    rules = BattleRules(catalog)
    state = rules.with_hand("Square", (card.card_id,))
    grid = [[Cell.EMPTY for _ in row] for row in state.board]
    grid[5][5] = Cell.SELF_SPECIAL
    grid[4][5] = Cell.OPPONENT_INK
    state = replace(state, board=tuple(map(tuple, grid)),
                    player_a=PlayerState(Side.A, (card.card_id,), ()))
    places = {a for a in rules.legal_actions(state, Side.A) if a.action_type is ActionType.PLACE}
    assert Action.place(card.card_id, 0, (4, 4)) in places  # diagonal contact
    assert Action.place(card.card_id, 0, (4, 5)) not in places  # occupied
    assert Action.place(card.card_id, 0, (5, 5)) not in places  # own special
    assert Action.place(card.card_id, 0, (3, 5)) not in places  # enemy-only contact
    assert Action.place(card.card_id, 0, (1, 1)) not in places  # isolated


def test_special_tile_distinguishes_symmetric_shape_and_missing_first_stripe(catalog):
    card = CardSpec("probe", "probe", "测试", 1, CardShape.from_matrix([[2, 1, 1]]), 3)
    catalog = replace(catalog, cards={**catalog.cards, card.card_id: card})
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog, VISION / "对战场地-正直大道.json", "Rectangle",
    )
    clean = reader.read(reference, region, "Rectangle", reference, profile)
    for rotation, special_col in ((0, 2), (2, 4)):
        board = replace(clean, readings=tuple(replace(
            reading, color_change=0.8,
            self_special_fraction=0.9 if (reading.row, reading.col) == (10, special_col) else 0,
            median_hsv=(20, 255, 255) if (reading.row, reading.col) == (10, special_col) else (32, 255, 255),
            preview_diagonal_score=0.5 if reading.row == 10 and reading.col in {2, 3, 4} - {special_col} else 0,
        ) for reading in clean.readings))
        preview = PlacementPreviewReader(catalog).read(board, card.card_id)
        assert preview is not None
        assert preview.rotation == rotation
        assert preview.origin == (10, 2)
        assert preview.anchor == (10, 2)


def test_task_runner_resets_battle_executor_on_new_redraw(catalog):
    from tableturf_rebuild.application.task_plan import NpcTaskDefinition
    from tableturf_rebuild.application.task_runner import TaskRunner

    executor, backend = executor_for(catalog)
    runtime = SimpleNamespace(
        update=lambda *args, **kwargs: observation(GameState.REDRAW),
        clear_expected_deck=lambda: None,
    )
    runner = TaskRunner(runtime, backend, battle_executor=executor)
    npc = next(iter(catalog.npcs.values()))
    runner.configure(NpcTaskDefinition(npc.npc_id, npc.name_zh, 3, 30), None)
    runner.start()
    executor._completed_turns = 12
    executor._revealed_opponent_cards.append("old")
    runner.tick(None, now=10)
    assert executor._completed_turns == 0
    assert executor._revealed_opponent_cards == []
    executor._completed_turns = 1
    runner.tick(None, now=10.1)
    assert executor._completed_turns == 1  # same redraw frame does not reset again


def test_grid_change_discards_old_plan_and_returns_to_card_selection(catalog):
    executor, backend = executor_for(catalog)
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog, VISION / "对战场地-正直大道.json", "Rectangle",
    )
    old = reader.read(reference, region, "Rectangle", reference, profile)
    executor._selection_board = old
    executor._selected_action = Action.place("MiniGame_ShockSonar", 0, (10, 1))
    executor._card_confirmed = True
    new = replace(old, stage_id="SShape")
    executor.step(replace(observation(GameState.TARGET_PLACEMENT), board=new), None)
    assert executor._selected_action is None
    assert executor._selection_board is None
    assert [s.buttons for s in backend.history if s.buttons] == [frozenset({Button.B})]


@pytest.mark.parametrize("sample, availability, second_card", [
    ("blocked-card", [False, True, True, True], "MiniGame_StringerShort00"),
    ("gray-hand", [False, False, True, True], "MiniGame_TakoStrike"),
])
def test_gray_selected_card_uses_white_arrow_and_keeps_its_shape(catalog, sample, availability, second_card):
    frame = cv2.imread(str(ROOT / f"artifacts/diagnostics/{sample}/analysis.png"))
    assert frame is not None
    resolver = SelectionProfileResolver.default()
    focus = resolver.resolve(frame)
    assert focus.focus_slot == (1, 1)
    assert resolver.hand_visibility(frame).cursor_slot == (1, 1)
    hand = HandReader(catalog, VISION).read(frame, focus.profile_path, tuple(catalog.cards))
    assert [card.available for card in hand.cards] == availability
    assert hand.cards[0].card_id == "MiniGame_SpGreatBarrier"
    assert hand.cards[1].card_id == second_card
    assert hand.confidence == 1.0


def selection_observation(executor, *, focus=(1, 2), available=True):
    card_id = "MiniGame_StringerShort00"
    hand = SimpleNamespace(
        cards=(SimpleNamespace(card_id=card_id, name="LACT-450", slot_row=1,
                               slot_col=2, available=available),),
        card_ids=(card_id,), confidence=1.0,
    )
    return replace(observation(GameState.TURN_SELECT), hand=hand,
                   selection_profile=SimpleNamespace(focus_slot=focus))


def test_pending_selection_checks_actual_focus_before_retrying_a(catalog):
    executor, backend = executor_for(catalog)
    executor._selected_action = Action.place("MiniGame_StringerShort00", 0, (1, 1))
    executor._selected_card_id = executor._selected_action.card_id
    executor._card_confirmed = True
    executor._card_confirmed_at = -10.0
    executor._card_confirm_attempts = 1
    executor._choose_and_select_card(selection_observation(executor, focus=(1, 1)))
    assert [s.buttons for s in backend.history if s.buttons] == [frozenset({Button.DPAD_RIGHT})]
    assert executor._card_confirmed is False


def test_selection_confirmation_is_bounded_and_latches_error(catalog):
    executor, backend = executor_for(catalog)
    executor._last_state = GameState.TURN_SELECT
    executor._selected_action = Action.place("MiniGame_StringerShort00", 0, (1, 1))
    executor._selected_card_id = executor._selected_action.card_id
    for _ in range(10):
        executor._card_confirmed_at = -10.0
        executor.step(selection_observation(executor), None)
    assert executor.phase is BattleExecutorPhase.ERROR
    assert executor._card_confirm_attempts == 3
    assert sum(Button.A in state.buttons for state in backend.history) == 3


@pytest.mark.parametrize("all_gray", [False, True])
def test_gray_cards_are_filtered_before_planning_and_all_gray_can_skip(catalog, all_gray):
    executor, backend = executor_for(catalog)
    obs = selection_observation(executor, focus=(2, 1) if all_gray else (1, 2),
                                available=not all_gray)
    gray_id = "MiniGame_SpGreatBarrier"
    gray = SimpleNamespace(card_id=gray_id, name="巨大防护罩", slot_row=1,
                           slot_col=1, available=False)
    obs = replace(obs, hand=SimpleNamespace(
        cards=(gray, *obs.hand.cards), card_ids=(gray_id, *obs.hand.card_ids), confidence=1.0))
    actions = (Action.place(gray_id, 0, (1, 1)),
               Action.place("MiniGame_StringerShort00", 0, (1, 1)),
               Action.skip("MiniGame_StringerShort00"))
    executor.rules = SimpleNamespace(legal_actions=lambda *args: actions)
    executor._build_battle_state = lambda *args: object()
    planned = []
    executor.policy = SimpleNamespace(
        set_opponent_context=lambda *args: None,
        choose=lambda rules, state, side, choices: planned.extend(choices) or choices[0])
    executor._choose_and_select_card(obs)
    if all_gray:
        assert planned == []
        assert executor._selected_action.action_type is ActionType.SKIP
        assert any(Button.DPAD_DOWN in s.buttons for s in backend.history)
    else:
        assert planned == [actions[1]]
        assert executor._selected_action.card_id == "MiniGame_StringerShort00"
