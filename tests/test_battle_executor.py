import time
from dataclasses import replace
from pathlib import Path

import cv2

from tableturf_rebuild.application.battle_executor import (
    BattleExecutorPhase,
    VisionMctsBattleExecutor,
    _path_movement_button,
)
from tableturf_rebuild.application.battle_vision import BattleVisionObservation
from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.state_machine import (
    GameState,
    StateMachine,
    StateObservation,
)
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.board_reader import (
    BoardCellReading,
    BoardGridReader,
    BoardObservation,
)
from tableturf_rebuild.vision.hand_reader import (
    HandCardObservation,
    HandObservation,
)
from tableturf_rebuild.vision.selection_profiles import SelectionProfileMatch


ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = (
    ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-221749"
    / "session-20261021-222704_annotations"
)


class ExecutorRuntime:
    def __init__(self, catalog: GameCatalog):
        self.catalog = catalog
        self.stage_id = "Rectangle"


def test_executor_places_when_preview_matches_mcts_action() -> None:
    catalog = GameCatalog.from_package()
    board_reader, profile, region, reference = BoardGridReader.from_profile(
        catalog,
        ROOT / "profiles" / "vision" / "对战场地-正直大道.json",
        "Rectangle",
    )
    frame = cv2.imread(
        str(
            ROOT
            / "artifacts"
            / "state_capture"
            / "run-20261021-110258"
            / "008_state.png"
        )
    )
    assert frame is not None
    board = board_reader.read(
        frame,
        region,
        "Rectangle",
        reference,
        profile,
    )
    observation = BattleVisionObservation(
        state=StateObservation(
            state=GameState.TARGET_PLACEMENT,
            confidence=0.9,
            candidates=(),
            actions=(),
            boxes=(),
        ),
        board=board,
        selection_profile=None,
        selection_candidates=(),
    )
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )
    executor._selected_action = Action.place(
        "MiniGame_Denchinamazu",
        0,
        (18, 2),
    )

    executor.step(observation, frame)
    executor.step(observation, frame)
    executor.step(observation, frame)

    assert executor.phase is BattleExecutorPhase.PLACING
    assert any(
        Button.A in state.buttons
        for state in backend.history
    )


def test_executor_does_not_confirm_gray_invalid_preview() -> None:
    catalog = GameCatalog.from_package()
    board_reader, profile, region, reference = BoardGridReader.from_profile(
        catalog,
        ROOT / "profiles" / "vision" / "对战场地-正直大道.json",
        "Rectangle",
    )
    frame = cv2.imread(str(FRAME_DIR / "017_video_state.png"))
    assert frame is not None
    board = board_reader.read(
        frame,
        region,
        "Rectangle",
        reference,
        profile,
    )
    observation = BattleVisionObservation(
        state=StateObservation(
            state=GameState.TARGET_PLACEMENT,
            confidence=0.9,
            candidates=(),
            actions=(),
            boxes=(),
        ),
        board=board,
        selection_profile=None,
        selection_candidates=(),
    )
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )
    executor._selected_action = Action.place(
        "MiniGame_Mother",
        2,
        (20, 2),
    )

    executor.step(observation, frame)
    executor.step(observation, frame)

    assert not any(
        Button.A in state.buttons
        for state in backend.history
    )


def test_executor_does_not_guess_cursor_when_preview_cannot_be_read() -> None:
    catalog = GameCatalog.from_package()
    board_reader, profile, region, reference = BoardGridReader.from_profile(
        catalog,
        ROOT / "profiles" / "vision" / "对战场地-正直大道.json",
        "Rectangle",
    )
    board = board_reader.read(
        reference,
        region,
        "Rectangle",
        reference,
        profile,
    )
    observation = BattleVisionObservation(
        state=StateObservation(
            state=GameState.TARGET_PLACEMENT,
            confidence=0.9,
            candidates=(),
            actions=(),
            boxes=(),
        ),
        board=board,
        selection_profile=None,
        selection_candidates=(),
    )
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )
    executor._selected_action = Action.place(
        "MiniGame_Mother",
        2,
        (20, 2),
    )

    executor.step(observation, reference)
    executor.step(observation, reference)
    executor.step(observation, reference)

    assert not any(state.buttons for state in backend.history)
    for _ in range(9):
        executor.step(observation, reference)
    assert [state.buttons for state in backend.history if state.buttons] == [
        frozenset({Button.B})
    ]


def test_executor_returns_with_b_when_placement_has_no_action() -> None:
    catalog = GameCatalog.from_package()
    observation = BattleVisionObservation(
        state=StateObservation(
            state=GameState.TARGET_PLACEMENT,
            confidence=0.9,
            candidates=(),
            actions=(),
            boxes=(),
        ),
        board=None,
        selection_profile=None,
        selection_candidates=(),
    )
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )

    executor.step(observation, None)

    assert executor.phase is BattleExecutorPhase.WAITING
    assert "连续按3次B返回选牌界面" in executor.message
    assert any(
        Button.B in state.buttons
        for state in backend.history
    )


def test_executor_tries_back_only_three_times_to_return_to_select() -> None:
    catalog = GameCatalog.from_package()
    observation = BattleVisionObservation(
        state=StateObservation(
            state=GameState.TARGET_PLACEMENT,
            confidence=0.9,
            candidates=(),
            actions=(),
            boxes=(),
        ),
        board=None,
        selection_profile=None,
        selection_candidates=(),
    )
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )

    for _ in range(4):
        executor._return_to_select_at = 0.0
        executor.step(observation, None)

    b_presses = [
        state
        for state in backend.history
        if Button.B in state.buttons
    ]
    assert len(b_presses) == 3
    assert executor._return_to_select_attempts == 3
    assert executor._force_selection_after_b is True
    assert "等待四张手牌" in executor.message


def test_executor_requests_selection_data_before_card_is_confirmed() -> None:
    catalog = GameCatalog.from_package()
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )

    assert executor.requires_selection_data is True

    executor._card_confirmed = True

    assert executor.requires_selection_data is False


def test_executor_rejects_low_confidence_selection_hand() -> None:
    catalog = GameCatalog.from_package()
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )
    hand = replace(
        _hand_observation(catalog, "MiniGame_SaberLight00"),
        confidence=0.20,
    )
    selection = SelectionProfileMatch(
        variant="focus_1_2",
        focus_slot=(1, 2),
        score=0.98,
        profile_path=ROOT / "profiles" / "vision" / "battle_select12.json",
    )

    executor.step(
        BattleVisionObservation(
            state=_state_observation(GameState.TURN_SELECT),
            board=None,
            selection_profile=selection,
            selection_candidates=(selection,),
            hand=hand,
        ),
        None,
    )

    assert executor.phase is BattleExecutorPhase.WAITING
    assert executor._selected_action is None
    assert "手牌匹配分数=0.20" in executor.message


def test_executor_completes_select_rotate_move_and_place_cycle() -> None:
    catalog = GameCatalog.from_package()
    card_id = "MiniGame_SaberLight00"
    target = Action.place(card_id, 0, (10, 5))
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )
    selection_state = _state_observation(GameState.TURN_SELECT)
    executor.step(
        BattleVisionObservation(
            state=selection_state,
            board=None,
            selection_profile=None,
            selection_candidates=(),
        ),
        None,
    )
    executor._selected_action = target
    executor._selected_card_id = target.card_id

    hand = _hand_observation(catalog, card_id)
    selection = SelectionProfileMatch(
        variant="focus_1_1",
        focus_slot=(1, 1),
        score=1.0,
        profile_path=ROOT / "profiles" / "vision" / "battle_select11.json",
    )
    executor.step(
        BattleVisionObservation(
            state=selection_state,
            board=None,
            selection_profile=selection,
            selection_candidates=(selection,),
            hand=hand,
        ),
        None,
    )

    selection = replace(selection, focus_slot=(1, 2))
    executor.step(
        BattleVisionObservation(
            state=selection_state,
            board=None,
            selection_profile=selection,
            selection_candidates=(selection,),
            hand=hand,
        ),
        None,
    )

    assert executor._card_confirmed is True
    settle_board = _preview_board(
        catalog,
        card_id,
        1,
        (8, 3),
        valid=False,
    )
    history_before_settle = len(backend.history)
    executor.step(
        BattleVisionObservation(
            state=_state_observation(GameState.TARGET_PLACEMENT),
            board=settle_board,
            selection_profile=None,
            selection_candidates=(),
            hand=hand,
        ),
        None,
    )
    assert len(backend.history) == history_before_settle
    assert "等待摆放界面" in executor.message

    executor._card_confirmed_at -= executor.PLACEMENT_SETTLE_SECONDS + 1.0

    rotation = 1
    origin = (8, 3)
    target_state = _state_observation(GameState.TARGET_PLACEMENT)
    for _ in range(40):
        if executor._place_sent:
            break
        board = _preview_board(
            catalog,
            card_id,
            rotation,
            origin,
            valid=(rotation % 2 == 0 and origin == (10, 5)),
        )
        history_before = len(backend.history)
        executor.step(
            BattleVisionObservation(
                state=target_state,
                board=board,
                selection_profile=None,
                selection_candidates=(),
                hand=hand,
            ),
            None,
        )
        new_buttons = {
            button
            for state in backend.history[history_before:]
            for button in state.buttons
        }
        if new_buttons:
            executor._preview_response_at = 0.0
        if Button.Y in new_buttons:
            rotation = (rotation + 1) % 4
        elif Button.X in new_buttons:
            rotation = (rotation - 1) % 4
        elif Button.DPAD_UP in new_buttons:
            origin = (origin[0] - 1, origin[1])
        elif Button.DPAD_DOWN in new_buttons:
            origin = (origin[0] + 1, origin[1])
        elif Button.DPAD_LEFT in new_buttons:
            origin = (origin[0], origin[1] - 1)
        elif Button.DPAD_RIGHT in new_buttons:
            origin = (origin[0], origin[1] + 1)

    assert executor._place_sent is True
    assert rotation % 2 == target.rotation % 2
    assert origin == target.origin
    assert executor.phase is BattleExecutorPhase.PLACING
    assert any(
        bool({Button.X, Button.Y} & state.buttons)
        for state in backend.history
    )
    assert any(
        state.buttons
        == frozenset(
            {
                Button.A,
            }
        )
        for state in backend.history[-4:]
    )


def test_executor_waits_for_preview_before_repeating_move() -> None:
    catalog = GameCatalog.from_package()
    card_id = "MiniGame_SaberLight00"
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )
    executor._selected_action = Action.place(card_id, 0, (10, 5))
    executor._selected_card_id = card_id
    executor._card_confirmed = True
    executor._card_confirmed_at = (
        time.perf_counter()
        - executor.PLACEMENT_SETTLE_SECONDS
        - 1.0
    )
    board = _preview_board(
        catalog,
        card_id,
        0,
        (8, 3),
        valid=False,
    )
    observation = BattleVisionObservation(
        state=_state_observation(GameState.TARGET_PLACEMENT),
        board=board,
        selection_profile=None,
        selection_candidates=(),
    )

    history_before = len(backend.history)
    for _ in range(3):
        executor.step(observation, None)
    first_move_count = len(backend.history) - history_before
    assert first_move_count > 0

    history_after_move = len(backend.history)
    for _ in range(4):
        executor.step(observation, None)

    assert len(backend.history) == history_after_move
    assert "等待预览响应" in executor.message


def test_action_anchor_uses_first_top_row_square() -> None:
    catalog = GameCatalog.from_package()
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )

    anchor = executor._action_anchor(
        Action.place("MiniGame_Ajio", 0, (10, 2))
    )

    assert anchor == (10, 4)


def test_path_movement_uses_the_route_with_fewer_walls() -> None:
    catalog = GameCatalog.from_package()
    board = _preview_board(
        catalog,
        "MiniGame_SaberLight00",
        0,
        (2, 2),
        valid=True,
    )
    rows = [list(row) for row in board.cells]
    rows[2][3] = Cell.WALL
    blocked_horizontal = replace(
        board,
        cells=tuple(tuple(row) for row in rows),
    )

    assert (
        _path_movement_button(
            blocked_horizontal,
            (2, 2),
            (4, 4),
        )
        is Button.DPAD_DOWN
    )

    rows = [list(row) for row in board.cells]
    rows[3][2] = Cell.WALL
    blocked_vertical = replace(
        board,
        cells=tuple(tuple(row) for row in rows),
    )

    assert (
        _path_movement_button(
            blocked_vertical,
            (2, 2),
            (4, 4),
        )
        is Button.DPAD_RIGHT
    )


def test_equivalent_rotations_are_deduplicated_and_canonicalized() -> None:
    catalog = GameCatalog.from_package()
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        ExecutorRuntime(catalog),
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
    )
    actions = tuple(
        Action.place("MiniGame_ShockSonar", rotation, (10, 2))
        for rotation in range(4)
    )

    unique = executor._deduplicate_rotations(actions)
    canonical = executor._canonicalize_rotation(actions[1])

    assert len(unique) == 1
    assert canonical.rotation == 0


def _state_observation(state: GameState) -> StateObservation:
    return StateObservation(
        state=state,
        confidence=0.9,
        candidates=(),
        actions=(),
        boxes=(),
    )


def _hand_observation(
    catalog: GameCatalog,
    target_card_id: str,
) -> HandObservation:
    card_ids = (
        "MiniGame_Denchinamazu",
        target_card_id,
        "MiniGame_Mother",
        "MiniGame_Ajio",
    )
    cards = []
    for index, card_id in enumerate(card_ids):
        card = catalog.cards[card_id]
        cards.append(
            HandCardObservation(
                slot_row=1 + index // 2,
                slot_col=1 + index % 2,
                card_id=card_id,
                name=card.name_zh,
                area=card.area,
                sp_cost=card.sp_cost,
                visible_cost_lights=0,
                match_score=1.0,
                candidates=(),
            )
        )
    return HandObservation(
        cards=tuple(cards),
        confidence=1.0,
        profile="test",
    )


def _preview_board(
    catalog: GameCatalog,
    card_id: str,
    rotation: int,
    origin: tuple[int, int],
    *,
    valid: bool,
) -> BoardObservation:
    rows = 26
    cols = 9
    shape = catalog.cards[card_id].shape.rotated(rotation)
    top = min(row for row, _ in shape.occupied)
    left = min(col for _, col in shape.occupied)
    preview_cells = {
        (origin[0] + row - top, origin[1] + col - left)
        for row, col in shape.occupied
    }
    special_cells = {
        (origin[0] + row - top, origin[1] + col - left)
        for row, col in shape.occupied if int(shape.cells[row][col]) == 2
    }
    readings = []
    for row in range(rows):
        for col in range(cols):
            is_preview = (row, col) in preview_cells
            readings.append(
                BoardCellReading(
                    row=row,
                    col=col,
                    stage_cell=Cell.EMPTY,
                    inferred_cell=(
                        Cell.SELF_INK if is_preview else Cell.EMPTY
                    ),
                    mean_bgr=(0, 0, 0),
                    median_hsv=(0, 0, 0),
                    self_fraction=0.8 if is_preview and valid else 0.0,
                    opponent_fraction=0.0,
                    self_special_fraction=0.0,
                    opponent_special_fraction=0.0,
                    self_special_pattern_delta=0.0,
                    opponent_special_pattern_delta=0.0,
                    center_brightness_contrast=0.0,
                    center_saturation=0.0,
                    special_active=False,
                    color_change=0.8 if is_preview else 0.0,
                    edge_density=0.3 if is_preview else 0.0,
                    preview_diagonal_score=0.5 if is_preview else 0.0,
                    preview_dot_score=1.0 if (row, col) in special_cells else 0.0,
                )
            )
    return BoardObservation(
        stage_id="Rectangle",
        rows=rows,
        cols=cols,
        cells=tuple(
            tuple(Cell.EMPTY for _ in range(cols))
            for _ in range(rows)
        ),
        readings=tuple(readings),
        confidence=1.0,
        placement_preview_score=len(preview_cells) / (rows * cols),
        placement_preview_present=True,
        playable_cells=rows * cols,
        reference_available=True,
    )


def test_debug_executor_can_play_without_npc_context() -> None:
    catalog = GameCatalog.from_package()
    runtime = BattleVisionRuntime(
        catalog,
        ROOT / "profiles" / "vision",
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.configure_stage("Rectangle")
    runtime.set_expected_deck(
        "Starter",
        catalog.decks["MiniGame_Starter"].card_ids,
    )
    runtime.state_machine.force_state(GameState.TURN_SELECT)
    frame = cv2.imread(str(FRAME_DIR / "045_video_state.png"))
    assert frame is not None
    observation = runtime.update(frame, include_selection=True)
    backend = DummyBackend()
    backend.connect()
    executor = VisionMctsBattleExecutor(
        runtime,
        backend,
        catalog.decks["MiniGame_Starter"].card_ids,
        fallback_opponent_deck_id="MiniGame_Kurage0",
    )

    executor.step(observation, frame)

    # The first HUD observation is deliberately unpublished until an
    # independent update confirms the balance; do not freeze a false zero.
    if executor._selected_action is None:
        observation = runtime.update(frame, include_selection=True)
        executor.step(observation, frame)

    assert executor._selected_action is not None
