from tableturf_rebuild.application.battle_vision import BattleVisionObservation
from tableturf_rebuild.application.deck_config import DeckConfig
from tableturf_rebuild.application.state_machine import (
    GameState,
    StateObservation,
)
from tableturf_rebuild.application.task_plan import (
    NpcTaskDefinition,
    build_npc_plan,
)
from tableturf_rebuild.application.task_runner import TaskRunner, TaskStatus
from tableturf_rebuild.application.task_runner import _deck_signature_matches
from tableturf_rebuild.control.buttons import Button
from tableturf_rebuild.control.dummy import DummyBackend
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.npc_menu_reader import NpcMenuObservation
from tableturf_rebuild.vision.deck_reader import DeckGridReader
from pathlib import Path
import cv2


class FakeRuntime:
    def __init__(self, observation: BattleVisionObservation):
        self.catalog = GameCatalog.from_package()
        self.observation = observation
        self.expected_deck = None

    def update(self, frame, include_selection=False, force_selection=False):
        del frame, include_selection, force_selection
        return self.observation

    def set_expected_deck(self, name, card_ids):
        self.expected_deck = (name, card_ids)

    def clear_expected_deck(self):
        self.expected_deck = None


def _observation(state: GameState, menu=None) -> BattleVisionObservation:
    return BattleVisionObservation(
        state=StateObservation(
            state=state,
            confidence=0.9,
            candidates=(),
            actions=(),
            boxes=(),
        ),
        board=None,
        selection_profile=None,
        selection_candidates=(),
        menu=menu,
    )


def test_deck_select_without_preset_waits_without_pressing_a() -> None:
    catalog = GameCatalog.from_package()
    npc = catalog.npcs["MiniGame_MiniGameKurage0"]
    menu = NpcMenuObservation(
        npc_id=npc.npc_id,
        npc_name=npc.name_zh,
        level_wins=(3, 3, 0),
        difficulty=3,
        stage_id="Rectangle",
        stage_name="正直大道",
        binding_consistent=True,
        name_confidence=1.0,
        wins_confidence=1.0,
        difficulty_confidence=1.0,
        stage_confidence=1.0,
    )
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.DECK_SELECT, menu)), backend)
    runner.configure(
        NpcTaskDefinition(
            npc_id=npc.npc_id,
            npc_name=npc.name_zh,
            target_level=3,
            target_wins=30,
        ),
        None,
    )
    runner.start()

    runner.tick(None, now=10.0)
    runner.tick(None, now=10.36)
    snapshot = runner.tick(None, now=10.72)

    assert snapshot.status is TaskStatus.WAITING_FOR_DECK
    assert backend.state.buttons == frozenset()


def test_redraw_uses_keep_current_hand_policy_once() -> None:
    catalog = GameCatalog.from_package()
    npc = catalog.npcs["MiniGame_MiniGameKurage0"]
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.REDRAW)), backend)
    runner.configure(
        NpcTaskDefinition(
            npc_id=npc.npc_id,
            npc_name=npc.name_zh,
            target_level=3,
            target_wins=30,
        ),
        None,
    )
    runner.start()

    waiting = runner.tick(None, now=10.0)
    first = runner.tick(None, now=11.0)
    second = runner.tick(None, now=11.36)
    third = runner.tick(None, now=11.72)

    pressed = [
        state.buttons
        for state in backend.history
        if state.buttons
    ]
    assert pressed == [
        frozenset({Button.DPAD_LEFT}),
        frozenset({Button.A}),
    ]
    assert waiting.message == "等待重抽弹窗和手牌稳定显示"
    assert first.message == "默认开局策略：向左选择维持现状"
    assert second.message == "确认维持现状"
    assert third.message == "已选择维持现状，等待发牌完成"


def test_npc_search_switches_to_up_after_down_stalls() -> None:
    catalog = GameCatalog.from_package()
    target = catalog.npcs["MiniGame_MiniGameKurage0"]
    other = catalog.npcs["MiniGame_Aori"]
    menu = NpcMenuObservation(
        npc_id=other.npc_id,
        npc_name=other.name_zh,
        level_wins=(0, 0, 0),
        difficulty=None,
        stage_id=None,
        stage_name=None,
        binding_consistent=None,
        name_confidence=1.0,
        wins_confidence=0.0,
        difficulty_confidence=0.0,
        stage_confidence=0.0,
    )
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.NPC_ROSTER, menu)), backend)
    runner.configure(
        NpcTaskDefinition(
            npc_id=target.npc_id,
            npc_name=target.name_zh,
            target_level=3,
            target_wins=30,
        ),
        None,
    )
    runner.start()

    for index in range(4):
        runner.tick(None, now=10.0 + index * 0.36)

    directions = [
        state.buttons
        for state in backend.history
        if state.buttons
        in {
            frozenset({Button.DPAD_DOWN}),
            frozenset({Button.DPAD_UP}),
        }
    ]
    assert directions == [
        frozenset({Button.DPAD_DOWN}),
        frozenset({Button.DPAD_DOWN}),
        frozenset({Button.DPAD_DOWN}),
        frozenset({Button.DPAD_UP}),
    ]


def test_replay_continues_when_same_difficulty_still_needs_wins() -> None:
    catalog = GameCatalog.from_package()
    npc = catalog.npcs["MiniGame_MiniGameKurage0"]
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.REPLAY_CHOICE)), backend)
    task = NpcTaskDefinition(
        npc_id=npc.npc_id,
        npc_name=npc.name_zh,
        target_level=3,
        target_wins=30,
    )
    runner.configure(task, None)
    runner.plan = build_npc_plan(task, (3, 3, 10))
    runner._last_match_outcome = "WIN"
    runner.start()

    runner.tick(None, now=10.0)
    runner.tick(None, now=10.36)

    pressed = [
        state.buttons
        for state in backend.history
        if state.buttons
    ]
    assert pressed == [
        frozenset({Button.DPAD_RIGHT}),
        frozenset({Button.A}),
    ]


def test_replay_exits_after_final_required_win() -> None:
    catalog = GameCatalog.from_package()
    npc = catalog.npcs["MiniGame_MiniGameKurage0"]
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.REPLAY_CHOICE)), backend)
    task = NpcTaskDefinition(
        npc_id=npc.npc_id,
        npc_name=npc.name_zh,
        target_level=3,
        target_wins=30,
    )
    runner.configure(task, None)
    runner.plan = build_npc_plan(task, (3, 3, 29))
    runner._last_match_outcome = "WIN"
    runner.start()

    runner.tick(None, now=10.0)
    runner.tick(None, now=10.36)

    pressed = [
        state.buttons
        for state in backend.history
        if state.buttons
    ]
    assert pressed == [
        frozenset({Button.DPAD_LEFT}),
        frozenset({Button.A}),
    ]


def test_match_setup_moves_toward_active_unlock_level() -> None:
    catalog = GameCatalog.from_package()
    npc = catalog.npcs["MiniGame_MiniGameKurage0"]
    menu = NpcMenuObservation(
        npc_id=npc.npc_id,
        npc_name=npc.name_zh,
        level_wins=(3, 0, 0),
        difficulty=1,
        stage_id="Rectangle",
        stage_name="正直大道",
        binding_consistent=True,
        name_confidence=1.0,
        wins_confidence=1.0,
        difficulty_confidence=1.0,
        stage_confidence=1.0,
    )
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.MATCH_SETUP, menu)), backend)
    runner.configure(
        NpcTaskDefinition(
            npc_id=npc.npc_id,
            npc_name=npc.name_zh,
            target_level=3,
            target_wins=30,
        ),
        DeckConfig("测试牌组", catalog.decks["MiniGame_Starter"].card_ids),
    )
    runner.start()

    runner.tick(None, now=10.0)
    runner.tick(None, now=10.36)
    snapshot = runner.tick(None, now=10.72)

    assert snapshot.plan is not None
    assert snapshot.plan.active_stage is not None
    assert snapshot.plan.active_stage.level == 2
    assert any(
        Button.DPAD_RIGHT in state.buttons
        for state in backend.history
    )


def test_win_counts_ignore_impossible_transient_ocr_jump() -> None:
    catalog = GameCatalog.from_package()
    npc = catalog.npcs["MiniGame_MiniGameKurage0"]
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.NPC_ROSTER)), backend)
    runner.configure(
        NpcTaskDefinition(
            npc_id=npc.npc_id,
            npc_name=npc.name_zh,
            target_level=3,
            target_wins=30,
        ),
        None,
    )

    def menu(wins):
        return NpcMenuObservation(
            npc_id=npc.npc_id,
            npc_name=npc.name_zh,
            level_wins=wins,
            difficulty=None,
            stage_id=None,
            stage_name=None,
            binding_consistent=None,
            name_confidence=1.0,
            wins_confidence=1.0,
            difficulty_confidence=0.0,
            stage_confidence=0.0,
        )

    for _ in range(3):
        stable = runner._confirmed_level_wins(menu((0, 0, 0)))
    assert stable == (0, 0, 0)

    for _ in range(3):
        invalid = runner._confirmed_level_wins(menu((0, 1, 0)))
    assert invalid == (0, 0, 0)

    for _ in range(3):
        changed_without_match = runner._confirmed_level_wins(
            menu((1, 0, 0))
        )
    assert changed_without_match == (0, 0, 0)

    runner._wins_change_allowed = True
    for _ in range(3):
        changed_after_win = runner._confirmed_level_wins(menu((1, 0, 0)))
    assert changed_after_win == (1, 0, 0)


def test_win_counts_from_other_npc_never_build_task_plan() -> None:
    catalog = GameCatalog.from_package()
    target = catalog.npcs["MiniGame_MiniGameKurage0"]
    other = catalog.npcs["MiniGame_Aori"]
    menu = NpcMenuObservation(
        npc_id=other.npc_id,
        npc_name=other.name_zh,
        level_wins=(3, 0, 0),
        difficulty=None,
        stage_id=None,
        stage_name=None,
        binding_consistent=None,
        name_confidence=1.0,
        wins_confidence=1.0,
        difficulty_confidence=0.0,
        stage_confidence=0.0,
    )
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.NPC_ROSTER, menu)), backend)
    runner.configure(
        NpcTaskDefinition(
            npc_id=target.npc_id,
            npc_name=target.name_zh,
            target_level=3,
            target_wins=30,
        ),
        None,
    )
    runner.start()

    for index in range(3):
        snapshot = runner.tick(None, now=10.0 + index * 0.36)

    assert snapshot.plan is None


def test_npc_dialogue_is_skipped_with_a() -> None:
    catalog = GameCatalog.from_package()
    npc = catalog.npcs["MiniGame_MiniGameKurage0"]
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.NPC_DIALOGUE)), backend)
    runner.configure(
        NpcTaskDefinition(
            npc_id=npc.npc_id,
            npc_name=npc.name_zh,
            target_level=3,
            target_wins=30,
        ),
        None,
    )
    runner.start()

    snapshot = runner.tick(None, now=10.0)

    assert "按A继续" in snapshot.message
    assert any(
        Button.A in state.buttons
        for state in backend.history
    )


def test_match_setup_moves_to_start_then_confirms_with_a() -> None:
    catalog = GameCatalog.from_package()
    npc = catalog.npcs["MiniGame_MiniGameKurage0"]
    menu = NpcMenuObservation(
        npc_id=npc.npc_id,
        npc_name=npc.name_zh,
        level_wins=(3, 3, 0),
        difficulty=3,
        stage_id="Rectangle",
        stage_name="正直大道",
        binding_consistent=True,
        name_confidence=1.0,
        wins_confidence=1.0,
        difficulty_confidence=1.0,
        stage_confidence=1.0,
    )
    backend = DummyBackend()
    backend.connect()
    runner = TaskRunner(FakeRuntime(_observation(GameState.MATCH_SETUP, menu)), backend)
    runner.configure(
        NpcTaskDefinition(
            npc_id=npc.npc_id,
            npc_name=npc.name_zh,
            target_level=3,
            target_wins=30,
        ),
        None,
    )
    runner.start()

    runner.tick(None, now=10.0)
    runner.tick(None, now=10.36)
    runner.tick(None, now=10.72)
    runner.tick(None, now=11.08)

    pressed = [
        state.buttons
        for state in backend.history
        if state.buttons
    ]
    assert pressed == [
        frozenset({Button.DPAD_DOWN}),
        frozenset({Button.A}),
    ]


def test_deck_signature_matches_custom_visible_deck() -> None:
    catalog = GameCatalog.from_package()
    frame = cv2.imread(
        "artifacts/state_capture/run-20261024-205207/001_state.png"
    )
    assert frame is not None
    reader = DeckGridReader(catalog, Path("profiles/vision"))
    visible = reader.read(frame)
    configured = DeckConfig(
        "自定义",
        tuple(card.card_id for card in visible.cards),
    )

    assert _deck_signature_matches(visible.cards, configured, catalog)
