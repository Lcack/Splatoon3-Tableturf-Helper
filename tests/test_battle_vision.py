from pathlib import Path

import cv2

from tableturf_rebuild.application.battle_vision import BattleVisionRuntime
from tableturf_rebuild.application.deck_config import load_deck_config
from tableturf_rebuild.application.state_machine import GameState, StateMachine
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.planning.mcts import MCTSConfig, MCTSDecisionPolicy


ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = ROOT / "profiles" / "vision"
FRAME_DIR = (
    ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-221749"
    / "session-20261021-222704_annotations"
)
LIVE_DIR = ROOT / "artifacts" / "state_capture" / "run-20261023-101741"
TASK_DEBUG_DIR = (
    ROOT
    / "artifacts"
    / "task_debug"
    / "run-20261025-103808"
    / "screenshots"
)


def test_runtime_combines_state_preview_and_board() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.configure_stage("Rectangle")
    frame = cv2.imread(str(FRAME_DIR / "017_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame)

    assert observation.state.state in {
        GameState.TARGET_PLACEMENT,
        GameState.TURN_SELECT,
    }
    assert observation.board is not None
    assert observation.board.placement_preview_present is True
    assert observation.board.self_count >= 1
    assert observation.board.opponent_count >= 1
    assert observation.initial_board_locked is True


def test_runtime_resolves_selected_hand_geometry() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.configure_stage("Rectangle")
    frame = cv2.imread(str(FRAME_DIR / "042_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame, include_selection=True)

    assert observation.state.state is GameState.TURN_SELECT
    assert observation.selection_profile is not None
    assert observation.selection_profile.variant == "focus_1_2"
    assert observation.selection_profile.focus_slot == (1, 2)


def test_runtime_resolves_turn_select_focus_2_1() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.configure_stage("Rectangle")
    runtime.state_machine.force_state(GameState.TURN_SELECT)
    frame = cv2.imread(str(FRAME_DIR / "044_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame, include_selection=True)

    assert observation.state.state is GameState.TURN_SELECT
    assert observation.selection_profile is not None
    assert observation.selection_profile.variant == "focus_2_1"


def test_redraw_does_not_resolve_hand_selection_geometry() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.state_machine.force_state(GameState.REDRAW)
    frame = cv2.imread(str(FRAME_DIR / "016_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame, include_selection=True)

    assert observation.state.state is GameState.REDRAW
    assert observation.selection_profile is None
    assert observation.selection_candidates == ()


def test_preview_evidence_prevents_placement_resolve_confusion() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.configure_stage("Rectangle")
    runtime.state_machine.force_state(GameState.TARGET_PLACEMENT)
    video = (
        ROOT
        / "artifacts"
        / "state_capture"
        / "run-20261021-221749"
        / "session-20261021-222704.mp4"
    )
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 1971)
    ok, frame = capture.read()
    capture.release()
    assert ok

    observation = runtime.update(frame)

    assert observation.board is not None
    assert observation.board.placement_preview_present is True
    assert observation.state.state is GameState.TARGET_PLACEMENT


def test_menu_states_do_not_trigger_board_reading() -> None:
    for filename in ("001_video_state.png", "003_video_state.png", "013_video_state.png"):
        runtime = BattleVisionRuntime(
            GameCatalog.from_package(),
            VISION_DIR,
            state_machine=StateMachine(stable_frames=1),
        )
        runtime.configure_stage("Rectangle")
        frame = cv2.imread(str(FRAME_DIR / filename))
        assert frame is not None

        observation = runtime.update(frame)

        assert observation.board is None


def test_runtime_reads_npc_menu_fields() -> None:
    mcts = MCTSDecisionPolicy(
        config=MCTSConfig(simulations=1, time_limit_seconds=0.1),
        seed=3,
    )
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
        mcts_policy=mcts,
    )
    runtime.state_machine.force_state(GameState.MATCH_SETUP)
    frame = cv2.imread(str(FRAME_DIR / "003_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame)

    assert observation.menu is not None
    assert observation.menu.npc_name == "新手水母"
    assert observation.menu.level_wins == (3, 3, 3)
    assert observation.menu.difficulty == 3
    assert observation.opponent_context is not None
    assert observation.opponent_context.npc_id == "MiniGame_MiniGameKurage0"
    assert observation.opponent_context.difficulty == 3
    assert observation.opponent_context.stage_id == "Rectangle"
    assert mcts.opponent_context == observation.opponent_context


def test_runtime_auto_selects_registered_board_profile() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.state_machine.force_state(GameState.NPC_ROSTER)
    frame = cv2.imread(str(LIVE_DIR / "002_state.png"))
    assert frame is not None

    observation = runtime.update(frame)

    assert observation.menu is not None
    assert observation.menu.stage_id == "Painted"
    assert runtime.stage_id == "Painted"


def test_runtime_reads_deck_total_points() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.state_machine.force_state(GameState.DECK_SELECT)
    frame = cv2.imread(str(FRAME_DIR / "013_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame)

    assert observation.deck is not None
    assert len(observation.deck.cards) == 15
    assert observation.deck.total_area == 131
    assert observation.deck.total_sp_cost == 51


def test_runtime_reads_four_cards_during_redraw() -> None:
    catalog = GameCatalog.from_package()
    runtime = BattleVisionRuntime(
        catalog,
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.set_expected_deck(
        "Starter",
        catalog.decks["MiniGame_Starter"].card_ids,
    )
    runtime.state_machine.force_state(GameState.REDRAW)
    frame = cv2.imread(str(FRAME_DIR / "016_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame, include_selection=True)

    assert observation.hand is not None
    assert [card.area for card in observation.hand.cards] == [9, 5, 3, 8]
    assert observation.hand.confidence == 1.0


def test_force_selection_corrects_false_placement_state() -> None:
    catalog = GameCatalog.from_package()
    runtime = BattleVisionRuntime(
        catalog,
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    deck = load_deck_config(
        ROOT / "profiles" / "decks" / "NGA Suggestion.json",
        catalog,
    )
    runtime.configure_stage("Zigzag")
    runtime.set_expected_deck(deck.name, deck.card_ids)
    runtime.state_machine.force_state(GameState.TARGET_PLACEMENT)
    frame = cv2.imread(
        str(
            TASK_DEBUG_DIR
            / "state-20261025-103810-929807.png"
        )
    )
    assert frame is not None

    observation = runtime.update(
        frame,
        include_selection=True,
        force_selection=True,
    )

    assert observation.state.state is GameState.TURN_SELECT
    assert observation.board is not None
    assert observation.board.placement_preview_present is False
    assert observation.hand is not None
    assert observation.hand.confidence >= 0.90
    assert observation.selection_profile is not None
    assert observation.selection_profile.focus_slot == (1, 1)


def test_force_selection_keeps_real_placement_state_when_hand_is_hidden() -> None:
    catalog = GameCatalog.from_package()
    runtime = BattleVisionRuntime(
        catalog,
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    deck = load_deck_config(
        ROOT / "profiles" / "decks" / "NGA Suggestion.json",
        catalog,
    )
    runtime.configure_stage("Zigzag")
    runtime.set_expected_deck(deck.name, deck.card_ids)
    runtime.state_machine.force_state(GameState.TARGET_PLACEMENT)
    frame = cv2.imread(
        str(
            ROOT
            / "artifacts"
            / "task_debug"
            / "run-20261025-133539"
            / "screenshots"
            / "state-20261025-133541-490871.png"
        )
    )
    assert frame is not None

    observation = runtime.update(
        frame,
        include_selection=True,
        force_selection=True,
    )

    assert observation.state.state is GameState.TARGET_PLACEMENT
    assert observation.board is not None
    assert observation.board.placement_preview_present is True
    assert observation.selection_profile is not None
    assert observation.selection_profile.score >= 0.90
    assert observation.hand is not None
    assert observation.hand.confidence < 0.30


def test_runtime_refreshes_visible_deck_after_content_changes() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    first_frame = cv2.imread(str(FRAME_DIR / "013_video_state.png"))
    changed_frame = cv2.imread(str(FRAME_DIR / "012_video_state.png"))
    assert first_frame is not None and changed_frame is not None
    runtime.state_machine.force_state(GameState.DECK_SELECT)
    first = runtime.update(first_frame).deck
    assert first is not None

    for _ in range(7):
        runtime.state_machine.force_state(GameState.DECK_SELECT)
        latest = runtime.update(changed_frame).deck

    assert latest is not None
    assert latest.cards[0].card_id != first.cards[0].card_id


def test_runtime_accepts_preconfigured_custom_deck() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    catalog = GameCatalog.from_package()
    card_ids = catalog.decks["MiniGame_Starter"].card_ids
    runtime.set_expected_deck("测试牌组", card_ids)
    frame = cv2.imread(str(FRAME_DIR / "045_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame)

    assert observation.deck is not None
    assert observation.deck.deck_id == "测试牌组"
    assert observation.deck.source == "configured"
    assert tuple(card.card_id for card in observation.deck.cards) == card_ids
    assert observation.deck.total_area == 131


def test_runtime_reads_revealed_cards_during_turn_reveal() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.state_machine.force_state(GameState.DECK_SELECT)
    deck_frame = cv2.imread(str(FRAME_DIR / "013_video_state.png"))
    assert deck_frame is not None
    deck_observation = runtime.update(deck_frame)
    assert deck_observation.deck is not None

    runtime.state_machine.force_state(GameState.TURN_REVEAL)
    reveal_frame = cv2.imread(str(FRAME_DIR / "034_video_state.png"))
    assert reveal_frame is not None
    observation = runtime.update(reveal_frame)

    assert observation.reveal is not None
    assert observation.reveal.own.card_id == "MiniGame_SaberLight00"
    assert observation.reveal.own.area == 5
    assert observation.reveal.own.sp_cost == 2
    assert observation.reveal.opponent.card_id == "MiniGame_Mother"
    assert observation.reveal.opponent.area == 15
    assert observation.reveal.opponent.sp_cost == 5


def test_runtime_reports_match_loss() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.configure_stage("Rectangle")
    runtime.state_machine.force_state(GameState.MATCH_RESULT)
    frame = cv2.imread(str(FRAME_DIR / "011_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame)

    assert observation.match_result is not None
    assert observation.match_result.outcome.value == "LOSS"


def test_finishing_animation_does_not_read_board() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.configure_stage("Rectangle")
    runtime.state_machine.force_state(GameState.FINISHING)
    frame = cv2.imread(str(FRAME_DIR / "028_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame)

    assert observation.board is None


def test_runtime_reports_hud_scores_and_active_specials() -> None:
    runtime = BattleVisionRuntime(
        GameCatalog.from_package(),
        VISION_DIR,
        state_machine=StateMachine(stable_frames=1),
    )
    runtime.configure_stage("Rectangle")
    runtime.state_machine.force_state(GameState.TURN_SELECT)
    frame = cv2.imread(str(FRAME_DIR / "045_video_state.png"))
    assert frame is not None

    observation = runtime.update(frame)

    assert observation.score is not None
    assert observation.score.self_value == 52
    assert observation.score.opponent_value == 48
    assert observation.board is not None
    assert observation.board.self_special_count >= 5
    assert observation.board.opponent_special_count >= 3
    assert observation.board.self_active_special_count >= 1
    assert observation.board.opponent_active_special_count >= 1
