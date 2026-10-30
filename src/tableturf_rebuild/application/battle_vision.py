from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from tableturf_rebuild.application.state_machine import (
    GameState,
    StateEvidence,
    StateMachine,
    StateObservation,
    StateScore,
    STATE_ACTIONS,
    STATE_BOXES,
)
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.planning.context import OpponentContext
from tableturf_rebuild.planning.mcts import MCTSDecisionPolicy
from tableturf_rebuild.vision.board_reader import (
    ActiveSpecialObservation,
    BoardGridReader,
    BoardObservation,
    SpecialActivityTracker,
    SpecialCellMemory,
)
from tableturf_rebuild.vision.calibration import VisionProfile, VisionRegion
from tableturf_rebuild.vision.deck_reader import (
    DeckFingerprint,
    DeckGridReader,
    DeckObservation,
    DeckSlotCandidate,
    DeckSlotObservation,
)
from tableturf_rebuild.vision.match_result_reader import (
    MatchOutcome,
    MatchResultObservation,
    MatchResultReader,
)
from tableturf_rebuild.vision.hand_reader import HandObservation, HandReader
from tableturf_rebuild.vision.legacy_grid import fit_empty_cell_grid
from tableturf_rebuild.vision.npc_menu_reader import (
    NpcMenuObservation,
    NpcMenuReader,
)
from tableturf_rebuild.vision.score_reader import (
    ScoreObservation,
    ScoreTemplateReader,
)
from tableturf_rebuild.vision.sp_reader import SpObservation, SpReader
from tableturf_rebuild.vision.battle_mode_reader import BattleModeObservation, BattleModeReader
from tableturf_rebuild.vision.skip_mode_reader import SkipModeObservation, SkipModeReader
from tableturf_rebuild.vision.replay_choice_reader import ReplayChoiceObservation, ReplayChoiceReader
from tableturf_rebuild.vision.reward_page_reader import RewardPageObservation, RewardPageReader
from tableturf_rebuild.vision.selection_profiles import (
    HandVisibilityObservation,
    SelectionProfileMatch,
    SelectionProfileResolver,
)
from tableturf_rebuild.vision.reveal_reader import (
    RevealObservation,
    RevealReader,
)


BOARD_READ_STATES = {
    GameState.TURN_SELECT,
    GameState.SKIP_DISCARD,
    GameState.SPECIAL_SELECT,
    GameState.TARGET_PLACEMENT,
    GameState.SPECIAL_PLACEMENT,
    GameState.TURN_REVEAL,
    GameState.TURN_RESOLVE,
}

CARD_SELECTION_STATES = {
    GameState.TURN_SELECT,
    GameState.SKIP_DISCARD,
    GameState.SPECIAL_SELECT,
}

HAND_READ_STATES = {
    GameState.REDRAW,
    GameState.TURN_SELECT,
    GameState.SKIP_DISCARD,
    GameState.SPECIAL_SELECT,
    GameState.TARGET_PLACEMENT,
}

SELECTION_RESOLUTION_STATES = CARD_SELECTION_STATES | {
    GameState.TARGET_PLACEMENT,
}

REVEAL_READ_STATES = {
    GameState.TURN_REVEAL,
    GameState.TURN_RESOLVE,
}


@dataclass(frozen=True, slots=True)
class BattleVisionObservation:
    state: StateObservation
    board: BoardObservation | None
    selection_profile: SelectionProfileMatch | None
    selection_candidates: tuple[SelectionProfileMatch, ...]
    initial_board_locked: bool = False
    match_result: MatchResultObservation | None = None
    score: ScoreObservation | None = None
    active_specials: ActiveSpecialObservation | None = None
    menu: NpcMenuObservation | None = None
    opponent_context: OpponentContext | None = None
    deck: DeckObservation | None = None
    reveal: RevealObservation | None = None
    hand: HandObservation | None = None
    hand_visibility: HandVisibilityObservation | None = None
    grid_fit: dict | None = None
    timings_ms: dict[str, float] | None = None
    sp: SpObservation | None = None
    special_mode: bool | None = None
    mode_evidence: BattleModeObservation | None = None
    discard_mode: bool | None = None
    discard_evidence: SkipModeObservation | None = None
    replay_choice_evidence: ReplayChoiceObservation | None = None
    reward_page_evidence: RewardPageObservation | None = None


class BattleVisionRuntime:
    SELECTION_OVERRIDE_HAND_CONFIDENCE = 0.65
    RESULT_OVERRIDE_CONFIDENCE = 0.85

    def __init__(
        self,
        catalog: GameCatalog,
        profile_dir: Path,
        state_machine: StateMachine | None = None,
        selection_resolver: SelectionProfileResolver | None = None,
        mcts_policy: MCTSDecisionPolicy | None = None,
    ):
        self.catalog = catalog
        self.profile_dir = profile_dir
        self.state_machine = state_machine or StateMachine()
        self.selection_resolver = (
            selection_resolver or SelectionProfileResolver(profile_dir)
        )
        self.mcts_policy = mcts_policy
        index = json.loads(
            (profile_dir / "index.json").read_text(encoding="utf-8")
        )
        self.board_profile_names = {
            str(stage): str(filename)
            for stage, filename in index.get("board_profiles", {}).items()
        }
        result_name = index.get("result_profiles", {}).get("MATCH_RESULT")
        self.match_result_reader = (
            MatchResultReader(profile_dir / str(result_name))
            if result_name is not None
            else None
        )
        score_name = index.get("score_profiles", {}).get("BATTLE")
        self.score_reader = (
            ScoreTemplateReader(profile_dir, profile_dir / "score_templates.json")
            if score_name is not None
            and (profile_dir / "score_templates.json").exists()
            else None
        )
        self.menu_reader = NpcMenuReader(profile_dir)
        self.stage_id: str | None = None
        self.board_reader: BoardGridReader | None = None
        self.board_region: VisionRegion | None = None
        self.preview_region: VisionRegion | None = None
        self.board_reference: np.ndarray | None = None
        self.board_profile: VisionProfile | None = None
        self.initial_board_locked = False
        self.special_activity_tracker = SpecialActivityTracker()
        self.special_cell_memory = SpecialCellMemory()
        self.sp_reader = SpReader()
        self.mode_reader = BattleModeReader()
        self.skip_mode_reader = SkipModeReader()
        self.replay_choice_reader = ReplayChoiceReader()
        self.reward_page_reader = RewardPageReader()
        self._sp_candidates = {"self": (None, 0), "opponent": (None, 0)}
        self.opponent_context: OpponentContext | None = None
        self.deck_reader: DeckGridReader | None = None
        self.reveal_reader: RevealReader | None = None
        self.hand_reader: HandReader | None = None
        self.hand_observation: HandObservation | None = None
        self.deck_observation: DeckObservation | None = None
        self._expected_deck_name: str | None = None
        self._expected_deck_ids: tuple[str, ...] | None = None
        self._deck_fingerprint: DeckFingerprint | None = None
        self._pending_deck_fingerprint: DeckFingerprint | None = None
        self._pending_deck_frames = 0
        self._reveal_observation: RevealObservation | None = None
        self._frame_counter = 0
        self._last_state = GameState.UNKNOWN
        self.grid_fit = None
        self._grid_fit_checked = False

    @classmethod
    def default(
        cls,
        mcts_policy: MCTSDecisionPolicy | None = None,
    ) -> "BattleVisionRuntime":
        # A running older terminal may cache the previous reload_core module.
        # Its reload callback calls this fresh factory only after module reload
        # has returned; refresh the support module and patch the existing UI
        # there, without creating a window or a second runtime worker.
        import sys
        patch_support = None
        patch_token = None
        ui = sys.modules.get("tableturf_rebuild.application.task_terminal_ui")
        if ui is not None:
            frame = sys._getframe(1)
            in_terminal_reload = False
            while frame is not None:
                if frame.f_code.co_name == "reload_core":
                    break
                if (frame.f_code.co_name == "reload_analysis_core"
                    and isinstance(frame.f_locals.get("self"), ui.TaskTerminalApp)):
                    in_terminal_reload = True
                    break
                frame = frame.f_back
            if in_terminal_reload:
                import importlib
                from tableturf_rebuild.application import core_reload
                patch_support = importlib.reload(core_reload)
                patch_token = patch_support.bootstrap_terminal_refresh()
        profile_dir = Path(__file__).resolve().parents[3] / "profiles" / "vision"
        try:
            return cls(
                GameCatalog.from_package(),
                profile_dir,
                mcts_policy=mcts_policy,
            )
        except Exception:
            if patch_support is not None:
                patch_support.rollback_terminal_refresh_patch(patch_token)
            raise

    def configure_stage(self, stage_id: str) -> None:
        if stage_id not in self.catalog.stages:
            raise KeyError(f"unknown stage: {stage_id}")
        profile_name = self.board_profile_names.get(stage_id)
        if profile_name is None:
            raise KeyError(f"no calibrated board profile for stage: {stage_id}")
        if self.stage_id != stage_id:
            self.special_activity_tracker.reset()
            self.special_cell_memory.reset()
            self._sp_candidates = {"self": (None, 0), "opponent": (None, 0)}
        reader, profile, region, reference = BoardGridReader.from_profile(
            self.catalog,
            self.profile_dir / profile_name,
            stage_id,
        )
        self.stage_id = stage_id
        self.board_reader = reader
        self.board_profile = profile
        self.board_region = region
        self.preview_region = None
        self.board_reference = reference
        self.initial_board_locked = False
        self.grid_fit = None
        self._grid_fit_checked = False

    def reset(self) -> None:
        self.state_machine.reset()
        self.initial_board_locked = False
        self.special_activity_tracker.reset()
        self.special_cell_memory.reset()
        self._sp_candidates = {"self": (None, 0), "opponent": (None, 0)}
        self.opponent_context = None
        self.deck_observation = None
        self._deck_fingerprint = None
        self._pending_deck_fingerprint = None
        self._pending_deck_frames = 0
        self._reveal_observation = None
        self.hand_observation = None
        self._frame_counter = 0
        self._last_state = GameState.UNKNOWN
        self.grid_fit = None
        self._grid_fit_checked = False
        if self.mcts_policy is not None:
            self.mcts_policy.set_opponent_context(None)
        self._restore_expected_deck()

    def set_expected_deck(
        self,
        name: str,
        card_ids: tuple[str, ...],
    ) -> None:
        if len(card_ids) != 15 or len(set(card_ids)) != 15:
            raise ValueError("expected deck must contain 15 unique card IDs")
        unknown = [card_id for card_id in card_ids if card_id not in self.catalog.cards]
        if unknown:
            raise KeyError(f"unknown card IDs: {', '.join(unknown)}")
        self._expected_deck_name = name
        self._expected_deck_ids = tuple(card_ids)
        self._restore_expected_deck()

    def clear_expected_deck(self) -> None:
        self._expected_deck_name = None
        self._expected_deck_ids = None
        self.deck_observation = None
        self._deck_fingerprint = None
        self._pending_deck_fingerprint = None
        self._pending_deck_frames = 0

    def bind_mcts(self, policy: MCTSDecisionPolicy) -> None:
        self.mcts_policy = policy
        policy.set_opponent_context(self.opponent_context)

    def update(
        self,
        frame: np.ndarray,
        include_selection: bool = False,
        force_selection: bool = False,
        placement_expected: bool = False,
    ) -> BattleVisionObservation:
        started = checkpoint = time.perf_counter()
        timings = {}

        def mark(name):
            nonlocal checkpoint
            now = time.perf_counter()
            timings[name] = round((now - checkpoint) * 1000, 3)
            checkpoint = now

        self._frame_counter += 1
        raw = self.state_machine.recognizer.classify(frame)
        mark("state_classifier")
        reward_page = self.reward_page_reader.read(frame)
        mark("reward_page")
        # The global layout classifier can mistake a different final board and
        # a bright name card for selectable hands. Read the small result words
        # independently, before that classifier can gate the evidence needed
        # to correct itself. The result reader already requires both labels
        # to be visible, complementary and above its absolute quality margin.
        match_result = self.match_result_reader.read(frame) if self.match_result_reader is not None else None
        mark("match_result")
        if reward_page.visible:
            # Read the modal itself before the result words in its background.
            # Its presence does not establish who won; the runner still needs
            # the result receipt from this match before acknowledging rewards.
            self.state_machine.force_state(GameState.MATCH_REWARD)
            self._last_state = GameState.MATCH_REWARD
            self.hand_observation = None
            self._reveal_observation = None
            self._sp_candidates = {"self": (None, 0), "opponent": (None, 0)}
            state = replace(raw, state=GameState.MATCH_REWARD,
                confidence=reward_page.confidence,
                candidates=(StateScore(GameState.MATCH_REWARD, reward_page.confidence),
                            *(item for item in raw.candidates if item.state is not GameState.MATCH_REWARD)),
                actions=STATE_ACTIONS[GameState.MATCH_REWARD],
                boxes=STATE_BOXES[GameState.MATCH_REWARD], focus_boxes=())
            timings["total"] = round((time.perf_counter() - started) * 1000, 3)
            return BattleVisionObservation(
                state=state, board=None, selection_profile=None, selection_candidates=(),
                initial_board_locked=self.initial_board_locked, match_result=match_result,
                opponent_context=self.opponent_context, deck=self.deck_observation,
                grid_fit=self.grid_fit, timings_ms=timings, reward_page_evidence=reward_page,
            )
        if (match_result is not None
            and match_result.outcome in {MatchOutcome.WIN, MatchOutcome.LOSS}
            and match_result.confidence >= self.RESULT_OVERRIDE_CONFIDENCE):
            self.state_machine.force_state(GameState.MATCH_RESULT)
            self._last_state = GameState.MATCH_RESULT
            self.hand_observation = None
            self._reveal_observation = None
            self._sp_candidates = {"self": (None, 0), "opponent": (None, 0)}
            state = replace(raw, state=GameState.MATCH_RESULT,
                confidence=match_result.confidence,
                candidates=(StateScore(GameState.MATCH_RESULT, match_result.confidence),
                            *(item for item in raw.candidates if item.state is not GameState.MATCH_RESULT)),
                actions=STATE_ACTIONS[GameState.MATCH_RESULT],
                boxes=STATE_BOXES[GameState.MATCH_RESULT], focus_boxes=())
            timings["total"] = round((time.perf_counter() - started) * 1000, 3)
            return BattleVisionObservation(
                state=state, board=None, selection_profile=None, selection_candidates=(),
                initial_board_locked=self.initial_board_locked, match_result=match_result,
                opponent_context=self.opponent_context, deck=self.deck_observation,
                grid_fit=self.grid_fit, timings_ms=timings,
            )
        # The independently readable toggle label can recover a stale global
        # state (for example FINISHING while the live UI is selecting a card).
        # Do not gate the evidence needed to correct that state behind it.
        mode = self.mode_reader.read(frame)
        mark("battle_mode")
        discard = self.skip_mode_reader.read(frame)
        mark("skip_mode")
        controls_visible = mode.special_mode is not None or discard.discard_mode is not None
        read_board = self._should_read_board(raw) or controls_visible
        if read_board and (not self._grid_fit_checked or self._frame_counter % 30 == 0 and self.grid_fit is None):
            fit = fit_empty_cell_grid(frame, self.catalog)
            self._grid_fit_checked = True
            if fit is not None and fit.stage_id in self.board_profile_names:
                requested = self.stage_id
                if self.stage_id != fit.stage_id:
                    self.configure_stage(fit.stage_id)
                self.preview_region = fit.region
                self._grid_fit_checked = True
                self.grid_fit = {
                    "source": "legacy_empty_cell_grid", "requested_stage": requested,
                    "detected_stage": fit.stage_id, "rows": fit.region.grid_rows,
                    "cols": fit.region.grid_cols, "empty_cells": fit.empty_cells,
                    "confidence": fit.confidence, "step_x": fit.step_x, "step_y": fit.step_y,
                }
        mark("grid_fit")
        board = None
        if (
            self.board_reader is not None
            and self.board_region is not None
            and self.board_reference is not None
            and read_board
        ):
            board = self.board_reader.read(
                frame,
                self.board_region,
                self.stage_id or "",
                self.board_reference,
                self.board_profile,
            )
        mark("board")
        hand_visibility = (
            self.selection_resolver.hand_visibility(frame)
            if read_board else None
        )
        mark("hand_visibility")
        evidence = StateEvidence(
            battle_controls_visible=controls_visible,
            discard_mode=discard.discard_mode,
            placement_expected=placement_expected,
            hand_cursor_visible=bool(hand_visibility and hand_visibility.cursor_slot is not None),
            hand_selection_visible=(
                hand_visibility.selection_visible
                if hand_visibility is not None else None
            ),
            placement_preview_present=(
                board.placement_preview_present
                if board is not None
                else None
            )
        )
        state = self.state_machine.update_observation(raw, evidence)
        mark("state_transition")
        selection_candidates: tuple[SelectionProfileMatch, ...] = ()
        selection_profile = None
        selection_hand = None
        if (
            include_selection
            and state.state in SELECTION_RESOLUTION_STATES
            and not (placement_expected and hand_visibility is not None
                     and hand_visibility.selection_visible is False)
            and (
                force_selection
                or
                state.state is not GameState.TARGET_PLACEMENT
                or board is None
                or not board.placement_preview_present
            )
        ):
            selection_candidates = self.selection_resolver.match(frame)
            selection_profile = _resolved_selection(selection_candidates)
            selection_hand = self._read_hand(
                frame,
                GameState.TURN_SELECT,
                selection_profile,
            )
            if (
                force_selection
                and state.state is GameState.TARGET_PLACEMENT
                and selection_profile is not None
                and selection_profile.focus_slot is not None
                and selection_profile.score >= 0.90
                and selection_hand is not None
                and selection_hand.confidence
                >= self.SELECTION_OVERRIDE_HAND_CONFIDENCE
                and hand_visibility is not None
                and hand_visibility.selection_visible is True
            ):
                state = replace(
                    state,
                    state=GameState.TURN_SELECT,
                    confidence=min(
                        1.0,
                        max(
                            state.confidence,
                            selection_profile.score,
                        ),
                    ),
                )
                self.state_machine.force_state(GameState.TURN_SELECT)
        mark("selection_profile_and_hand")
        if state.state is not self._last_state:
            if state.state in {GameState.DECK_SELECT, GameState.REDRAW}:
                self._sp_candidates = {"self": (None, 0), "opponent": (None, 0)}
                self.special_cell_memory.reset()
                self.special_activity_tracker.reset()
            if state.state is GameState.DECK_SELECT:
                # A previous configured deck is not visual evidence about
                # the highlighted deck in this new selection screen.
                self._expected_deck_name = None
                self._expected_deck_ids = None
                self.deck_observation = None
                self._deck_fingerprint = None
                self._pending_deck_fingerprint = None
                self._pending_deck_frames = 0
            if (
                state.state is GameState.TURN_REVEAL
                and self._last_state is not GameState.TURN_REVEAL
            ):
                self._reveal_observation = None
            self._last_state = state.state
        if state.state in {GameState.DECK_SELECT, GameState.REDRAW}:
            self.initial_board_locked = False
        if board is not None:
            board = self.special_cell_memory.update(board, allow_new=(
                state.state in CARD_SELECTION_STATES
                and hand_visibility is not None
                and hand_visibility.selection_visible is True
            ))
        if (
            state.state in BOARD_READ_STATES
            and board is not None
            and board.initial_board_ready
        ):
            self.initial_board_locked = True
        hand = None if (
            placement_expected and hand_visibility is not None
            and hand_visibility.selection_visible is False
        ) else (
            selection_hand
            if selection_hand is not None
            and state.state in HAND_READ_STATES
            else self._read_hand(
                frame,
                state.state,
                selection_profile,
            )
        )
        mark("hand")
        match_result = (match_result
                        if state.state in {GameState.MATCH_RESULT, GameState.MATCH_REWARD}
                        else None)
        active_specials = (
            self.special_activity_tracker.update(board)
            if board is not None
            else None
        )
        menu = (
            self.menu_reader.read(frame, state.state.value)
            if state.state in {GameState.NPC_ROSTER, GameState.MATCH_SETUP}
            else None
        )
        if (
            menu is not None
            and menu.stage_id is not None
            and menu.stage_id != self.stage_id
        ):
            try:
                self.configure_stage(menu.stage_id)
            except (KeyError, ValueError):
                pass
        if state.state is GameState.DECK_SELECT:
            self._refresh_deck(frame)
        if menu is not None:
            opponent_context = self._opponent_context(menu)
            if opponent_context is not None:
                self.opponent_context = opponent_context
                if self.mcts_policy is not None:
                    self.mcts_policy.set_opponent_context(opponent_context)
        mark("scene_details")
        score = (
            self.score_reader.read(frame)
            if self.score_reader is not None
            and state.state in BOARD_READ_STATES
            else None
        )
        reveal = self._read_reveal(frame, state.state)
        mark("score_and_reveal")
        sp = self._read_sp(frame, state.state)
        mark("sp_hud")
        replay_choice = (self.replay_choice_reader.read(frame)
                         if state.state is GameState.REPLAY_CHOICE else None)
        mark("replay_choice")
        timings["total"] = round((time.perf_counter() - started) * 1000, 3)
        return BattleVisionObservation(
            state=state,
            board=board,
            selection_profile=selection_profile,
            selection_candidates=selection_candidates,
            initial_board_locked=self.initial_board_locked,
            match_result=match_result,
            score=score,
            active_specials=active_specials,
            menu=menu,
            opponent_context=self.opponent_context,
            deck=self.deck_observation,
            reveal=reveal,
            hand=hand,
            hand_visibility=hand_visibility,
            grid_fit=self.grid_fit,
            timings_ms=timings,
            sp=sp,
            special_mode=mode.special_mode if mode is not None else None,
            mode_evidence=mode,
            discard_mode=discard.discard_mode,
            discard_evidence=discard,
            replay_choice_evidence=replay_choice,
            reward_page_evidence=reward_page if state.state is GameState.MATCH_REWARD else None,
        )

    def _read_sp(self, frame: np.ndarray, state: GameState) -> SpObservation | None:
        if state not in BOARD_READ_STATES:
            self._sp_candidates = {"self": (None, 0), "opponent": (None, 0)}
            return None
        raw = self.sp_reader.read(frame)
        updates = {}
        for side in ("self", "opponent"):
            value = getattr(raw, f"{side}_value")
            confidence = getattr(raw, f"{side}_confidence")
            if value is None or confidence < 0.8:
                self._sp_candidates[side] = (None, 0)
                updates[f"{side}_value"] = None
                updates[f"{side}_confidence"] = 0.0
                continue
            previous, count = self._sp_candidates[side]
            count = count + 1 if previous == value else 1
            self._sp_candidates[side] = (value, count)
            if count < 2:
                # Publish unknown during a balance change rather than exposing
                # the previously confirmed, possibly already spent SP balance.
                updates[f"{side}_value"] = None
                updates[f"{side}_confidence"] = 0.0
                updates[f"{side}_reason"] = "AWAITING_TWO_IDENTICAL_FRAMES"
        return replace(raw, **updates)

    def _read_hand(
        self,
        frame: np.ndarray,
        state: GameState,
        selection_profile: SelectionProfileMatch | None,
    ) -> HandObservation | None:
        if state not in HAND_READ_STATES:
            return None
        profile_path = (
            selection_profile.profile_path
            if selection_profile is not None
            else self.profile_dir / "battle_select_clean.json"
        )
        candidate_ids = (
            self._expected_deck_ids
            if self._expected_deck_ids is not None
            else tuple(
                card.card_id
                for card in self.deck_observation.cards
            )
            if self.deck_observation is not None
            else tuple(self.catalog.cards)
        )
        try:
            if self.hand_reader is None:
                if self.deck_reader is None:
                    self.deck_reader = DeckGridReader(
                        self.catalog,
                        self.profile_dir,
                    )
                self.hand_reader = HandReader(
                    self.catalog,
                    self.profile_dir,
                    shape_reader=self.deck_reader,
                )
            self.hand_observation = self.hand_reader.read(
                frame,
                profile_path,
                candidate_ids,
            )
            return self.hand_observation
        except (ValueError, KeyError):
            self.hand_observation = None
            return None

    def _refresh_deck(self, frame: np.ndarray) -> None:
        try:
            if self.deck_observation is not None and self.deck_observation.source == "configured":
                self.deck_observation = None
                self._deck_fingerprint = None
            if self.deck_reader is None:
                self.deck_reader = DeckGridReader(
                    self.catalog,
                    self.profile_dir,
                )
            if self.deck_observation is None:
                self.deck_observation = self.deck_reader.read(frame)
                self._deck_fingerprint = self.deck_reader.fingerprint(frame)
                self._pending_deck_fingerprint = None
                self._pending_deck_frames = 0
                return
            if self._frame_counter % 3 != 0:
                return

            fingerprint = self.deck_reader.fingerprint(frame)
            if (
                self._deck_fingerprint is not None
                and not self._deck_fingerprint.materially_different(
                    fingerprint
                )
            ):
                self._pending_deck_fingerprint = None
                self._pending_deck_frames = 0
                return
            if (
                self._pending_deck_fingerprint is not None
                and not self._pending_deck_fingerprint.materially_different(
                    fingerprint
                )
            ):
                self._pending_deck_frames += 1
            else:
                self._pending_deck_fingerprint = fingerprint
                self._pending_deck_frames = 1
            if self._pending_deck_frames < 2:
                return

            observation = self.deck_reader.read(frame)
            self.deck_observation = observation
            self._deck_fingerprint = fingerprint
            self._pending_deck_fingerprint = None
            self._pending_deck_frames = 0
        except (ValueError, KeyError):
            self._pending_deck_fingerprint = None
            self._pending_deck_frames = 0

    def _restore_expected_deck(self) -> None:
        if self._expected_deck_ids is None:
            return
        cards = []
        for index, card_id in enumerate(self._expected_deck_ids):
            card = self.catalog.cards[card_id]
            candidate = DeckSlotCandidate(
                card_id=card.card_id,
                name=card.name_zh or card.name,
                area=card.area,
                sp_cost=card.sp_cost,
                score=1.0,
            )
            cards.append(
                DeckSlotObservation(
                    slot_row=index // 3 + 1,
                    slot_col=index % 3 + 1,
                    card_id=card.card_id,
                    name=card.name_zh or card.name,
                    area=card.area,
                    sp_cost=card.sp_cost,
                    visible_cost_lights=min(card.sp_cost, 5),
                    match_score=1.0,
                    candidates=(candidate,),
                )
            )
        self.deck_observation = DeckObservation(
            deck_id=self._expected_deck_name or "CONFIGURED",
            deck_score=1.0,
            deck_margin=1.0,
            total_area=sum(card.area for card in cards),
            total_sp_cost=sum(card.sp_cost for card in cards),
            cards=tuple(cards),
            confidence=1.0,
            source="configured",
        )

    def _read_reveal(
        self,
        frame: np.ndarray,
        state: GameState,
    ) -> RevealObservation | None:
        if state not in REVEAL_READ_STATES:
            return None
        try:
            if self.reveal_reader is None:
                if self.deck_reader is None:
                    self.deck_reader = DeckGridReader(
                        self.catalog,
                        self.profile_dir,
                    )
                self.reveal_reader = RevealReader(
                    self.catalog,
                    self.profile_dir,
                    shape_reader=self.deck_reader,
                )
            own_card_ids = (
                tuple(card.card_id for card in self.deck_observation.cards)
                if self.deck_observation is not None
                else None
            )
            opponent_card_ids = (
                self.catalog.decks[self.opponent_context.deck_id].card_ids
                if (
                    self.opponent_context is not None
                    and self.opponent_context.deck_id in self.catalog.decks
                )
                else None
            )
            refresh_frames = (
                3
                if (
                    own_card_ids is not None
                    and opponent_card_ids is not None
                )
                else 15
            )
            if (
                self._reveal_observation is not None
                and self._frame_counter % refresh_frames != 0
            ):
                return self._reveal_observation
            observation = self.reveal_reader.read(
                frame,
                own_card_ids=own_card_ids,
                opponent_card_ids=opponent_card_ids,
            )
            self._reveal_observation = observation
            return observation
        except (ValueError, KeyError):
            return None

    def _should_read_board(self, raw: StateObservation) -> bool:
        if self.state_machine.current in BOARD_READ_STATES:
            return True
        return (
            raw.state in BOARD_READ_STATES
            and raw.confidence >= 0.45
        )

    def _opponent_context(
        self,
        menu: NpcMenuObservation,
    ) -> OpponentContext | None:
        if (
            menu.npc_id is None
            or menu.difficulty is None
            or menu.stage_id is None
        ):
            return None
        npc = self.catalog.npcs[menu.npc_id]
        index = menu.difficulty - 1
        if not 0 <= index < len(npc.deck_ids):
            return None
        return OpponentContext(
            npc_id=npc.npc_id,
            npc_name=menu.npc_name or npc.name_zh,
            difficulty=menu.difficulty,
            ai_type=(
                npc.ai_types[index]
                if index < len(npc.ai_types)
                else ""
            ),
            deck_id=npc.deck_ids[index],
            stage_id=menu.stage_id,
            stage_name=menu.stage_name or self.catalog.stages[menu.stage_id].name_zh,
            confidence=min(
                menu.name_confidence,
                menu.wins_confidence,
                menu.difficulty_confidence,
                menu.stage_confidence,
            ),
        )


def _resolved_selection(
    candidates: tuple[SelectionProfileMatch, ...],
    minimum_score: float = 0.75,
    minimum_margin: float = 0.01,
) -> SelectionProfileMatch | None:
    if not candidates or candidates[0].score < minimum_score:
        return None
    if len(candidates) > 1 and candidates[0].score - candidates[1].score < minimum_margin:
        return None
    return candidates[0]
