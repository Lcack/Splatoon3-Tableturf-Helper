from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from tableturf_rebuild.application.state_machine import (
    GameState,
    StateEvidence,
    StateMachine,
    StateObservation,
)
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.planning.context import OpponentContext
from tableturf_rebuild.planning.mcts import MCTSDecisionPolicy
from tableturf_rebuild.vision.board_reader import (
    ActiveSpecialObservation,
    BoardGridReader,
    BoardObservation,
    SpecialActivityTracker,
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
    MatchResultObservation,
    MatchResultReader,
)
from tableturf_rebuild.vision.hand_reader import HandObservation, HandReader
from tableturf_rebuild.vision.npc_menu_reader import (
    NpcMenuObservation,
    NpcMenuReader,
)
from tableturf_rebuild.vision.score_reader import (
    ScoreObservation,
    ScoreTemplateReader,
)
from tableturf_rebuild.vision.selection_profiles import (
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


class BattleVisionRuntime:
    SELECTION_OVERRIDE_HAND_CONFIDENCE = 0.65

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
        self.board_reference: np.ndarray | None = None
        self.board_profile: VisionProfile | None = None
        self.initial_board_locked = False
        self.special_activity_tracker = SpecialActivityTracker()
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

    @classmethod
    def default(
        cls,
        mcts_policy: MCTSDecisionPolicy | None = None,
    ) -> "BattleVisionRuntime":
        profile_dir = Path(__file__).resolve().parents[3] / "profiles" / "vision"
        return cls(
            GameCatalog.from_package(),
            profile_dir,
            mcts_policy=mcts_policy,
        )

    def configure_stage(self, stage_id: str) -> None:
        if stage_id not in self.catalog.stages:
            raise KeyError(f"unknown stage: {stage_id}")
        profile_name = self.board_profile_names.get(stage_id)
        if profile_name is None:
            raise KeyError(f"no calibrated board profile for stage: {stage_id}")
        reader, profile, region, reference = BoardGridReader.from_profile(
            self.catalog,
            self.profile_dir / profile_name,
            stage_id,
        )
        self.stage_id = stage_id
        self.board_reader = reader
        self.board_profile = profile
        self.board_region = region
        self.board_reference = reference
        self.initial_board_locked = False

    def reset(self) -> None:
        self.state_machine.reset()
        self.initial_board_locked = False
        self.special_activity_tracker.reset()
        self.opponent_context = None
        self.deck_observation = None
        self._deck_fingerprint = None
        self._pending_deck_fingerprint = None
        self._pending_deck_frames = 0
        self._reveal_observation = None
        self.hand_observation = None
        self._frame_counter = 0
        self._last_state = GameState.UNKNOWN
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
    ) -> BattleVisionObservation:
        self._frame_counter += 1
        raw = self.state_machine.recognizer.classify(frame)
        board = None
        if (
            self.board_reader is not None
            and self.board_region is not None
            and self.board_reference is not None
            and self._should_read_board(raw)
        ):
            board = self.board_reader.read(
                frame,
                self.board_region,
                self.stage_id or "",
                self.board_reference,
                self.board_profile,
            )
        evidence = StateEvidence(
            placement_preview_present=(
                board.placement_preview_present
                if board is not None
                else None
            )
        )
        state = self.state_machine.update_observation(raw, evidence)
        selection_candidates: tuple[SelectionProfileMatch, ...] = ()
        selection_profile = None
        selection_hand = None
        if (
            include_selection
            and state.state in SELECTION_RESOLUTION_STATES
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
        if state.state is not self._last_state:
            if state.state is GameState.DECK_SELECT:
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
        if (
            state.state in BOARD_READ_STATES
            and board is not None
            and board.initial_board_ready
        ):
            self.initial_board_locked = True
        hand = (
            selection_hand
            if selection_hand is not None
            and state.state in HAND_READ_STATES
            else self._read_hand(
                frame,
                state.state,
                selection_profile,
            )
        )
        match_result = (
            self.match_result_reader.read(frame)
            if state.state is GameState.MATCH_RESULT
            and self.match_result_reader is not None
            else None
        )
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
        score = (
            self.score_reader.read(frame)
            if self.score_reader is not None
            and state.state in BOARD_READ_STATES
            else None
        )
        reveal = self._read_reveal(frame, state.state)
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
        )

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
            return self.hand_observation

    def _refresh_deck(self, frame: np.ndarray) -> None:
        if self._expected_deck_ids is not None:
            self._restore_expected_deck()
            return
        try:
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
