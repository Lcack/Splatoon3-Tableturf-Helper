from __future__ import annotations

from dataclasses import asdict, replace
from enum import StrEnum
import threading
import time

import numpy as np

from tableturf_rebuild.application.battle_vision import (
    BattleVisionObservation,
    BattleVisionRuntime,
)
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.control.backend import ControllerBackend
from tableturf_rebuild.control.buttons import Button, ControllerState
from tableturf_rebuild.control.sequences import pulse_buttons
from tableturf_rebuild.domain.actions import Action, ActionType
from tableturf_rebuild.domain.card import CardShape
from tableturf_rebuild.domain.grid import PLAYABLE_CELLS, Cell
from tableturf_rebuild.domain.rules import BattleRules
from tableturf_rebuild.domain.state import BattleState, PlayerState, Side
from tableturf_rebuild.planning.mcts import (
    MCTSConfig,
    MCTSDecisionPolicy,
)
from tableturf_rebuild.planning.policies import GreedyPolicy
from tableturf_rebuild.vision.board_reader import BoardObservation
from tableturf_rebuild.vision.placement_reader import PlacementPreviewReader


class BattleExecutorPhase(StrEnum):
    WAITING = "WAITING"
    CHOOSING_CARD = "CHOOSING_CARD"
    POSITIONING = "POSITIONING"
    PLACING = "PLACING"
    WAITING_RESULT = "WAITING_RESULT"
    ERROR = "ERROR"


class VisionMctsBattleExecutor:
    """Closed-loop executor for normal card placement turns."""

    HAND_CONFIDENCE_THRESHOLD = 0.55
    PLACEMENT_SETTLE_SECONDS = 1.2
    PREVIEW_STABLE_FRAMES = 3
    PREVIEW_MOTION_STABLE_FRAMES = 2
    PREVIEW_RESPONSE_SECONDS = 0.75
    MAX_ROTATION_FEEDBACK_READS = 8
    MAX_CONFIRM_ATTEMPTS = 3
    MAX_GRAY_RECOVERIES = 3
    COARSE_MOVEMENT_STEPS = 3
    COARSE_MOVEMENT_MIN_DISTANCE = 5
    COMPLETION_HAND_CONFIDENCE = 0.85

    def __init__(
        self,
        runtime: BattleVisionRuntime,
        controller: ControllerBackend,
        own_deck_ids: tuple[str, ...],
        *,
        mcts_config: MCTSConfig | None = None,
        seed: int = 0,
        fallback_opponent_deck_id: str | None = None,
    ):
        self.runtime = runtime
        self.controller = controller
        self.catalog = runtime.catalog
        self.own_deck_ids = tuple(own_deck_ids)
        self.fallback_opponent_deck_id = fallback_opponent_deck_id
        self.rules = BattleRules(self.catalog)
        self.policy = MCTSDecisionPolicy(
            config=mcts_config
            or MCTSConfig(
                simulations=16,
                time_limit_seconds=0.35,
                root_action_limit=12,
                tree_action_limit=8,
            ),
            seed=seed,
            opponent_model=GreedyPolicy("npc-greedy"),
            prior_policy=GreedyPolicy("mcts-prior"),
        )
        self.placement_reader = PlacementPreviewReader(self.catalog)
        self.phase = BattleExecutorPhase.WAITING
        self.message = "等待战斗状态"
        self._last_state = GameState.UNKNOWN
        self._selected_action: Action | None = None
        self._selected_card_id: str | None = None
        self._selected_target_slot = None
        self._selection_navigation_steps = 0
        self._hand_mismatch_reads = 0
        self._last_mismatch_hand = None
        self._current_hand = None
        self._card_confirmed = False
        self._card_confirmed_at = 0.0
        self._card_confirm_attempts = 0
        self._place_sent = False
        self._place_sent_at = 0.0
        self._placement_confirm_attempts = 0
        self._skip_down_sent = False
        self._skip_focus_missing_reads = 0
        self._last_selection_focus = None
        self._skip_activate_sent = False
        self._skip_activate_at = 0.0
        self._discard_confirm_sent = False
        self._discard_confirm_at = 0.0
        self._discard_confirm_attempts = 0
        self._return_to_select_attempts = 0
        self._return_to_select_at = 0.0
        self._force_selection_after_b = False
        self._preview_missing_reads = 0
        self._preview_signature: tuple[int, tuple[int, int], bool] | None = None
        self._preview_signature_count = 0
        self._preview_response_at = 0.0
        self._pending_preview_signature: (
            tuple[int, tuple[int, int], bool] | None
        ) = None
        self._preview_stall_count = 0
        self._completed_turns = 0
        self._reset_rejected_placements()
        self._turn_result_counted = False
        self._selection_board: BoardObservation | None = None
        self._preview_recoveries = 0
        self._revealed_own_cards: list[str] = []
        self._revealed_opponent_cards: list[str] = []
        self._reveal_pending: dict[str, tuple[str, int]] = {}
        self._last_reveal_observation = None
        self._last_preview = None
        self._last_preview_attempts: list[dict] = []
        self._last_command: dict | None = None
        self._visual_shapes: dict = {}
        self._rotation_commands = 0
        self._reset_rotation_feedback()
        self._motion_cancelled = threading.Event()
        self._reset_movement_feedback()
        self._selection_sp: tuple[int, int] | None = None
        self._sp_confirmation_waits = 0
        self._reset_special_flow()
        self._reset_hand_completion()

    @property
    def requires_selection_data(self) -> bool:
        # A false placement-preview detection can make the vision layer hide
        # focus data even while the game is still on the card-selection UI.
        # Keep resolving focus until the executor has actually confirmed a card.
        return not self._card_confirmed

    @property
    def awaiting_placement(self) -> bool:
        return (self._card_confirmed and not self._place_sent
                and self._selected_action is not None
                and self._selected_action.action_type in {ActionType.PLACE, ActionType.SPECIAL}
                and self.phase is not BattleExecutorPhase.ERROR)

    def reset_match(self) -> None:
        """Clear live-game history before the next match, keeping configuration."""
        self._completed_turns = 0
        self._reset_rejected_placements()
        self._turn_result_counted = False
        self._revealed_own_cards.clear()
        self._revealed_opponent_cards.clear()
        self._reveal_pending.clear()
        self._last_reveal_observation = None
        self._selected_action = None
        self._selected_card_id = None
        self._selected_target_slot = None
        self._selection_navigation_steps = 0
        self._hand_mismatch_reads = 0
        self._last_mismatch_hand = None
        self._current_hand = None
        self._selection_board = None
        self._card_confirmed = False
        self._card_confirmed_at = 0.0
        self._card_confirm_attempts = 0
        self._place_sent = False
        self._place_sent_at = 0.0
        self._placement_confirm_attempts = 0
        self._skip_down_sent = False
        self._skip_focus_missing_reads = 0
        self._last_selection_focus = None
        self._skip_activate_sent = False
        self._skip_activate_at = 0.0
        self._discard_confirm_sent = False
        self._discard_confirm_at = 0.0
        self._discard_confirm_attempts = 0
        self._return_to_select_attempts = 0
        self._force_selection_after_b = False
        self._preview_missing_reads = 0
        self._preview_recoveries = 0
        self._preview_signature = None
        self._preview_signature_count = 0
        self._preview_response_at = 0.0
        self._pending_preview_signature = None
        self._preview_stall_count = 0
        self._last_preview = None
        self._last_preview_attempts = []
        self._last_command = None
        self._last_state = GameState.UNKNOWN
        self._rotation_commands = 0
        self._reset_rotation_feedback()
        self._reset_movement_feedback()
        self._selection_sp = None
        self._sp_confirmation_waits = 0
        self._reset_special_flow()
        self.phase = BattleExecutorPhase.WAITING
        self.message = "等待新一局战斗"
        self._reset_hand_completion()

    def step(
        self,
        observation: BattleVisionObservation,
        frame: np.ndarray,
    ) -> bool:
        if self._motion_cancelled.is_set():
            if self.phase is not BattleExecutorPhase.ERROR:
                self.phase = BattleExecutorPhase.WAITING
                self.message = "自动按键已暂停，等待继续任务"
            return False
        state = observation.state.state
        profile = observation.selection_profile
        focus = profile.focus_slot if profile is not None else None
        if focus is None and observation.hand_visibility is not None:
            focus = observation.hand_visibility.cursor_slot
        self._last_selection_focus = tuple(focus) if focus is not None else None
        if self._rejection_turn != self._completed_turns:
            self._reset_rejected_placements()
        if self._gray_recovery is not None:
            return self._observe_gray_recovery(observation)
        if (self._selection_board is not None and observation.board is not None
            and (self._selection_board.stage_id, self._selection_board.rows, self._selection_board.cols)
            != (observation.board.stage_id, observation.board.rows, observation.board.cols)):
            # Automatic grid fitting can correct a manually selected map.
            # An action planned in the old coordinates must never be reused.
            self._selected_action = None
            self._selected_card_id = None
            self._selection_board = None
            self._selection_sp = None
            self._reset_special_flow()
            self._current_hand = None
            self._card_confirmed = False
            self._card_confirm_attempts = 0
            self._place_sent = False
            self._last_preview = None
            self._preview_signature = None
            self._preview_signature_count = 0
            self._reset_rotation_feedback()
            self._last_state = GameState.UNKNOWN
            self.phase = BattleExecutorPhase.WAITING
            self.message = "实际网格已校正，废弃旧坐标动作并返回选牌重新规划"
            if state is GameState.TARGET_PLACEMENT:
                self._force_selection_after_b = True
                return self._pulse(Button.B)
            return False
        if state is GameState.MATCH_RESULT:
            self.phase = BattleExecutorPhase.WAITING_RESULT
            self.message = "对战已结束，等待查看结算"
            self._last_preview = None
            self._last_preview_attempts = []
            return False
        if self.phase is BattleExecutorPhase.ERROR and state not in {
            GameState.DECK_SELECT, GameState.BATTLE_INTRO, GameState.REDRAW,
        }:
            return False
        if (state is GameState.SPECIAL_SELECT
            and (self._place_sent or self._discard_confirm_sent)
            and observation.hand is not None
            and observation.hand.confidence >= self.HAND_CONFIDENCE_THRESHOLD
            and self._selected_card_id not in observation.hand.card_ids
            and observation.hand_visibility is not None
            and observation.hand_visibility.selection_visible is True):
            # The mode template can lag after the result. A new selectable
            # hand acknowledges this turn regardless of that template's name.
            state = GameState.TURN_SELECT
        if (
            state in {GameState.TURN_REVEAL, GameState.TURN_RESOLVE}
            and self._card_confirmed and not self._place_sent
            and self._selected_action is not None
        ):
            # A card-selection A is not a placement A. Do not count a turn or
            # wait indefinitely merely because the image resembles a reveal.
            return self._position_selected_card(observation, frame)
        if (
            state in {GameState.DECK_SELECT, GameState.BATTLE_INTRO, GameState.REDRAW}
            and state is not self._last_state
        ):
            self.reset_match()
        if self.phase is BattleExecutorPhase.ERROR:
            return False
        if (self._selected_action is not None
            and self._selected_action.action_type is ActionType.SPECIAL
            and not self._card_confirmed and not self._place_sent
            and state in {GameState.TURN_SELECT, GameState.SPECIAL_SELECT, GameState.TARGET_PLACEMENT,
                          GameState.SPECIAL_PLACEMENT, GameState.TURN_REVEAL, GameState.TURN_RESOLVE}):
            return self._choose_and_select_card(observation)
        skip_pending = (self._selected_action is not None
                        and self._selected_action.action_type is ActionType.SKIP)
        if (skip_pending and self._discard_confirm_sent
            and state in {GameState.TARGET_PLACEMENT, GameState.SPECIAL_PLACEMENT}):
            self.phase = BattleExecutorPhase.WAITING_RESULT
            self.message = "弃牌已确认，等待回合结束"
            return False
        if skip_pending and not self._discard_confirm_sent:
            if observation.discard_mode is True:
                # The button now says Return. This also recovers a reload or
                # manual entry into the discard dialog without pressing A
                # on Return and accidentally closing it again.
                self._skip_activate_sent = True
                return self._discard_for_skip(observation)
            if not self._skip_activate_sent:
                # A previous Down is not proof that focus reached the button.
                # The animated squid identifies all six menu slots, even if
                # the selected button's background does not change.
                return self._open_skip_dialog(self._last_selection_focus)
            self.phase = BattleExecutorPhase.WAITING
            self.message = "已请求跳过，等待返回文字确认弃牌界面"
            if time.perf_counter() - self._skip_activate_at > 8.0:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "跳过确认后未识别到返回文字，已停止按键并保留诊断"
            return False
        if state is GameState.SKIP_DISCARD and not self._skip_activate_sent:
            # Four unavailable gray cards resemble the discard template even
            # before the skip button has been activated.
            state = GameState.TURN_SELECT
        self._observe_revealed_cards(observation)
        same_state_new_hand = False
        if (state is GameState.TURN_SELECT and state is self._last_state
            and (self._place_sent or self._discard_confirm_sent)):
            # Skip handlers can return before recording the dialog/reveal
            # states. A new hand is still a new turn when the label stayed
            # TURN_SELECT; never try to select the just-consumed card again.
            same_state_new_hand = self._confirm_new_hand_after_commit(observation)
            if not same_state_new_hand:
                self.phase = BattleExecutorPhase.WAITING_RESULT
                self.message = ("动作已提交，等待新手牌、旧牌消失及选牌焦点确认 "
                                f"({self._completion_hand_reads}/2)")
                if time.perf_counter() - self._completion_wait_at >= 20:
                    self.phase = BattleExecutorPhase.ERROR
                    self.message = "提交动作后未能确认手牌更新，已停止按键并保留诊断"
                return False
        if state is not self._last_state or same_state_new_hand:
            if (state is GameState.TURN_SELECT
                and (self._place_sent or self._discard_confirm_sent)
                and not self._turn_result_counted):
                # Like the original manager, a return to selectable hands
                # acknowledges the completed move even if reveal was missed.
                visibility = observation.hand_visibility
                if visibility is not None and visibility.selection_visible is not True:
                    self.phase = BattleExecutorPhase.WAITING_RESULT
                    self.message = "等待手牌光标重新出现以确认本回合结束"
                    return False
                self._completed_turns += 1
                self._turn_result_counted = True
                if (self._selected_card_id is not None
                    and self._selected_card_id not in self._revealed_own_cards):
                    self._revealed_own_cards.append(self._selected_card_id)
            if state is GameState.TURN_SELECT and (
                self._selected_action is None
                or self._place_sent
                or self._discard_confirm_sent
                or self._last_state
                in {
                    GameState.UNKNOWN,
                    GameState.REDRAW,
                    GameState.TURN_REVEAL,
                    GameState.TURN_RESOLVE,
                    GameState.BATTLE_INTRO,
                    GameState.MATCH_SETUP,
                }
            ):
                self._selected_action = None
                self._selected_card_id = None
                self._selected_target_slot = None
                self._selection_navigation_steps = 0
                self._current_hand = None
                self._selection_board = None
                self._selection_sp = None
                self._sp_confirmation_waits = 0
                self._reset_special_flow()
                self._turn_result_counted = False
                self._reveal_pending.clear()
                self._last_reveal_observation = None
                self._card_confirmed = False
                self._card_confirmed_at = 0.0
                self._card_confirm_attempts = 0
                self._rotation_commands = 0
                self._reset_rotation_feedback()
                self._reset_movement_feedback()
                self._place_sent = False
                self._place_sent_at = 0.0
                self._placement_confirm_attempts = 0
                self._skip_down_sent = False
                self._skip_focus_missing_reads = 0
                self._skip_activate_sent = False
                self._skip_activate_at = 0.0
                self._discard_confirm_sent = False
                self._discard_confirm_attempts = 0
                self._preview_missing_reads = 0
                self._preview_signature = None
                self._preview_signature_count = 0
                self._preview_response_at = 0.0
                self._pending_preview_signature = None
                self._preview_stall_count = 0
                self._force_selection_after_b = False
                self._reset_hand_completion()
                if self._last_state in {GameState.TURN_REVEAL, GameState.TURN_RESOLVE}:
                    self._preview_recoveries = 0
            if state is not GameState.TARGET_PLACEMENT:
                self._return_to_select_attempts = 0
                self._return_to_select_at = 0.0
            if (
                state in {GameState.TURN_REVEAL, GameState.TURN_RESOLVE}
                and not self._turn_result_counted
                and (self._place_sent or self._discard_confirm_sent)
            ):
                self._completed_turns += 1
                self._turn_result_counted = True
                if (self._selected_card_id in self.catalog.cards
                    and self._selected_card_id not in self._revealed_own_cards):
                    self._revealed_own_cards.append(self._selected_card_id)
            self._last_state = state

        if state in {GameState.TURN_SELECT, GameState.SPECIAL_SELECT}:
            return self._choose_and_select_card(observation)
        if state is GameState.TARGET_PLACEMENT:
            if self._force_selection_after_b and not self._card_confirmed:
                return self._choose_and_select_card(observation)
            if (
                self._selected_action is None
                and observation.hand is not None
                and observation.selection_profile is not None
                and (observation.hand_visibility is None
                     or observation.hand_visibility.selection_visible is not False)
            ):
                return self._choose_and_select_card(observation)
            return self._position_selected_card(observation, frame)
        if state is GameState.SKIP_DISCARD:
            return self._discard_for_skip(observation)
        if state in {
            GameState.TURN_REVEAL,
            GameState.TURN_RESOLVE,
            GameState.REDRAW,
            GameState.BATTLE_INTRO,
        }:
            self.phase = BattleExecutorPhase.WAITING_RESULT
            self.message = f"等待阶段完成: {state.value}"
            return False
        if state is GameState.SPECIAL_PLACEMENT:
            return self._position_selected_card(observation, frame)
        return False

    def _reset_hand_completion(self) -> None:
        self._completion_hand_key = None
        self._completion_hand_reads = 0
        self._last_completion_hand = None
        self._completion_wait_at = 0.0

    def _reset_rejected_placements(self) -> None:
        self._rejection_turn = self._completed_turns
        self._rejected_placements = set()
        self._gray_recovery_count = 0
        self._gray_recovery = None
        self._last_rejected_placement = None

    def _placement_rejection_key(self, action):
        shape = self.catalog.cards[action.card_id].shape.rotated(action.rotation)
        footprint = tuple(sorted((action.origin[0] + row, action.origin[1] + col,
                                  int(shape.cells[row][col])) for row, col in shape.occupied))
        return action.action_type.value, action.card_id, footprint

    def _recover_gray_target(self, observation) -> bool:
        hand = self._current_hand
        ids = tuple(getattr(hand, "card_ids", ()))
        if (self._place_sent or len(ids) != 4 or len(set(ids)) != 4
            or any(card not in self.catalog.cards for card in ids)):
            return False
        if self._gray_recovery_count >= self.MAX_GRAY_RECOVERIES:
            self.message = "本回合已恢复3次灰色目标仍未成功，已停止按键并保留诊断"
            return False
        action = self._selected_action
        self._rejected_placements.add(self._placement_rejection_key(action))
        self._last_rejected_placement = {"turn": self._completed_turns, "action": asdict(action)}
        if not self._pulse(Button.B):
            return False
        self._gray_recovery_count += 1
        self._gray_recovery = {"hand_ids": frozenset(ids), "reads": 0,
                               "last_hand": hand, "last_board": self._selection_board,
                               "at": time.perf_counter()}
        self._selected_action = self._selected_card_id = self._selected_target_slot = None
        self._selection_board = self._selection_sp = self._current_hand = None
        self._card_confirmed = False
        self.reset_preview_feedback()
        self._last_preview = None
        self._last_preview_attempts = []
        self.phase = BattleExecutorPhase.WAITING
        self.message = f"目标灰色不可放，已按B取消；等待原手牌和新棋盘重读（恢复{self._gray_recovery_count}/3）"
        return True

    def _observe_gray_recovery(self, observation) -> bool:
        pending = self._gray_recovery
        hand, board, visibility = observation.hand, observation.board, observation.hand_visibility
        cards = tuple(getattr(hand, "cards", ()))
        valid = (observation.state.state in {GameState.TURN_SELECT, GameState.SPECIAL_SELECT}
                 and hand is not None and not getattr(hand, "deck_mismatch", False)
                 and hand.confidence >= self.COMPLETION_HAND_CONFIDENCE
                 and len(cards) == 4 and frozenset(hand.card_ids) == pending["hand_ids"]
                 and all(getattr(card, "match_score", 0) >= self.COMPLETION_HAND_CONFIDENCE for card in cards)
                 and board is not None and not board.placement_preview_present
                 and visibility is not None and visibility.selection_visible is True
                 and self._last_selection_focus in {(1, 1), (1, 2), (2, 1), (2, 2)}
                 and getattr(observation, "discard_mode", None) is not True)
        if not valid:
            pending["reads"] = 0
        elif hand is not pending["last_hand"] and board is not pending["last_board"]:
            pending["reads"] += 1
            pending["last_hand"], pending["last_board"] = hand, board
        self.phase = BattleExecutorPhase.WAITING
        self.message = f"灰色目标已取消，等待原四张手牌和新棋盘独立确认（{pending['reads']}/2），不重发A或B"
        if pending["reads"] >= 2:
            self._gray_recovery = None
            self._selected_action = self._selected_card_id = self._selected_target_slot = None
            self._selection_board = self._selection_sp = self._current_hand = None
            self._card_confirmed = False
            self._card_confirm_attempts = 0
            self._last_state = GameState.UNKNOWN
            self._reset_special_flow()
            self._reset_rotation_feedback()
            self._reset_movement_feedback()
            self.message = "取消已确认，重新规划并排除本回合被拒绝的落点"
        elif time.perf_counter() - pending["at"] > 12:
            self.phase = BattleExecutorPhase.ERROR
            self.message = "取消灰色目标后未能确认原手牌与新棋盘，已停止按键并保留诊断"
        return False

    def _confirm_new_hand_after_commit(self, observation) -> bool:
        if not self._completion_wait_at:
            self._completion_wait_at = time.perf_counter()
        hand = observation.hand
        visibility = observation.hand_visibility
        cards = tuple(getattr(hand, "cards", ()))
        ids = tuple(getattr(hand, "card_ids", ()))
        valid = (
            self._selected_card_id is not None
            and hand is not None and not getattr(hand, "deck_mismatch", False)
            and getattr(hand, "confidence", 0.0) >= self.COMPLETION_HAND_CONFIDENCE
            and len(cards) == len(set(ids)) == 4
            and all(getattr(card, "match_score", 1.0) >= self.COMPLETION_HAND_CONFIDENCE
                    for card in cards)
            and self._selected_card_id not in ids
            and visibility is not None and visibility.selection_visible is True
            and self._last_selection_focus in {(1, 1), (1, 2), (2, 1), (2, 2)}
            and (not self._discard_confirm_sent or observation.discard_mode is False)
        )
        if not valid:
            self._completion_hand_key = None
            self._completion_hand_reads = 0
            self._last_completion_hand = None
            return False
        key = tuple((card.slot_row, card.slot_col, card.card_id) for card in cards)
        if key != self._completion_hand_key:
            self._completion_hand_key = key
            self._completion_hand_reads = 0
            self._last_completion_hand = None
        if hand is not self._last_completion_hand:
            self._completion_hand_reads += 1
            self._last_completion_hand = hand
        return self._completion_hand_reads >= 2

    def battle_memory_checkpoint(self) -> dict | None:
        count = getattr(self, "_completed_turns", None)
        if not isinstance(count, int):
            return None
        same_turn = getattr(self, "_rejection_turn", count) == count
        rejected = set(getattr(self, "_rejected_placements", ())) if same_turn else set()
        recoveries = getattr(self, "_gray_recovery_count", 0) if same_turn else 0
        pending_recovery = getattr(self, "_gray_recovery", None) if same_turn else None
        old_action = getattr(self, "_selected_action", None)
        if (same_turn and old_action is not None and not getattr(self, "_place_sent", False)
            and str(getattr(self, "message", "")).startswith("目标预览为灰色不可放")):
            rejected.add(VisionMctsBattleExecutor._placement_rejection_key(self, old_action))
            recoveries = max(1, recoveries)
        return {"completed_turns": count,
                "own_deck_ids": tuple(self.own_deck_ids),
                "own_cards": tuple(self._revealed_own_cards),
                "opponent_cards": tuple(self._revealed_opponent_cards),
                "rejected_placements": tuple(rejected), "gray_recoveries": recoveries,
                "gray_recovery_hand": tuple(pending_recovery["hand_ids"])
                    if pending_recovery is not None else None}

    def restore_battle_memory(self, checkpoint: dict) -> None:
        count = checkpoint.get("completed_turns")
        own = tuple(checkpoint.get("own_cards", ()))
        opponent = tuple(checkpoint.get("opponent_cards", ()))
        if (not isinstance(count, int) or not 0 <= count <= 12
            or count < self._completed_turns
            or tuple(checkpoint.get("own_deck_ids", ())) != tuple(self.own_deck_ids)
            or any(card not in self.catalog.cards for card in (*own, *opponent))):
            return
        # Restore confirmed history only. An unsubmitted action, its frozen
        # board and error must be re-read and planned after the core reload.
        self._completed_turns = count
        self._revealed_own_cards = list(own)
        self._revealed_opponent_cards = list(opponent)
        self._rejection_turn = count
        self._rejected_placements = {
            (kind, card, tuple(tuple(cell) for cell in footprint))
            for kind, card, footprint in checkpoint.get("rejected_placements", ())
        }
        self._gray_recovery_count = checkpoint.get("gray_recoveries", 0)
        pending_ids = checkpoint.get("gray_recovery_hand")
        if pending_ids is not None and len(set(pending_ids)) == 4:
            self._gray_recovery = {"hand_ids": frozenset(pending_ids), "reads": 0,
                                   "last_hand": None, "last_board": None,
                                   "at": time.perf_counter()}

    def pending_discard_checkpoint(self) -> dict | None:
        """Keep a submitted discard across reload until its hand change is read."""
        action = getattr(self, "_selected_action", None)
        if (not getattr(self, "_discard_confirm_sent", False) or action is None
            or action.action_type is not ActionType.SKIP):
            return None
        return {"card_id": action.card_id, "completed_turns": self._completed_turns,
                "turn_result_counted": self._turn_result_counted,
                "own_cards": tuple(self._revealed_own_cards),
                "opponent_cards": tuple(self._revealed_opponent_cards),
                "attempts": self._discard_confirm_attempts}

    def restore_pending_discard(self, checkpoint: dict) -> None:
        card_id = checkpoint.get("card_id")
        count = checkpoint.get("completed_turns")
        if (card_id not in self.catalog.cards or not isinstance(count, int)
            or self._completed_turns > count):
            return
        self._completed_turns = count
        self._revealed_own_cards = list(checkpoint["own_cards"])
        self._revealed_opponent_cards = list(checkpoint["opponent_cards"])
        self._turn_result_counted = bool(checkpoint["turn_result_counted"])
        self._selected_action = Action.skip(card_id)
        self._selected_card_id = card_id
        self._skip_activate_sent = True
        self._discard_confirm_sent = True
        self._discard_confirm_attempts = checkpoint["attempts"]
        self._discard_confirm_at = time.perf_counter()
        self._last_state = GameState.TURN_SELECT
        self.phase = BattleExecutorPhase.WAITING_RESULT
        self._reset_hand_completion()

    def pending_placement_checkpoint(self) -> dict | None:
        action = getattr(self, "_selected_action", None)
        if (not getattr(self, "_place_sent", False) or action is None
            or action.action_type not in {ActionType.PLACE, ActionType.SPECIAL}):
            return None
        return {"action": asdict(action), "completed_turns": self._completed_turns,
                "turn_result_counted": self._turn_result_counted,
                "own_cards": tuple(self._revealed_own_cards),
                "opponent_cards": tuple(self._revealed_opponent_cards),
                "attempts": self._placement_confirm_attempts}

    def restore_pending_placement(self, checkpoint: dict) -> None:
        saved = checkpoint.get("action") or {}
        count = checkpoint.get("completed_turns")
        if (saved.get("card_id") not in self.catalog.cards or not isinstance(count, int)
            or self._completed_turns > count):
            return
        action = Action(ActionType(saved["action_type"]), saved["card_id"], saved["rotation"],
                        tuple(saved["origin"]))
        if action.action_type not in {ActionType.PLACE, ActionType.SPECIAL}:
            return
        self._completed_turns = count
        self._revealed_own_cards = list(checkpoint["own_cards"])
        self._revealed_opponent_cards = list(checkpoint["opponent_cards"])
        self._turn_result_counted = bool(checkpoint["turn_result_counted"])
        self._selected_action = action
        self._selected_card_id = action.card_id
        self._card_confirmed = True
        self._place_sent = True
        self._place_sent_at = time.perf_counter()
        self._placement_confirm_attempts = checkpoint["attempts"]
        # A receipt cannot reconstruct the frozen board/SP/visual card shape.
        # Observe completion after reload; never retry A from this receipt.
        self._selection_board = None
        self._selection_sp = None
        self._last_state = GameState.TURN_SELECT
        self.phase = BattleExecutorPhase.WAITING_RESULT
        self._reset_hand_completion()

    def _choose_and_select_card(
        self,
        observation: BattleVisionObservation,
    ) -> bool:
        if getattr(observation.hand, "deck_mismatch", False):
            if observation.hand is not self._last_mismatch_hand:
                self._hand_mismatch_reads += 1
                self._last_mismatch_hand = observation.hand
            self._selected_action = None
            self._selected_card_id = None
            self._selected_target_slot = None
            self._current_hand = None
            self._card_confirmed = False
            self.controller.send_state(ControllerState.neutral())
            self.phase = (BattleExecutorPhase.ERROR if self._hand_mismatch_reads >= 2
                          else BattleExecutorPhase.WAITING)
            self.message = ("实际手牌与预设卡组不符，已取消旧动作；请核对本局使用的卡组"
                            if self._hand_mismatch_reads >= 2
                            else "手牌出现预设卡组之外的牌形，等待再次确认，不移动选牌焦点")
            return False
        self._hand_mismatch_reads = 0
        self._last_mismatch_hand = None
        if (self._selected_action is not None
            and self._selected_action.action_type is ActionType.SPECIAL
            and not self._card_confirmed):
            progress = self._prepare_special_mode(observation)
            if progress is not None:
                return progress
        elif self._selected_action is None and getattr(observation, "special_mode", None) is True:
            self.phase = BattleExecutorPhase.CHOOSING_CARD
            self.message = "先退出现有特殊攻击模式，再读取普通手牌规划"
            return self._pulse(Button.B)
        if (observation.hand_visibility is not None
            and observation.hand_visibility.selection_visible is False):
            self.phase = BattleExecutorPhase.WAITING
            self.message = "手牌区域已变灰，等待摆放阶段识别，不发送选牌按键"
            return False
        if (
            observation.hand is None
            or observation.selection_profile is None
            or observation.hand.confidence
            < self.HAND_CONFIDENCE_THRESHOLD
        ):
            self.phase = BattleExecutorPhase.WAITING
            self.message = (
                "等待四张手牌和当前选牌焦点稳定（高置信） "
                f"(手牌={'有' if observation.hand is not None else '无'}, "
                f"手牌匹配分数="
                f"{observation.hand.confidence if observation.hand is not None else 0:.2f}, "
                f"焦点={'有' if observation.selection_profile is not None else '无'})"
            )
            return False
        self._current_hand = observation.hand
        if self._selected_action is None:
            self._selected_target_slot = None
            self._selection_navigation_steps = 0
            self._reset_rotation_feedback()
            self._reset_movement_feedback()
            sp = getattr(observation, "sp", None)
            if (sp is not None and sp.self_value is None
                and getattr(sp, "self_reason", None) == "AWAITING_TWO_IDENTICAL_FRAMES"
                and self._sp_confirmation_waits < 3):
                self._sp_confirmation_waits += 1
                self.phase = BattleExecutorPhase.WAITING
                self.message = "等待HUD的SP余额连续两帧一致后规划"
                return False
            reliable_cards = self._bind_hand_shapes(observation.hand)
            self._selection_board = observation.board
            self._selection_sp = ((sp.self_value or 0, sp.opponent_value or 0)
                                  if sp is not None else (0, 0))
            state = self._build_battle_state(observation)
            if state is None:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "无法组装BattleState，缺少牌组、场地或手牌"
                return False
            legal = self.rules.legal_actions(state, Side.A)
            unavailable_cards = {
                card.card_id for card in observation.hand.cards
                if getattr(card, "available", None) is False
            }
            placements = tuple(
                action
                for action in legal
                if ((action.action_type is ActionType.PLACE and action.card_id not in unavailable_cards)
                or (action.action_type is ActionType.SPECIAL and self._selection_sp[0] > 0
                    and getattr(observation, "special_mode", None) is not None))
                and self._placement_rejection_key(action) not in self._rejected_placements
            )
            if observation.discard_mode is True:
                placements = ()  # An already open discard dialog accepts only a skip.
            reliable_placements = tuple(a for a in placements if a.card_id in reliable_cards)
            if placements and not reliable_placements:
                self.phase = BattleExecutorPhase.WAITING
                self.message = "手牌图形格数或特殊格未完整读出，等待重读后再规划"
                return False
            placements = reliable_placements
            if not placements:
                skips = tuple(
                    action
                    for action in legal
                    if action.action_type is ActionType.SKIP
                )
                if not skips:
                    self.phase = BattleExecutorPhase.ERROR
                    self.message = "当前没有可执行的普通摆放或弃牌动作"
                    return False
                self._selected_action = min(
                    skips,
                    key=lambda action: (
                        self.catalog.cards[action.card_id or ""].area
                    ),
                )
            else:
                placements = self._deduplicate_rotations(placements)
                self.policy.set_opponent_context(observation.opponent_context)
                self._selected_action = self.policy.choose(
                    self.rules,
                    state,
                    Side.A,
                    placements,
                )
                self._selected_action = self._canonicalize_rotation(
                    self._selected_action
                )
            self._selected_card_id = self._selected_action.card_id
            selected_card = next(card for card in observation.hand.cards
                                 if card.card_id == self._selected_card_id)
            self._selected_target_slot = (selected_card.slot_row, selected_card.slot_col)
            if self._selected_action.action_type is ActionType.SPECIAL:
                progress = self._prepare_special_mode(observation)
                if progress is not None:
                    return progress

        target = next(
            (
                card
                for card in observation.hand.cards
                if card.card_id == self._selected_card_id
            ),
            None,
        )
        if target is None:
            self.phase = BattleExecutorPhase.ERROR
            self.message = "当前动作引用的卡不在四张手牌中，已停止并保留诊断"
            return False
        focus = observation.selection_profile.focus_slot
        if focus is None:
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待选牌焦点槽位识别"
            return False
        target_slot = (target.slot_row, target.slot_col)
        if self._selected_target_slot is not None and target_slot != self._selected_target_slot:
            self._selected_action = None
            self._selected_card_id = None
            self._selected_target_slot = None
            self._card_confirmed = False
            self.phase = BattleExecutorPhase.WAITING
            self.message = "同一张牌的识别槽位发生变化，已取消旧动作，等待重新识别"
            return False
        self._selected_target_slot = target_slot
        action = self._selected_action
        assert action is not None
        if action.action_type is ActionType.SKIP:
            if observation.discard_mode is True:
                self._skip_activate_sent = True
                return self._discard_for_skip(observation)
            return self._open_skip_dialog(
                focus,
            )
        if getattr(target, "available", None) is False:
            self._selected_action = None
            self._selected_card_id = None
            self._card_confirmed = False
            self._card_confirm_attempts = 0
            self.phase = BattleExecutorPhase.WAITING
            self.message = f"{target.name} 已变灰不可用，重新规划其余亮色卡牌"
            return False
        direction = _hand_focus_direction(focus, target_slot)
        if direction is not None:
            if self._selection_navigation_steps >= 8:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "选牌焦点连续移动8次仍未到达目标，已停止循环并保留诊断"
                return False
            self._card_confirmed = False
            self.phase = BattleExecutorPhase.CHOOSING_CARD
            self.message = (
                f"MCTS选择 {target.name}，目标旋转{action.rotation}，"
                f"起点{action.origin}；移动选牌焦点 "
                f"{focus}->{target_slot}"
            )
            if self._pulse(direction):
                self._selection_navigation_steps += 1
                return True
            return False

        if self._card_confirmed:
            if time.perf_counter() - self._card_confirmed_at < 1.2:
                self.phase = BattleExecutorPhase.WAITING
                self.message = "卡牌已确认，等待进入摆放阶段"
                return False
            if self._card_confirm_attempts >= self.MAX_CONFIRM_ATTEMPTS:
                self.phase = BattleExecutorPhase.ERROR
                self.message = (f"{target.name} 在槽位{target_slot}连续确认"
                                f"{self._card_confirm_attempts}次仍未进入摆放；已停止按键，保留诊断")
                return False
            self.message = f"卡牌确认未生效，重试A（{self._card_confirm_attempts + 1}/{self.MAX_CONFIRM_ATTEMPTS}）"
            if self._pulse(Button.A):
                self._card_confirm_attempts += 1
                self._card_confirmed_at = time.perf_counter()
            return False

        if self._card_confirm_attempts >= self.MAX_CONFIRM_ATTEMPTS:
            self.phase = BattleExecutorPhase.ERROR
            self.message = "选牌确认次数达到上限，已停止按键并保留诊断"
            return False

        self.phase = BattleExecutorPhase.CHOOSING_CARD
        self.message = (
            f"MCTS选择 {target.name}，目标旋转{action.rotation}，"
            f"起点{action.origin}；确认卡牌"
        )
        if self._pulse(Button.A):
            self._card_confirmed = True
            self._card_confirm_attempts += 1
            self._card_confirmed_at = time.perf_counter()
            return True
        return False

    def _reset_special_flow(self) -> None:
        self._special_down_sent = False
        self._special_activation_attempts = 0
        self._special_activate_at = 0.0
        self._special_return_attempts = 0
        self._special_mode_missing_reads = 0
        self._special_mode_wait_reads = 0

    def _observe_revealed_cards(self, observation) -> None:
        """Only confirmed turns may add two-frame, unambiguous reveal IDs."""
        if observation.reveal is None or not (self._place_sent or self._discard_confirm_sent):
            self._reveal_pending.clear()
            self._last_reveal_observation = None
            return
        if observation.reveal is self._last_reveal_observation:
            return  # Runtime may reuse a cached reveal between reader samples.
        self._last_reveal_observation = observation.reveal
        for side, known in (("own", self._revealed_own_cards),
                            ("opponent", self._revealed_opponent_cards)):
            reading = getattr(observation.reveal, side)
            card_id = reading.card_id
            options = getattr(reading, "candidates", ())
            margin = (options[0].score - options[1].score if len(options) >= 2 else 1.0)
            reliable = (card_id in self.catalog.cards
                        and getattr(reading, "confidence", 0.0) >= 0.85
                        and (side == "own" and card_id == self._selected_card_id or margin >= 0.05))
            if side == "own" and card_id != self._selected_card_id:
                reliable = False
            if not reliable:
                self._reveal_pending.pop(side, None)
                continue
            previous, reads = self._reveal_pending.get(side, (None, 0))
            reads = reads + 1 if previous == card_id else 1
            self._reveal_pending[side] = (card_id, reads)
            if reads >= 2 and card_id not in known:
                known.append(card_id)

    def _prepare_special_mode(self, observation) -> bool | None:
        """Open the mode, require visual acknowledgement, then reach a card."""
        mode = getattr(observation, "special_mode", None)
        profile = observation.selection_profile
        focus = profile.focus_slot if profile is not None else None
        if mode is None:
            self._special_mode_wait_reads += 1
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待特殊攻击按钮文字明确，不在未知模式下发送A"
            if self._special_mode_wait_reads >= 8:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "特殊攻击模式持续不明，已停止按键并保留诊断"
            return False
        self._special_mode_wait_reads = 0
        if mode is True:
            if focus is not None and focus[0] in {1, 2}:
                return None
            if self._special_return_attempts >= 3:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "特殊攻击已开启，但无法确认手牌焦点；已停止按键"
                return False
            self.message = "特殊攻击已开启，返回手牌选择"
            if self._pulse(Button.DPAD_UP):
                self._special_return_attempts += 1
                return True
            return False
        if focus == (3, 2):
            if time.perf_counter() - self._special_activate_at < 1.0:
                self.phase = BattleExecutorPhase.WAITING
                self.message = "等待特殊攻击按钮切换为关闭提示"
                return False
            if self._special_activation_attempts >= 3:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "未检测到特殊攻击模式开启，已停止确认并保留诊断"
                return False
            self.message = "开启特殊攻击，等待视觉确认"
            if self._pulse(Button.A):
                self._special_down_sent = True
                self._special_activation_attempts += 1
                self._special_activate_at = time.perf_counter()
                return True
            return False
        if focus is None:
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待手牌或特殊攻击按钮的实际焦点"
            return False
        self._special_down_sent = False
        direction = _hand_focus_direction(focus, (2, 2))
        if direction is not None:
            self.message = f"移动到特殊攻击入口 {focus}->(2, 2)"
            return self._pulse(direction)
        self.message = "移动到开启特殊攻击按钮"
        if self._pulse(Button.DPAD_DOWN):
            self._special_down_sent = True
            return True
        return False

    def _open_skip_dialog(
        self,
        focus: tuple[int, int] | None,
    ) -> bool:
        focus = tuple(focus) if focus is not None else None
        self._last_selection_focus = focus
        if self._skip_activate_sent:
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待弃牌选择界面"
            return False
        if focus is None:
            self._skip_focus_missing_reads += 1
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待跳过按钮的小鱿鱼焦点，不凭上次向下操作发送A"
            if self._skip_focus_missing_reads >= 8:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "跳过按钮焦点持续不明，已停止按键并保留诊断"
            return False
        self._skip_focus_missing_reads = 0
        if focus == (3, 1):
            self.phase = BattleExecutorPhase.CHOOSING_CARD
            self.message = "已确认跳过按钮焦点，打开弃牌界面"
            if self._pulse(Button.A):
                self._skip_activate_sent = True
                self._skip_activate_at = time.perf_counter()
                return True
            return False
        direction = _hand_focus_direction(focus, (3, 1))
        self.phase = BattleExecutorPhase.CHOOSING_CARD
        self.message = f"移动到跳过按钮 {focus}->(3, 1)"
        if self._pulse(direction):
            self._skip_down_sent = direction is Button.DPAD_DOWN
            return True
        return False

    def _discard_for_skip(
        self,
        observation: BattleVisionObservation,
    ) -> bool:
        action = self._selected_action
        if (
            action is None
            or action.action_type is not ActionType.SKIP
            or observation.hand is None
            or observation.hand.confidence < self.HAND_CONFIDENCE_THRESHOLD
            or observation.selection_profile is None
        ):
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待弃牌卡牌和焦点识别"
            return False
        if observation.discard_mode is not True:
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待返回文字确认弃牌模式，不把亮色手牌当作跳过成功"
            return False
        target = next(
            (
                card
                for card in observation.hand.cards
                if card.card_id == action.card_id
            ),
            None,
        )
        focus = observation.selection_profile.focus_slot
        if target is None or focus is None:
            self.phase = BattleExecutorPhase.ERROR
            self.message = "无法在弃牌界面定位MCTS选择的卡"
            return False
        target_slot = (target.slot_row, target.slot_col)
        direction = _hand_focus_direction(focus, target_slot)
        if direction is not None:
            self.phase = BattleExecutorPhase.CHOOSING_CARD
            self.message = f"选择弃牌 {target.name}"
            return self._pulse(direction)
        if self._discard_confirm_sent:
            if time.perf_counter() - self._discard_confirm_at >= 1.2:
                if self._discard_confirm_attempts >= self.MAX_CONFIRM_ATTEMPTS:
                    self.phase = BattleExecutorPhase.ERROR
                    self.message = "弃牌连续确认未生效，已停止按键并保留诊断"
                    return False
                self.message = "弃牌确认未生效，重试A"
                if self._pulse(Button.A):
                    self._discard_confirm_attempts += 1
                    self._discard_confirm_at = time.perf_counter()
            else:
                self.phase = BattleExecutorPhase.WAITING
                self.message = f"已确认弃牌 {target.name}"
            return False
        self.phase = BattleExecutorPhase.CHOOSING_CARD
        self.message = f"确认弃牌 {target.name}"
        if self._pulse(Button.A):
            self._discard_confirm_sent = True
            self._discard_confirm_attempts += 1
            self._discard_confirm_at = time.perf_counter()
            return True
        return False

    def _position_selected_card(
        self,
        observation: BattleVisionObservation,
        frame: np.ndarray | None,
    ) -> bool:
        action = self._selected_action
        if action is None or action.card_id is None:
            self.phase = BattleExecutorPhase.WAITING
            if self._return_to_select_attempts >= 3:
                self.message = "已连续发送3次B，等待选牌阶段状态更新"
                return False
            self.message = "缺少MCTS动作，连续按3次B返回选牌界面"
            for _ in range(3 - self._return_to_select_attempts):
                if not self._pulse(Button.B):
                    break
                self._return_to_select_attempts += 1
            self._return_to_select_at = time.perf_counter()
            self._force_selection_after_b = True
            self._card_confirmed = False
            self._card_confirmed_at = 0.0
            self._reset_rotation_feedback()
            return False
        if self._place_sent and time.perf_counter() - self._place_sent_at > 20:
            self._wait_for_submitted_placement("等待回合推进")
            return False
        if (action.action_type is ActionType.SPECIAL
            and getattr(observation, "special_mode", None) is not True):
            if self._place_sent:
                self._wait_for_submitted_placement("特殊攻击模式已变化，等待回合推进")
                return False
            self._special_mode_missing_reads += 1
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待确认摆放界面仍处于特殊攻击模式"
            if self._special_mode_missing_reads >= 8:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "摆放时特殊攻击模式证据丢失，已停止确认并保留诊断"
            return False
        self._special_mode_missing_reads = 0
        if self._place_sent:
            if (observation.hand_visibility is not None
                and observation.hand_visibility.selection_visible is True):
                self._wait_for_submitted_placement("手牌光标已恢复，等待选牌阶段确认")
                return False
            if time.perf_counter() - self._place_sent_at < 1.2:
                self.phase = BattleExecutorPhase.PLACING
                self.message = "已发送放置A，等待亮牌"
                return False
            # A retry must pass the same position/color checks as the first A.
        if (
            self._card_confirmed
            and time.perf_counter() - self._card_confirmed_at
            < self.PLACEMENT_SETTLE_SECONDS
        ):
            remaining = (
                self.PLACEMENT_SETTLE_SECONDS
                - (time.perf_counter() - self._card_confirmed_at)
            )
            self.phase = BattleExecutorPhase.WAITING
            self.message = (
                "卡牌已确认，等待摆放界面和预览动画稳定 "
                f"({remaining:.1f}s)"
            )
            return False
        if observation.board is None:
            if self._place_sent:
                self._wait_for_submitted_placement("等待亮牌或新手牌")
                return False
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待棋盘与摆放预览"
            return False
        if time.perf_counter() - self._preview_response_at < self.PREVIEW_RESPONSE_SECONDS:
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待预览响应上一条旋转或移动"
            return False
        preview = None
        native_preview = False
        self._last_preview_attempts = []
        board_region = (getattr(self.runtime, "preview_region", None)
                        or getattr(self.runtime, "board_region", None))
        stage_id = getattr(self.runtime, "stage_id", None)
        if frame is not None and board_region is not None and stage_id is not None:
            preview = self.placement_reader.read_frame(
                frame, board_region, stage_id, action.card_id, observation.board,
                temporal=True, background=self._selection_board,
                special_attack=action.action_type is ActionType.SPECIAL,
            )
            native_preview = preview is not None
            self._last_preview_attempts.append(dict(getattr(self.placement_reader, "last_diagnostics", {})))
        if (preview is None and observation.board.placement_preview_present
            and getattr(self.placement_reader, "last_diagnostics", {}).get("failure")
            != "AMBIGUOUS_SPECIAL_ORIENTATION"):
            if action.action_type is ActionType.SPECIAL:
                preview = self.placement_reader.read(observation.board, action.card_id, special_attack=True)
            else:
                preview = self.placement_reader.read(observation.board, action.card_id)
            self._last_preview_attempts.append(dict(getattr(self.placement_reader, "last_diagnostics", {})))
        self._last_preview = preview
        if self._place_sent and not self._placement_retry_is_visible(observation, preview):
            self._wait_for_submitted_placement("预览已变化，等待亮牌或新手牌")
            return False
        if preview is None:
            if self._place_sent:
                self.phase = BattleExecutorPhase.WAITING_RESULT
                self.message = "放置已确认且预览已消失，等待亮牌或选牌焦点恢复"
                if time.perf_counter() - self._place_sent_at > 20:
                    self.phase = BattleExecutorPhase.ERROR
                    self.message = "放置确认后长时间未观察到回合推进，已保留现场并停止按键"
                return False
            self.phase = BattleExecutorPhase.WAITING
            self._preview_signature = None
            self._preview_signature_count = 0
            self._preview_missing_reads += 1
            failure = next((attempt.get("failure") for attempt in self._last_preview_attempts
                            if attempt.get("failure")), None)
            reason = {
                "AMBIGUOUS_SPECIAL_ORIENTATION": "斜线已检出，特殊点位置不明，无法确定朝向",
                "SHAPE_MISMATCH": "预览纹理已检出，但格子与当前卡牌形状不匹配",
                "NO_DIAGONAL_CELLS": "未检出斜线预览格子",
            }.get(failure, "未识别到可信摆放预览")
            self.message = f"{reason}，等待重读 ({self._preview_missing_reads}/12)"
            if self._preview_missing_reads < 12:
                return False
            if self._preview_recoveries >= 2:
                self.phase = BattleExecutorPhase.ERROR
                self.message = f"预览识别连续失败，已停止操作；{reason}"
                return False
            if self._pulse(Button.B):
                self._preview_recoveries += 1
                self._preview_missing_reads = 0
                self._card_confirmed = False
                self._place_sent = False
                self._force_selection_after_b = True
                self._reset_rotation_feedback()
                if self._pending_movement is not None:
                    self._coarse_movement_disabled = True
                    self._pending_movement = None
                self.message = "预览识别失败，按B返回选牌后重新识别"
                return True
            return False
        self._preview_missing_reads = 0
        observed_rotation_signature = self._rotation_signature(action.card_id, preview.rotation)
        observed_rotation = next(rotation for rotation in range(4)
                                 if self._rotation_signature(action.card_id, rotation)
                                 == observed_rotation_signature)
        signature = (observed_rotation, preview.anchor, preview.valid)
        assert action.rotation is not None
        target_anchor = self._action_anchor(action)
        rotation_matches = (
            observed_rotation_signature
            == self._rotation_signature(action.card_id, action.rotation)
        )
        now = time.perf_counter()
        if now - self._preview_response_at < self.PREVIEW_RESPONSE_SECONDS:
            self.phase = BattleExecutorPhase.WAITING
            self.message = (
                "等待预览响应上一条旋转或移动 "
                f"当前{preview.anchor} 目标{target_anchor} "
                f"({self.PREVIEW_RESPONSE_SECONDS - (now - self._preview_response_at):.1f}s)"
            )
            return False
        expected_rotation = (self._pending_rotation["expected_rotation"]
                             if self._pending_rotation is not None else self._verified_rotation)
        if self._pending_rotation is not None:
            self._rotation_feedback_reads += 1
        if (expected_rotation is not None
            and observed_rotation_signature
            != self._rotation_signature(action.card_id, expected_rotation)):
            self._preview_signature = None
            self._preview_signature_count = 0
            self._rotation_mismatch_reads += 1
            self.phase = BattleExecutorPhase.WAITING
            self.message = (
                f"旋转反馈不一致：观测r={preview.rotation}，期望r={expected_rotation}；"
                "等待重读，不反向纠正"
            )
            if (self._rotation_feedback_reads >= self.MAX_ROTATION_FEEDBACK_READS
                or self._rotation_mismatch_reads >= self.MAX_ROTATION_FEEDBACK_READS):
                self.phase = BattleExecutorPhase.ERROR
                self.message = (
                    f"旋转反馈持续不一致：观测r={preview.rotation}，期望r={expected_rotation}；"
                    "已停止按键并保留牌形与旋转记录"
                )
            return False
        self._rotation_mismatch_reads = 0
        if self._pending_preview_signature is not None:
            if signature == self._pending_preview_signature:
                self._preview_stall_count += 1
            else:
                self._preview_stall_count = 0
            self._pending_preview_signature = None
        if (
            self._preview_stall_count >= 2
            and (preview.anchor != target_anchor or not rotation_matches)
        ):
            self.phase = BattleExecutorPhase.ERROR
            self.message = "预览连续未响应旋转或移动，已停止操作；检查识别坐标和控制器响应"
            return False
        if signature == self._preview_signature:
            self._preview_signature_count += 1
        else:
            self._preview_signature = signature
            self._preview_signature_count = 1
        if self._preview_signature_count >= self.PREVIEW_MOTION_STABLE_FRAMES:
            self._verified_rotation = preview.rotation
            self._pending_rotation = None
            self._rotation_feedback_reads = 0
        elif self._rotation_feedback_reads >= self.MAX_ROTATION_FEEDBACK_READS:
            self.phase = BattleExecutorPhase.ERROR
            self.message = "旋转反馈未能连续稳定确认，已停止按键并保留牌形与旋转记录"
            return False
        required_frames = (self.PREVIEW_STABLE_FRAMES
                           if preview.anchor == target_anchor and rotation_matches
                           else self.PREVIEW_MOTION_STABLE_FRAMES)
        if self._preview_signature_count < required_frames:
            self.phase = BattleExecutorPhase.WAITING
            self.message = (
                "确认摆放预览稳定 "
                f"{preview.anchor} r={preview.rotation} "
                f"{'黄色可放' if preview.valid else '灰色不可放'} "
                f"score={preview.score:.2f} "
                f"({self._preview_signature_count}/"
                f"{required_frames})"
            )
            return False

        memory = getattr(self.runtime, "special_cell_memory", None)
        if memory is not None and native_preview and preview.score >= .95 and not self._place_sent:
            shape = self.catalog.cards[action.card_id].shape.rotated(preview.rotation)
            excluded = frozenset((preview.origin[0] + row + dr, preview.origin[1] + col + dc)
                                 for row, col in shape.occupied
                                 for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            memory.update(observation.board, allow_new=True, trusted_preview_exclusions=excluded)

        if self._pending_movement is not None:
            movement = self._pending_movement
            movement["actual_anchor"] = preview.anchor
            movement["difference"] = tuple(actual - expected for actual, expected
                                           in zip(preview.anchor, movement["expected_anchor"]))
            movement["matched"] = preview.anchor == movement["expected_anchor"]
            self._last_movement_batch = movement
            self._pending_movement = None
            if not movement["matched"]:
                self._coarse_movement_disabled = True
                self.phase = BattleExecutorPhase.WAITING
                self.message = (
                    f"粗移反馈偏差：预期{movement['expected_anchor']}，实际{preview.anchor}；"
                    "本张牌改为逐格精调"
                )
                return False

        if not rotation_matches:
            if self._rotation_commands >= 8:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "旋转方向连续未收敛，已停止按键并保留点阵/斜线诊断"
                return False
            self.phase = BattleExecutorPhase.POSITIONING
            target_signature = self._rotation_signature(action.card_id, action.rotation)
            delta = min(
                ((rotation - preview.rotation) % 4 for rotation in range(4)
                 if self._rotation_signature(action.card_id, rotation) == target_signature),
                key=lambda value: (min(value, 4 - value), value),
            )
            button = Button.X if delta == 3 else Button.Y
            presses = min(delta, 4 - delta)
            self.message = (
                f"预览{preview.rotation}->{action.rotation}，"
                f"还需要按{button.value} {presses}次，本次按1次后等待视觉确认"
            )
            if self._pulse(button):
                self._rotation_commands += 1
                self._record_preview_command(signature, rotation_button=button)
                return True
            return False

        if preview.anchor != target_anchor:
            button = _path_movement_button(
                observation.board,
                preview.anchor,
                target_anchor,
            )
            self.phase = BattleExecutorPhase.POSITIONING
            count = self._movement_pulse_count(observation.board, preview, target_anchor, button)
            if count > 1:
                self.message = (
                    f"粗移预览锚点 {preview.anchor}->{target_anchor} "
                    f"使用 {button.value} 连续{count}格，随后视觉重读"
                )
                return self._pulse_movement_batch(button, count, signature)
            self.message = (
                f"移动预览锚点 {preview.anchor}->{target_anchor} "
                f"使用 {button.value}"
            )
            if self._pulse(button):
                self._record_preview_command(signature)
                return True
            return False

        if not preview.valid:
            self.phase = BattleExecutorPhase.ERROR
            self.message = "目标预览为灰色不可放，已停止确认；检查地图、棋盘识别和目标坐标"
            cells = getattr(observation.board, "cells", ())
            if action.action_type is ActionType.SPECIAL and cells:
                shape = self.catalog.cards[action.card_id].shape.rotated(action.rotation)
                blocked = [(action.origin[0] + row, action.origin[1] + col)
                           for row, col in shape.occupied
                           if 0 <= action.origin[0] + row < len(cells)
                           and 0 <= action.origin[1] + col < len(cells[0])
                           and cells[action.origin[0] + row][action.origin[1] + col]
                               in {Cell.SELF_SPECIAL, Cell.OPPONENT_SPECIAL}]
                if blocked:
                    self.message += f"；当前棋盘识别到覆盖特殊点 {sorted(blocked)}"
            if self._recover_gray_target(observation):
                return True
            return False

        self.phase = BattleExecutorPhase.PLACING
        if self._selection_board is not None:
            state = self._build_battle_state(observation)
            if state is None or action not in self.rules.legal_actions(state, Side.A):
                self.phase = BattleExecutorPhase.ERROR
                self.message = "目标不满足选牌时棋盘与SP的放牌规则，已停止确认"
                return False
        self.message = f"位置已到达 {target_anchor}，按A放置"
        if self._placement_confirm_attempts >= self.MAX_CONFIRM_ATTEMPTS:
            self.phase = BattleExecutorPhase.ERROR
            self.message = "放置连续确认未生效，已停止按键并保留诊断"
            return False
        if self._pulse(Button.A):
            self._placement_confirm_attempts += 1
            self._place_sent = True
            self._place_sent_at = time.perf_counter()
            self._preview_signature = None
            self._preview_signature_count = 0
            return True
        return False

    def _wait_for_submitted_placement(self, reason: str) -> None:
        self._preview_signature = None
        self._preview_signature_count = 0
        self.phase = BattleExecutorPhase.WAITING_RESULT
        self.message = f"已发送放置A，{reason}，不调整已提交位置"
        if time.perf_counter() - self._place_sent_at > 20:
            self.phase = BattleExecutorPhase.ERROR
            self.message = "放置提交后长时间未确认回合推进，已停止按键并保留诊断"

    def _placement_retry_is_visible(self, observation, preview) -> bool:
        action = self._selected_action
        visibility = observation.hand_visibility
        hand = observation.hand
        return (
            self._selection_board is not None
            and preview is not None and preview.valid and preview.score >= 0.95
            and preview.anchor == self._action_anchor(action)
            and self._rotation_signature(action.card_id, preview.rotation)
                == self._rotation_signature(action.card_id, action.rotation)
            and visibility is not None and visibility.selection_visible is False
            and self._last_selection_focus is None
            and hand is not None and not getattr(hand, "deck_mismatch", False)
            and hand.confidence >= self.COMPLETION_HAND_CONFIDENCE
            and action.card_id in hand.card_ids
            and (action.action_type is not ActionType.SPECIAL or self._selection_sp is not None)
        )

    def _action_anchor(self, action: Action) -> tuple[int, int]:
        assert action.card_id is not None
        assert action.rotation is not None
        assert action.origin is not None
        occupied = self.catalog.cards[action.card_id].shape.rotated(
            action.rotation
        ).occupied
        first = min(occupied)
        return (
            action.origin[0] + first[0],
            action.origin[1] + first[1],
        )

    def _deduplicate_rotations(
        self,
        actions: tuple[Action, ...],
    ) -> tuple[Action, ...]:
        unique = {}
        for action in actions:
            assert action.card_id is not None
            assert action.rotation is not None
            key = (
                action.card_id,
                self._rotation_signature(action.card_id, action.rotation),
                action.origin,
            )
            unique.setdefault(key, action)
        return tuple(unique.values())

    def _canonicalize_rotation(self, action: Action) -> Action:
        assert action.card_id is not None
        assert action.rotation is not None
        signature = self._rotation_signature(
            action.card_id,
            action.rotation,
        )
        canonical = next(
            rotation
            for rotation in range(4)
            if self._rotation_signature(action.card_id, rotation)
            == signature
        )
        return replace(action, rotation=canonical)

    def _rotation_signature(
        self,
        card_id: str,
        rotation: int,
    ) -> tuple[tuple[int, int, int], ...]:
        shape = self.catalog.cards[card_id].shape.rotated(rotation)
        anchor = min(shape.occupied)
        return tuple(
            (
                row - anchor[0],
                col - anchor[1],
                int(shape.cells[row][col]),
            )
            for row, col in sorted(shape.occupied)
        )

    def _build_battle_state(
        self,
        observation: BattleVisionObservation,
    ) -> BattleState | None:
        if (
            observation.board is None
            or observation.hand is None
            and self._current_hand is None
            or self.runtime.stage_id is None
        ):
            return None
        base_stage = self.catalog.stages[self.runtime.stage_id]
        # The striped yellow preview is not committed ink. Plan and recheck
        # against the board captured while the hand was still selectable.
        board = (self._selection_board or observation.board).cells
        stage = replace(
            base_stage,
            cells=board,
            special_points=(),
            source_max_cells=sum(
                cell in PLAYABLE_CELLS
                for row in board
                for cell in row
            ),
        )
        hand_observation = self._current_hand or observation.hand
        assert hand_observation is not None
        hand = hand_observation.card_ids
        own_draw = tuple(
            card_id
            for card_id in self.own_deck_ids
            if (
                card_id not in hand
                and card_id not in self._revealed_own_cards
            )
        )
        opponent_deck_id = (
            observation.opponent_context.deck_id
            if observation.opponent_context is not None
            else self.fallback_opponent_deck_id
        )
        if (
            opponent_deck_id is None
            or opponent_deck_id not in self.catalog.decks
        ):
            return None
        opponent_deck = self.catalog.decks[opponent_deck_id].card_ids
        opponent_remaining = [
            card_id
            for card_id in opponent_deck
            if card_id not in self._revealed_opponent_cards
        ]
        opponent_hand = tuple(opponent_remaining[:4])
        opponent_draw = tuple(opponent_remaining[4:])
        if len(opponent_hand) < 4:
            return None
        # Activated board tiles are not a spendable balance. The HUD includes
        # pass income and past Special Attack expenses; unknown means no SP
        # can be safely assumed for planning a live action.
        sp = getattr(observation, "sp", None)
        own_sp = (sp.self_value if sp is not None and sp.self_value is not None else 0)
        opponent_sp = (sp.opponent_value if sp is not None and sp.opponent_value is not None else 0)
        if self._selection_sp is not None and self._selection_board is not None:
            own_sp, opponent_sp = self._selection_sp
        return BattleState(
            stage=stage,
            board=board,
            player_a=PlayerState(
                side=Side.A,
                hand=hand,
                draw_pile=own_draw,
                sp=own_sp,
            ),
            player_b=PlayerState(
                side=Side.B,
                hand=opponent_hand,
                draw_pile=opponent_draw,
                sp=opponent_sp,
            ),
            turns_remaining=max(0, 12 - self._completed_turns),
        )

    def _pulse(self, button: Button) -> bool:
        if self._motion_cancelled.is_set():
            if self.phase is not BattleExecutorPhase.ERROR:
                self.phase = BattleExecutorPhase.WAITING
                self.message = "自动按键已暂停，等待继续任务"
            return False
        try:
            pulse_buttons(self.controller, (button,))
            if hasattr(self.placement_reader, "reset_history"):
                self.placement_reader.reset_history()
            self._last_command = {"button": button.value, "sent_at": time.time(), "success": True}
            return True
        except RuntimeError:
            self._last_command = {"button": button.value, "sent_at": time.time(), "success": False}
            self.phase = BattleExecutorPhase.ERROR
            self.message = "控制器发送失败"
            return False

    def cancel_pending_motion(self) -> None:
        """Latch cancellation so an in-flight short batch cannot send more keys."""
        self._motion_cancelled.set()

    def resume_pending_motion(self) -> None:
        self._motion_cancelled.clear()

    def _reset_movement_feedback(self) -> None:
        self._pending_movement = None
        self._last_movement_batch = None
        self._coarse_movement_disabled = False

    def _movement_pulse_count(self, board, preview, target, button: Button) -> int:
        if (self._coarse_movement_disabled or self._verified_rotation is None
            or self._pending_rotation is not None or preview.score < 0.95):
            return 1
        delta = _direction_delta(button)
        distance = abs(target[0] - preview.anchor[0]) if delta[0] else abs(target[1] - preview.anchor[1])
        if distance < self.COARSE_MOVEMENT_MIN_DISTANCE:
            return 1
        count = min(self.COARSE_MOVEMENT_STEPS, distance - 2)
        # A wall/void can change cursor response. Keep the full short path and
        # one extra square inside the board, without crossing an axis change.
        if not all(_playable_anchor(board, (preview.anchor[0] + delta[0] * offset,
                                             preview.anchor[1] + delta[1] * offset))
                   for offset in range(count + 2)):
            return 1
        if not all(_path_movement_button(board,
                       (preview.anchor[0] + delta[0] * offset, preview.anchor[1] + delta[1] * offset),
                       target) is button for offset in range(1, count + 1)):
            return 1
        return count

    def _pulse_movement_batch(self, button: Button, count: int, signature) -> bool:
        delta = _direction_delta(button)
        sent = 0
        for _ in range(count):
            if self._motion_cancelled.is_set() or not self._pulse(button):
                break
            sent += 1
        if sent:
            self._record_preview_command(signature)
            self._pending_movement = {
                "button": button.value, "requested_steps": count, "sent_steps": sent,
                "before_anchor": signature[1],
                "expected_anchor": (signature[1][0] + delta[0] * sent,
                                    signature[1][1] + delta[1] * sent),
                "cancelled": self._motion_cancelled.is_set(),
            }
            self._last_movement_batch = self._pending_movement
        if self._motion_cancelled.is_set() and self.phase is not BattleExecutorPhase.ERROR:
            self.phase = BattleExecutorPhase.WAITING
            self.message = f"粗移已暂停，已发送{sent}/{count}格；继续时先重读实际预览"
        return sent > 0

    def diagnostic_snapshot(self) -> dict:
        preview = self._last_preview
        action = self._selected_action
        return {
            "phase": self.phase.value, "message": self.message,
            "completed_turns": self._completed_turns,
            "action": asdict(action) if action else None,
            "card_name": self.catalog.cards[action.card_id].name_zh if action and action.card_id else None,
            "selected_target_slot": self._selected_target_slot,
            "selection_navigation_steps": self._selection_navigation_steps,
            "hand_mismatch_reads": self._hand_mismatch_reads,
            "card_confirmed": self._card_confirmed, "placement_sent": self._place_sent,
            "card_confirm_attempts": self._card_confirm_attempts,
            "placement_confirm_attempts": self._placement_confirm_attempts,
            "discard_confirm_attempts": self._discard_confirm_attempts,
            "turn_completion": {
                "new_hand_reads": self._completion_hand_reads,
                "counted": self._turn_result_counted,
                "discard_submitted": self._discard_confirm_sent,
            },
            "rejection_recovery": {
                "pending": self._gray_recovery is not None,
                "attempts": self._gray_recovery_count,
                "excluded_targets": len(self._rejected_placements),
                "last_rejected": self._last_rejected_placement,
                "selection_reads": self._gray_recovery["reads"] if self._gray_recovery else 0,
            },
            "remembered_special_cells": [
                {"row": row, "col": col, "cell": int(owner)}
                for (row, col), owner in sorted(getattr(
                    getattr(self.runtime, "special_cell_memory", None), "_cells", {}).items())
            ],
            "rotation_commands": self._rotation_commands,
            "rotation_feedback": {
                "verified_rotation": self._verified_rotation,
                "pending": self._pending_rotation,
                "reads": self._rotation_feedback_reads,
                "mismatch_reads": self._rotation_mismatch_reads,
            },
            "movement_feedback": {
                "pending": self._pending_movement,
                "last_batch": self._last_movement_batch,
                "coarse_disabled": self._coarse_movement_disabled,
                "cancelled": self._motion_cancelled.is_set(),
            },
            "selection_sp": self._selection_sp,
            "revealed_own_cards": self._revealed_own_cards,
            "revealed_opponent_cards": self._revealed_opponent_cards,
            "reveal_pending": self._reveal_pending,
            "special_activation_attempts": self._special_activation_attempts,
            "skip_button_focused": self._last_selection_focus == (3, 1),
            "selection_focus": self._last_selection_focus,
            "skip_navigation_down_sent": self._skip_down_sent,
            "skip_focus_missing_reads": self._skip_focus_missing_reads,
            "skip_activated": self._skip_activate_sent,
            "preview": {
                "rotation": preview.rotation, "origin": preview.origin,
                "anchor": preview.anchor, "score": preview.score, "valid": preview.valid,
            } if preview is not None else None,
            "preview_attempts": self._last_preview_attempts,
            "preview_missing_reads": self._preview_missing_reads,
            "preview_stable_reads": self._preview_signature_count,
            "preview_stalls": self._preview_stall_count,
            "last_command": self._last_command,
            "selection_board": self._selection_board.cells if self._selection_board else None,
            "cached_hand": self._current_hand.card_ids if self._current_hand else None,
            "visual_shapes": self._visual_shapes,
            "search": asdict(self.policy.last_stats) if self.policy.last_stats else None,
        }

    def _bind_hand_shapes(self, hand) -> set[str]:
        """Use the actual visible orientation, including chirality, for play.

        Identification permits mirrored templates. That must not silently make
        the planner assume a mirrored card can be produced with rotation keys.
        """
        cards = dict(self.catalog.cards)
        reliable = set()
        changed = False
        for observed in hand.cards:
            if getattr(observed, "match_score", 1.0) < 0.85:
                continue
            matrix = getattr(observed, "observed_shape", None)
            if matrix is None:  # Older observers/test adapters have no matrix.
                reliable.add(observed.card_id)
                continue
            try:
                shape = CardShape.from_matrix(matrix)
            except ValueError:
                continue
            card = cards[observed.card_id]
            specials = sum(int(cell) == 2 for row in shape.cells for cell in row)
            expected_specials = sum(int(cell) == 2 for row in card.shape.cells for cell in row)
            if shape.size != card.area or specials != expected_specials:
                continue
            reliable.add(observed.card_id)
            if shape != card.shape:
                cards[observed.card_id] = replace(card, shape=shape)
                changed = True
            self._visual_shapes[observed.card_id] = tuple(tuple(int(cell) for cell in row) for row in shape.cells)
        if changed:
            self.catalog = replace(self.catalog, cards=cards)
            self.rules = BattleRules(self.catalog)
            self.placement_reader.catalog = self.catalog
        return reliable

    def _record_preview_command(
        self,
        signature: tuple[int, tuple[int, int], bool],
        *,
        rotation_button: Button | None = None,
    ) -> None:
        self._preview_response_at = time.perf_counter()
        self._pending_preview_signature = signature if rotation_button is None else None
        if rotation_button is not None:
            self._pending_rotation = {
                "card_id": self._selected_action.card_id,
                "button": rotation_button.value,
                "before_rotation": signature[0],
                "expected_rotation": (signature[0] + (1 if rotation_button is Button.Y else -1)) % 4,
            }
            self._rotation_feedback_reads = 0
            self._rotation_mismatch_reads = 0
        self._preview_signature = None
        self._preview_signature_count = 0

    def _reset_rotation_feedback(self) -> None:
        self._verified_rotation = None
        self._pending_rotation = None
        self._rotation_feedback_reads = 0
        self._rotation_mismatch_reads = 0

    def reset_preview_feedback(self) -> None:
        """Reobserve after manual input, keeping the plan and revealed cards.

        Call after a manual pulse is released. Manual input may have changed
        rotation or position without passing through the automatic executor.
        An executor already in ERROR still requires a core reload.
        """
        self._reset_rotation_feedback()
        self._reset_movement_feedback()
        self._preview_signature = None
        self._preview_signature_count = 0
        self._pending_preview_signature = None
        self._preview_stall_count = 0
        self._preview_missing_reads = 0
        self._rotation_commands = 0
        self._preview_response_at = time.perf_counter()
        if hasattr(self.placement_reader, "reset_history"):
            self.placement_reader.reset_history()


def _hand_focus_direction(
    current: tuple[int, int],
    target: tuple[int, int],
) -> Button | None:
    row_delta = target[0] - current[0]
    col_delta = target[1] - current[1]
    if row_delta > 0:
        return Button.DPAD_DOWN
    if row_delta < 0:
        return Button.DPAD_UP
    if col_delta > 0:
        return Button.DPAD_RIGHT
    if col_delta < 0:
        return Button.DPAD_LEFT
    return None


def _movement_button(row_delta: int, col_delta: int) -> Button:
    if abs(row_delta) >= abs(col_delta) and row_delta != 0:
        return Button.DPAD_DOWN if row_delta > 0 else Button.DPAD_UP
    return Button.DPAD_RIGHT if col_delta > 0 else Button.DPAD_LEFT


def _direction_delta(button: Button) -> tuple[int, int]:
    return {
        Button.DPAD_UP: (-1, 0), Button.DPAD_DOWN: (1, 0),
        Button.DPAD_LEFT: (0, -1), Button.DPAD_RIGHT: (0, 1),
    }[button]


def _playable_anchor(board, anchor: tuple[int, int]) -> bool:
    return (hasattr(board, "rows") and hasattr(board, "cols") and hasattr(board, "cells")
            and 0 <= anchor[0] < board.rows and 0 <= anchor[1] < board.cols
            and board.cells[anchor[0]][anchor[1]] in PLAYABLE_CELLS)


def _path_movement_button(
    board: BoardObservation,
    current: tuple[int, int],
    target: tuple[int, int],
) -> Button:
    row_delta = target[0] - current[0]
    col_delta = target[1] - current[1]
    if row_delta == 0 or col_delta == 0:
        return _movement_button(row_delta, col_delta)

    horizontal_first_walls = (
        _wall_count_inclusive(
            board,
            (current[0], current[1]),
            (current[0], target[1]),
        )
        + _wall_count_inclusive(
            board,
            (current[0], target[1]),
            (target[0], target[1]),
        )
    )
    vertical_first_walls = (
        _wall_count_inclusive(
            board,
            (current[0], current[1]),
            (target[0], current[1]),
        )
        + _wall_count_inclusive(
            board,
            (target[0], current[1]),
            (target[0], target[1]),
        )
    )
    move_horizontal_first = horizontal_first_walls < vertical_first_walls
    if move_horizontal_first:
        return _movement_button(0, col_delta)
    return _movement_button(row_delta, 0)


def _wall_count_inclusive(
    board: BoardObservation,
    start: tuple[int, int],
    end: tuple[int, int],
) -> int:
    if start[0] != end[0] and start[1] != end[1]:
        return 0
    if start[0] == end[0]:
        step = 1 if end[1] >= start[1] else -1
        cells = (
            (start[0], col)
            for col in range(start[1], end[1] + step, step)
        )
    else:
        step = 1 if end[0] >= start[0] else -1
        cells = (
            (row, start[1])
            for row in range(start[0], end[0] + step, step)
        )
    count = 0
    for row, col in cells:
        if (
            row < 0
            or row >= board.rows
            or col < 0
            or col >= board.cols
            or board.cells[row][col] in {Cell.VOID, Cell.WALL}
        ):
            count += 1
    return count
