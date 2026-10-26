from __future__ import annotations

from dataclasses import replace
from enum import StrEnum
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
    PREVIEW_RESPONSE_SECONDS = 0.75

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
        self._current_hand = None
        self._card_confirmed = False
        self._card_confirmed_at = 0.0
        self._place_sent = False
        self._place_sent_at = 0.0
        self._rotation_probe: tuple[int, int] | None = None
        self._x_rotation_delta: int | None = None
        self._skip_down_sent = False
        self._skip_activate_sent = False
        self._discard_confirm_sent = False
        self._discard_confirm_at = 0.0
        self._return_to_select_attempts = 0
        self._return_to_select_at = 0.0
        self._force_selection_after_b = False
        self._preview_missing_reads = 0
        self._preview_nudge_index = 0
        self._preview_sweep_horizontal = Button.DPAD_RIGHT
        self._preview_sweep_remaining = 4
        self._preview_retry_at = 0.0
        self._cursor_guess: tuple[int, int] | None = None
        self._preview_signature: tuple[int, tuple[int, int], bool] | None = None
        self._preview_signature_count = 0
        self._preview_response_at = 0.0
        self._pending_preview_signature: (
            tuple[int, tuple[int, int], bool] | None
        ) = None
        self._preview_stall_count = 0
        self._completed_turns = 0
        self._counted_reveal_turn = -1
        self._revealed_own_cards: list[str] = []
        self._revealed_opponent_cards: list[str] = []

    @property
    def requires_selection_data(self) -> bool:
        # A false placement-preview detection can make the vision layer hide
        # focus data even while the game is still on the card-selection UI.
        # Keep resolving focus until the executor has actually confirmed a card.
        return not self._card_confirmed

    def step(
        self,
        observation: BattleVisionObservation,
        frame: np.ndarray,
    ) -> bool:
        state = observation.state.state
        if observation.reveal is not None:
            own = observation.reveal.own.card_id
            opponent = observation.reveal.opponent.card_id
            if own is not None and own not in self._revealed_own_cards:
                self._revealed_own_cards.append(own)
            if (
                opponent is not None
                and opponent not in self._revealed_opponent_cards
            ):
                self._revealed_opponent_cards.append(opponent)
        if state is not self._last_state:
            if state is GameState.TURN_SELECT and (
                self._selected_action is None
                or self._place_sent
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
                self._current_hand = None
                self._card_confirmed = False
                self._card_confirmed_at = 0.0
                self._place_sent = False
                self._place_sent_at = 0.0
                self._rotation_probe = None
                self._skip_down_sent = False
                self._skip_activate_sent = False
                self._preview_missing_reads = 0
                self._preview_nudge_index = 0
                self._preview_sweep_horizontal = Button.DPAD_RIGHT
                self._preview_sweep_remaining = 4
                self._preview_retry_at = 0.0
                self._preview_signature = None
                self._preview_signature_count = 0
                self._preview_response_at = 0.0
                self._pending_preview_signature = None
                self._preview_stall_count = 0
                self._force_selection_after_b = False
            if state is GameState.SKIP_DISCARD:
                self._discard_confirm_sent = False
                self._discard_confirm_at = 0.0
            if state is not GameState.TARGET_PLACEMENT:
                self._return_to_select_attempts = 0
                self._return_to_select_at = 0.0
            if (
                state is GameState.TURN_REVEAL
                and self._counted_reveal_turn != self._completed_turns
            ):
                self._completed_turns += 1
                self._counted_reveal_turn = self._completed_turns
            self._last_state = state

        if state is GameState.TURN_SELECT:
            return self._choose_and_select_card(observation)
        if state is GameState.TARGET_PLACEMENT:
            if self._force_selection_after_b and not self._card_confirmed:
                return self._choose_and_select_card(observation)
            if (
                self._selected_action is None
                and observation.hand is not None
                and observation.selection_profile is not None
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
            self.phase = BattleExecutorPhase.ERROR
            self.message = "特殊攻击自动执行尚未接入"
            return False
        return False

    def _choose_and_select_card(
        self,
        observation: BattleVisionObservation,
    ) -> bool:
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
                f"手牌置信度="
                f"{observation.hand.confidence if observation.hand is not None else 0:.2f}, "
                f"焦点={'有' if observation.selection_profile is not None else '无'})"
            )
            return False
        self._current_hand = observation.hand
        if self._selected_action is None:
            state = self._build_battle_state(observation)
            if state is None:
                self.phase = BattleExecutorPhase.ERROR
                self.message = "无法组装BattleState，缺少牌组、场地或手牌"
                return False
            legal = self.rules.legal_actions(state, Side.A)
            placements = tuple(
                action
                for action in legal
                if action.action_type is ActionType.PLACE
            )
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
        if self._card_confirmed:
            if time.perf_counter() - self._card_confirmed_at >= 1.2:
                self.message = "卡牌确认未生效，重试A"
                if self._pulse(Button.A):
                    self._card_confirmed_at = time.perf_counter()
            else:
                self.phase = BattleExecutorPhase.WAITING
                self.message = "卡牌已确认，等待进入摆放阶段"
            return False

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
            self.message = "MCTS选择的卡不在当前四张手牌中"
            return False
        focus = observation.selection_profile.focus_slot
        if focus is None:
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待选牌焦点槽位识别"
            return False
        target_slot = (target.slot_row, target.slot_col)
        action = self._selected_action
        assert action is not None
        if action.action_type is ActionType.SKIP:
            return self._open_skip_dialog(
                focus,
            )
        direction = _hand_focus_direction(focus, target_slot)
        if direction is not None:
            self.phase = BattleExecutorPhase.CHOOSING_CARD
            self.message = (
                f"MCTS选择 {target.name}，目标旋转{action.rotation}，"
                f"起点{action.origin}；移动选牌焦点 "
                f"{focus}->{target_slot}"
            )
            return self._pulse(direction)

        self.phase = BattleExecutorPhase.CHOOSING_CARD
        self.message = (
            f"MCTS选择 {target.name}，目标旋转{action.rotation}，"
            f"起点{action.origin}；确认卡牌"
        )
        if self._pulse(Button.A):
            self._card_confirmed = True
            self._card_confirmed_at = time.perf_counter()
            return True
        return False

    def _open_skip_dialog(
        self,
        focus: tuple[int, int],
    ) -> bool:
        anchor = (2, 1)
        if focus != anchor:
            direction = _hand_focus_direction(focus, anchor)
            assert direction is not None
            self.phase = BattleExecutorPhase.CHOOSING_CARD
            self.message = "没有可摆放位置，移动选牌焦点到跳过按钮"
            return self._pulse(direction)
        if not self._skip_down_sent:
            self.phase = BattleExecutorPhase.CHOOSING_CARD
            self.message = "移动到跳过按钮"
            if self._pulse(Button.DPAD_DOWN):
                self._skip_down_sent = True
            return True
        if not self._skip_activate_sent:
            self.phase = BattleExecutorPhase.CHOOSING_CARD
            self.message = "打开跳过并弃牌界面"
            if self._pulse(Button.A):
                self._skip_activate_sent = True
            return True
        self.phase = BattleExecutorPhase.WAITING
        self.message = "等待弃牌选择界面"
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
            or observation.selection_profile is None
        ):
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待弃牌卡牌和焦点识别"
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
                self.message = "弃牌确认未生效，重试A"
                if self._pulse(Button.A):
                    self._discard_confirm_at = time.perf_counter()
            else:
                self.phase = BattleExecutorPhase.WAITING
                self.message = f"已确认弃牌 {target.name}"
            return False
        self.phase = BattleExecutorPhase.CHOOSING_CARD
        self.message = f"确认弃牌 {target.name}"
        if self._pulse(Button.A):
            self._discard_confirm_sent = True
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
            return False
        if self._place_sent:
            if time.perf_counter() - self._place_sent_at >= 1.2:
                self.message = "放置确认未生效，重试A"
                if self._pulse(Button.A):
                    self._place_sent_at = time.perf_counter()
            else:
                self.phase = BattleExecutorPhase.PLACING
                self.message = "已发送放置A，等待亮牌"
            return False
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
            self.phase = BattleExecutorPhase.WAITING
            self.message = "等待棋盘与摆放预览"
            return False
        preview = None
        if observation.board.placement_preview_present:
            board_region = getattr(self.runtime, "board_region", None)
            stage_id = getattr(self.runtime, "stage_id", None)
            if (
                frame is not None
                and board_region is not None
                and stage_id is not None
            ):
                preview = self.placement_reader.read_frame(
                    frame,
                    board_region,
                    stage_id,
                    action.card_id,
                    observation.board,
                )
            if preview is None:
                preview = self.placement_reader.read(
                    observation.board,
                    action.card_id,
                )
        if preview is None:
            self.phase = BattleExecutorPhase.WAITING
            self._preview_signature = None
            self._preview_signature_count = 0
            self._preview_missing_reads += 1
            if self._preview_missing_reads <= 2:
                self.message = "等待摆放预览动画稳定"
                return False
            if time.perf_counter() < self._preview_retry_at:
                self.message = "移动一步后等待摆放预览稳定"
                return False
            if self._cursor_guess is None:
                self._cursor_guess = (
                    observation.board.rows // 2,
                    observation.board.cols // 2,
                )
            if self._cursor_guess is not None and action.origin is not None:
                row_delta = action.origin[0] - self._cursor_guess[0]
                col_delta = action.origin[1] - self._cursor_guess[1]
                if row_delta == 0 and col_delta == 0:
                    button = self._next_sweep_button()
                else:
                    button = _movement_button(row_delta, col_delta)
            else:
                button = self._next_sweep_button()
            self.message = (
                "未识别到摆放预览，移动光标脱困 "
                f"{button.value}"
            )
            self._pulse(button)
            self._preview_retry_at = time.perf_counter() + 0.9
            if self._cursor_guess is not None:
                row_delta = (
                    1
                    if button is Button.DPAD_DOWN
                    else -1
                    if button is Button.DPAD_UP
                    else 0
                )
                col_delta = (
                    1
                    if button is Button.DPAD_RIGHT
                    else -1
                    if button is Button.DPAD_LEFT
                    else 0
                )
                self._cursor_guess = (
                    min(
                        observation.board.rows - 1,
                        max(0, self._cursor_guess[0] + row_delta),
                    ),
                    min(
                        observation.board.cols - 1,
                        max(0, self._cursor_guess[1] + col_delta),
                    ),
                )
            return True
        self._preview_missing_reads = 0
        self._preview_retry_at = 0.0
        self._cursor_guess = preview.origin
        signature = (preview.rotation, preview.anchor, preview.valid)
        assert action.rotation is not None
        target_anchor = self._action_anchor(action)
        now = time.perf_counter()
        if now - self._preview_response_at < self.PREVIEW_RESPONSE_SECONDS:
            self.phase = BattleExecutorPhase.WAITING
            self.message = (
                "等待预览响应上一条旋转或移动 "
                f"当前{preview.anchor} 目标{target_anchor} "
                f"({self.PREVIEW_RESPONSE_SECONDS - (now - self._preview_response_at):.1f}s)"
            )
            return False
        if self._pending_preview_signature is not None:
            if signature == self._pending_preview_signature:
                self._preview_stall_count += 1
            else:
                self._preview_stall_count = 0
            self._pending_preview_signature = None
        if (
            self._preview_stall_count >= 2
            and preview.anchor != target_anchor
        ):
            button = self._next_sweep_button()
            self.phase = BattleExecutorPhase.POSITIONING
            self.message = (
                "预览连续未响应，停止重复同一方向并换方向探测 "
                f"{button.value}"
            )
            self._pulse(button)
            self._record_preview_command(signature)
            self._preview_stall_count = 0
            return True
        if signature == self._preview_signature:
            self._preview_signature_count += 1
        else:
            self._preview_signature = signature
            self._preview_signature_count = 1
        if self._preview_signature_count < self.PREVIEW_STABLE_FRAMES:
            self.phase = BattleExecutorPhase.WAITING
            self.message = (
                "确认摆放预览稳定 "
                f"{preview.anchor} r={preview.rotation} "
                f"{'黄色可放' if preview.valid else '灰色不可放'} "
                f"score={preview.score:.2f} "
                f"({self._preview_signature_count}/"
                f"{self.PREVIEW_STABLE_FRAMES})"
            )
            return False

        if preview.rotation != action.rotation:
            self.phase = BattleExecutorPhase.POSITIONING
            delta = (action.rotation - preview.rotation) % 4
            self.message = (
                f"预览{preview.rotation}->{action.rotation}，"
                f"还需要按Y {delta}次"
            )
            if self._pulse(Button.Y):
                self._record_preview_command(signature)
                return True
            return False

        if preview.anchor != target_anchor:
            button = _path_movement_button(
                observation.board,
                preview.anchor,
                target_anchor,
            )
            self.phase = BattleExecutorPhase.POSITIONING
            self.message = (
                f"移动预览锚点 {preview.anchor}->{target_anchor} "
                f"使用 {button.value}"
            )
            if self._pulse(button):
                self._record_preview_command(signature)
                return True
            return False

        if not preview.valid:
            replacement = self._nearest_alternative_placement(
                observation,
                action,
                preview.origin,
            )
            if replacement is not None:
                self._selected_action = replacement
                self.phase = BattleExecutorPhase.POSITIONING
                self.message = (
                    "目标预览为灰色不可放，改选邻近合法起点 "
                    f"{replacement.origin}"
                )
                return True
            button = self._next_sweep_button()
            self.phase = BattleExecutorPhase.POSITIONING
            self.message = (
                "目标仍为灰色预览，移动一步寻找可放区域 "
                f"{button.value}"
            )
            self._pulse(button)
            self._cursor_guess = preview.origin
            return True

        self.phase = BattleExecutorPhase.PLACING
        self.message = f"位置已到达 {target_anchor}，按A放置"
        if self._pulse(Button.A):
            self._place_sent = True
            self._place_sent_at = time.perf_counter()
            self._cursor_guess = action.origin
            return True
        return False

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

    def _nearest_alternative_placement(
        self,
        observation: BattleVisionObservation,
        current: Action,
        origin: tuple[int, int],
    ) -> Action | None:
        state = self._build_battle_state(observation)
        if state is None or current.card_id is None:
            return None
        candidates = [
            action
            for action in self.rules.legal_actions(state, Side.A)
            if (
                action.action_type is ActionType.PLACE
                and action.card_id == current.card_id
                and action != current
                and action.origin is not None
            )
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda action: (
                abs(action.origin[0] - origin[0])
                + abs(action.origin[1] - origin[1])
            ),
        )

    def _next_sweep_button(self) -> Button:
        if self._preview_sweep_remaining <= 0:
            self._preview_sweep_horizontal = (
                Button.DPAD_LEFT
                if self._preview_sweep_horizontal is Button.DPAD_RIGHT
                else Button.DPAD_RIGHT
            )
            self._preview_sweep_remaining = 4
            return Button.DPAD_DOWN
        self._preview_sweep_remaining -= 1
        return self._preview_sweep_horizontal

    def _choose_rotation_button(
        self,
        current: int,
        target: int,
    ) -> Button:
        delta = self._x_rotation_delta
        if delta == 0:
            return Button.Y
        if delta is None:
            delta = 1
        x_steps = (
            (target - current) % 4
            if delta == 1
            else (current - target) % 4
        )
        y_steps = (4 - x_steps) % 4
        return Button.X if x_steps <= y_steps else Button.Y

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
        board = observation.board.cells
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
        hand_observation = observation.hand or self._current_hand
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
        own_sp = (
            observation.active_specials.self_count
            if observation.active_specials is not None
            else observation.board.self_active_special_count
        )
        opponent_sp = (
            observation.active_specials.opponent_count
            if observation.active_specials is not None
            else observation.board.opponent_active_special_count
        )
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
        try:
            pulse_buttons(self.controller, (button,))
            return True
        except RuntimeError:
            self.phase = BattleExecutorPhase.ERROR
            self.message = "控制器发送失败"
            return False

    def _record_preview_command(
        self,
        signature: tuple[int, tuple[int, int], bool],
    ) -> None:
        self._preview_response_at = time.perf_counter()
        self._pending_preview_signature = signature
        self._preview_signature = None
        self._preview_signature_count = 0


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
