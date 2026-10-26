from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from tableturf_rebuild.application.battle_vision import (
    BattleVisionObservation,
    BattleVisionRuntime,
)
from tableturf_rebuild.application.battle_executor import (
    BattleExecutorPhase,
    VisionMctsBattleExecutor,
)
from tableturf_rebuild.application.deck_config import DeckConfig
from tableturf_rebuild.application.state_machine import GameState
from tableturf_rebuild.application.task_plan import (
    NpcTaskDefinition,
    NpcWinPlan,
    build_npc_plan,
)
from tableturf_rebuild.control.backend import ControllerBackend
from tableturf_rebuild.control.buttons import Button, ControllerState
from tableturf_rebuild.control.sequences import (
    DEFAULT_ACTION_GAP_SECONDS,
    DEFAULT_PRESS_SECONDS,
    pulse_buttons,
)
from tableturf_rebuild.domain.card import CardShape


class TaskStatus(StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    WAITING_RECOGNITION = "WAITING_RECOGNITION"
    WAITING_FOR_DECK = "WAITING_FOR_DECK"
    WAITING_FOR_BATTLE_EXECUTOR = "WAITING_FOR_BATTLE_EXECUTOR"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class BattleStepExecutor(Protocol):
    def step(
        self,
        observation: BattleVisionObservation,
        frame: np.ndarray,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    status: TaskStatus
    message: str
    state: GameState
    observation: BattleVisionObservation | None
    plan: NpcWinPlan | None
    current_npc_name: str | None
    current_difficulty: int | None
    observed_level_wins: tuple[int | None, int | None, int | None] | None


class TaskRunner:
    REDRAW_READY_DELAY_SECONDS = 1.0
    WINS_STABLE_READINGS = 3

    BATTLE_STATES = {
        GameState.REDRAW,
        GameState.TURN_SELECT,
        GameState.SKIP_DISCARD,
        GameState.SPECIAL_SELECT,
        GameState.TARGET_PLACEMENT,
        GameState.SPECIAL_PLACEMENT,
        GameState.TURN_REVEAL,
        GameState.TURN_RESOLVE,
        GameState.FINISHING,
    }

    def __init__(
        self,
        runtime: BattleVisionRuntime,
        controller: ControllerBackend,
        *,
        action_cooldown_seconds: float = DEFAULT_ACTION_GAP_SECONDS,
        deck_probe_seconds: float = 1.8,
        battle_executor: BattleStepExecutor | None = None,
    ):
        self.runtime = runtime
        self.controller = controller
        self.action_cooldown_seconds = action_cooldown_seconds
        self.deck_probe_seconds = deck_probe_seconds
        self.battle_executor = battle_executor
        self._external_battle_executor = battle_executor is not None
        self.task: NpcTaskDefinition | None = None
        self.deck: DeckConfig | None = None
        self.plan: NpcWinPlan | None = None
        self.status = TaskStatus.IDLE
        self.message = "任务未开始"
        self._last_action_at = 0.0
        self._last_deck_probe_at = 0.0
        self._deck_navigation_steps = 0
        self._deck_selection_confirmed = False
        self._last_state = GameState.UNKNOWN
        self._last_observation: BattleVisionObservation | None = None
        self._match_setup_start_prepared = False
        self._match_setup_start_sent = False
        self._redraw_choice_prepared = False
        self._redraw_confirm_sent = False
        self._redraw_entered_at: float | None = None
        self._npc_search_direction = Button.DPAD_DOWN
        self._npc_last_seen: str | None = None
        self._npc_unchanged_steps = 0
        self._last_match_outcome: str | None = None
        self._replay_choice_prepared = False
        self._replay_choice_sent = False
        self._wins_candidate: tuple[int, int, int] | None = None
        self._wins_candidate_count = 0
        self._stable_level_wins: tuple[int, int, int] | None = None
        self._wins_change_allowed = True
        self._wins_npc_id: str | None = None

    def configure(
        self,
        task: NpcTaskDefinition,
        deck: DeckConfig | None,
    ) -> None:
        self.task = task
        self.deck = deck
        self.plan = None
        self.status = TaskStatus.IDLE
        self.message = "任务已配置"
        self._last_action_at = 0.0
        self._last_deck_probe_at = 0.0
        self._deck_navigation_steps = 0
        self._deck_selection_confirmed = False
        self._last_state = GameState.UNKNOWN
        self._last_observation = None
        self._match_setup_start_prepared = False
        self._match_setup_start_sent = False
        self._redraw_choice_prepared = False
        self._redraw_confirm_sent = False
        self._redraw_entered_at = None
        self._npc_search_direction = Button.DPAD_DOWN
        self._npc_last_seen = None
        self._npc_unchanged_steps = 0
        self._last_match_outcome = None
        self._replay_choice_prepared = False
        self._replay_choice_sent = False
        self._wins_candidate = None
        self._wins_candidate_count = 0
        self._stable_level_wins = None
        self._wins_change_allowed = True
        self._wins_npc_id = None
        if self.deck is not None and not self._external_battle_executor:
            self.battle_executor = VisionMctsBattleExecutor(
                self.runtime,
                self.controller,
                self.deck.card_ids,
            )
        elif not self._external_battle_executor:
            self.battle_executor = None
        self.runtime.clear_expected_deck()

    def start(self) -> None:
        if self.task is None:
            raise RuntimeError("task is not configured")
        if not self.controller.connected:
            raise RuntimeError("controller is not connected")
        self.status = TaskStatus.RUNNING
        self.message = "任务启动"

    def pause(self) -> None:
        if self.status is not TaskStatus.COMPLETED:
            self.status = TaskStatus.PAUSED
            self.message = "任务已暂停"
            self.controller.send_state(ControllerState.neutral())

    def resume(self) -> None:
        if self.status is TaskStatus.PAUSED:
            self.status = TaskStatus.RUNNING
            self.message = "任务继续"
            self._wins_change_allowed = True

    def stop(self) -> None:
        self.status = TaskStatus.IDLE
        self.message = "任务已停止"
        self.controller.send_state(ControllerState.neutral())

    def tick(
        self,
        frame: np.ndarray,
        *,
        now: float | None = None,
    ) -> TaskSnapshot:
        timestamp = time.perf_counter() if now is None else now
        observation = self.runtime.update(
            frame,
            include_selection=True,
            force_selection=bool(
                getattr(self.battle_executor, "requires_selection_data", False)
            ),
        )
        self._last_observation = observation
        state = observation.state.state
        if state is not self._last_state:
            if (
                state is GameState.DECK_SELECT
                and self._last_state is not GameState.DECK_SELECT
            ):
                self._deck_selection_confirmed = False
                self._deck_navigation_steps = 0
                self._last_deck_probe_at = timestamp
            if state is not GameState.MATCH_SETUP:
                self._match_setup_start_prepared = False
                self._match_setup_start_sent = False
            if state is not GameState.REDRAW:
                self._redraw_choice_prepared = False
                self._redraw_confirm_sent = False
                self._redraw_entered_at = None
            elif self._redraw_entered_at is None:
                self._redraw_entered_at = timestamp
            if state is not GameState.NPC_ROSTER:
                self._npc_search_direction = Button.DPAD_DOWN
                self._npc_last_seen = None
                self._npc_unchanged_steps = 0
            if state is not GameState.REPLAY_CHOICE:
                self._replay_choice_prepared = False
                self._replay_choice_sent = False
            if state is GameState.MATCH_RESULT:
                self._last_match_outcome = None
            self._last_state = state

        if self.status in {TaskStatus.IDLE, TaskStatus.PAUSED, TaskStatus.ERROR}:
            return self.snapshot()
        if self.status is TaskStatus.COMPLETED:
            return self.snapshot()
        if self.task is None:
            self.status = TaskStatus.ERROR
            self.message = "任务配置缺失"
            return self.snapshot()

        menu = observation.menu
        npc_name = (
            menu.npc_name
            if menu is not None
            else observation.opponent_context.npc_name
            if observation.opponent_context is not None
            else None
        )
        difficulty = (
            menu.difficulty
            if menu is not None
            else observation.opponent_context.difficulty
            if observation.opponent_context is not None
            else None
        )
        level_wins = (
            self._confirmed_level_wins(menu)
            if menu is not None
            else self._stable_level_wins
        )
        if (
            state in {GameState.NPC_ROSTER, GameState.MATCH_SETUP}
            and menu is not None
            and menu.npc_id == self.task.npc_id
            and level_wins is not None
        ):
            if self.plan is None:
                self.plan = build_npc_plan(self.task, level_wins)
            else:
                self.plan = self.plan.with_observed_wins(level_wins)
            if self.plan.complete:
                self.status = TaskStatus.COMPLETED
                self.message = "目标胜利数已达到"
                self.controller.send_state(ControllerState.neutral())
                return self.snapshot()

        if state is GameState.NPC_ROSTER:
            self._handle_npc_roster(observation, timestamp)
        elif state is GameState.NPC_DIALOGUE:
            self.status = TaskStatus.RUNNING
            self.message = "NPC对白中，按A继续直到难度选择界面"
            self._pulse(Button.A, timestamp)
        elif state is GameState.MATCH_SETUP:
            self._handle_match_setup(observation, timestamp)
        elif state is GameState.DECK_SELECT:
            self._handle_deck_select(observation, timestamp)
        elif state is GameState.REDRAW:
            self.status = TaskStatus.RUNNING
            if (
                self._redraw_entered_at is None
                or timestamp - self._redraw_entered_at
                < self.REDRAW_READY_DELAY_SECONDS
            ):
                self.message = "等待重抽弹窗和手牌稳定显示"
                self.controller.send_state(ControllerState.neutral())
                return self.snapshot()
            if self._redraw_confirm_sent:
                self.message = "已选择维持现状，等待发牌完成"
                self.controller.send_state(ControllerState.neutral())
            elif not self._redraw_choice_prepared:
                self.message = "默认开局策略：向左选择维持现状"
                if self._pulse(Button.DPAD_LEFT, timestamp):
                    self._redraw_choice_prepared = True
            else:
                self.message = "确认维持现状"
                if self._pulse(Button.A, timestamp):
                    self._redraw_confirm_sent = True
        elif state in self.BATTLE_STATES:
            if self.battle_executor is None:
                self.status = TaskStatus.WAITING_FOR_BATTLE_EXECUTOR
                self.message = (
                    "已进入对战，但视觉得分→BattleState→MCTS动作执行桥尚未接入"
                )
                self.controller.send_state(ControllerState.neutral())
            else:
                self.battle_executor.step(observation, frame)
                self.message = getattr(
                    self.battle_executor,
                    "message",
                    "MCTS战斗执行器运行中",
                )
                self.status = (
                    TaskStatus.ERROR
                    if getattr(
                        self.battle_executor,
                        "phase",
                        None,
                    )
                    is BattleExecutorPhase.ERROR
                    else TaskStatus.RUNNING
                )
        elif state is GameState.MATCH_RESULT:
            self.status = TaskStatus.RUNNING
            result = (
                observation.match_result.outcome.value
                if observation.match_result is not None
                else "UNKNOWN"
            )
            if result != "UNKNOWN":
                self._last_match_outcome = result
                if result == "WIN":
                    self._wins_change_allowed = True
            self.message = f"比赛结果 {result}，前往奖励界面"
            self._pulse(Button.A, timestamp)
        elif state is GameState.MATCH_REWARD:
            self.status = TaskStatus.RUNNING
            self.message = "领取奖励并前进"
            self._pulse(Button.A, timestamp)
        elif state is GameState.REPLAY_CHOICE:
            self._handle_replay_choice(timestamp)
        else:
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = f"等待稳定状态，当前 {state.value}"

        return self.snapshot(
            npc_name=npc_name,
            difficulty=difficulty,
            level_wins=level_wins,
        )

    def _handle_npc_roster(
        self,
        observation: BattleVisionObservation,
        timestamp: float,
    ) -> None:
        assert self.task is not None
        menu = observation.menu
        if menu is None or menu.npc_id is None:
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = "等待NPC名称稳定识别"
            return
        if menu.npc_id == self.task.npc_id:
            self.status = TaskStatus.RUNNING
            self.message = f"选择NPC {self.task.npc_name}"
            self._npc_search_direction = Button.DPAD_DOWN
            self._npc_last_seen = menu.npc_id
            self._npc_unchanged_steps = 0
            self._pulse(Button.A, timestamp)
            return
        if self._npc_last_seen == menu.npc_id:
            self._npc_unchanged_steps += 1
        else:
            self._npc_last_seen = menu.npc_id
            self._npc_unchanged_steps = 0
        if self._npc_unchanged_steps >= 3:
            self._npc_search_direction = (
                Button.DPAD_UP
                if self._npc_search_direction is Button.DPAD_DOWN
                else Button.DPAD_DOWN
            )
            self._npc_unchanged_steps = 0
        self.status = TaskStatus.RUNNING
        direction = (
            "向上"
            if self._npc_search_direction is Button.DPAD_UP
            else "向下"
        )
        self.message = (
            f"寻找NPC {self.task.npc_name}，{direction}查找，"
            f"当前 {menu.npc_name}"
        )
        self._pulse(self._npc_search_direction, timestamp)

    def _confirmed_level_wins(
        self,
        menu,
    ) -> tuple[int, int, int] | None:
        assert self.task is not None
        if menu.npc_id != self.task.npc_id:
            self._wins_candidate = None
            self._wins_candidate_count = 0
            return (
                self._stable_level_wins
                if self._wins_npc_id == self.task.npc_id
                else None
            )
        if self._wins_npc_id != self.task.npc_id:
            self._wins_candidate = None
            self._wins_candidate_count = 0
            self._stable_level_wins = None
            self._wins_npc_id = self.task.npc_id
        values = menu.level_wins
        if (
            menu.wins_confidence < 0.55
            or any(value is None for value in values)
        ):
            self._wins_candidate = None
            self._wins_candidate_count = 0
            return self._stable_level_wins
        candidate = tuple(int(value) for value in values)
        if (
            any(value < 0 for value in candidate)
            or candidate[1] > 0
            and candidate[0] < 3
            or candidate[2] > 0
            and candidate[1] < 3
        ):
            self._wins_candidate = None
            self._wins_candidate_count = 0
            return self._stable_level_wins
        if candidate == self._wins_candidate:
            self._wins_candidate_count += 1
        else:
            self._wins_candidate = candidate
            self._wins_candidate_count = 1
        if self._wins_candidate_count < self.WINS_STABLE_READINGS:
            return self._stable_level_wins
        if self._stable_level_wins is None:
            self._stable_level_wins = candidate
            self._wins_change_allowed = False
        elif candidate != self._stable_level_wins:
            if self._wins_change_allowed:
                self._stable_level_wins = candidate
                self._wins_change_allowed = False
        self._wins_change_allowed = False
        self._wins_candidate_count = 0
        self._wins_candidate = None
        return self._stable_level_wins

    def _handle_match_setup(
        self,
        observation: BattleVisionObservation,
        timestamp: float,
    ) -> None:
        assert self.task is not None
        menu = observation.menu
        if menu is None or menu.npc_id != self.task.npc_id:
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = "等待目标NPC与胜场信息稳定识别"
            return
        if self.plan is not None and self._wins_change_allowed:
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = "等待本场结束后的最新三档胜场稳定识别"
            return
        if self.plan is None:
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = "等待三档胜利数识别"
            return
        active = self.plan.active_stage
        if active is None:
            self.status = TaskStatus.COMPLETED
            self.message = "目标胜利数已达到"
            return
        if menu.difficulty is None:
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = "等待当前难度数字稳定识别"
            return
        if menu.difficulty < active.level:
            self._match_setup_start_prepared = False
            self._match_setup_start_sent = False
            self.status = TaskStatus.RUNNING
            self.message = f"切换到难度{active.level}"
            self._pulse(Button.DPAD_RIGHT, timestamp)
        elif menu.difficulty > active.level:
            self._match_setup_start_prepared = False
            self._match_setup_start_sent = False
            self.status = TaskStatus.RUNNING
            self.message = f"切换到难度{active.level}"
            self._pulse(Button.DPAD_LEFT, timestamp)
        else:
            self.status = TaskStatus.RUNNING
            if not self._match_setup_start_prepared:
                self.message = f"难度{active.level}已选，向下移动到开始"
                if self._pulse(Button.DPAD_DOWN, timestamp):
                    self._match_setup_start_prepared = True
                return
            if not self._match_setup_start_sent:
                self.message = (
                    f"确认开始，难度{active.level} "
                    f"还剩{active.remaining_wins}胜"
                )
                if self._pulse(Button.A, timestamp):
                    self._match_setup_start_sent = True
                return
            self.message = "开始指令已发送，等待进入卡组选择"

    def _handle_replay_choice(self, timestamp: float) -> None:
        if self.plan is None or self._last_match_outcome is None:
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = "等待比赛结果和剩余胜场信息"
            self.controller.send_state(ControllerState.neutral())
            return
        active = self.plan.active_stage
        if active is None:
            should_continue = False
        else:
            remaining_after_match = active.remaining_wins - (
                1 if self._last_match_outcome == "WIN" else 0
            )
            should_continue = remaining_after_match > 0
        direction = (
            Button.DPAD_RIGHT
            if should_continue
            else Button.DPAD_LEFT
        )
        choice = "继续当前难度" if should_continue else "退出并切换难度"
        if not self._replay_choice_prepared:
            self.status = TaskStatus.RUNNING
            self.message = f"比赛结束：{choice}"
            if self._pulse(direction, timestamp):
                self._replay_choice_prepared = True
            return
        if not self._replay_choice_sent:
            self.status = TaskStatus.RUNNING
            self.message = f"比赛结束：{choice}，确认"
            if self._pulse(Button.A, timestamp):
                self._replay_choice_sent = True
            return
        self.status = TaskStatus.WAITING_RECOGNITION
        self.message = f"已选择{choice}，等待界面切换"
        self.controller.send_state(ControllerState.neutral())

    def _handle_deck_select(
        self,
        observation: BattleVisionObservation,
        timestamp: float,
    ) -> None:
        if self.deck is None:
            self.status = TaskStatus.WAITING_FOR_DECK
            self.message = "任务未指定卡组，请在卡组选择界面手动选择"
            self.controller.send_state(ControllerState.neutral())
            return
        if self._deck_selection_confirmed:
            self.status = TaskStatus.RUNNING
            self.message = "卡组已确认，等待进入对战"
            self.controller.send_state(ControllerState.neutral())
            return
        if timestamp - self._last_deck_probe_at < self.deck_probe_seconds:
            return
        self._last_deck_probe_at = timestamp
        visible = observation.deck
        if visible is not None and _deck_signature_matches(
            visible.cards,
            self.deck,
            self.runtime.catalog,
        ):
            self.status = TaskStatus.RUNNING
            self.message = f"找到卡组 {self.deck.name}，确认并锁定"
            if self._pulse(Button.A, timestamp):
                self.runtime.set_expected_deck(
                    self.deck.name,
                    self.deck.card_ids,
                )
                self._deck_selection_confirmed = True
            return

        self.status = TaskStatus.WAITING_FOR_DECK
        self._deck_navigation_steps += 1
        navigation_position = (self._deck_navigation_steps - 1) % 9
        if navigation_position < 8:
            self.message = (
                f"寻找卡组 {self.deck.name}，向下浏览 "
                f"{navigation_position + 1}/8"
            )
            self._pulse(Button.DPAD_DOWN, timestamp)
        elif self._deck_navigation_steps <= 36:
            self.message = f"寻找卡组 {self.deck.name}，翻页"
            self._pulse(Button.R, timestamp)
        else:
            self.message = (
                f"未自动找到卡组 {self.deck.name}，"
                "请手动选中；识别到后会继续"
            )

    def _pulse(self, button: Button, timestamp: float) -> bool:
        if timestamp - self._last_action_at < self.action_cooldown_seconds:
            return False
        try:
            pulse_buttons(
                self.controller,
                (button,),
                DEFAULT_PRESS_SECONDS,
                DEFAULT_ACTION_GAP_SECONDS,
            )
        except RuntimeError:
            self.status = TaskStatus.ERROR
            self.message = "控制器发送失败"
            return False
        self._last_action_at = timestamp
        return True

    def snapshot(
        self,
        *,
        npc_name: str | None = None,
        difficulty: int | None = None,
        level_wins: tuple[int | None, int | None, int | None] | None = None,
    ) -> TaskSnapshot:
        observation = self._last_observation
        if observation is not None:
            if npc_name is None:
                npc_name = (
                    observation.menu.npc_name
                    if observation.menu is not None
                    else observation.opponent_context.npc_name
                    if observation.opponent_context is not None
                    else None
                )
            if difficulty is None:
                difficulty = (
                    observation.menu.difficulty
                    if observation.menu is not None
                    else observation.opponent_context.difficulty
                    if observation.opponent_context is not None
                    else None
                )
            if level_wins is None and observation.menu is not None:
                level_wins = observation.menu.level_wins
        return TaskSnapshot(
            status=self.status,
            message=self.message,
            state=(
                observation.state.state
                if observation is not None
                else GameState.UNKNOWN
            ),
            observation=observation,
            plan=self.plan,
            current_npc_name=npc_name,
            current_difficulty=difficulty,
            observed_level_wins=level_wins,
        )


def _deck_signature_matches(
    visible_cards,
    configured: DeckConfig,
    catalog,
) -> bool:
    visible_signature = sorted(
        (
            _canonical_shape(catalog.cards[card.card_id].shape),
            card.sp_cost,
        )
        for card in visible_cards
        if card.card_id in catalog.cards
    )
    configured_signature = sorted(
        (
            _canonical_shape(catalog.cards[card_id].shape),
            catalog.cards[card_id].sp_cost,
        )
        for card_id in configured.card_ids
    )
    return tuple(visible_signature) == tuple(configured_signature)


def _canonical_shape(shape: CardShape | tuple[tuple[object, ...], ...]) -> bytes:
    if isinstance(shape, CardShape):
        matrix = np.array(
            [
                [1 if cell > 0 else 0 for cell in row]
                for row in shape.cells
            ],
            dtype=np.uint8,
        )
    else:
        matrix = np.array(
            [
                [1 if cell > 0 else 0 for cell in row]
                for row in shape
            ],
            dtype=np.uint8,
        )
    variants = []
    for transform in (
        lambda value: value,
        lambda value: np.rot90(value, 1),
        lambda value: np.rot90(value, 2),
        lambda value: np.rot90(value, 3),
        lambda value: np.fliplr(value),
        lambda value: np.flipud(value),
        lambda value: value.T,
        lambda value: np.rot90(value.T, 2),
    ):
        variants.append(np.ascontiguousarray(transform(matrix)).tobytes())
    return min(variants)
