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
    RESULT_STABLE_READINGS = 2
    RESULT_MIN_CONFIDENCE = 0.85
    SETTLEMENT_READY_SECONDS = 0.75
    SETTLEMENT_RETRY_SECONDS = 1.5
    SETTLEMENT_MAX_ATTEMPTS = 4
    REWARD_TOTAL_MAX_ATTEMPTS = 12
    SETTLEMENT_SCENE_CONFIDENCE = 0.80
    SETTLEMENT_SCENE_MARGIN = 0.15

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
        self._awaiting_deck_reselection = False
        self._deck_select_seen = False
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
        self._match_identity: tuple[str, int] | None = None
        self._match_win_applied = False
        self._result_candidate: str | None = None
        self._result_candidate_count = 0
        self._last_result_reading = None
        self._result_confirmed = False
        self._reset_result_advancement()
        self._reset_menu_routing()

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
        self._awaiting_deck_reselection = False
        self._deck_select_seen = False
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
        self._match_identity = None
        self._match_win_applied = False
        self._result_candidate = None
        self._result_candidate_count = 0
        self._last_result_reading = None
        self._result_confirmed = False
        self._reset_result_advancement()
        self._reset_menu_routing()
        # A new task must not inherit the previous NPC's vision or deck state.
        # Core reload callers restore their saved context/progress afterwards.
        self.runtime.clear_expected_deck()
        reset_runtime = getattr(self.runtime, "reset", None)
        if reset_runtime is not None:
            reset_runtime()
        elif hasattr(self.runtime, "opponent_context"):
            self.runtime.opponent_context = None
        if self.deck is not None and not self._external_battle_executor:
            self.battle_executor = VisionMctsBattleExecutor(
                self.runtime,
                self.controller,
                self.deck.card_ids,
            )
        elif not self._external_battle_executor:
            self.battle_executor = None

    def start(self) -> None:
        if self.task is None:
            raise RuntimeError("task is not configured")
        if not self.controller.connected:
            raise RuntimeError("controller is not connected")
        resume_motion = getattr(self.battle_executor, "resume_pending_motion", None)
        if resume_motion is not None:
            resume_motion()
        self.status = TaskStatus.RUNNING
        self.message = "任务启动"

    def pause(self) -> None:
        cancel_motion = getattr(self.battle_executor, "cancel_pending_motion", None)
        if cancel_motion is not None:
            cancel_motion()
        if self.status is not TaskStatus.COMPLETED:
            self.status = TaskStatus.PAUSED
            self.message = "任务已暂停"
            self.controller.send_state(ControllerState.neutral())

    def resume(self) -> None:
        if self.status is TaskStatus.PAUSED:
            resume_motion = getattr(self.battle_executor, "resume_pending_motion", None)
            if resume_motion is not None:
                resume_motion()
            self.status = TaskStatus.RUNNING
            self.message = "任务继续"
            self._wins_change_allowed = True

    def stop(self) -> None:
        cancel_motion = getattr(self.battle_executor, "cancel_pending_motion", None)
        if cancel_motion is not None:
            cancel_motion()
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
        expectation = ({"placement_expected": True}
                       if getattr(self.battle_executor, "awaiting_placement", False) else {})
        observation = self.runtime.update(
            frame,
            include_selection=True,
            force_selection=bool(
                getattr(self.battle_executor, "requires_selection_data", False)
            ),
            **expectation,
        )
        self._last_observation = observation
        state = observation.state.state
        if state not in {GameState.MATCH_RESULT, GameState.MATCH_REWARD}:
            self._settlement_state = None
            self._settlement_evidence_reads = 0
            self._settlement_last_observation = None
        if state is not GameState.REPLAY_CHOICE or not self._settlement_scene_visible(observation):
            self._replay_evidence_reads = 0
            self._last_replay_observation = None
        elif observation is not self._last_replay_observation:
            self._replay_evidence_reads += 1
            self._last_replay_observation = observation
        if state is not self._last_state:
            if (
                state is GameState.DECK_SELECT
                and not self._deck_select_seen
            ):
                self._deck_select_seen = True
                self._awaiting_deck_reselection = True
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
                reset_match = getattr(self.battle_executor, "reset_match", None)
                if reset_match is not None:
                    reset_match()
            if state is not GameState.NPC_ROSTER:
                self._npc_search_direction = Button.DPAD_DOWN
                self._npc_last_seen = None
                self._npc_unchanged_steps = 0
            if state is not GameState.REPLAY_CHOICE:
                self._replay_choice_prepared = False
                self._replay_choice_sent = False
            self._last_state = state

        if self.status in {TaskStatus.IDLE, TaskStatus.PAUSED, TaskStatus.ERROR}:
            return self.snapshot()

        self._observe_completion_menu(observation)
        if self._setup_return_sent:
            if (state is GameState.NPC_ROSTER
                and self.safe_menu_visible(observation)):
                self._setup_return_sent = False
                self._setup_return_at = 0.0
            else:
                self.status = TaskStatus.WAITING_RECOGNITION
                self.message = "已按B返回NPC列表，等待真实列表页面，不重复按B"
                if timestamp - self._setup_return_at > 12:
                    self.status = TaskStatus.ERROR
                    self.message = "返回NPC列表后未确认真实列表页面，已停止按键并保留诊断"
                self.controller.send_state(ControllerState.neutral())
                return self.snapshot()
        if self.status is TaskStatus.COMPLETED:
            return self.snapshot()
        if self.task is None:
            self.status = TaskStatus.ERROR
            self.message = "任务配置缺失"
            return self.snapshot()

        if self._match_identity is None and state in self.BATTLE_STATES:
            context = observation.opponent_context
            if (context is not None and context.npc_id == self.task.npc_id
                and context.difficulty in {1, 2, 3} and context.confidence >= 0.55):
                self._match_identity = (context.npc_id, context.difficulty)

        if (self.plan is not None and self.plan.complete
            and state not in {GameState.MATCH_RESULT, GameState.MATCH_REWARD,
                              GameState.REPLAY_CHOICE}):
            self._finish_from_safe_menu()
            self.controller.send_state(ControllerState.neutral())
            return self.snapshot()

        if self._awaiting_deck_reselection and state in self.BATTLE_STATES | {
            GameState.BATTLE_INTRO, GameState.MATCH_RESULT, GameState.MATCH_REWARD,
        }:
            self.status = TaskStatus.WAITING_FOR_DECK
            self.message = "本局尚未重新确认15张实际卡牌，已停止自动操作；请返回卡组选择并确认预设卡组"
            self.controller.send_state(ControllerState.neutral())
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
                self._finish_from_safe_menu()
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
            self._observe_settlement_page(observation, timestamp)
            if not self._ready_match_result(observation.match_result):
                return self.snapshot()
            self._advance_settlement_page(state, timestamp)
        elif state is GameState.MATCH_REWARD:
            self._observe_settlement_page(observation, timestamp)
            if not self._ready_match_result(observation.match_result):
                return self.snapshot()
            self._advance_settlement_page(state, timestamp)
        elif state is GameState.REPLAY_CHOICE:
            if (not self._settlement_scene_visible(observation)
                or self._replay_choice_prepared and self._replay_evidence_reads < 2):
                self.status = TaskStatus.WAITING_RECOGNITION
                evidence = getattr(observation, "replay_choice_evidence", None)
                detail = (f"标题={evidence.title_score:.2f}，取消={evidence.cancel_score:.2f}，"
                          f"继续={evidence.continue_score:.2f}，原因={evidence.reason or '已确认'}"
                          if evidence is not None else "未读取局部弹窗")
                self.message = ("等待继续对战页面连续确认，不发送菜单A "
                                f"({self._replay_evidence_reads}/2，整图={observation.state.confidence:.2f}，{detail})")
                return self.snapshot()
            if (self._replay_evidence_reads >= 2 and self._result_confirmed
                and not self._awaiting_deck_reselection):
                self._result_advance_acknowledged = True
                self._reward_advance_acknowledged = True
            self._handle_replay_choice(timestamp)
        else:
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = f"等待稳定状态，当前 {state.value}"

        return self.snapshot(
            npc_name=npc_name,
            difficulty=difficulty,
            level_wins=level_wins,
        )

    def _begin_match(self, identity: tuple[str, int] | None) -> None:
        self._match_identity = identity
        self._match_win_applied = False
        self._last_match_outcome = None
        self._result_candidate = None
        self._result_candidate_count = 0
        self._last_result_reading = None
        self._result_confirmed = False
        self._reset_result_advancement()

    def _reset_menu_routing(self) -> None:
        self._completion_menu_key = None
        self._completion_menu_reads = 0
        self._completion_last_observation = None
        self._setup_return_sent = False
        self._setup_return_at = 0.0

    @staticmethod
    def safe_menu_visible(observation) -> bool:
        if observation is None:
            return False
        state, menu = observation.state, observation.menu
        if (state.state not in {GameState.NPC_ROSTER, GameState.MATCH_SETUP}
            or state.confidence < .55 or not state.candidates
            or state.candidates[0].state is not state.state
            or state.candidates[0].score < .55 or menu is None
            or getattr(menu, "npc_id", None) is None
            or getattr(menu, "name_confidence", 1.0) < .55
            or getattr(menu, "wins_confidence", 0.0) < .55):
            return False
        wins = getattr(menu, "level_wins", ())
        return (len(wins) == 3 and all(value is not None and value >= 0 for value in wins)
                and (state.state is not GameState.MATCH_SETUP
                     or getattr(menu, "difficulty", None) in {1, 2, 3}))

    def _observe_completion_menu(self, observation) -> None:
        if not self.safe_menu_visible(observation):
            self._completion_menu_key = None
            self._completion_menu_reads = 0
            self._completion_last_observation = None
            return
        key = (observation.state.state, observation.menu.npc_id)
        if key != self._completion_menu_key:
            self._completion_menu_key = key
            self._completion_menu_reads = 0
            self._completion_last_observation = None
        if observation is not self._completion_last_observation:
            self._completion_menu_reads += 1
            self._completion_last_observation = observation

    def _finish_from_safe_menu(self) -> None:
        if self._completion_menu_reads >= 2:
            self.status = TaskStatus.COMPLETED
            self.message = "目标胜利数已达到，已确认返回NPC菜单"
        else:
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = "目标胜利数已达到，等待真实NPC菜单连续确认后完成任务"

    def _reset_result_advancement(self) -> None:
        self._result_advance_sent = False
        self._reward_advance_sent = False
        self._result_advance_attempts = 0
        self._reward_advance_attempts = 0
        self._result_advance_at = 0.0
        self._reward_advance_at = 0.0
        self._result_advance_acknowledged = False
        self._reward_advance_acknowledged = False
        self._reward_page_kind = None
        self._reward_total_attempts = 0
        self._settlement_state = None
        self._settlement_entered_at = 0.0
        self._settlement_evidence_key = None
        self._settlement_evidence_reads = 0
        self._settlement_last_observation = None
        self._replay_evidence_reads = 0
        self._last_replay_observation = None
        self._replay_exit_attempts = 0
        self._replay_exit_at = 0.0

    def _observe_settlement_page(self, observation, timestamp: float) -> None:
        state = observation.state.state
        if state is not self._settlement_state:
            self._settlement_state = state
            self._settlement_entered_at = timestamp
            self._settlement_evidence_key = None
            self._settlement_evidence_reads = 0
            self._settlement_last_observation = None
        if state is GameState.MATCH_RESULT:
            reading = observation.match_result
            outcome = getattr(getattr(reading, "outcome", None), "value", None)
            valid = (outcome in {"WIN", "LOSS", "DRAW"}
                     and reading.confidence >= self.RESULT_MIN_CONFIDENCE
                     and self._settlement_scene_visible(observation)
                     and (not self._result_confirmed or outcome == self._last_match_outcome))
            key = (state, outcome)
        else:
            valid = self._settlement_scene_visible(observation)
            page = getattr(observation, "reward_page_evidence", None)
            kind = page.page_kind if page is not None and page.visible else None
            key = (state, kind)
        if not valid:
            self._settlement_evidence_key = None
            self._settlement_evidence_reads = 0
            self._settlement_last_observation = None
            return
        if key != self._settlement_evidence_key:
            self._settlement_entered_at = timestamp
            self._settlement_evidence_key = key
            self._settlement_evidence_reads = 0
            self._settlement_last_observation = None
        if observation is not self._settlement_last_observation:
            self._settlement_evidence_reads += 1
            self._settlement_last_observation = observation

    def _settlement_scene_visible(self, observation) -> bool:
        # State hysteresis may retain MATCH_REWARD while confidence belongs
        # to the new top-ranked REPLAY_CHOICE candidate. Such a frame cannot
        # authorize another reward A, even if confidence is close to 1.
        state = observation.state
        candidates = state.candidates
        minimum = (self.RESULT_MIN_CONFIDENCE if state.state is GameState.MATCH_RESULT
                   else self.SETTLEMENT_SCENE_CONFIDENCE)
        margin_ok = (state.state is GameState.MATCH_RESULT or len(candidates) < 2
                     or candidates[0].score - candidates[1].score >= self.SETTLEMENT_SCENE_MARGIN)
        replay = getattr(observation, "replay_choice_evidence", None)
        if state.state is GameState.REPLAY_CHOICE and replay is not None:
            # The final board and WIN/LOSE background change the whole-frame
            # similarity. Confirm the actual modal text without weakening the
            # reward/result guards or trusting the state machine's hysteresis.
            return (replay.visible and bool(candidates)
                    and candidates[0].state is state.state and margin_ok)
        reward = getattr(observation, "reward_page_evidence", None)
        if state.state is GameState.MATCH_REWARD and reward is not None:
            return (reward.visible and bool(candidates)
                    and candidates[0].state is state.state)
        return (state.confidence >= minimum and bool(candidates)
                and candidates[0].state is state.state
                and candidates[0].score >= minimum and margin_ok)

    def _advance_settlement_page(self, state: GameState, timestamp: float) -> None:
        prefix = "_result" if state is GameState.MATCH_RESULT else "_reward"
        label = "结果页" if state is GameState.MATCH_RESULT else "奖励页"
        if (self._settlement_evidence_reads < 2
            or timestamp - self._settlement_entered_at < self.SETTLEMENT_READY_SECONDS):
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = f"比赛结果 {self._last_match_outcome} 已确认，等待当前{label}稳定后按A"
            if state is GameState.MATCH_REWARD and self._last_observation is not None:
                page = getattr(self._last_observation, "reward_page_evidence", None)
                if page is not None:
                    self.message += (f" ({self._settlement_evidence_reads}/2，"
                                     f"标题={page.title_score:.2f}，原因={page.reason or page.page_kind})")
            return
        if state is GameState.MATCH_REWARD:
            self._result_advance_acknowledged = True
            kind = self._settlement_evidence_key[1]
            self._reward_total_attempts = max(self._reward_total_attempts, self._reward_advance_attempts)
            if kind is not None and kind != self._reward_page_kind:
                # Commit a new subpage only after its own two observations
                # and dwell. The whole reward sequence also keeps a budget.
                if self._reward_page_kind is not None:
                    self._reward_advance_attempts = 0
                    self._reward_advance_sent = False
                    self._reward_advance_at = 0.0
                self._reward_page_kind = kind
        if getattr(self, prefix + "_advance_acknowledged"):
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = f"已确认离开{label}，等待后续页面稳定，不重发旧页面A"
            return
        attempts = getattr(self, prefix + "_advance_attempts")
        if getattr(self, prefix + "_advance_sent") and attempts == 0:
            attempts = 1  # A cached older core recorded only the sent latch.
            setattr(self, prefix + "_advance_attempts", attempts)
        if attempts and timestamp - getattr(self, prefix + "_advance_at") < self.SETTLEMENT_RETRY_SECONDS:
            self.status = TaskStatus.RUNNING
            self.message = f"已按A {attempts}/{self.SETTLEMENT_MAX_ATTEMPTS}次，等待{label}切换"
            return
        if attempts >= self.SETTLEMENT_MAX_ATTEMPTS:
            self.status = TaskStatus.ERROR
            self.message = f"{label}连续确认{attempts}次仍未切换，已停止按键并保留诊断"
            self.controller.send_state(ControllerState.neutral())
            return
        if state is GameState.MATCH_REWARD:
            self._reward_total_attempts = max(self._reward_total_attempts, attempts)
            if self._reward_total_attempts >= self.REWARD_TOTAL_MAX_ATTEMPTS:
                self.status = TaskStatus.ERROR
                self.message = "本场奖励确认已达到总次数上限，已停止按键并保留诊断"
                self.controller.send_state(ControllerState.neutral())
                return
        self.status = TaskStatus.RUNNING
        self.message = (f"比赛结果 {self._last_match_outcome}，确认离开{label}"
                        if attempts == 0 else f"{label}仍未切换，重试A（{attempts + 1}/{self.SETTLEMENT_MAX_ATTEMPTS}）")
        if self._pulse(Button.A, timestamp):
            setattr(self, prefix + "_advance_sent", True)
            setattr(self, prefix + "_advance_attempts", attempts + 1)
            setattr(self, prefix + "_advance_at", timestamp)
            if state is GameState.MATCH_REWARD:
                self._reward_total_attempts += 1

    def _ready_match_result(self, reading) -> bool:
        if not self._confirm_match_result(reading):
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = "等待比赛结果连续两次明确识别，不跳过未知结算"
        elif self._last_match_outcome == "WIN" and not self._apply_match_win():
            self.status = TaskStatus.WAITING_RECOGNITION
            self.message = "已确认胜利，等待本场NPC、难度和累计胜场基准，保留结算"
        else:
            return True
        self.controller.send_state(ControllerState.neutral())
        return False

    def _confirm_match_result(self, reading) -> bool:
        if self._result_confirmed:
            return True
        if reading is self._last_result_reading:
            return False
        self._last_result_reading = reading
        outcome = getattr(getattr(reading, "outcome", None), "value", "UNKNOWN")
        if outcome not in {"WIN", "LOSS", "DRAW"} or reading.confidence < self.RESULT_MIN_CONFIDENCE:
            self._result_candidate = None
            self._result_candidate_count = 0
            return False
        if outcome == self._result_candidate:
            self._result_candidate_count += 1
        else:
            self._result_candidate = outcome
            self._result_candidate_count = 1
        if self._result_candidate_count < self.RESULT_STABLE_READINGS:
            return False
        self._last_match_outcome = outcome
        self._result_confirmed = True
        return True

    def _apply_match_win(self) -> bool:
        if self._match_win_applied:
            return True
        if (self.task is None or self.plan is None or self._match_identity is None
            or self._match_identity[0] != self.task.npc_id):
            return False
        level = self._match_identity[1]
        if level not in {1, 2, 3}:
            return False
        values = list(self._stable_level_wins or tuple(
            max(int(value or 0), next((stage.current_wins for stage in self.plan.stages
                if stage.level == index), 0))
            for index, value in enumerate(self.plan.initial_level_wins, 1)
        ))
        values[level - 1] += 1
        self._stable_level_wins = tuple(values)
        self._wins_npc_id = self.task.npc_id
        self.plan = self.plan.with_observed_wins(self._stable_level_wins)
        self._match_win_applied = True
        self._wins_change_allowed = True
        return True

    def progress_checkpoint(self) -> dict:
        """Carry confirmed progress across an analysis-core reload."""
        names = ("_stable_level_wins", "_wins_npc_id", "_wins_change_allowed",
                 "_match_identity", "_match_win_applied", "_last_match_outcome",
                 "_result_confirmed", "_result_advance_sent", "_reward_advance_sent",
                 "_result_advance_attempts", "_reward_advance_attempts",
                 "_result_advance_at", "_reward_advance_at",
                 "_result_advance_acknowledged", "_reward_advance_acknowledged",
                 "_awaiting_deck_reselection", "_deck_selection_confirmed",
                 "_deck_navigation_steps", "_last_deck_probe_at", "_deck_select_seen",
                 "_replay_choice_prepared", "_replay_choice_sent", "_reward_page_kind", "_reward_total_attempts",
                 "_replay_exit_attempts", "_replay_exit_at", "_setup_return_sent", "_setup_return_at")
        executor = getattr(self, "battle_executor", None)
        pending = (VisionMctsBattleExecutor.pending_discard_checkpoint(executor)
                   if executor is not None else None)
        placement = (VisionMctsBattleExecutor.pending_placement_checkpoint(executor)
                     if executor is not None else None)
        memory = (VisionMctsBattleExecutor.battle_memory_checkpoint(executor)
                  if executor is not None else None)
        return {"task": self.task, "pending_discard": pending,
                "pending_placement": placement,
                "battle_memory": memory,
                "values": {name: getattr(self, name, None) for name in names}}

    def restore_progress(self, checkpoint: dict) -> None:
        if checkpoint.get("task") != self.task:
            return
        for name, value in checkpoint.get("values", {}).items():
            if name in {"_stable_level_wins", "_wins_npc_id", "_wins_change_allowed",
                        "_match_identity", "_match_win_applied", "_last_match_outcome",
                        "_result_confirmed", "_result_advance_sent", "_reward_advance_sent",
                        "_result_advance_attempts", "_reward_advance_attempts",
                        "_result_advance_at", "_reward_advance_at",
                        "_result_advance_acknowledged", "_reward_advance_acknowledged",
                        "_awaiting_deck_reselection", "_deck_selection_confirmed",
                        "_deck_navigation_steps", "_last_deck_probe_at", "_deck_select_seen",
                        "_replay_choice_prepared", "_replay_choice_sent", "_reward_page_kind", "_reward_total_attempts",
                        "_replay_exit_attempts", "_replay_exit_at", "_setup_return_sent", "_setup_return_at"} and value is not None:
                setattr(self, name, value)
        memory = checkpoint.get("battle_memory")
        restore = getattr(self.battle_executor, "restore_battle_memory", None)
        if memory is not None and restore is not None:
            restore(memory)
        pending = checkpoint.get("pending_discard")
        restore = getattr(self.battle_executor, "restore_pending_discard", None)
        if pending is not None and restore is not None:
            restore(pending)
        placement = checkpoint.get("pending_placement")
        restore = getattr(self.battle_executor, "restore_pending_placement", None)
        if placement is not None and restore is not None:
            restore(placement)

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
        if self._stable_level_wins is not None and any(
            value < previous for value, previous in zip(candidate, self._stable_level_wins, strict=True)
        ):
            # Result feedback is a lower bound on cumulative wins. The menu
            # can briefly show an old count before its counter animation ends.
            self._wins_candidate = None
            self._wins_candidate_count = 0
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
            if menu is not None and self.safe_menu_visible(observation):
                self.status = TaskStatus.RUNNING
                self.message = "当前难度页属于其他NPC，按B返回NPC列表"
                if self._pulse(Button.B, timestamp):
                    self._setup_return_sent = True
                    self._setup_return_at = timestamp
            else:
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
            self._finish_from_safe_menu()
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
                    self._begin_match((menu.npc_id, menu.difficulty))
                    self._require_deck_reselection()
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
                1 if self._last_match_outcome == "WIN" and not self._match_win_applied else 0
            )
            should_continue = remaining_after_match > 0 and (
                self._match_identity is None or active.level == self._match_identity[1]
            )
        direction = (
            Button.DPAD_RIGHT
            if should_continue
            else Button.DPAD_LEFT
        )
        choice = "继续当前难度" if should_continue else "退出并切换难度"
        if not should_continue:
            if self._replay_choice_sent and self._replay_exit_attempts == 0:
                self._replay_exit_attempts = 1  # Migrate a previous core's sent latch.
            if self._replay_exit_attempts:
                if timestamp - self._replay_exit_at < self.SETTLEMENT_RETRY_SECONDS:
                    self.status = TaskStatus.WAITING_RECOGNITION
                    self.message = "已确认退出继续对战，等待NPC菜单，不启动下一任务"
                    self.controller.send_state(ControllerState.neutral())
                    return
                if self._replay_exit_attempts >= self.SETTLEMENT_MAX_ATTEMPTS:
                    self.status = TaskStatus.ERROR
                    self.message = "退出继续对战连续确认4次仍未返回菜单，已停止按键并保留诊断"
                    self.controller.send_state(ControllerState.neutral())
                    return
        if not self._replay_choice_prepared:
            self.status = TaskStatus.RUNNING
            self.message = f"比赛结束：{choice}"
            if self._pulse(direction, timestamp):
                self._replay_choice_prepared = True
            return
        if not self._replay_choice_sent or not should_continue:
            self.status = TaskStatus.RUNNING
            self.message = f"比赛结束：{choice}，确认"
            if self._pulse(Button.A, timestamp):
                self._replay_choice_sent = True
                if should_continue:
                    self._begin_match(self._match_identity)
                    self._require_deck_reselection()
                else:
                    self._replay_exit_attempts += 1
                    self._replay_exit_at = timestamp
                    # Reassert Cancel before a retry, rather than trusting a
                    # focus that a lost input or manual action could change.
                    self._replay_choice_prepared = False
                    self.status = TaskStatus.WAITING_RECOGNITION
                    self.message = "已确认退出继续对战，等待NPC菜单，不启动下一任务"
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
            # A reload may have recreated the vision runtime after this
            # match's visual confirmation. Restore that binding without
            # pressing A again or reusing it for a later replay.
            self.runtime.set_expected_deck(self.deck.name, self.deck.card_ids)
            self.status = TaskStatus.RUNNING
            self.message = "卡组已确认，等待进入对战"
            self.controller.send_state(ControllerState.neutral())
            return
        if timestamp - self._last_deck_probe_at < self.deck_probe_seconds:
            return
        self._last_deck_probe_at = timestamp
        visible = observation.deck
        if visible is not None and getattr(visible, "source", "auto") == "configured":
            self.status = TaskStatus.WAITING_FOR_DECK
            self.message = "等待读取本局实际15张卡牌，预设配置不能作为已选中卡组的证据"
            self.controller.send_state(ControllerState.neutral())
            return
        if visible is not None and len(visible.cards) == 15 and _deck_signature_matches(
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
                self._awaiting_deck_reselection = False
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

    def _require_deck_reselection(self) -> None:
        self._awaiting_deck_reselection = True
        self._deck_selection_confirmed = False
        self._deck_select_seen = False
        self._deck_navigation_steps = 0
        self._last_deck_probe_at = 0.0
        self.runtime.clear_expected_deck()

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
            if self._stable_level_wins is not None:
                level_wins = self._stable_level_wins
            elif level_wins is None and observation.menu is not None:
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
