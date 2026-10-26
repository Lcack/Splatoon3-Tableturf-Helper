from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from tableturf_rebuild.application.battle_vision import (
    BattleVisionObservation,
    BattleVisionRuntime,
)
from tableturf_rebuild.capture.models import CaptureConfig
from tableturf_rebuild.capture.worker import CaptureWorker


@dataclass(frozen=True, slots=True)
class LiveVisionRecord:
    sequence: int
    timestamp: float
    state: str
    confidence: float
    self_cells: int | None
    opponent_cells: int | None
    self_special_cells: int | None
    opponent_special_cells: int | None
    self_active_special_cells: int | None
    opponent_active_special_cells: int | None
    placement_preview_present: bool | None
    selection_variant: str | None
    match_result: str | None
    npc_name: str | None
    level_wins: tuple[int | None, int | None, int | None] | None
    difficulty: int | None
    stage_id: str | None
    stage_name: str | None
    ai_type: str | None
    deck_id: str | None
    deck_source: str | None
    deck_total_area: int | None
    deck_total_sp_cost: int | None
    card_areas: tuple[int, ...] | None
    card_sp_costs: tuple[int, ...] | None
    hand_card_ids: tuple[str, ...] | None
    hand_areas: tuple[int, ...] | None
    hand_sp_costs: tuple[int, ...] | None
    self_reveal_card: str | None
    self_reveal_area: int | None
    self_reveal_sp_cost: int | None
    opponent_reveal_card: str | None
    opponent_reveal_area: int | None
    opponent_reveal_sp_cost: int | None


@dataclass(frozen=True, slots=True)
class LiveVisionSummary:
    frames: int
    elapsed_seconds: float
    stopped_reason: str


def run_live_vision(
    capture_config: CaptureConfig,
    runtime: BattleVisionRuntime,
    *,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    poll_seconds: float = 0.005,
    on_record: Callable[[LiveVisionRecord], None] | None = None,
    worker_factory: Callable[[CaptureConfig], CaptureWorker] = CaptureWorker,
) -> LiveVisionSummary:
    worker = worker_factory(capture_config)
    worker.start()
    started = time.perf_counter()
    deadline = started + max_seconds if max_seconds is not None else None
    frames = 0
    last_sequence = -1
    stopped_reason = "limit"
    try:
        while True:
            if max_frames is not None and frames >= max_frames:
                stopped_reason = "max_frames"
                break
            if deadline is not None and time.perf_counter() >= deadline:
                stopped_reason = "max_seconds"
                break
            error = worker.error
            if error:
                raise RuntimeError(error)
            packet = worker.latest()
            if packet is None or packet.sequence == last_sequence:
                time.sleep(poll_seconds)
                continue
            last_sequence = packet.sequence
            observation = runtime.update(
                packet.frame,
                include_selection=(
                    frames % 5 == 0
                ),
            )
            record = _record(packet.sequence, packet.timestamp, observation)
            if on_record is not None:
                on_record(record)
            frames += 1
    finally:
        worker.stop()
    return LiveVisionSummary(
        frames=frames,
        elapsed_seconds=time.perf_counter() - started,
        stopped_reason=stopped_reason,
    )


def _record(
    sequence: int,
    timestamp: float,
    observation: BattleVisionObservation,
) -> LiveVisionRecord:
    board = observation.board
    return LiveVisionRecord(
        sequence=sequence,
        timestamp=timestamp,
        state=observation.state.state.value,
        confidence=observation.state.confidence,
        self_cells=board.self_count if board is not None else None,
        opponent_cells=board.opponent_count if board is not None else None,
        self_special_cells=(
            observation.active_specials.self_special_count
            if observation.active_specials is not None
            else None
        ),
        opponent_special_cells=(
            observation.active_specials.opponent_special_count
            if observation.active_specials is not None
            else None
        ),
        self_active_special_cells=(
            observation.active_specials.self_count
            if observation.active_specials is not None
            else None
        ),
        opponent_active_special_cells=(
            observation.active_specials.opponent_count
            if observation.active_specials is not None
            else None
        ),
        placement_preview_present=(
            board.placement_preview_present
            if board is not None
            else None
        ),
        selection_variant=(
            observation.selection_profile.variant
            if observation.selection_profile is not None
            else None
        ),
        match_result=(
            observation.match_result.outcome.value
            if observation.match_result is not None
            else None
        ),
        npc_name=(
            observation.menu.npc_name
            if observation.menu is not None
            else observation.opponent_context.npc_name
            if observation.opponent_context is not None
            else None
        ),
        level_wins=(
            observation.menu.level_wins
            if observation.menu is not None
            else None
        ),
        difficulty=(
            observation.menu.difficulty
            if observation.menu is not None
            else observation.opponent_context.difficulty
            if observation.opponent_context is not None
            else None
        ),
        stage_id=(
            observation.opponent_context.stage_id
            if observation.opponent_context is not None
            else None
        ),
        stage_name=(
            observation.opponent_context.stage_name
            if observation.opponent_context is not None
            else None
        ),
        ai_type=(
            observation.opponent_context.ai_type
            if observation.opponent_context is not None
            else None
        ),
        deck_id=(
            observation.deck.deck_id
            if observation.deck is not None
            else observation.opponent_context.deck_id
            if observation.opponent_context is not None
            else None
        ),
        deck_source=(
            observation.deck.source
            if observation.deck is not None
            else None
        ),
        deck_total_area=(
            observation.deck.total_area
            if observation.deck is not None
            else None
        ),
        deck_total_sp_cost=(
            observation.deck.total_sp_cost
            if observation.deck is not None
            else None
        ),
        card_areas=(
            tuple(card.area for card in observation.deck.cards)
            if observation.deck is not None
            else None
        ),
        card_sp_costs=(
            tuple(card.sp_cost for card in observation.deck.cards)
            if observation.deck is not None
            else None
        ),
        hand_card_ids=(
            observation.hand.card_ids
            if observation.hand is not None
            else None
        ),
        hand_areas=(
            tuple(card.area for card in observation.hand.cards)
            if observation.hand is not None
            else None
        ),
        hand_sp_costs=(
            tuple(card.sp_cost for card in observation.hand.cards)
            if observation.hand is not None
            else None
        ),
        self_reveal_card=(
            observation.reveal.own.name
            if observation.reveal is not None
            else None
        ),
        self_reveal_area=(
            observation.reveal.own.area
            if observation.reveal is not None
            else None
        ),
        self_reveal_sp_cost=(
            observation.reveal.own.sp_cost
            if observation.reveal is not None
            else None
        ),
        opponent_reveal_card=(
            observation.reveal.opponent.name
            if observation.reveal is not None
            else None
        ),
        opponent_reveal_area=(
            observation.reveal.opponent.area
            if observation.reveal is not None
            else None
        ),
        opponent_reveal_sp_cost=(
            observation.reveal.opponent.sp_cost
            if observation.reveal is not None
            else None
        ),
    )
