from __future__ import annotations

from dataclasses import dataclass


UNLOCK_WINS = 3


@dataclass(frozen=True, slots=True)
class NpcTaskDefinition:
    npc_id: str
    npc_name: str
    target_level: int
    target_wins: int
    deck_name: str | None = None

    def __post_init__(self) -> None:
        if self.target_level not in {1, 2, 3}:
            raise ValueError("target level must be 1, 2, or 3")
        if self.target_wins <= 0:
            raise ValueError("target wins must be positive")


@dataclass(frozen=True, slots=True)
class WinStage:
    level: int
    goal_wins: int
    current_wins: int

    @property
    def remaining_wins(self) -> int:
        return max(0, self.goal_wins - self.current_wins)

    @property
    def complete(self) -> bool:
        return self.remaining_wins == 0


@dataclass(frozen=True, slots=True)
class NpcWinPlan:
    task: NpcTaskDefinition
    initial_level_wins: tuple[int | None, int | None, int | None]
    stages: tuple[WinStage, ...]

    @property
    def complete(self) -> bool:
        return all(stage.complete for stage in self.stages)

    @property
    def active_stage(self) -> WinStage | None:
        return next(
            (stage for stage in self.stages if not stage.complete),
            None,
        )

    @property
    def total_remaining_wins(self) -> int:
        return sum(stage.remaining_wins for stage in self.stages)

    def with_observed_wins(
        self,
        level_wins: tuple[int | None, int | None, int | None],
    ) -> "NpcWinPlan":
        return build_npc_plan(
            self.task,
            level_wins,
            initial_level_wins=self.initial_level_wins,
        )


def build_npc_plan(
    task: NpcTaskDefinition,
    level_wins: tuple[int | None, int | None, int | None],
    *,
    initial_level_wins: tuple[int | None, int | None, int | None] | None = None,
) -> NpcWinPlan:
    baseline = initial_level_wins or level_wins
    observed = tuple(
        max(
            int(current or 0),
            int(initial or 0),
        )
        for current, initial in zip(level_wins, baseline, strict=True)
    )
    stages = []
    for level in range(1, task.target_level + 1):
        goal = UNLOCK_WINS if level < task.target_level else task.target_wins
        stages.append(
            WinStage(
                level=level,
                goal_wins=goal,
                current_wins=min(observed[level - 1], goal),
            )
        )
    return NpcWinPlan(
        task=task,
        initial_level_wins=tuple(baseline),
        stages=tuple(stages),
    )
