from tableturf_rebuild.application.task_plan import (
    NpcTaskDefinition,
    build_npc_plan,
)


def test_plan_unlocks_intermediate_levels_before_target() -> None:
    task = NpcTaskDefinition(
        npc_id="MiniGame_MiniGameKurage0",
        npc_name="新手水母",
        target_level=3,
        target_wins=30,
    )

    plan = build_npc_plan(task, (3, 0, 0))

    assert [(stage.level, stage.goal_wins, stage.remaining_wins) for stage in plan.stages] == [
        (1, 3, 0),
        (2, 3, 3),
        (3, 30, 30),
    ]
    assert plan.active_stage is not None
    assert plan.active_stage.level == 2
    assert plan.total_remaining_wins == 33


def test_plan_tracks_new_wins_without_losing_initial_progress() -> None:
    task = NpcTaskDefinition(
        npc_id="MiniGame_MiniGameKurage0",
        npc_name="新手水母",
        target_level=3,
        target_wins=30,
    )
    initial = build_npc_plan(task, (3, 0, 0))

    updated = initial.with_observed_wins((3, 2, 0))

    assert updated.active_stage is not None
    assert updated.active_stage.level == 2
    assert updated.active_stage.remaining_wins == 1
    assert updated.total_remaining_wins == 31
