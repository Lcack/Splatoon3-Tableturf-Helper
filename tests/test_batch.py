import json

import pytest

from tableturf_rebuild.application.batch import BatchSelfPlayRunner
from tableturf_rebuild.application.selfplay import SelfPlayRunner
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.planning.policies import GreedyPolicy


def make_runner(tmp_path):
    catalog = GameCatalog.from_package()
    batch = BatchSelfPlayRunner(SelfPlayRunner(catalog))
    scenarios = catalog.npc_scenarios()[:2]
    output = tmp_path / "run.jsonl"
    metadata = {
        "seed": 77,
        "policy_a": "greedy",
        "policy_b": "greedy",
        "level": 1,
        "simulations": 8,
        "time_limit_seconds": 0.1,
    }
    return batch, scenarios, output, metadata


def test_streaming_run_and_resume(tmp_path) -> None:
    batch, scenarios, output, metadata = make_runner(tmp_path)

    first = batch.run(
        output=output,
        games=2,
        scenarios=scenarios,
        seed=77,
        policy_a=GreedyPolicy(name="greedy-a"),
        policy_b=GreedyPolicy(name="greedy-b"),
        metadata=metadata,
    )

    assert first.stopped_reason == "games"
    assert first.summary["overall"]["games"] == 2
    assert output.exists()
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert output.with_name("run.jsonl.meta.json").exists()
    assert output.with_name("run.jsonl.summary.json").exists()

    second = batch.run(
        output=output,
        games=1,
        scenarios=scenarios,
        seed=77,
        policy_a=GreedyPolicy(name="greedy-a"),
        policy_b=GreedyPolicy(name="greedy-b"),
        metadata=metadata,
        resume=True,
    )

    assert second.summary["overall"]["games"] == 3
    assert second.first_game_index == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_resume_rejects_different_configuration(tmp_path) -> None:
    batch, scenarios, output, metadata = make_runner(tmp_path)
    batch.run(
        output=output,
        games=1,
        scenarios=scenarios,
        seed=77,
        policy_a=GreedyPolicy(name="greedy-a"),
        policy_b=GreedyPolicy(name="greedy-b"),
        metadata=metadata,
    )
    changed = dict(metadata)
    changed["seed"] = 999

    with pytest.raises(ValueError, match="does not match"):
        batch.run(
            output=output,
            games=1,
            scenarios=scenarios,
            seed=999,
            policy_a=GreedyPolicy(name="greedy-a"),
            policy_b=GreedyPolicy(name="greedy-b"),
            metadata=changed,
            resume=True,
        )


def test_resume_discards_incomplete_tail(tmp_path) -> None:
    batch, scenarios, output, metadata = make_runner(tmp_path)
    batch.run(
        output=output,
        games=1,
        scenarios=scenarios,
        seed=77,
        policy_a=GreedyPolicy(name="greedy-a"),
        policy_b=GreedyPolicy(name="greedy-b"),
        metadata=metadata,
    )
    with output.open("ab") as file:
        file.write(b'{"incomplete":')

    result = batch.run(
        output=output,
        games=1,
        scenarios=scenarios,
        seed=77,
        policy_a=GreedyPolicy(name="greedy-a"),
        policy_b=GreedyPolicy(name="greedy-b"),
        metadata=metadata,
        resume=True,
    )

    assert result.summary["overall"]["games"] == 2
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line) for line in lines)


def test_time_limit_can_stop_before_a_game(tmp_path) -> None:
    batch, scenarios, output, metadata = make_runner(tmp_path)
    result = batch.run(
        output=output,
        games=10,
        scenarios=scenarios,
        seed=77,
        policy_a=GreedyPolicy(name="greedy-a"),
        policy_b=GreedyPolicy(name="greedy-b"),
        metadata=metadata,
        max_seconds=0,
    )

    assert result.games_run == 0
    assert result.stopped_reason == "time_limit"

