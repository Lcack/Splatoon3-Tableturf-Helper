from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tableturf_rebuild.application.selfplay import GameRecord, SelfPlayRunner
from tableturf_rebuild.data.catalog import NpcScenario
from tableturf_rebuild.planning.policies import Policy


@dataclass(slots=True)
class Bucket:
    games: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    score_a: int = 0
    score_b: int = 0
    elapsed_seconds: float = 0.0

    def add(self, record: GameRecord) -> None:
        self.games += 1
        self.wins_a += record.winner == "A"
        self.wins_b += record.winner == "B"
        self.draws += record.winner == "draw"
        self.score_a += record.score_a
        self.score_b += record.score_b
        self.elapsed_seconds += record.elapsed_seconds

    def as_dict(self) -> dict[str, Any]:
        games = max(1, self.games)
        return {
            "games": self.games,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "draws": self.draws,
            "average_score_a": self.score_a / games,
            "average_score_b": self.score_b / games,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(slots=True)
class PolicyBucket:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    score_for: int = 0
    score_against: int = 0

    def as_dict(self) -> dict[str, Any]:
        games = max(1, self.games)
        return {
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / games,
            "average_score_for": self.score_for / games,
            "average_score_against": self.score_against / games,
        }


@dataclass(slots=True)
class SummaryAccumulator:
    overall: Bucket = field(default_factory=Bucket)
    groups: dict[str, dict[str, Bucket]] = field(
        default_factory=lambda: {
            "matchup": {},
            "npc_a": {},
            "npc_level_a": {},
            "stage": {},
            "scenario_a": {},
        }
    )
    policies: dict[str, PolicyBucket] = field(default_factory=dict)
    last_game_index: int = -1

    def add(self, record: GameRecord) -> None:
        self.overall.add(record)
        self.last_game_index = max(self.last_game_index, record.game_index)
        labels = {
            "matchup": f"{record.policy_a} vs {record.policy_b}",
            "npc_a": record.scenario_a_npc,
            "npc_level_a": (
                f"{record.scenario_a_npc}:L{record.scenario_a_level}"
                if record.scenario_a_npc
                else ""
            ),
            "stage": record.stage_id,
            "scenario_a": (
                f"{record.scenario_a_npc}:L{record.scenario_a_level}:"
                f"{record.stage_id}"
                if record.scenario_a_npc
                else record.stage_id
            ),
        }
        for group_name, label in labels.items():
            self.groups[group_name].setdefault(label, Bucket()).add(record)

        for label, own_side, own_score, opponent_score in (
            (record.policy_a, "A", record.score_a, record.score_b),
            (record.policy_b, "B", record.score_b, record.score_a),
        ):
            policy = self.policies.setdefault(label, PolicyBucket())
            policy.games += 1
            policy.score_for += own_score
            policy.score_against += opponent_score
            if record.winner == "draw":
                policy.draws += 1
            elif record.winner == own_side:
                policy.wins += 1
            else:
                policy.losses += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "last_game_index": self.last_game_index,
            "overall": self.overall.as_dict(),
            "policies": {
                name: bucket.as_dict()
                for name, bucket in sorted(self.policies.items())
            },
            "groups": {
                group_name: {
                    key: bucket.as_dict()
                    for key, bucket in sorted(group.items())
                }
                for group_name, group in self.groups.items()
            },
        }


@dataclass(frozen=True, slots=True)
class BatchResult:
    games_run: int
    first_game_index: int | None
    last_game_index: int | None
    elapsed_seconds: float
    stopped_reason: str
    summary: dict[str, Any]


def _sidecar_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.name}.{suffix}")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    """Read complete JSONL records and trim one incomplete crash tail."""

    if not path.exists():
        return
    with path.open("rb+") as file:
        while True:
            position = file.tell()
            line = file.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                remaining = file.read()
                if remaining.strip():
                    raise ValueError(f"corrupt JSONL record at byte {position}") from None
                file.truncate(position)
                file.flush()
                os.fsync(file.fileno())
                break
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at byte {position} is not an object")
            yield value


def append_json_record(path: Path, record: GameRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        asdict(record),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    with path.open("ab") as file:
        file.write(encoded)
        file.write(b"\n")
        file.flush()
        os.fsync(file.fileno())


class BatchSelfPlayRunner:
    def __init__(self, runner: SelfPlayRunner):
        self.runner = runner

    def run(
        self,
        output: Path,
        games: int,
        scenarios: Iterable[NpcScenario],
        seed: int,
        policy_a: Policy,
        policy_b: Policy,
        metadata: dict[str, Any],
        max_seconds: float | None = None,
        resume: bool = False,
    ) -> BatchResult:
        output = output.resolve()
        with _exclusive_run_lock(_sidecar_path(output, "lock")):
            return self._run_unlocked(
                output=output,
                games=games,
                scenarios=scenarios,
                seed=seed,
                policy_a=policy_a,
                policy_b=policy_b,
                metadata=metadata,
                max_seconds=max_seconds,
                resume=resume,
            )

    def _run_unlocked(
        self,
        output: Path,
        games: int,
        scenarios: Iterable[NpcScenario],
        seed: int,
        policy_a: Policy,
        policy_b: Policy,
        metadata: dict[str, Any],
        max_seconds: float | None,
        resume: bool,
    ) -> BatchResult:
        if games < 0:
            raise ValueError("games cannot be negative")
        scenario_list = tuple(scenarios)
        if not scenario_list:
            raise ValueError("at least one NPC scenario is required")

        metadata_path = _sidecar_path(output, "meta.json")
        summary_path = _sidecar_path(output, "summary.json")
        full_metadata = {
            "format_version": 1,
            **metadata,
        }
        summary = SummaryAccumulator()

        if resume:
            if not output.exists() or not metadata_path.exists():
                raise ValueError("resume requires existing JSONL and metadata files")
            existing_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if existing_metadata != full_metadata:
                raise ValueError(
                    "resume configuration does not match the existing run"
                )
            for value in iter_json_records(output):
                summary.add(GameRecord(**value))
        else:
            if output.exists() and output.stat().st_size:
                raise ValueError(
                    f"output already exists and is not empty: {output}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.touch(exist_ok=True)
            _write_json_atomic(metadata_path, full_metadata)

        next_index = summary.last_game_index + 1
        first_index: int | None = None
        games_run = 0
        stopped_reason = "games"
        started = time.monotonic()
        try:
            for _ in range(games):
                if (
                    max_seconds is not None
                    and time.monotonic() - started >= max_seconds
                ):
                    stopped_reason = "time_limit"
                    break
                index = next_index + games_run
                scenario_a = scenario_list[index % len(scenario_list)]
                scenario_b = scenario_list[(index + 1) % len(scenario_list)]
                record = self.runner.play_game(
                    game_index=index,
                    seed=seed + index * 104729,
                    stage_id=scenario_a.stage_id,
                    deck_a_id=scenario_a.deck_id,
                    deck_b_id=scenario_b.deck_id,
                    policy_a=policy_a,
                    policy_b=policy_b,
                    scenario_a=scenario_a,
                    scenario_b=scenario_b,
                )
                if first_index is None:
                    first_index = index
                append_json_record(output, record)
                summary.add(record)
                games_run += 1
        except KeyboardInterrupt:
            stopped_reason = "interrupted"
        finally:
            _write_json_atomic(summary_path, summary.as_dict())

        return BatchResult(
            games_run=games_run,
            first_game_index=first_index,
            last_game_index=summary.last_game_index if games_run else None,
            elapsed_seconds=time.monotonic() - started,
            stopped_reason=stopped_reason,
            summary=summary.as_dict(),
        )


@contextmanager
def _exclusive_run_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    file = lock_path.open("a+b")
    try:
        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            file.write(b"\0")
            file.flush()
        file.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError(
                    f"another process is already writing this run: {lock_path}"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError(
                    f"another process is already writing this run: {lock_path}"
                ) from error
        yield
    finally:
        try:
            file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        file.close()

