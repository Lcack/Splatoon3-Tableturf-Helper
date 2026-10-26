from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from tableturf_rebuild.application.batch import BatchSelfPlayRunner
from tableturf_rebuild.application.deck_search import (
    DeckSearchConfig,
    DeckSearchRunner,
    select_level3_scenarios,
)
from tableturf_rebuild.application.selfplay import SelfPlayRunner
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.grid import PLAYABLE_CELLS, Cell
from tableturf_rebuild.planning.mcts import MCTSConfig, MCTSDecisionPolicy
from tableturf_rebuild.planning.policies import GreedyPolicy, RandomPolicy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tableturf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-data", help="validate bundled game data")

    card_parser = subparsers.add_parser("describe-card", help="show one card")
    card_parser.add_argument("card_id")

    stage_parser = subparsers.add_parser("describe-stage", help="show one stage")
    stage_parser.add_argument("stage_id")

    selfplay_parser = subparsers.add_parser(
        "selfplay", help="run reproducible offline AI self-play"
    )
    selfplay_parser.add_argument("--games", type=int, default=1)
    selfplay_parser.add_argument("--seed", type=int, default=0)
    selfplay_parser.add_argument(
        "--policy-a", choices=("random", "greedy", "mcts"), default="greedy"
    )
    selfplay_parser.add_argument(
        "--policy-b", choices=("random", "greedy", "mcts"), default="greedy"
    )
    selfplay_parser.add_argument("--simulations", type=int, default=64)
    selfplay_parser.add_argument("--time-limit", type=float, default=0.5)
    selfplay_parser.add_argument("--max-minutes", type=float, default=None)
    selfplay_parser.add_argument("--resume", action="store_true")
    selfplay_parser.add_argument(
        "--level", type=int, choices=(1, 2, 3), default=None
    )
    selfplay_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/selfplay/latest.jsonl"),
    )
    subparsers.add_parser(
        "controller-debug",
        help="open the interactive EasyCon controller debugger",
    )
    subparsers.add_parser(
        "capture-debug",
        help="open the capture-card diagnostics window",
    )
    subparsers.add_parser(
        "task-terminal",
        help="open the NPC win task, deck preset, and automation terminal",
    )
    video_review_parser = subparsers.add_parser(
        "video-review",
        help="review a recorded video and annotate paused frames",
    )
    video_review_parser.add_argument(
        "--video",
        type=Path,
        default=None,
    )
    live_parser = subparsers.add_parser(
        "vision-live",
        help="analyze the capture card without opening any window",
    )
    live_parser.add_argument(
        "--backend",
        choices=("DSHOW", "MSMF", "ANY"),
        default="DSHOW",
    )
    live_parser.add_argument("--device-index", type=int, default=0)
    live_parser.add_argument("--width", type=int, default=1920)
    live_parser.add_argument("--height", type=int, default=1080)
    live_parser.add_argument("--fps", type=float, default=60)
    live_parser.add_argument("--stage", default="Rectangle")
    live_parser.add_argument("--max-frames", type=int, default=None)
    live_parser.add_argument("--max-seconds", type=float, default=None)
    live_parser.add_argument("--log-every", type=int, default=60)
    live_parser.add_argument("--output", type=Path, default=None)

    deck_search_parser = subparsers.add_parser(
        "deck-search",
        help="evaluate unique candidate decks against NPC proxy scenarios",
    )
    deck_search_parser.add_argument("--decks", type=int, default=120)
    deck_search_parser.add_argument("--games-per-deck", type=int, default=20)
    deck_search_parser.add_argument("--seed", type=int, default=20261022)
    deck_search_parser.add_argument("--simulations", type=int, default=32)
    deck_search_parser.add_argument("--time-limit", type=float, default=1.0)
    deck_search_parser.add_argument("--max-minutes", type=float, default=240)
    deck_search_parser.add_argument("--resume", action="store_true")
    deck_search_parser.add_argument("--no-monitor", action="store_true")
    deck_search_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/deck_search/main"),
    )
    deck_monitor_parser = subparsers.add_parser(
        "deck-search-monitor",
        help="open a live deck-search progress and leaderboard window",
    )
    deck_monitor_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/deck_search/main"),
    )
    calibration_parser = subparsers.add_parser(
        "calibrate-vision",
        help="open the high-precision ROI calibration tool",
    )
    calibration_parser.add_argument("--image", type=Path, default=None)
    calibration_parser.add_argument(
        "--profile",
        type=Path,
        default=Path("profiles/vision/splat3-final-1920x1080.json"),
    )
    board_parser = subparsers.add_parser(
        "inspect-board",
        help="extract and classify a calibrated board grid",
    )
    board_parser.add_argument("--image", type=Path, required=True)
    board_parser.add_argument(
        "--profile",
        type=Path,
        default=Path("profiles/vision/对战场地-正直大道.json"),
    )
    board_parser.add_argument("--stage", default="Rectangle")
    board_parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    command_args = list(sys.argv[1:] if argv is None else argv)
    if not command_args:
        from tableturf_rebuild.vision.calibration_ui import run_calibration_ui

        project_root = Path(__file__).resolve().parents[2]
        image = (
            project_root
            / "artifacts"
            / "state_capture"
            / "run-20261021-221749"
            / "session-20261021-222704_annotations"
            / "008_video_state.png"
        )
        profile = project_root / "profiles" / "vision" / "battle_select.json"
        return run_calibration_ui(
            image if image.exists() else None,
            profile,
        )
    args = _build_parser().parse_args(command_args)
    if args.command == "controller-debug":
        from tableturf_rebuild.control.debug_ui import run_debug_ui

        return run_debug_ui()
    if args.command == "capture-debug":
        from tableturf_rebuild.capture.debug_ui import run_capture_debug

        return run_capture_debug()
    if args.command == "task-terminal":
        from tableturf_rebuild.application.task_terminal_ui import (
            run_task_terminal,
        )

        return run_task_terminal()
    if args.command == "video-review":
        from tableturf_rebuild.capture.video_review_ui import run_video_review

        return run_video_review(args.video)
    if args.command == "vision-live":
        from tableturf_rebuild.application.battle_vision import (
            BattleVisionRuntime,
        )
        from tableturf_rebuild.capture.live_vision import run_live_vision
        from tableturf_rebuild.capture.models import CaptureConfig

        runtime = BattleVisionRuntime.default()
        runtime.configure_stage(args.stage)
        previous_state = None
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        output_file = (
            args.output.open("a", encoding="utf-8", newline="\n")
            if args.output is not None
            else None
        )

        def report(record) -> None:
            nonlocal previous_state
            if (
                record.state != previous_state
                or args.log_every > 0
                and record.sequence % args.log_every == 0
            ):
                print(
                    f"frame={record.sequence} state={record.state} "
                    f"confidence={record.confidence:.3f} "
                    f"self={record.self_cells} "
                    f"opponent={record.opponent_cells} "
                    f"self_special={record.self_special_cells} "
                    f"opponent_special={record.opponent_special_cells} "
                    f"self_active_special={record.self_active_special_cells} "
                    f"opponent_active_special="
                    f"{record.opponent_active_special_cells} "
                    f"preview={record.placement_preview_present} "
                    f"selection={record.selection_variant} "
                    f"result={record.match_result} "
                    f"npc={record.npc_name} "
                    f"wins={record.level_wins} "
                    f"difficulty={record.difficulty} "
                    f"stage={record.stage_name} "
                    f"ai={record.ai_type} "
                    f"deck={record.deck_id} "
                    f"deck_source={record.deck_source} "
                    f"deck_area={record.deck_total_area} "
                    f"hand={record.hand_card_ids} "
                    f"reveal_self={record.self_reveal_card}/"
                    f"{record.self_reveal_area}/{record.self_reveal_sp_cost} "
                    f"reveal_opponent={record.opponent_reveal_card}/"
                    f"{record.opponent_reveal_area}/"
                    f"{record.opponent_reveal_sp_cost}"
                )
            previous_state = record.state
            if output_file is not None:
                output_file.write(
                    json.dumps(asdict(record), ensure_ascii=False) + "\n"
                )
                output_file.flush()

        try:
            summary = run_live_vision(
                CaptureConfig(
                    device_index=args.device_index,
                    backend=args.backend,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                ),
                runtime,
                max_frames=args.max_frames,
                max_seconds=args.max_seconds,
                on_record=report,
            )
        except KeyboardInterrupt:
            return 130
        finally:
            if output_file is not None:
                output_file.close()
        print(
            f"frames={summary.frames} "
            f"elapsed={summary.elapsed_seconds:.2f}s "
            f"stop={summary.stopped_reason}"
        )
        return 0
    if args.command == "deck-search-monitor":
        from tableturf_rebuild.application.deck_search_monitor import (
            run_deck_search_monitor,
        )

        return run_deck_search_monitor(args.output)
    if args.command == "calibrate-vision":
        from tableturf_rebuild.vision.calibration_ui import run_calibration_ui

        return run_calibration_ui(args.image, args.profile)
    if args.command == "deck-search":
        if not args.no_monitor:
            _launch_deck_monitor(args.output)
        catalog = GameCatalog.from_package()
        config = DeckSearchConfig(
            candidate_count=args.decks,
            games_per_deck=args.games_per_deck,
            seed=args.seed,
            simulations=args.simulations,
            time_limit_seconds=args.time_limit,
        )
        scenarios = select_level3_scenarios(catalog, args.games_per_deck)
        runner = DeckSearchRunner(catalog)
        result = runner.run(
            output_dir=args.output,
            config=config,
            scenarios=scenarios,
            resume=args.resume,
            max_seconds=(
                args.max_minutes * 60
                if args.max_minutes is not None
                else None
            ),
        )
        print(
            f"games={result.games_run} decks_completed={result.decks_completed}/"
            f"{args.decks} stop={result.stopped_reason} "
            f"elapsed={result.elapsed_seconds:.1f}s"
        )
        for rank, item in enumerate(result.leaderboard[:5], start=1):
            print(
                f"{rank}. {item['candidate_id']} "
                f"win={item['win_rate']:.1%} "
                f"margin={item['average_score_margin']:.2f} "
                f"games={item['games']}"
            )
        return 130 if result.stopped_reason == "interrupted" else 0
    catalog = GameCatalog.from_package()

    if args.command == "inspect-board":
        import cv2

        from tableturf_rebuild.vision.board_reader import BoardGridReader

        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read image: {args.image}")
        reader, profile, region, reference = BoardGridReader.from_profile(
            catalog,
            args.profile,
            args.stage,
        )
        observation = reader.read(
            image,
            region,
            args.stage,
            reference,
            profile,
        )
        symbols = {
            Cell.EMPTY: ".",
            Cell.SELF_INK: "a",
            Cell.SELF_SPECIAL: "A",
            Cell.OPPONENT_INK: "b",
            Cell.OPPONENT_SPECIAL: "B",
            Cell.VOID: " ",
            Cell.WALL: "#",
        }
        print(
            f"stage={observation.stage_id} self={observation.self_count} "
            f"opponent={observation.opponent_count} "
            f"self_special={observation.self_special_count} "
            f"opponent_special={observation.opponent_special_count} "
            f"self_active_special={observation.self_active_special_count} "
            f"opponent_active_special={observation.opponent_active_special_count} "
            f"confidence={observation.confidence:.3f} "
            f"preview={observation.placement_preview_present} "
            f"preview_score={observation.placement_preview_score:.3f} "
            f"initial={observation.initial_board_ready}"
        )
        for row in observation.cells:
            print("".join(symbols[cell] for cell in row))
        if args.output is not None:
            payload = {
                "stage_id": observation.stage_id,
                "rows": observation.rows,
                "cols": observation.cols,
                "confidence": observation.confidence,
                "placement_preview_score": observation.placement_preview_score,
                "placement_preview_present": (
                    observation.placement_preview_present
                ),
                "playable_cells": observation.playable_cells,
                "empty_cells": observation.empty_count,
                "self_special_count": observation.self_special_count,
                "opponent_special_count": observation.opponent_special_count,
                "self_active_special_count": (
                    observation.self_active_special_count
                ),
                "opponent_active_special_count": (
                    observation.opponent_active_special_count
                ),
                "initial_board_ready": observation.initial_board_ready,
                "cells": [
                    [cell.name for cell in row] for row in observation.cells
                ],
                "readings": [
                    asdict(reading) for reading in observation.readings
                ],
                "profile": str(profile.profile_id),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return 0

    if args.command == "validate-data":
        print(
            f"cards={len(catalog.cards)} decks={len(catalog.decks)} "
            f"npcs={len(catalog.npcs)} stages={len(catalog.stages)}"
        )
        return 0

    if args.command == "describe-card":
        card = catalog.cards[args.card_id]
        print(
            f"{card.card_id}: {card.name} / {card.name_zh} "
            f"area={card.area} sp={card.sp_cost}"
        )
        for row in card.shape.cells:
            print("".join({0: ".", 1: "#", 2: "*"}[int(cell)] for cell in row))
        return 0

    if args.command == "describe-stage":
        stage = catalog.stages[args.stage_id]
        print(
            f"{stage.stage_id}: {stage.name_zh} "
            f"size={stage.rows}x{stage.cols} playable={stage.playable_count}"
        )
        symbols = {
            Cell.EMPTY: ".",
            Cell.SELF_INK: "a",
            Cell.SELF_SPECIAL: "A",
            Cell.OPPONENT_INK: "b",
            Cell.OPPONENT_SPECIAL: "B",
            Cell.VOID: " ",
            Cell.WALL: "#",
        }
        for row in stage.cells:
            print("".join(symbols[cell] for cell in row))
        return 0

    if args.command == "selfplay":
        scenarios = catalog.npc_scenarios()
        if args.level is not None:
            scenarios = tuple(
                scenario for scenario in scenarios if scenario.level == args.level
            )
        policy_a = _build_policy(
            args.policy_a,
            args.seed,
            f"{args.policy_a}-a",
            args.simulations,
            args.time_limit,
        )
        policy_b = _build_policy(
            args.policy_b,
            args.seed ^ 0xBEEF,
            f"{args.policy_b}-b",
            args.simulations,
            args.time_limit,
        )
        runner = SelfPlayRunner(catalog)
        batch_runner = BatchSelfPlayRunner(runner)
        result = batch_runner.run(
            output=args.output,
            games=args.games,
            scenarios=scenarios,
            seed=args.seed,
            policy_a=policy_a,
            policy_b=policy_b,
            metadata={
                "seed": args.seed,
                "policy_a": args.policy_a,
                "policy_b": args.policy_b,
                "level": args.level,
                "simulations": args.simulations,
                "time_limit_seconds": args.time_limit,
            },
            max_seconds=(
                args.max_minutes * 60
                if args.max_minutes is not None
                else None
            ),
            resume=args.resume,
        )
        summary = result.summary["overall"]
        print(
            f"games={summary['games']} A={summary['wins_a']} "
            f"B={summary['wins_b']} draw={summary['draws']} avg_score="
            f"{summary['average_score_a']:.2f}:"
            f"{summary['average_score_b']:.2f} "
            f"run={result.games_run} stop={result.stopped_reason} "
            f"elapsed={result.elapsed_seconds:.3f}s output={args.output}"
        )
        print(f"summary={args.output.resolve().with_name(args.output.name + '.summary.json')}")
        return 130 if result.stopped_reason == "interrupted" else 0

    raise AssertionError(f"unhandled command: {args.command}")


def _build_policy(
    name: str,
    seed: int,
    label: str,
    simulations: int,
    time_limit: float,
):
    if name == "random":
        return RandomPolicy(seed=seed, name=label)
    if name == "greedy":
        return GreedyPolicy(name=label)
    if name == "mcts":
        return MCTSDecisionPolicy(
            config=MCTSConfig(
                simulations=simulations,
                time_limit_seconds=time_limit,
            ),
            seed=seed,
            name=label,
        )
    raise ValueError(f"unknown policy: {name}")


def _launch_deck_monitor(output_dir: Path) -> None:
    python = Path(sys.executable)
    pythonw = python.with_name("pythonw.exe")
    executable = pythonw if pythonw.exists() else python
    command = [
        str(executable),
        "-m",
        "tableturf_rebuild",
        "deck-search-monitor",
        "--output",
        str(output_dir.resolve()),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW
    try:
        subprocess.Popen(command, creationflags=creationflags, close_fds=True)
    except OSError:
        pass

