"""Evaluate the state machine over an annotated video."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tableturf_rebuild.application.state_machine import StateMachine
from tableturf_rebuild.capture.video_source import OpenCvVideoSource


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--stable-frames", type=int, default=2)
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="JSON object mapping frame indices to canonical state names",
    )
    args = parser.parse_args()

    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "state_templates"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_by_source = {
        str(Path(entry["source"]).resolve()): entry["state"]
        for entry in manifest
    }
    labels = {}
    if args.labels_file is not None:
        labels = {
            int(frame): state
            for frame, state in json.loads(
                args.labels_file.read_text(encoding="utf-8")
            ).items()
        }
    else:
        for path in sorted(args.annotations.glob("[0-9][0-9][0-9]_*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            source = str((args.annotations / value["original_image"]).resolve())
            if source not in state_by_source:
                continue
            labels[int(value["capture"]["frame_index"])] = state_by_source[source]

    source = OpenCvVideoSource(args.video)
    source.open()
    machine = StateMachine(stable_frames=args.stable_frames)
    max_frame = min(source.frame_count, max(labels) + 2 if labels else source.frame_count)
    predictions = {}
    frame_index = 0
    try:
        while frame_index < max_frame:
            frame = source.read()
            if frame is None:
                break
            observation = machine.update(frame)
            if frame_index in labels:
                predictions[frame_index] = observation.state.value
            frame_index += 1
    finally:
        source.close()

    correct = sum(predictions.get(index) == state for index, state in labels.items())
    confusion = Counter(
        (state, predictions.get(index, "MISSING"))
        for index, state in labels.items()
    )
    print(f"sequential accuracy: {correct}/{len(labels)} = {correct / len(labels):.3f}")
    for index, state in sorted(labels.items()):
        predicted = predictions.get(index, "MISSING")
        marker = "OK" if predicted == state else "MISS"
        print(f"{marker} frame={index:5d} expected={state:18s} predicted={predicted}")
    print("confusion:")
    for (expected, predicted), count in sorted(confusion.items()):
        if expected != predicted:
            print(f"{expected} -> {predicted}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

