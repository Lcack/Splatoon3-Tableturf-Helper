"""Import the most recent detailed battle-state annotations."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from tableturf_rebuild.application.state_machine import (
    GameState,
    save_manual_state_template,
)


LABELS = {
    1: GameState.NPC_ROSTER,
    2: GameState.MATCH_SETUP,
    3: GameState.MATCH_SETUP,
    4: GameState.TARGET_PLACEMENT,
    5: GameState.TARGET_PLACEMENT,
    6: GameState.TURN_REVEAL,
    7: GameState.TURN_REVEAL,
    8: GameState.TURN_SELECT,
    9: GameState.SKIP_DISCARD,
    10: GameState.SPECIAL_SELECT,
    11: GameState.MATCH_RESULT,
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    annotation_dir = (
        root
        / "artifacts"
        / "state_capture"
        / "run-20261021-221749"
        / "session-20261021-222704_annotations"
    )
    labels = {}
    for sequence, state in LABELS.items():
        json_path = annotation_dir / f"{sequence:03d}_video_state.json"
        value = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = annotation_dir / value["original_image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        save_manual_state_template(image, state)
        labels[int(value["capture"]["frame_index"])] = state.value
        print(f"imported frame={value['capture']['frame_index']} state={state.value}")
    with (annotation_dir / "detailed_state_labels.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(labels, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

