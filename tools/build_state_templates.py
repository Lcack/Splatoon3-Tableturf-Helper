"""Build compact state templates from the two annotated capture sessions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    PROJECT_ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-110258": {
        1: "NPC_ROSTER",
        2: "MATCH_SETUP",
        3: "MATCH_SETUP",
        4: "DECK_SELECT",
        5: "REDRAW",
        6: "TURN_SELECT",
        7: "TARGET_PLACEMENT",
        8: "TARGET_PLACEMENT",
        9: "TURN_REVEAL",
        10: "SKIP_DISCARD",
        11: "SPECIAL_SELECT",
        12: "SPECIAL_PLACEMENT",
        13: "REPLAY_CHOICE",
    },
    PROJECT_ROOT
    / "artifacts"
    / "state_capture"
    / "run-20261021-221749"
    / "session-20261021-222704_annotations": {
        1: "NPC_ROSTER",
        2: "NPC_DIALOGUE",
        3: "NPC_DIALOGUE",
        4: "NPC_ROSTER",
        5: "NPC_ROSTER",
        6: "NPC_DIALOGUE",
        7: "MATCH_SETUP",
        8: "MATCH_SETUP",
        9: "MATCH_SETUP",
        10: "DECK_SELECT",
        11: "DECK_SELECT",
        12: "DECK_SELECT",
        13: "DECK_SELECT",
        14: "BATTLE_INTRO",
        15: "BATTLE_INTRO",
        16: "REDRAW",
        17: "TARGET_PLACEMENT",
        18: "TURN_REVEAL",
        19: "TURN_RESOLVE",
        20: "TURN_RESOLVE",
        21: "TURN_RESOLVE",
        22: "TURN_RESOLVE",
        23: "SKIP_DISCARD",
        24: "SPECIAL_SELECT",
        25: "SKIP_DISCARD",
        26: "SPECIAL_SELECT",
        27: "SPECIAL_PLACEMENT",
        28: "FINISHING",
        29: "FINISHING",
        30: "MATCH_RESULT",
        31: "MATCH_REWARD",
        32: "REPLAY_CHOICE",
        33: "REDRAW",
        34: "TURN_RESOLVE",
        35: "TURN_RESOLVE",
        36: "TURN_RESOLVE",
        37: "FINISHING",
        38: "MATCH_RESULT",
        39: "REPLAY_CHOICE",
        40: "MATCH_SETUP",
        41: "NPC_ROSTER",
    },
}


def source_image(directory: Path, sequence: int) -> Path:
    matches = sorted(directory.glob(f"{sequence:03d}_*.png"))
    matches = [path for path in matches if ".annotated." not in path.name]
    if len(matches) != 1:
        raise RuntimeError(
            f"{directory}: expected one image for sequence {sequence}, got {matches}"
        )
    return matches[0]


def main() -> int:
    output_dir = PROJECT_ROOT / "assets" / "state_templates"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source_dir, labels in SOURCES.items():
        for sequence, state in labels.items():
            source = source_image(source_dir, sequence)
            image = cv2.imread(str(source))
            if image is None:
                raise RuntimeError(f"cannot read {source}")
            thumbnail = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
            relative = Path(state) / f"{source_dir.name}-{sequence:03d}.jpg"
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 88])
            manifest.append(
                {
                    "state": state,
                    "image": relative.as_posix(),
                    "source": str(source.resolve()),
                    "sequence": sequence,
                }
            )
    with (output_dir / "manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"wrote {len(manifest)} state templates to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

