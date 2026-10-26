"""Search a small ROI that separates NPC_ROSTER from MATCH_SETUP."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import cv2
import numpy as np


def normalize(value: np.ndarray) -> np.ndarray:
    flat = value.reshape(-1).astype(np.float32)
    flat -= flat.mean()
    length = np.linalg.norm(flat)
    return flat / length if length else flat


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "assets" / "state_templates"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for entry in manifest:
        if entry["state"] not in {"NPC_ROSTER", "MATCH_SETUP"}:
            continue
        image = cv2.imread(str(root / entry["image"]))
        gray = cv2.cvtColor(
            cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        records.append((entry["state"], gray))

    best = []
    xs = (0, 20, 40, 80, 120, 160, 200, 240, 280)
    ys = (0, 20, 40, 60, 80, 100, 120, 140)
    widths = (40, 60, 80, 100, 120, 160)
    heights = (40, 60, 80, 100)
    for x, y, width, height in itertools.product(xs, ys, widths, heights):
        if x + width > 320 or y + height > 180:
            continue
        vectors = [
            normalize(
                cv2.resize(
                    gray[y : y + height, x : x + width],
                    (40, 24),
                    interpolation=cv2.INTER_AREA,
                )
            )
            for _, gray in records
        ]
        correct = 0
        for index, (state, _) in enumerate(records):
            scores = []
            for other, vector in enumerate(vectors):
                if other == index:
                    continue
                scores.append((float(np.dot(vectors[index], vector)), records[other][0]))
            prediction = max(scores)[1]
            correct += prediction == state
        best.append(
            (
                correct,
                x / 320,
                y / 180,
                (x + width) / 320,
                (y + height) / 180,
            )
        )
    for item in sorted(best, reverse=True)[:20]:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

