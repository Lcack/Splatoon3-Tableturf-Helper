"""Leave-one-template-out evaluation for the state recognizer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2

from tableturf_rebuild.application.state_machine import (
    TemplateStateRecognizer,
)


def main() -> int:
    recognizer = TemplateStateRecognizer.default()
    correct = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for record in recognizer.records:
        others = tuple(item for item in recognizer.records if item is not record)
        recognizer.records, original = others, recognizer.records
        image = cv2.imread(str(record.path))
        prediction = recognizer.classify(image)
        recognizer.records = original
        predicted = prediction.state.value
        actual = record.state.value
        confusion[(actual, predicted)] += 1
        if actual == predicted:
            correct += 1
        else:
            print(
                f"MISS {record.path.name}: {actual} -> {predicted} "
                f"({prediction.confidence:.3f})"
            )
    total = len(recognizer.records)
    print(f"leave-one-out accuracy: {correct}/{total} = {correct / total:.3f}")
    print("confusion:")
    for pair, count in sorted(confusion.items()):
        if count and pair[0] != pair[1]:
            print(pair, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

