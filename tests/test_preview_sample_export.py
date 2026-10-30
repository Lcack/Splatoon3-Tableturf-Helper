import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_preview_samples", ROOT / "tools/export_preview_samples.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def saved_sample(tmp_path, *, frozen=True):
    folder = tmp_path / "sample"
    folder.mkdir()
    image = np.zeros((80, 80, 3), np.uint8)
    image[0:40, 0:40] = (20, 120, 220)
    exporter.save_png(folder / "analysis.png", image)
    value = {"stage_id": "probe", "observation": {"board": {"rows": 2, "cols": 2}},
             "executor": {"selection_board": [[1, 2], [4, 8]] if frozen else None,
                          "visual_shapes": {"probe": [[2], [1]]},
                          "preview_attempts": [{"source": "native_grid", "rows": 2, "cols": 2,
                                                "region": [[0, 0], [1, 1]],
                                                "candidate_cells": [[0, 0]], "dot_cells": [[1, 1]]}]}}
    (folder / "latest.json").write_text(json.dumps(value), encoding="utf-8")
    return folder


def test_predictions_and_explicit_geometry_do_not_become_human_truth(tmp_path):
    folder = saved_sample(tmp_path)
    reference = {"card_id": "probe", "origin": [0, 1], "rotation": 2}
    sample, _ = exporter.load_sample(folder, reference)
    entries = {(item["row"], item["col"]): item for item in sample["entries"]}
    assert entries[(0, 1)]["explicit_reference"]["class"] == "stripe"
    assert entries[(1, 1)]["explicit_reference"]["class"] == "dot"
    assert all(item["human_label"] is None for item in entries.values())
    assert all(not item["background_prediction"]["human_confirmed"] for item in entries.values())
    assert all(item["human_confirmed_count"] == 0 for item in exporter.coverage([sample]))


def test_frozen_background_is_required_and_current_preview_is_not_fallback(tmp_path):
    folder = saved_sample(tmp_path, frozen=False)
    sample, _ = exporter.load_sample(folder)
    assert all(item["background_prediction"]["class"] == "unknown" for item in sample["entries"])


def test_human_labels_require_deliberate_confirmation_and_reviewer(tmp_path):
    folder = saved_sample(tmp_path)
    label = {"row": 0, "col": 0, "background": "empty", "preview": "stripe", "confirmed": True}
    with pytest.raises(ValueError, match="reviewer"):
        exporter.load_sample(folder, human_labels=[label])
    label["reviewer"] = "test-reviewer"
    sample, _ = exporter.load_sample(folder, human_labels=[label])
    assert sample["entries"][0]["human_label"] == label
    assert sum(item["human_confirmed_count"] for item in exporter.coverage([sample])) == 1


def test_export_keeps_raw_pixels_and_refuses_silent_metadata_overwrite(tmp_path):
    folder = saved_sample(tmp_path)
    output = tmp_path / "output"
    result = exporter.export_samples([folder], output, make_contact=False)
    first = result["samples"][0]["entries"][0]
    patch = cv2.imdecode(np.frombuffer((output / first["full_image"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert np.all(patch == (20, 120, 220))
    assert (output / "metadata.json").is_file()
    assert (output / "说明.md").is_file()
    assert (output / result["samples"][0]["archived_analysis"]).read_bytes() == (folder / "analysis.png").read_bytes()
    assert (output / result["samples"][0]["archived_snapshot"]).read_bytes() == (folder / "latest.json").read_bytes()
    with pytest.raises(ValueError, match="覆盖人工复核"):
        exporter.export_samples([folder], output, make_contact=False)


def test_long_card_reference_keeps_actual_dot_at_other_end():
    sample, _ = exporter.load_sample(
        ROOT / "artifacts/diagnostics/long-card-stall",
        {"card_id": "MiniGame_SaberLight00", "origin": [11, 3], "rotation": 2,
         "shape": [[2], [1], [1], [1], [1]]},
    )
    references = {(item["row"], item["col"]): item["explicit_reference"]["class"]
                  for item in sample["entries"] if item["explicit_reference"]}
    assert references == {(11, 3): "stripe", (12, 3): "stripe", (13, 3): "stripe",
                          (14, 3): "stripe", (15, 3): "dot"}
    assert not any(item["human_label"] for item in sample["entries"])
