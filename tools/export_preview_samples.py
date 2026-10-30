"""Export saved preview cells for visual review; never label predictions as truth.

No capture device or controller is opened. All inputs are saved local files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tableturf_rebuild.domain.card import CardShape


BACKGROUND_NAMES = {
    "empty": "空格", "self_ink": "我方普通墨水", "self_special": "我方特殊格",
    "opponent_ink": "敌方普通墨水", "opponent_special": "敌方特殊格", "outside": "场外/墙障",
    "unknown": "底图未知",
}
CELL_BACKGROUNDS = {1: "empty", 2: "self_ink", 4: "self_special", 8: "opponent_ink",
                    16: "opponent_special", 32: "outside", 64: "outside"}
PREVIEW_NAMES = {"stripe": "斜线", "dot": "点阵", "both": "斜线及点阵", "none": "未检出"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def reference_cells(snapshot, reference):
    """Explicit geometry is a review reference, not a confirmed human label."""
    if not reference:
        return {}
    for key in ("card_id", "origin", "rotation"):
        if key not in reference:
            raise ValueError(f"参考标注缺少 {key}")
    card_id = reference["card_id"]
    shape = reference.get("shape")
    if shape is None:
        shape = ((snapshot.get("executor") or {}).get("visual_shapes") or {}).get(card_id)
    if shape is None:
        from tableturf_rebuild.data.catalog import GameCatalog
        shape = GameCatalog.from_package().cards[card_id].shape.cells
    origin = reference["origin"]
    if len(origin) != 2 or reference["rotation"] not in range(4):
        raise ValueError("origin 必须是行列二元组，rotation 必须为 0 至 3")
    rotated = CardShape.from_matrix(shape).rotated(reference["rotation"])
    return {(int(origin[0]) + row, int(origin[1]) + col):
            "dot" if int(rotated.cells[row][col]) == 2 else "stripe"
            for row, col in rotated.occupied}


def load_sample(path, reference=None, human_labels=()):
    path = Path(path).resolve()
    snapshot_path = (path if path.is_file() else next(
        (path / name for name in ("latest.json", "snapshot.json", "source_snapshot.json") if (path / name).exists()),
        path / "latest.json"))
    snapshot = read_json(snapshot_path)
    folder = snapshot_path.parent
    # The analysis image belongs to these predictions. capture.jpg can be newer.
    image_path = folder / "analysis.png"
    if not image_path.exists():
        image_path = folder / "source_analysis.png"
    if not image_path.exists():
        raise ValueError(f"缺少与识别结果对应的 analysis.png：{folder}")
    payload = image_path.read_bytes()
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"图片读取失败：{image_path}")
    executor = snapshot.get("executor") or {}
    observation = snapshot.get("observation") or {}
    board = observation.get("board") or {}
    attempts = executor.get("preview_attempts") or []
    attempt = next((item for item in attempts if item.get("source") == "native_grid"),
                   attempts[0] if attempts else {})
    points = attempt.get("region") or snapshot.get("board_region")
    rows, cols = attempt.get("rows", board.get("rows")), attempt.get("cols", board.get("cols"))
    if not points or len(points) not in {2, 4} or not rows or not cols:
        raise ValueError(f"快照缺少网格坐标/行列：{snapshot_path}")
    stripes = {tuple(cell) for cell in attempt.get("candidate_cells", [])}
    dots = {tuple(cell) for cell in attempt.get("dot_cells", [])}
    refs = reference_cells(snapshot, reference)
    labels = {}
    for label in human_labels:
        if (label.get("confirmed") is not True or not isinstance(label.get("reviewer"), str)
                or not label["reviewer"].strip()):
            raise ValueError("人工标签必须显式 confirmed=true，且 reviewer 非空")
        if (label.get("background") not in set(BACKGROUND_NAMES) - {"unknown"}
                or label.get("preview") not in {"stripe", "dot", "none"}):
            raise ValueError("人工标签 background/preview 类别无效")
        cell = (int(label["row"]), int(label["col"]))
        if cell in labels:
            raise ValueError(f"人工标签包含重复格子：{cell}")
        labels[cell] = dict(label)
    frozen = executor.get("selection_board")
    cells = sorted(stripes | dots | set(refs) | set(labels))
    scores = {tuple(item["cell"]): item.get("score") for item in attempt.get("strongest_cells", [])}
    entries = []
    for row, col in cells:
        if not 0 <= row < rows or not 0 <= col < cols:
            background, code = "outside", None
        elif frozen is not None and row < len(frozen) and col < len(frozen[row]):
            code = int(frozen[row][col])
            background = CELL_BACKGROUNDS.get(code, "unknown")
        else:
            code, background = None, "unknown"
        predicted = ("both" if (row, col) in stripes and (row, col) in dots
                     else "stripe" if (row, col) in stripes else "dot" if (row, col) in dots else "none")
        entries.append({
            "sample": folder.name, "row": row, "col": col,
            "background_prediction": {"class": background, "name_zh": BACKGROUND_NAMES[background],
                                      "cell_code": code,
                                      "source": "executor.selection_board" if frozen is not None else "missing_frozen_board",
                                      "human_confirmed": False},
            "preview_prediction": {"class": predicted, "name_zh": PREVIEW_NAMES[predicted],
                                   "diagonal_score": scores.get((row, col)),
                                   "source": "executor.preview_attempts", "human_confirmed": False},
            "explicit_reference": ({"class": refs[(row, col)], "name_zh": PREVIEW_NAMES[refs[(row, col)]],
                                    "source": "explicit_card_origin_rotation", "human_confirmed": False}
                                   if (row, col) in refs else None),
            "human_label": labels.get((row, col)),
        })
    return {
        "sample": folder.name, "snapshot": str(snapshot_path), "image": str(image_path),
        "image_sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot_updated_at": snapshot.get("updated_at"), "analysis": snapshot.get("analysis"),
        "stage_id": snapshot.get("stage_id"), "reference_geometry": reference,
        "source_preview_prediction": executor.get("preview"),
        "source_planned_action": executor.get("action"),
        "geometry": {"points": points, "rows": rows, "cols": cols,
                     "source": "executor.preview_attempts.region" if attempt.get("region") else "board_region"},
        "background_source": "selection_frozen_prediction" if frozen is not None else "missing",
        "entries": entries,
    }, frame


def cell_crop(frame, geometry, row, col, *, inset=0):
    points = np.asarray(geometry["points"], np.float32) * (frame.shape[1] - 1, frame.shape[0] - 1)
    rows, cols = geometry["rows"], geometry["cols"]
    if len(points) == 2:
        dx, dy = (points[1] - points[0]) / (cols, rows)
        x1, y1 = np.round(points[0] + (col * dx + inset, row * dy + inset)).astype(int)
        x2, y2 = np.round(points[0] + ((col + 1) * dx - inset, (row + 1) * dy - inset)).astype(int)
        bounds = [int(x1), int(y1), int(x2), int(y2)]
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        patch = frame[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
        return patch, {"method": "native_rect", "unclipped_pixel_bounds": bounds,
                       "clipped_to_frame": bounds != [x1, y1, x2, y2]}
    target = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], np.float32)
    logical_to_image = cv2.getPerspectiveTransform(target, points.astype(np.float32))
    corners = np.array([[[col, row], [col + 1, row], [col + 1, row + 1], [col, row + 1]]], np.float32)
    corners = cv2.perspectiveTransform(corners, logical_to_image)[0]
    size = max(8, round(float(np.mean(np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)))))
    destination = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    patch = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(corners, destination), (size, size))
    margin = round(inset)
    if margin:
        patch = patch[margin:-margin, margin:-margin]
    return patch, {"method": "perspective_quad", "source_pixel_corners": corners.tolist()}


def save_png(path, frame):
    ok, data = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError(f"图片编码失败：{path}")
    path.write_bytes(data.tobytes())


def coverage(samples):
    predicted, reference, confirmed = Counter(), Counter(), Counter()
    for sample in samples:
        for entry in sample["entries"]:
            bg = entry["background_prediction"]["class"]
            predicted[(bg, entry["preview_prediction"]["class"])] += 1
            if entry["explicit_reference"]:
                reference[(bg, entry["explicit_reference"]["class"])] += 1
            human = entry["human_label"]
            if human:
                confirmed[(human["background"], human["preview"])] += 1
    combinations = [(bg, preview) for bg in BACKGROUND_NAMES if bg != "unknown" for preview in ("stripe", "dot")]
    return [{"background": bg, "background_zh": BACKGROUND_NAMES[bg],
             "preview": preview, "preview_zh": PREVIEW_NAMES[preview],
             "predicted_count": predicted[(bg, preview)], "reference_count": reference[(bg, preview)],
             "human_confirmed_count": confirmed[(bg, preview)],
             "reference_missing": reference[(bg, preview)] == 0,
             "human_confirmation_missing": confirmed[(bg, preview)] == 0}
            for bg, preview in combinations]


def contact_sheet(output, samples):
    from PIL import Image, ImageDraw, ImageFont
    font_path = next((p for p in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"),
                                 Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")) if p.exists()), None)
    if font_path is None:
        return {"created": False, "reason": "未找到可靠中文字体，已保留原始小图与中文说明"}
    font = ImageFont.truetype(str(font_path), 15)
    title_font = ImageFont.truetype(str(font_path), 23)
    entries = [(sample, entry) for sample in samples for entry in sample["entries"]]
    if not entries:
        return {"created": False, "reason": "没有可导出的候选或参考格"}
    columns, card_w, card_h = 5, 240, 270
    sheet = Image.new("RGB", (columns * card_w, 92 + ((len(entries) + columns - 1) // columns) * card_h), "#f1f3f5")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 9), "预览逐格复核联系表", font=title_font, fill="#182430")
    draw.text((14, 43), "底色与识别预测不是训练真值；指定参考也须人工核验。", font=font, fill="#a52b30")
    for index, (sample, entry) in enumerate(entries):
        x, y = (index % columns) * card_w, 92 + (index // columns) * card_h
        draw.rectangle((x + 5, y + 5, x + card_w - 5, y + card_h - 5), fill="white", outline="#ccd2d8")
        name = sample["sample"].replace("long-card-stall", "长条牌卡住").replace("two-cell-preview", "两格牌黄色底").replace("two-cell-rotation", "两格牌旋转")
        draw.text((x + 12, y + 10), f"{name} [{entry['row']},{entry['col']}]", font=font, fill="#182430")
        if entry.get("inset_image"):
            patch = Image.open(output / entry["inset_image"]).convert("RGB")
            sheet.paste(patch.resize((128, 128), Image.Resampling.NEAREST), (x + 56, y + 37))
        else:
            draw.text((x + 20, y + 75), "无画面像素", font=font, fill="#a52b30")
        draw.text((x + 12, y + 174), "底图预测：" + entry["background_prediction"]["name_zh"], font=font, fill="#36444f")
        draw.text((x + 12, y + 195), "识别预测：" + entry["preview_prediction"]["name_zh"], font=font, fill="#36444f")
        ref = entry["explicit_reference"]
        draw.text((x + 12, y + 216), "指定参考：" + (ref["name_zh"] if ref else "无"), font=font, fill="#315b91")
        human = entry["human_label"]
        draw.text((x + 12, y + 239), "人工确认：" + ("已确认" if human else "未确认"), font=font, fill="#26733d" if human else "#a52b30")
    sheet.save(output / "预览复核联系表.png")
    return {"created": True, "image": "预览复核联系表.png", "font": str(font_path)}


def export_samples(paths, output, references=None, human_labels=None, *, overwrite=False, make_contact=True):
    output = Path(output).resolve()
    if (output / "metadata.json").exists() and not overwrite:
        raise ValueError("输出已存在 metadata.json；请另选目录，或显式 --overwrite，避免覆盖人工复核")
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, path in enumerate(paths):
        name = Path(path).parent.name if Path(path).is_file() else Path(path).name
        sample, frame = load_sample(path, (references or {}).get(name), (human_labels or {}).get(name, ()))
        slug = re.sub(r"[^\w-]", "_", sample["sample"])
        destination = output / f"{index:02d}_{slug}"
        destination.mkdir(exist_ok=True)
        archived_image = destination / "source_analysis.png"
        archived_snapshot = destination / "source_snapshot.json"
        archived_image.write_bytes(Path(sample["image"]).read_bytes())
        archived_snapshot.write_bytes(Path(sample["snapshot"]).read_bytes())
        sample["archived_analysis"] = archived_image.relative_to(output).as_posix()
        sample["archived_snapshot"] = archived_snapshot.relative_to(output).as_posix()
        for entry in sample["entries"]:
            stem = f"r{entry['row']}_c{entry['col']}"
            for key, inset in (("full_image", 0), ("inset_image", 3 * frame.shape[0] / 1080)):
                patch, crop = cell_crop(frame, sample["geometry"], entry["row"], entry["col"], inset=inset)
                entry[key.replace("image", "crop")] = crop
                entry[key] = None
                if patch is not None and patch.size:
                    image_path = destination / f"{stem}_{key}.png"
                    save_png(image_path, patch)
                    entry[key] = image_path.relative_to(output).as_posix()
        samples.append(sample)
    result = {"schema_version": 1, "purpose": "保存原始预览格及可复核预测；不训练、不自动生成真值", "samples": samples,
              "coverage": coverage(samples), "contact_sheet": contact_sheet(output, samples) if make_contact else {"created": False}}
    (output / "metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 预览背景复核样本", "", "所有小图来自保存的 analysis.png，不读取采集卡，不发送控制器按键。",
             "每组同时归档 source_analysis.png 和 source_snapshot.json；即使原采集目录移动，仍可回放。",
             "", "`background_prediction` 取自选牌时冻结的识别棋盘，仍是自动预测；缺少冻结底图时记为未知，不用预览画面反推底色。",
             "`preview_prediction` 保留保存快照当时的斜线/点阵预测，包括误检和漏检。",
             "`explicit_reference` 来自显式指定的卡牌、起点和旋转，只是复核参考；不会自动写入人工标签。",
             "只有 `human_label` 中显式 confirmed=true 且 reviewer 非空的记录才计入人工确认覆盖。",
             "格图分完整原生格和去边框小图；quad 标定才作透视采样，metadata 保留采样方法和坐标。",
             "场外与墙障在六类背景表合并展示，原始 cell_code 仍保留，可再次细分。", "",
             "## 当前样本覆盖（底色仍是预测）", "", "| 背景 | 预览 | 识别预测格数 | 指定参考格数 | 人工确认格数 |", "|---|---|---:|---:|---:|"]
    for item in result["coverage"]:
        lines.append(f"| {item['background_zh']} | {item['preview_zh']} | {item['predicted_count']} | {item['reference_count']} | {item['human_confirmed_count']} |")
    missing = [f"{item['background_zh']}＋{item['preview_zh']}" for item in result["coverage"] if item["reference_missing"]]
    lines += ["", "当前导出尚无指定参考的组合：" + "、".join(missing) + "。",
              "这只统计当前保存样本，不表示游戏不会出现这些组合；自动误检不计为已采到的参考组合。",
              "所有未人工确认的记录均不得直接当训练真值。", "",
              "## 人工确认文件格式", "", "使用 --human-labels 指定 JSON；键为采集目录名。保留原预测，并只添加人工标签。", "", "```json",
              json.dumps({samples[0]["sample"] if samples else "采集目录名": [{"row": 0, "col": 0, "background": "empty", "preview": "stripe", "confirmed": True, "reviewer": "填写实际复核者"}]}, ensure_ascii=False, indent=2),
              "```", "", "此格式示例不是本次采集的标签。背景键见 metadata.coverage；预览键使用 stripe/dot/none。"]
    (output / "说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="导出保存快照中的预览格；自动预测与人工标签严格分开")
    parser.add_argument("samples", nargs="+", type=Path, help="包含 latest.json 和 analysis.png 的目录，或快照 JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--references", type=Path, help="按采集目录名给出 card_id/origin/rotation/可选shape 的参考 JSON")
    parser.add_argument("--human-labels", type=Path, help="显式人工确认文件；不会由识别结果自动填入")
    parser.add_argument("--card-id")
    parser.add_argument("--origin", nargs=2, type=int, metavar=("ROW", "COL"))
    parser.add_argument("--rotation", type=int, choices=range(4))
    parser.add_argument("--shape", help="显式二维形状 JSON，例如 [[2],[1],[1],[1],[1]]")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-contact-sheet", action="store_true")
    args = parser.parse_args(argv)
    references = read_json(args.references) if args.references else {}
    if any(value is not None for value in (args.card_id, args.origin, args.rotation, args.shape)):
        if len(args.samples) != 1 or args.card_id is None or args.origin is None or args.rotation is None:
            parser.error("单样本显式参考必须同时提供 --card-id、--origin、--rotation；多样本请用 --references")
        name = args.samples[0].parent.name if args.samples[0].is_file() else args.samples[0].name
        references[name] = {"card_id": args.card_id, "origin": args.origin, "rotation": args.rotation}
        if args.shape:
            references[name]["shape"] = json.loads(args.shape)
    try:
        result = export_samples(args.samples, args.output, references,
                                read_json(args.human_labels) if args.human_labels else None,
                                overwrite=args.overwrite, make_contact=not args.no_contact_sheet)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"已导出 {len(result['samples'])} 组、{sum(len(s['entries']) for s in result['samples'])} 格：{args.output.resolve()}")
    print("自动预测和指定参考均未冒充人工训练真值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
