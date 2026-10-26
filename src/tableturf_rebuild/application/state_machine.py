from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from tableturf_rebuild.vision.profile_state_verifier import (
    ProfileStateScore,
    VisionProfileStateVerifier,
)


class GameState(str, Enum):
    UNKNOWN = "UNKNOWN"
    NPC_ROSTER = "NPC_ROSTER"
    NPC_DIALOGUE = "NPC_DIALOGUE"
    MATCH_SETUP = "MATCH_SETUP"
    DECK_SELECT = "DECK_SELECT"
    BATTLE_INTRO = "BATTLE_INTRO"
    REDRAW = "REDRAW"
    TURN_SELECT = "TURN_SELECT"
    SKIP_DISCARD = "SKIP_DISCARD"
    SPECIAL_SELECT = "SPECIAL_SELECT"
    TARGET_PLACEMENT = "TARGET_PLACEMENT"
    SPECIAL_PLACEMENT = "SPECIAL_PLACEMENT"
    TURN_REVEAL = "TURN_REVEAL"
    TURN_RESOLVE = "TURN_RESOLVE"
    FINISHING = "FINISHING"
    MATCH_RESULT = "MATCH_RESULT"
    MATCH_REWARD = "MATCH_REWARD"
    REPLAY_CHOICE = "REPLAY_CHOICE"

STATE_LABELS = {
    GameState.UNKNOWN: "未知",
    GameState.NPC_ROSTER: "NPC选择",
    GameState.NPC_DIALOGUE: "NPC对白",
    GameState.MATCH_SETUP: "NPC难度与开始",
    GameState.DECK_SELECT: "卡组选择",
    GameState.BATTLE_INTRO: "对战开场动画",
    GameState.REDRAW: "重抽手牌",
    GameState.TURN_SELECT: "回合选牌",
    GameState.SKIP_DISCARD: "跳过并弃牌",
    GameState.SPECIAL_SELECT: "特殊攻击选牌",
    GameState.TARGET_PLACEMENT: "普通卡牌摆放",
    GameState.SPECIAL_PLACEMENT: "特殊攻击摆放",
    GameState.TURN_REVEAL: "双方亮牌",
    GameState.TURN_RESOLVE: "落墨与特殊点结算",
    GameState.FINISHING: "最终结算动画",
    GameState.MATCH_RESULT: "胜负结算",
    GameState.MATCH_REWARD: "经验与奖励",
    GameState.REPLAY_CHOICE: "是否继续对战",
}

STATE_DISPLAY_NAMES = {
    state: f"{state.value} | {label}" for state, label in STATE_LABELS.items()
}


TRANSITIONS = {
    GameState.UNKNOWN: set(GameState),
    GameState.NPC_ROSTER: {GameState.NPC_DIALOGUE, GameState.MATCH_SETUP},
    GameState.NPC_DIALOGUE: {GameState.NPC_DIALOGUE, GameState.MATCH_SETUP},
    GameState.MATCH_SETUP: {GameState.NPC_ROSTER, GameState.DECK_SELECT},
    GameState.DECK_SELECT: {GameState.MATCH_SETUP, GameState.BATTLE_INTRO, GameState.REDRAW},
    GameState.BATTLE_INTRO: {GameState.REDRAW},
    GameState.REDRAW: {GameState.TURN_SELECT},
    GameState.TURN_SELECT: {
        GameState.SKIP_DISCARD,
        GameState.SPECIAL_SELECT,
        GameState.TARGET_PLACEMENT,
    },
    GameState.SKIP_DISCARD: {GameState.TURN_SELECT, GameState.TURN_REVEAL},
    GameState.SPECIAL_SELECT: {GameState.TURN_SELECT, GameState.SPECIAL_PLACEMENT},
    GameState.TARGET_PLACEMENT: {GameState.TURN_SELECT, GameState.TURN_REVEAL},
    GameState.SPECIAL_PLACEMENT: {GameState.SPECIAL_SELECT, GameState.TURN_REVEAL},
    GameState.TURN_REVEAL: {GameState.TURN_RESOLVE, GameState.TURN_SELECT},
    GameState.TURN_RESOLVE: {GameState.TURN_SELECT, GameState.FINISHING},
    GameState.FINISHING: {GameState.MATCH_RESULT},
    GameState.MATCH_RESULT: {GameState.MATCH_REWARD, GameState.REPLAY_CHOICE},
    GameState.MATCH_REWARD: {GameState.REPLAY_CHOICE},
    GameState.REPLAY_CHOICE: {GameState.DECK_SELECT, GameState.MATCH_SETUP},
}

STATE_ACTIONS = {
    GameState.NPC_ROSTER: ("上下选择 NPC", "A 确认", "B 返回"),
    GameState.NPC_DIALOGUE: ("A 跳过对白",),
    GameState.MATCH_SETUP: ("左右选择难度", "上下切换焦点", "A 确认", "B 返回"),
    GameState.DECK_SELECT: ("方向键选择卡组", "L/R 翻页", "A 确认"),
    GameState.BATTLE_INTRO: ("等待动画结束",),
    GameState.REDRAW: ("左右选择维持/重抽", "A 确认"),
    GameState.TURN_SELECT: ("方向键选择", "A 确认", "B 返回"),
    GameState.SKIP_DISCARD: ("方向键选择弃牌", "A 确认", "B 返回"),
    GameState.SPECIAL_SELECT: ("方向键选择", "A 确认", "B 返回"),
    GameState.TARGET_PLACEMENT: ("方向键移动", "X/Y 旋转", "A 放置", "B 返回"),
    GameState.SPECIAL_PLACEMENT: ("方向键移动", "X/Y 旋转", "A 放置", "B 返回"),
    GameState.TURN_REVEAL: ("等待卡牌揭示",),
    GameState.TURN_RESOLVE: ("等待回合结算",),
    GameState.FINISHING: ("等待最终结算",),
    GameState.MATCH_RESULT: ("A 继续",),
    GameState.MATCH_REWARD: ("A 继续",),
    GameState.REPLAY_CHOICE: ("左右选择", "A 确认"),
}

STATE_BOXES = {
    GameState.NPC_ROSTER: ((0.38, 0.18, 0.69, 0.82),),
    GameState.NPC_DIALOGUE: ((0.18, 0.64, 0.82, 0.92),),
    GameState.MATCH_SETUP: ((0.64, 0.36, 0.92, 0.70),),
    GameState.DECK_SELECT: ((0.21, 0.16, 0.59, 0.92), (0.69, 0.13, 0.91, 0.95)),
    GameState.REDRAW: ((0.01, 0.14, 0.28, 0.75), (0.42, 0.62, 0.78, 0.83)),
    GameState.TURN_SELECT: ((0.00, 0.10, 0.28, 0.87), (0.81, 0.50, 0.98, 0.89)),
    GameState.SKIP_DISCARD: ((0.01, 0.09, 0.29, 0.89),),
    GameState.SPECIAL_SELECT: ((0.01, 0.09, 0.28, 0.86), (0.01, 0.89, 0.20, 0.96)),
    GameState.TARGET_PLACEMENT: ((0.49, 0.55, 0.72, 0.92),),
    GameState.SPECIAL_PLACEMENT: ((0.49, 0.22, 0.72, 0.92),),
    GameState.TURN_REVEAL: ((0.80, 0.09, 0.98, 0.89),),
    GameState.FINISHING: (),
    GameState.MATCH_RESULT: ((0.25, 0.20, 0.75, 0.85),),
    GameState.MATCH_REWARD: ((0.20, 0.20, 0.80, 0.85),),
    GameState.REPLAY_CHOICE: ((0.33, 0.55, 0.71, 0.84),),
}

STATE_FOCUS_BOXES = {
    GameState.TURN_SELECT: ((0.18, 0.10, 0.82, 0.94),),
    GameState.SKIP_DISCARD: ((0.18, 0.10, 0.82, 0.94),),
    GameState.SPECIAL_SELECT: ((0.18, 0.10, 0.82, 0.94),),
    GameState.TARGET_PLACEMENT: ((0.18, 0.10, 0.82, 0.94),),
    GameState.SPECIAL_PLACEMENT: ((0.18, 0.10, 0.82, 0.94),),
    GameState.TURN_REVEAL: ((0.18, 0.10, 0.82, 0.94),),
    GameState.TURN_RESOLVE: ((0.18, 0.10, 0.82, 0.94),),
}

STATE_STABLE_FRAMES = {
    GameState.BATTLE_INTRO: 8,
    GameState.MATCH_RESULT: 6,
    GameState.REPLAY_CHOICE: 5,
}

TRANSITION_STABLE_FRAMES = {
    (GameState.NPC_ROSTER, GameState.MATCH_SETUP): 8,
    (GameState.TURN_SELECT, GameState.TARGET_PLACEMENT): 2,
    (GameState.TURN_REVEAL, GameState.TARGET_PLACEMENT): 6,
    (GameState.TURN_RESOLVE, GameState.TARGET_PLACEMENT): 8,
}


@dataclass(frozen=True, slots=True)
class TemplateRecord:
    state: GameState
    path: Path
    gray: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True, slots=True)
class StateScore:
    state: GameState
    score: float


@dataclass(frozen=True, slots=True)
class StateObservation:
    state: GameState
    confidence: float
    candidates: tuple[StateScore, ...]
    actions: tuple[str, ...]
    boxes: tuple[tuple[float, float, float, float], ...]
    focus_boxes: tuple[tuple[float, float, float, float], ...] = ()
    profile_candidates: tuple[ProfileStateScore, ...] = ()


@dataclass(frozen=True, slots=True)
class StateEvidence:
    placement_preview_present: bool | None = None


class TemplateStateRecognizer:
    FEATURE_SIZE = (160, 90)
    PROFILE_STATE_WEIGHT = 0.35
    PROFILE_STATE_BASELINE = 0.60
    PROFILE_TRIGGER_MARGIN = 0.12
    PROFILE_CANDIDATE_LIMIT = 4

    def __init__(
        self,
        manifest_path: Path,
        manual_manifest_path: Path | None = None,
        profile_verifier: VisionProfileStateVerifier | None = None,
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manual_path = manual_manifest_path or (
            manifest_path.parent / "manual_manifest.json"
        )
        if manual_path.exists():
            manifest.extend(json.loads(manual_path.read_text(encoding="utf-8")))
        records = []
        for entry in manifest:
            image = cv2.imread(str(manifest_path.parent / entry["image"]))
            if image is None:
                raise RuntimeError(f"cannot read state template: {entry['image']}")
            gray, edges = self._features(image)
            records.append(
                TemplateRecord(
                    state=GameState(entry["state"]),
                    path=manifest_path.parent / entry["image"],
                    gray=gray,
                    edges=edges,
                )
            )
        self.records = tuple(records)
        self.profile_verifier = profile_verifier

    @classmethod
    def default(cls) -> "TemplateStateRecognizer":
        root = Path(__file__).resolve().parents[3] / "assets" / "state_templates"
        return cls(
            root / "manifest.json",
            profile_verifier=VisionProfileStateVerifier.default(),
        )

    def reload(self) -> None:
        replacement = type(self).default()
        self.records = replacement.records
        self.profile_verifier = replacement.profile_verifier

    @classmethod
    def _features(cls, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        resized = cv2.resize(image, cls.FEATURE_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 100)
        return (
            cls._normalize(blur.astype(np.float32)),
            cls._normalize(edges.astype(np.float32)),
        )

    @staticmethod
    def _normalize(value: np.ndarray) -> np.ndarray:
        flat = value.reshape(-1)
        flat = flat - flat.mean()
        length = np.linalg.norm(flat)
        return flat / length if length else flat

    def classify(self, frame: np.ndarray) -> StateObservation:
        gray, edges = self._features(frame)
        scores: dict[GameState, float] = {}
        for record in self.records:
            gray_score = float(np.dot(gray, record.gray))
            edge_score = float(np.dot(edges, record.edges))
            score = 0.72 * gray_score + 0.28 * edge_score
            scores[record.state] = max(scores.get(record.state, -1.0), score)
        base_ranked = tuple(
            StateScore(state, score)
            for state, score in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        profile_candidates: tuple[ProfileStateScore, ...] = ()
        margin = (
            base_ranked[0].score - base_ranked[1].score
            if len(base_ranked) > 1
            else 1.0
        )
        if (
            self.profile_verifier is not None
            and base_ranked
            and base_ranked[0].score >= 0.30
            and margin < self.PROFILE_TRIGGER_MARGIN
        ):
            candidate_states = {
                item.state.value
                for item in base_ranked[: self.PROFILE_CANDIDATE_LIMIT]
            }
            profile_candidates = self.profile_verifier.score(
                frame,
                states=candidate_states,
            )
            for item in profile_candidates:
                if item.score < self.PROFILE_STATE_BASELINE:
                    continue
                state = GameState(item.state)
                scores[state] += self.PROFILE_STATE_WEIGHT * (
                    item.score - self.PROFILE_STATE_BASELINE
                )
        ranked = tuple(
            StateScore(state, score)
            for state, score in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        best = ranked[0]
        state = best.state if best.score >= 0.30 else GameState.UNKNOWN
        return StateObservation(
            state=state,
            confidence=max(0.0, min(1.0, best.score)),
            candidates=ranked,
            actions=STATE_ACTIONS.get(state, ()),
            boxes=STATE_BOXES.get(state, ()),
            focus_boxes=STATE_FOCUS_BOXES.get(state, ()),
            profile_candidates=profile_candidates,
        )


class StateMachine:
    def __init__(
        self,
        recognizer: TemplateStateRecognizer | None = None,
        stable_frames: int = 4,
        transition_bonus: float = 0.08,
    ):
        self.recognizer = recognizer or TemplateStateRecognizer.default()
        self.stable_frames = stable_frames
        self.transition_bonus = transition_bonus
        self.current = GameState.UNKNOWN
        self._pending = GameState.UNKNOWN
        self._pending_count = 0

    def reset(self) -> None:
        self.current = GameState.UNKNOWN
        self._pending = GameState.UNKNOWN
        self._pending_count = 0

    def force_state(self, state: GameState) -> None:
        self.current = state
        self._pending = state
        self._pending_count = self.stable_frames

    def update(
        self,
        frame: np.ndarray,
        evidence: StateEvidence | None = None,
    ) -> StateObservation:
        return self.update_observation(
            self.recognizer.classify(frame),
            evidence,
        )

    def update_observation(
        self,
        raw: StateObservation,
        evidence: StateEvidence | None = None,
    ) -> StateObservation:
        adjusted = []
        for candidate in raw.candidates:
            bonus = 0.0
            if candidate.state == self.current:
                bonus = self.transition_bonus * 0.5
            elif candidate.state in TRANSITIONS.get(self.current, set()):
                bonus = self.transition_bonus
            else:
                bonus = -self.transition_bonus
            if evidence is not None:
                if evidence.placement_preview_present is True:
                    if candidate.state in {
                        GameState.TARGET_PLACEMENT,
                        GameState.SPECIAL_PLACEMENT,
                    }:
                        bonus += 0.20
                    elif candidate.state in {
                        GameState.TURN_REVEAL,
                        GameState.TURN_RESOLVE,
                        GameState.TURN_SELECT,
                    }:
                        bonus -= 0.12
                elif evidence.placement_preview_present is False:
                    if candidate.state is GameState.TURN_SELECT:
                        bonus += 0.18
                    elif candidate.state is GameState.TARGET_PLACEMENT:
                        bonus -= 0.14
                    elif candidate.state is GameState.SPECIAL_PLACEMENT:
                        bonus -= 0.08
            adjusted.append(
                StateScore(candidate.state, candidate.score + bonus)
            )
        adjusted.sort(key=lambda item: item.score, reverse=True)
        selected = adjusted[0].state
        if selected == self._pending:
            self._pending_count += 1
        else:
            self._pending = selected
            self._pending_count = 1
        required_frames = max(
            STATE_STABLE_FRAMES.get(selected, 0),
            TRANSITION_STABLE_FRAMES.get(
                (self.current, selected),
                self.stable_frames,
            ),
        )
        if self._pending_count >= required_frames:
            self.current = selected
        state = selected if self._pending_count >= required_frames else self.current
        top_score = adjusted[0].score
        return StateObservation(
            state=state,
            confidence=max(0.0, min(1.0, top_score)),
            candidates=tuple(adjusted[:5]),
            actions=STATE_ACTIONS.get(state, ()),
            boxes=STATE_BOXES.get(state, ()),
            focus_boxes=STATE_FOCUS_BOXES.get(state, ()),
            profile_candidates=raw.profile_candidates,
        )


def save_manual_state_template(
    frame: np.ndarray,
    state: GameState,
    root_dir: Path | None = None,
) -> Path:
    if state is GameState.UNKNOWN:
        raise ValueError("cannot save an UNKNOWN state template")
    root = root_dir or (
        Path(__file__).resolve().parents[3] / "assets" / "state_templates"
    )
    target_dir = root / "manual" / state.value
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = target_dir / f"corrected-{stamp}.jpg"
    import cv2

    thumbnail = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg",
        thumbnail,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if not ok:
        raise RuntimeError(f"cannot write manual template: {target}")
    target.write_bytes(encoded.tobytes())

    manifest_path = root / "manual_manifest.json"
    values = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else []
    )
    values.append(
        {
            "state": state.value,
            "image": target.relative_to(root).as_posix(),
            "source": "manual",
            "sequence": 0,
        }
    )
    temporary = manifest_path.with_name(f"{manifest_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(values, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(manifest_path)
    return target

