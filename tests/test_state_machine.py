import numpy as np
import json
from PIL import Image

from tableturf_rebuild.application.state_machine import (
    GameState,
    STATE_LABELS,
    StateMachine,
    StateEvidence,
    StateObservation,
    StateScore,
    save_manual_state_template,
)


class SequenceRecognizer:
    def __init__(self, predictions):
        self.predictions = list(predictions)

    def classify(self, frame):
        del frame
        state = self.predictions.pop(0)
        return StateObservation(
            state=state,
            confidence=0.9,
            candidates=(StateScore(state, 0.9),),
            actions=(),
            boxes=(),
        )


class PlacementEvidenceRecognizer:
    def classify(self, frame):
        del frame
        return StateObservation(
            state=GameState.TURN_RESOLVE,
            confidence=0.9,
            candidates=(
                StateScore(GameState.TURN_RESOLVE, 0.9),
                StateScore(GameState.TARGET_PLACEMENT, 0.7),
            ),
            actions=(),
            boxes=(),
        )


class MissingPreviewRecognizer:
    def classify(self, frame):
        del frame
        return StateObservation(
            state=GameState.TARGET_PLACEMENT,
            confidence=0.75,
            candidates=(
                StateScore(GameState.TARGET_PLACEMENT, 0.75),
                StateScore(GameState.TURN_SELECT, 0.69),
            ),
            actions=(),
            boxes=(),
        )


def test_state_machine_requires_stable_frames() -> None:
    machine = StateMachine(
        recognizer=SequenceRecognizer(
            [GameState.NPC_ROSTER, GameState.NPC_ROSTER, GameState.NPC_DIALOGUE]
        ),
        stable_frames=2,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    assert machine.update(frame).state is GameState.UNKNOWN
    assert machine.update(frame).state is GameState.NPC_ROSTER
    assert machine.update(frame).state is GameState.NPC_ROSTER


def test_state_machine_bonus_prefers_legal_transition() -> None:
    machine = StateMachine(stable_frames=1, transition_bonus=0.2)
    machine.current = GameState.NPC_ROSTER
    machine.recognizer = SequenceRecognizer([GameState.NPC_DIALOGUE])
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    assert machine.update(frame).state is GameState.NPC_DIALOGUE


def test_manual_state_template_is_saved(tmp_path) -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    path = save_manual_state_template(
        image,
        GameState.MATCH_SETUP,
        root_dir=tmp_path,
    )

    assert path.exists()
    assert Image.open(path).size == (640, 360)
    manifest = json.loads(
        (tmp_path / "manual_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest[0]["state"] == "MATCH_SETUP"
    assert manifest[0]["image"] == path.relative_to(tmp_path).as_posix()


def test_battle_intro_requires_extra_stable_frames() -> None:
    predictions = [GameState.DECK_SELECT] * 2 + [GameState.BATTLE_INTRO] * 9
    machine = StateMachine(
        recognizer=SequenceRecognizer(predictions),
        stable_frames=2,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    assert machine.update(frame).state is GameState.UNKNOWN
    assert machine.update(frame).state is GameState.DECK_SELECT
    for _ in range(7):
        assert machine.update(frame).state is GameState.DECK_SELECT
    assert machine.update(frame).state is GameState.BATTLE_INTRO


def test_all_states_have_chinese_labels() -> None:
    assert set(STATE_LABELS) == set(GameState)
    assert STATE_LABELS[GameState.TARGET_PLACEMENT] == "普通卡牌摆放"


def test_npc_to_difficulty_requires_more_frames() -> None:
    predictions = [GameState.NPC_ROSTER] * 4 + [GameState.MATCH_SETUP] * 8
    machine = StateMachine(
        recognizer=SequenceRecognizer(predictions),
        stable_frames=4,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    for _ in range(4):
        machine.update(frame)
    for _ in range(7):
        assert machine.update(frame).state is GameState.NPC_ROSTER
    assert machine.update(frame).state is GameState.MATCH_SETUP


def test_turn_select_enters_placement_quickly() -> None:
    predictions = [GameState.TURN_SELECT] * 4 + [GameState.TARGET_PLACEMENT] * 2
    machine = StateMachine(
        recognizer=SequenceRecognizer(predictions),
        stable_frames=4,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    for _ in range(4):
        machine.update(frame)
    assert machine.update(frame).state is GameState.TURN_SELECT
    assert machine.update(frame).state is GameState.TARGET_PLACEMENT


def test_placement_preview_evidence_overrides_resolve_candidate() -> None:
    machine = StateMachine(
        recognizer=PlacementEvidenceRecognizer(),
        stable_frames=1,
        transition_bonus=0.05,
    )
    machine.force_state(GameState.TARGET_PLACEMENT)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    observation = machine.update(
        frame,
        StateEvidence(placement_preview_present=True),
    )

    assert observation.state is GameState.TARGET_PLACEMENT


def test_missing_preview_evidence_selects_turn_select() -> None:
    machine = StateMachine(
        recognizer=MissingPreviewRecognizer(),
        stable_frames=1,
        transition_bonus=0.05,
    )
    machine.force_state(GameState.TARGET_PLACEMENT)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    observation = machine.update(
        frame,
        StateEvidence(placement_preview_present=False),
    )

    assert observation.state is GameState.TURN_SELECT

