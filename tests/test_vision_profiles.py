import json
from pathlib import Path

import cv2

from tableturf_rebuild.application.state_machine import (
    GameState,
    TemplateStateRecognizer,
)
from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.board_reader import BoardGridReader
from tableturf_rebuild.vision.calibration import VisionProfile, VisionRegion
from tableturf_rebuild.vision.profile_state_verifier import (
    VisionProfileStateVerifier,
)
from tableturf_rebuild.vision.profile_semantics import (
    ProfileSemantics,
    RegionRole,
)
from tableturf_rebuild.vision.selection_profiles import SelectionProfileResolver


ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = ROOT / "profiles" / "vision"


def _profile_source(path: Path):
    profile = VisionProfile.load(path)
    source = Path(profile.notes.removeprefix("source="))
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def test_state_profile_verifier_identifies_npc_screen() -> None:
    image = _profile_source(VISION_DIR / "npc选择.json")
    scores = VisionProfileStateVerifier.default().score(image)

    assert scores[0].state == GameState.NPC_ROSTER.value
    assert scores[0].score > scores[1].score + 0.3


def test_state_recognizer_uses_profile_evidence_for_npc_ambiguity() -> None:
    image = _profile_source(VISION_DIR / "npc选择.json")
    observation = TemplateStateRecognizer.default().classify(image)

    assert observation.state is GameState.NPC_ROSTER
    assert observation.profile_candidates
    assert observation.profile_candidates[0].state == GameState.NPC_ROSTER.value


def test_focus_2_1_profile_supports_turn_select_state() -> None:
    image = _profile_source(VISION_DIR / "battle_select21.json")
    observation = TemplateStateRecognizer.default().classify(image)

    assert observation.state is GameState.TURN_SELECT
    assert observation.profile_candidates


def test_selection_resolver_identifies_focus_1_1_reference() -> None:
    image = _profile_source(VISION_DIR / "battle_select11.json")
    result = SelectionProfileResolver.default().resolve(image)

    assert result is not None
    assert result.variant == "focus_1_1"
    assert result.focus_slot == (1, 1)


def test_selection_resolver_uses_brightness_to_resolve_idle_reference() -> None:
    image = _profile_source(VISION_DIR / "battle_select_clean.json")
    resolver = SelectionProfileResolver.default()
    scores = resolver.match(image)
    result = resolver.resolve(image)

    assert scores[0].variant == "idle"
    assert result is not None and result.variant == "idle"


def test_selection_resolver_identifies_focus_2_1_reference() -> None:
    image = _profile_source(VISION_DIR / "battle_select21.json")
    resolver = SelectionProfileResolver.default()
    result = resolver.resolve(image)

    assert result is not None
    assert result.variant == "focus_2_1"
    assert result.brightness_margin > 30


def test_hand_profile_semantics_resolve_slots() -> None:
    semantics = ProfileSemantics.load(VISION_DIR / "battle_select_clean.json")

    assert len(semantics.by_role(RegionRole.HAND_CARD)) == 4
    assert len(semantics.by_role(RegionRole.HAND_GRID)) == 4
    assert len(semantics.by_role(RegionRole.HAND_AREA_NUMBER)) == 4
    assert len(semantics.by_role(RegionRole.HAND_SPECIAL_COST)) == 4
    assert semantics.by_slot(RegionRole.HAND_GRID, 2, 2) is not None


def test_deck_and_reveal_profile_semantics() -> None:
    deck = ProfileSemantics.load(VISION_DIR / "卡组选择.json")
    reveal = ProfileSemantics.load(VISION_DIR / "结算阶段开牌.json")

    assert len(deck.by_role(RegionRole.DECK_CARD_GRID)) == 15
    assert len(deck.by_role(RegionRole.DECK_CARD_COST)) == 15
    assert len(reveal.by_role(RegionRole.REVEAL_CARD)) == 2
    assert len(reveal.by_role(RegionRole.REVEAL_GRID)) == 2
    assert len(reveal.by_role(RegionRole.REVEAL_SPECIAL_COST)) == 2


def test_battle_hud_score_regions_are_semantic(tmp_path) -> None:
    profile = VisionProfile(
        profile_id="hud",
        source_width=1920,
        source_height=1080,
        regions=(
            VisionRegion(
                name="我方分数",
                kind="rect",
                points=((0.1, 0.1), (0.2, 0.2)),
            ),
            VisionRegion(
                name="敌方分数",
                kind="rect",
                points=((0.8, 0.1), (0.9, 0.2)),
            ),
        ),
    )
    path = tmp_path / "hud.json"
    profile.save(path)

    semantics = ProfileSemantics.load(path)

    assert len(semantics.by_role(RegionRole.SELF_SCORE)) == 1
    assert len(semantics.by_role(RegionRole.OPPONENT_SCORE)) == 1


def test_irregular_profile_can_override_pack_dimensions() -> None:
    catalog = GameCatalog.from_package()

    _, profile, region, _ = BoardGridReader.from_profile(
        catalog,
        VISION_DIR / "对战场地-酷似大道.json",
        "Divide",
    )

    assert profile.stage_id == "Divide"
    assert (region.grid_rows, region.grid_cols) == (18, 15)
    assert (catalog.stages["Divide"].rows, catalog.stages["Divide"].cols) == (
        18,
        16,
    )
    overrides = {
        (override.row, override.col): override.cell
        for override in profile.cell_overrides
    }
    assert overrides[(15, 3)] == Cell.SELF_SPECIAL
    assert overrides[(2, 3)] == Cell.OPPONENT_SPECIAL
    assert overrides[(15, 11)] == Cell.SELF_SPECIAL
    assert overrides[(2, 11)] == Cell.OPPONENT_SPECIAL


def test_painted_profile_keeps_initial_ink_cells() -> None:
    catalog = GameCatalog.from_package()
    reader, profile, region, reference = BoardGridReader.from_profile(
        catalog,
        VISION_DIR / "对战场地-间隔墙.json",
        "Painted",
    )
    source = _profile_source(VISION_DIR / "对战场地-间隔墙.json")

    observation = reader.read(
        source,
        region,
        "Painted",
        reference,
        profile,
    )

    assert sum(
        reading.stage_cell is Cell.SELF_INK
        and reading.inferred_cell in {Cell.SELF_INK, Cell.SELF_SPECIAL}
        for reading in observation.readings
    ) == 9
    assert sum(
        reading.stage_cell is Cell.OPPONENT_INK
        and reading.inferred_cell
        in {Cell.OPPONENT_INK, Cell.OPPONENT_SPECIAL}
        for reading in observation.readings
    ) == 9


def test_all_stages_have_registered_board_profiles() -> None:
    catalog = GameCatalog.from_package()
    index = json.loads(
        (VISION_DIR / "index.json").read_text(encoding="utf-8")
    )

    assert set(index["board_profiles"]) == set(catalog.stages)
