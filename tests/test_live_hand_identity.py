"""A deck prior must not force visible starter cards into an unrelated deck."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.vision.hand_reader import HandReader
from tableturf_rebuild.vision.selection_profiles import SelectionProfileResolver


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "profiles/vision"


@pytest.fixture(scope="module")
def live():
    frame = cv2.imread(str(ROOT / "artifacts/diagnostics/win-plan-and-missing-card/analysis.png"))
    assert frame is not None
    catalog = GameCatalog.from_package()
    reader = HandReader(catalog, VISION)
    profile = SelectionProfileResolver.default().resolve(frame)
    assert profile is not None and profile.focus_slot == (1, 1)
    return catalog, reader, frame, profile.profile_path


def test_actual_starter_cards_are_exact_when_the_candidate_deck_is_correct(live):
    catalog, reader, frame, profile = live
    hand = reader.read(frame, profile, catalog.decks["MiniGame_Starter"].card_ids)
    assert hand.card_ids == ("MiniGame_Batoroika", "MiniGame_ChargerNormal00",
                            "MiniGame_Denchinamazu", "MiniGame_ShooterNormal00")
    assert [card.area for card in hand.cards] == [10, 8, 9, 8]
    assert [card.visible_cost_lights for card in hand.cards] == [4, 3, 4, 3]
    assert [card.match_score for card in hand.cards] == [1.0] * 4
    assert hand.confidence == 1.0
    assert hand.deck_mismatch is False
    assert hand.unknown_card_ids == ()


def test_unrelated_expected_deck_returns_unknown_instead_of_inventing_lact(live):
    _, reader, frame, profile = live
    preset = json.loads((ROOT / "profiles/decks/NGA Suggestion.json").read_text(encoding="utf-8"))
    hand = reader.read(frame, profile, tuple(preset["cards"]))
    assert hand.deck_mismatch is True
    assert hand.cards == ()
    assert hand.confidence == 0.0
    assert set(hand.unknown_card_ids) == {
        "MiniGame_Batoroika", "MiniGame_ChargerNormal00", "MiniGame_ChargerNormal01",
        "MiniGame_Denchinamazu", "MiniGame_ShooterNormal00", "MiniGame_ShooterNormal01",
    }
    assert "MiniGame_StringerShort00" not in hand.card_ids
    assert not set(hand.unknown_card_ids).intersection(preset["cards"])


def test_full_catalog_does_not_claim_same_shape_weapon_variants_are_unique(live):
    catalog, reader, frame, profile = live
    hand = reader.read(frame, profile, tuple(catalog.cards))
    assert hand.confidence == 1.0
    assert hand.deck_mismatch is False
    assert {x.card_id for x in hand.cards[1].candidates if x.score == 1.0} == {
        "MiniGame_ChargerNormal00", "MiniGame_ChargerNormal01",
    }
    assert {x.card_id for x in hand.cards[3].candidates if x.score == 1.0} == {
        "MiniGame_ShooterNormal00", "MiniGame_ShooterNormal01",
    }


def test_low_quality_frame_without_exact_external_evidence_is_not_a_deck_mismatch(live):
    catalog, reader, frame, profile = live
    obscured = np.zeros_like(frame)
    hand = reader.read(obscured, profile, catalog.decks["MiniGame_Starter"].card_ids)
    assert hand.deck_mismatch is False
    assert hand.unknown_card_ids == ()
    assert hand.confidence < 0.5


def test_known_exact_hand_does_not_need_a_global_catalog_rescan(live, monkeypatch):
    catalog, reader, frame, profile = live
    candidates = catalog.decks["MiniGame_Starter"].card_ids
    original = reader.shape_reader.match_shape
    calls = []

    def tracked(pattern, candidate_ids, lights=None):
        calls.append(tuple(candidate_ids))
        assert tuple(candidate_ids) == candidates
        return original(pattern, candidate_ids, lights)

    monkeypatch.setattr(reader.shape_reader, "match_shape", tracked)
    assert reader.read(frame, profile, candidates).confidence == 1.0
    assert len(calls) == 4


@pytest.mark.parametrize("empty_value", (153, 200, 255))
def test_selected_gray_cost_slot_is_not_a_fifth_yellow_light(empty_value):
    hsv = np.zeros((20, 100, 3), dtype=np.uint8)
    hsv[:, :80] = (30, 220, 230)
    hsv[:, 80:] = (30, 20, empty_value)
    assert HandReader._cost_lights(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), minimum_value=150) == 4


def test_disabled_yellow_cost_lights_use_the_dim_threshold():
    hsv = np.zeros((20, 100, 3), dtype=np.uint8)
    hsv[:, :60] = (30, 180, 110)
    hsv[:, 60:] = (30, 10, 153)
    patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert HandReader._cost_lights(patch, minimum_value=90) == 3
    assert HandReader._cost_lights(patch, minimum_value=150) == 0


def test_small_yellow_illustration_under_empty_dim_cost_slot_is_not_a_light():
    hsv = np.zeros((20, 100, 3), dtype=np.uint8)
    hsv[:, :60] = (30, 220, 110)
    hsv[:, 60:] = (30, 20, 65)
    hsv[:8, 60:80] = (30, 220, 110)
    patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert HandReader._cost_lights(patch, minimum_value=90) == 3
