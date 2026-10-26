from tableturf_rebuild.domain.grid import Cell
from tableturf_rebuild.vision.board_reader import (
    BoardCellReading,
    BoardObservation,
    SpecialActivityTracker,
)


def _reading(
    owner: Cell,
    row: int,
    col: int,
    active: bool,
) -> BoardCellReading:
    return BoardCellReading(
        row=row,
        col=col,
        stage_cell=Cell.EMPTY,
        inferred_cell=owner,
        mean_bgr=(0, 0, 0),
        median_hsv=(0, 0, 0),
        self_fraction=1.0 if owner is Cell.SELF_SPECIAL else 0.0,
        opponent_fraction=1.0 if owner is Cell.OPPONENT_SPECIAL else 0.0,
        self_special_fraction=1.0 if owner is Cell.SELF_SPECIAL else 0.0,
        opponent_special_fraction=1.0 if owner is Cell.OPPONENT_SPECIAL else 0.0,
        self_special_pattern_delta=0.0,
        opponent_special_pattern_delta=0.0,
        center_brightness_contrast=0.0,
        center_saturation=0.0,
        special_active=active,
        color_change=0.0,
        edge_density=0.0,
    )


def _observation(
    self_active: set[tuple[int, int]],
    opponent_active: set[tuple[int, int]],
    transient_self: set[tuple[int, int]],
) -> BoardObservation:
    readings = []
    for row, col in {(13, 3), (13, 4), (17, 5), (21, 5), (22, 4), *transient_self}:
        readings.append(
            _reading(
                Cell.SELF_SPECIAL,
                row,
                col,
                (row, col) in self_active or (row, col) in transient_self,
            )
        )
    for row, col in {(0, 8), (3, 4), (4, 3), (7, 4), (8, 1)}:
        readings.append(
            _reading(
                Cell.OPPONENT_SPECIAL,
                row,
                col,
                (row, col) in opponent_active,
            )
        )
    return BoardObservation(
        stage_id="Rectangle",
        rows=1,
        cols=1,
        cells=((Cell.EMPTY,),),
        readings=tuple(readings),
        confidence=1.0,
        placement_preview_score=0.0,
        placement_preview_present=False,
        playable_cells=1,
    )


def test_special_activity_tracker_bridges_flicker_and_rejects_transient() -> None:
    tracker = SpecialActivityTracker()
    stable_self = {(13, 4)}
    stable_opponent = {(0, 8), (3, 4), (7, 4)}
    for _ in range(18):
        tracker.update(_observation(stable_self, stable_opponent, set()))
    for _ in range(13):
        tracker.update(_observation(set(), set(), set()))
    for _ in range(18):
        tracker.update(_observation(set(), set(), {(21, 5)}))
    for _ in range(81):
        observation = tracker.update(
            _observation(stable_self, stable_opponent, set())
        )

    assert observation.self_cells == frozenset(stable_self)
    assert observation.opponent_cells == frozenset(stable_opponent)
    assert observation.self_special_count == 5
    assert observation.opponent_special_count == 5
