from __future__ import annotations

import random
from collections.abc import Iterable

from tableturf_rebuild.data.catalog import GameCatalog
from tableturf_rebuild.domain.actions import Action, ActionType
from tableturf_rebuild.domain.card import CardCell
from tableturf_rebuild.domain.grid import BLOCKED_CELLS, Cell
from tableturf_rebuild.domain.state import BattleState, PlayerState, Side
from tableturf_rebuild.domain.stage import StageSpec


class RulesError(ValueError):
    pass


NEIGHBORS_8 = tuple(
    (row_delta, col_delta)
    for row_delta in (-1, 0, 1)
    for col_delta in (-1, 0, 1)
    if row_delta != 0 or col_delta != 0
)

OWN_NORMAL = {Side.A: Cell.SELF_INK, Side.B: Cell.OPPONENT_INK}
OWN_SPECIAL = {Side.A: Cell.SELF_SPECIAL, Side.B: Cell.OPPONENT_SPECIAL}


class BattleRules:
    """Deterministic simulation of the published Tableturf Battle rules.

    Each game turn commits one action for each player simultaneously. The
    resolution order for overlapping placements follows the card-size and
    Special Space rules documented by SplatoonWiki.
    """

    def __init__(self, catalog: GameCatalog):
        self.catalog = catalog
        self._legal_cache: dict[tuple[BattleState, Side], tuple[Action, ...]] = {}

    def new_battle(
        self,
        stage_id: str,
        deck_a_id: str,
        deck_b_id: str,
        seed: int = 0,
        hand_size: int = 4,
        turns_per_player: int = 12,
    ) -> BattleState:
        if turns_per_player < 1:
            raise RulesError("turns_per_player must be positive")
        deck_a = list(self.catalog.decks[deck_a_id].card_ids)
        deck_b = list(self.catalog.decks[deck_b_id].card_ids)
        return self.new_battle_from_cards(
            stage_id=stage_id,
            deck_a=deck_a,
            deck_b=deck_b,
            seed=seed,
            hand_size=hand_size,
            turns_per_player=turns_per_player,
        )

    def new_battle_from_cards(
        self,
        stage_id: str,
        deck_a: Iterable[str],
        deck_b: Iterable[str],
        seed: int = 0,
        hand_size: int = 4,
        turns_per_player: int = 12,
    ) -> BattleState:
        if turns_per_player < 1:
            raise RulesError("turns_per_player must be positive")
        stage = self.catalog.stages[stage_id]
        deck_a = list(deck_a)
        deck_b = list(deck_b)
        self._validate_deck_cards(deck_a)
        self._validate_deck_cards(deck_b)
        if not 1 <= hand_size <= min(len(deck_a), len(deck_b)):
            raise RulesError("invalid starting hand size")

        random.Random(seed).shuffle(deck_a)
        random.Random(seed ^ 0x5A17).shuffle(deck_b)
        return BattleState(
            stage=stage,
            board=stage.cells,
            player_a=PlayerState(
                side=Side.A,
                hand=tuple(deck_a[:hand_size]),
                draw_pile=tuple(deck_a[hand_size:]),
            ),
            player_b=PlayerState(
                side=Side.B,
                hand=tuple(deck_b[:hand_size]),
                draw_pile=tuple(deck_b[hand_size:]),
            ),
            turns_remaining=turns_per_player,
        )

    def _validate_deck_cards(self, deck: Iterable[str]) -> None:
        card_ids = tuple(deck)
        if len(card_ids) != 15:
            raise RulesError(f"deck must contain 15 cards, got {len(card_ids)}")
        if len(set(card_ids)) != len(card_ids):
            raise RulesError("deck contains duplicate cards")
        unknown = [card_id for card_id in card_ids if card_id not in self.catalog.cards]
        if unknown:
            raise RulesError(f"deck contains unknown cards: {unknown}")

    def with_hand(
        self,
        stage_id: str,
        hand: Iterable[str],
        draw_pile: Iterable[str] = (),
        round_index: int = 12,
        self_sp: int = 0,
        opponent_hand: Iterable[str] = (),
        opponent_draw_pile: Iterable[str] = (),
        opponent_sp: int = 0,
    ) -> BattleState:
        stage = self.catalog.stages[stage_id]
        hand_tuple = tuple(hand)
        draw_pile_tuple = tuple(draw_pile)
        opponent_hand_tuple = tuple(opponent_hand)
        opponent_draw_tuple = tuple(opponent_draw_pile)
        for card_id in (
            hand_tuple
            + draw_pile_tuple
            + opponent_hand_tuple
            + opponent_draw_tuple
        ):
            if card_id not in self.catalog.cards:
                raise RulesError(f"unknown card: {card_id}")
        return BattleState(
            stage=stage,
            board=stage.cells,
            player_a=PlayerState(
                side=Side.A,
                hand=hand_tuple,
                draw_pile=draw_pile_tuple,
                sp=self_sp,
            ),
            player_b=PlayerState(
                side=Side.B,
                hand=opponent_hand_tuple,
                draw_pile=opponent_draw_tuple,
                sp=opponent_sp,
            ),
            turns_remaining=round_index,
        )

    def redraw(self, state: BattleState, side: Side, seed: int) -> BattleState:
        player = state.player(side)
        if not player.redraw_available:
            raise RulesError(f"{side.value}: redraw already used")
        hand_size = len(player.hand)
        cards = list(player.hand + player.draw_pile)
        random.Random(seed).shuffle(cards)
        updated = PlayerState(
            side=side,
            hand=tuple(cards[:hand_size]),
            draw_pile=tuple(cards[hand_size:]),
            sp=player.sp,
            redraw_available=False,
        )
        return BattleState(
            stage=state.stage,
            board=state.board,
            player_a=updated if side is Side.A else state.player_a,
            player_b=updated if side is Side.B else state.player_b,
            turns_remaining=state.turns_remaining,
        )

    def legal_actions(self, state: BattleState, side: Side) -> tuple[Action, ...]:
        cache_key = (state, side)
        cached = self._legal_cache.get(cache_key)
        if cached is not None:
            return cached
        if state.terminal:
            self._legal_cache[cache_key] = ()
            return ()

        actions: list[Action] = []
        player = state.player(side)
        empty_frontier = self._empty_frontier(state, side)
        special_frontier = self._special_frontier(state, side)
        for card_id in player.hand:
            actions.append(Action.skip(card_id))
            card = self.catalog.cards[card_id]
            for rotation in range(4):
                shape = card.shape.rotated(rotation)
                place_origins = {
                    (row - row_delta, col - col_delta)
                    for row, col in empty_frontier
                    for row_delta, col_delta in shape.occupied
                }
                for origin in sorted(place_origins):
                    targets = self._targets(state, shape, origin)
                    if targets is None:
                        continue

                    if all(
                        state.board[row][col] == Cell.EMPTY
                        for row, col, _ in targets
                    ) and self._touches_own_ink(state, targets, side):
                        actions.append(Action.place(card_id, rotation, origin))

                if player.sp >= card.sp_cost:
                    special_origins = {
                        (row - row_delta, col - col_delta)
                        for row, col in special_frontier
                        for row_delta, col_delta in shape.occupied
                    }
                    for origin in sorted(special_origins):
                        targets = self._targets(state, shape, origin)
                        if targets is None:
                            continue
                        if (
                            all(
                                state.board[row][col]
                                not in {
                                    Cell.SELF_SPECIAL,
                                    Cell.OPPONENT_SPECIAL,
                                    Cell.VOID,
                                    Cell.WALL,
                                }
                                for row, col, _ in targets
                            )
                            and self._touches_own_special(state, targets, side)
                        ):
                            actions.append(Action.special(card_id, rotation, origin))

        result = tuple(actions)
        if len(self._legal_cache) >= 8192:
            self._legal_cache.clear()
        self._legal_cache[cache_key] = result
        return result

    def apply_turn(
        self,
        state: BattleState,
        action_a: Action,
        action_b: Action,
        legal_a: tuple[Action, ...] | None = None,
        legal_b: tuple[Action, ...] | None = None,
        validate: bool = True,
    ) -> BattleState:
        if state.terminal:
            raise RulesError("game is already terminal")
        if validate:
            candidates_a = (
                legal_a if legal_a is not None else self.legal_actions(state, Side.A)
            )
            candidates_b = (
                legal_b if legal_b is not None else self.legal_actions(state, Side.B)
            )
            if action_a not in candidates_a:
                raise RulesError(f"illegal player A action: {action_a}")
            if action_b not in candidates_b:
                raise RulesError(f"illegal player B action: {action_b}")

        board = [list(row) for row in state.board]
        actions = {Side.A: action_a, Side.B: action_b}
        claims: dict[tuple[int, int], list[tuple[Side, CardCell]]] = {}

        for side, action in actions.items():
            if action.action_type is ActionType.SKIP:
                continue
            for row, col, card_cell in self.action_targets(state, action):
                claims.setdefault((row, col), []).append((side, card_cell))

        for (row, col), cell_claims in claims.items():
            if len(cell_claims) == 1:
                side, card_cell = cell_claims[0]
                board[row][col] = (
                    OWN_NORMAL[side]
                    if card_cell is CardCell.INK
                    else OWN_SPECIAL[side]
                )
                continue
            board[row][col] = self._resolve_overlap(
                state,
                actions,
                cell_claims,
            )

        next_players: dict[Side, PlayerState] = {}
        for side, action in actions.items():
            player = state.player(side)
            hand = list(player.hand)
            draw_pile = list(player.draw_pile)
            sp = player.sp
            if action.action_type is ActionType.SKIP:
                sp += 1
            else:
                assert action.card_id is not None
                card = self.catalog.cards[action.card_id]
                if action.action_type is ActionType.SPECIAL:
                    sp -= card.sp_cost

            assert action.card_id is not None
            hand.remove(action.card_id)
            if draw_pile:
                hand.append(draw_pile.pop(0))

            active_before = self._active_special_count(state.board, side)
            active_after = self._active_special_count(board, side)
            sp += max(0, active_after - active_before)
            next_players[side] = PlayerState(
                side=side,
                hand=tuple(hand),
                draw_pile=tuple(draw_pile),
                sp=sp,
                redraw_available=False,
            )

        return BattleState(
            stage=state.stage,
            board=tuple(tuple(row) for row in board),
            player_a=next_players[Side.A],
            player_b=next_players[Side.B],
            turns_remaining=state.turns_remaining - 1,
        )

    def action_targets(
        self, state: BattleState, action: Action
    ) -> tuple[tuple[int, int, CardCell], ...]:
        if action.action_type is ActionType.SKIP:
            return ()
        assert action.card_id is not None
        assert action.rotation is not None
        assert action.origin is not None
        card = self.catalog.cards[action.card_id]
        shape = card.shape.rotated(action.rotation)
        targets = self._targets(state, shape, action.origin)
        if targets is None:
            raise RulesError(f"action targets are outside the board: {action}")
        return targets

    def sample_action(
        self,
        state: BattleState,
        side: Side,
        rng: random.Random,
    ) -> Action:
        """Sample a legal action without materializing the full action list."""

        if state.terminal:
            raise RulesError("cannot sample an action from a terminal state")
        player = state.player(side)
        empty_frontier = tuple(self._empty_frontier(state, side))
        special_frontier = tuple(self._special_frontier(state, side))
        hand = list(player.hand)
        rng.shuffle(hand)
        for card_id in hand:
            card = self.catalog.cards[card_id]
            rotations = list(range(4))
            rng.shuffle(rotations)
            for rotation in rotations:
                shape = card.shape.rotated(rotation)
                offsets = shape.occupied
                if empty_frontier:
                    for _ in range(12):
                        row, col = empty_frontier[rng.randrange(len(empty_frontier))]
                        row_delta, col_delta = offsets[rng.randrange(len(offsets))]
                        origin = (row - row_delta, col - col_delta)
                        targets = self._targets(state, shape, origin)
                        if targets is not None and all(
                            state.board[target_row][target_col] == Cell.EMPTY
                            for target_row, target_col, _ in targets
                        ):
                            return Action.place(card_id, rotation, origin)
                if player.sp >= card.sp_cost and special_frontier:
                    for _ in range(12):
                        row, col = special_frontier[
                            rng.randrange(len(special_frontier))
                        ]
                        row_delta, col_delta = offsets[rng.randrange(len(offsets))]
                        origin = (row - row_delta, col - col_delta)
                        targets = self._targets(state, shape, origin)
                        if targets is not None and all(
                            state.board[target_row][target_col]
                            not in {
                                Cell.SELF_SPECIAL,
                                Cell.OPPONENT_SPECIAL,
                                Cell.VOID,
                                Cell.WALL,
                            }
                            for target_row, target_col, _ in targets
                        ):
                            return Action.special(card_id, rotation, origin)
        return Action.skip(rng.choice(tuple(player.hand)))

    def _resolve_overlap(
        self,
        state: BattleState,
        actions: dict[Side, Action],
        claims: list[tuple[Side, CardCell]],
    ) -> Cell:
        side_a, cell_a = claims[0]
        side_b, cell_b = claims[1]
        card_a_id = actions[side_a].card_id
        card_b_id = actions[side_b].card_id
        assert card_a_id is not None and card_b_id is not None
        size_a = self.catalog.cards[card_a_id].shape.size
        size_b = self.catalog.cards[card_b_id].shape.size

        special_a = cell_a is CardCell.SPECIAL
        special_b = cell_b is CardCell.SPECIAL
        if size_a == size_b:
            if special_a and special_b:
                return Cell.WALL
            if special_a:
                return OWN_SPECIAL[side_a]
            if special_b:
                return OWN_SPECIAL[side_b]
            return Cell.WALL

        if special_a or special_b:
            if special_a and special_b:
                winner = side_a if size_a < size_b else side_b
                return OWN_SPECIAL[winner]
            return OWN_SPECIAL[side_a] if special_a else OWN_SPECIAL[side_b]

        winner = side_a if size_a < size_b else side_b
        return OWN_NORMAL[winner]

    @staticmethod
    def _empty_frontier(state: BattleState, side: Side) -> set[tuple[int, int]]:
        own_cells = {OWN_NORMAL[side], OWN_SPECIAL[side]}
        frontier: set[tuple[int, int]] = set()
        rows, cols = state.stage.rows, state.stage.cols
        for row in range(rows):
            for col in range(cols):
                if state.board[row][col] not in own_cells:
                    continue
                for row_delta, col_delta in NEIGHBORS_8:
                    neighbor_row = row + row_delta
                    neighbor_col = col + col_delta
                    if (
                        0 <= neighbor_row < rows
                        and 0 <= neighbor_col < cols
                        and state.board[neighbor_row][neighbor_col] == Cell.EMPTY
                    ):
                        frontier.add((neighbor_row, neighbor_col))
        return frontier

    @staticmethod
    def _special_frontier(
        state: BattleState, side: Side
    ) -> set[tuple[int, int]]:
        own_special = OWN_SPECIAL[side]
        frontier: set[tuple[int, int]] = set()
        rows, cols = state.stage.rows, state.stage.cols
        for row in range(rows):
            for col in range(cols):
                if state.board[row][col] != own_special:
                    continue
                for row_delta, col_delta in NEIGHBORS_8:
                    neighbor_row = row + row_delta
                    neighbor_col = col + col_delta
                    if not (
                        0 <= neighbor_row < rows and 0 <= neighbor_col < cols
                    ):
                        continue
                    if state.board[neighbor_row][neighbor_col] not in {
                        Cell.SELF_SPECIAL,
                        Cell.OPPONENT_SPECIAL,
                        Cell.VOID,
                        Cell.WALL,
                    }:
                        frontier.add((neighbor_row, neighbor_col))
        return frontier

    @staticmethod
    def _targets(
        state: BattleState, shape, origin: tuple[int, int]
    ) -> tuple[tuple[int, int, CardCell], ...] | None:
        row_origin, col_origin = origin
        targets: list[tuple[int, int, CardCell]] = []
        for row_delta, col_delta in shape.occupied:
            row = row_origin + row_delta
            col = col_origin + col_delta
            if not (0 <= row < state.stage.rows and 0 <= col < state.stage.cols):
                return None
            if state.board[row][col] in BLOCKED_CELLS:
                return None
            targets.append((row, col, shape.cells[row_delta][col_delta]))
        return tuple(targets)

    @staticmethod
    def _touches_own_ink(
        state: BattleState,
        targets: Iterable[tuple[int, int, CardCell]],
        side: Side,
    ) -> bool:
        own_cells = {OWN_NORMAL[side], OWN_SPECIAL[side]}
        target_set = {(row, col) for row, col, _ in targets}
        for row, col in target_set:
            for row_delta, col_delta in NEIGHBORS_8:
                neighbor_row = row + row_delta
                neighbor_col = col + col_delta
                if (neighbor_row, neighbor_col) in target_set:
                    continue
                if not (
                    0 <= neighbor_row < state.stage.rows
                    and 0 <= neighbor_col < state.stage.cols
                ):
                    continue
                if state.board[neighbor_row][neighbor_col] in own_cells:
                    return True
        return False

    @staticmethod
    def _touches_own_special(
        state: BattleState,
        targets: Iterable[tuple[int, int, CardCell]],
        side: Side,
    ) -> bool:
        own_special = OWN_SPECIAL[side]
        target_set = {(row, col) for row, col, _ in targets}
        for row, col in target_set:
            for row_delta, col_delta in NEIGHBORS_8:
                neighbor_row = row + row_delta
                neighbor_col = col + col_delta
                if (neighbor_row, neighbor_col) in target_set:
                    continue
                if not (
                    0 <= neighbor_row < state.stage.rows
                    and 0 <= neighbor_col < state.stage.cols
                ):
                    continue
                if state.board[neighbor_row][neighbor_col] == own_special:
                    return True
        return False

    @staticmethod
    def _active_special_count(board, side: Side) -> int:
        own_special = OWN_SPECIAL[side]
        active = 0
        rows = len(board)
        cols = len(board[0])
        for row in range(rows):
            for col in range(cols):
                if board[row][col] != own_special:
                    continue
                neighbors = (
                    board[row + row_delta][col + col_delta]
                    for row_delta, col_delta in NEIGHBORS_8
                    if 0 <= row + row_delta < rows and 0 <= col + col_delta < cols
                )
                if all(cell != Cell.EMPTY for cell in neighbors):
                    active += 1
        return active
