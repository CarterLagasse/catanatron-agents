"""The engine against the official rules (Catan 5th edition Game Rules & Almanac).

The positions are in tests/rules_scenarios.py; tests/jax_rl/test_parity.py checks
that the JAX environment matches the engine on them.
"""

import random

import pytest

from catanatron.models.board import Board
from catanatron.models.enums import ActionRecord
from catanatron.models.map import build_map
from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_color,
    get_longest_road_length,
    player_num_resource_cards,
)
from tests import rules_scenarios
from tests.rules_scenarios import BLUE, RED, WHITE


def execute(game, action, dice):
    record = ActionRecord(action=action, result=dice) if dice else None
    game.execute(action, action_record=record)


def test_broken_road_below_five_sets_longest_road_aside():
    game, action, dice = rules_scenarios.broken_road()
    assert get_longest_road_color(game.state) == RED
    assert get_actual_victory_points(game.state, RED) == 3

    execute(game, action, dice)
    assert get_longest_road_length(game.state, RED) == 3  # 2 + 3, not 5
    assert get_longest_road_color(game.state) is None
    assert get_actual_victory_points(game.state, RED) == 1
    assert game.state.board.road_length == 3


@pytest.mark.parametrize(
    "lengths, holder, expected",
    [
        # The holder is still tied for the longest road: keeps the card
        ({RED: 5, BLUE: 5, WHITE: 3}, RED, RED),
        # One other player now has the longest road: takes it
        ({RED: 4, BLUE: 6, WHITE: 5}, RED, BLUE),
        # Two others tie for the longest road: set aside
        ({RED: 4, BLUE: 6, WHITE: 6}, RED, None),
        # Nobody has 5+ roads: set aside
        ({RED: 4, BLUE: 4, WHITE: 2}, RED, None),
        # A card set aside comes back once only 1 player has the longest road
        ({RED: 6, BLUE: 4, WHITE: 5}, None, RED),
        ({RED: 6, BLUE: 6, WHITE: 5}, None, None),
    ],
)
def test_longest_road_after_a_broken_road(lengths, holder, expected):
    board = Board(build_map("BASE"))
    board.road_lengths.update(lengths)
    board.road_color = holder
    board._reassign_longest_road_after_cut()
    assert board.road_color == expected
    assert board.road_length == max(lengths.values())


def test_road_ending_at_an_opponents_settlement_counts():
    game, action, dice = rules_scenarios.road_into_settlement()
    assert get_longest_road_length(game.state, RED) == 4

    execute(game, action, dice)
    assert get_longest_road_length(game.state, RED) == 5
    assert get_longest_road_color(game.state) == RED


def test_players_only_win_on_their_own_turn():
    game, action, dice = rules_scenarios.win_waits_for_own_turn()
    assert get_actual_victory_points(game.state, BLUE) == 10
    assert game.winning_color() is None  # it's RED's turn

    execute(game, action, dice)
    assert game.winning_color() == BLUE


def test_bank_short_for_one_player_gives_them_the_rest():
    game, action, dice = rules_scenarios.bank_short_for_one_player()
    bank = sum(game.state.resource_freqdeck)
    hand = player_num_resource_cards(game.state, RED)

    execute(game, action, dice)
    assert player_num_resource_cards(game.state, RED) == hand + 1  # owed 2
    assert sum(game.state.resource_freqdeck) == bank - 1


def test_bank_short_for_two_players_gives_nobody_anything():
    game, action, dice = rules_scenarios.bank_short_for_two_players()
    bank = sum(game.state.resource_freqdeck)
    hands = [player_num_resource_cards(game.state, c) for c in (RED, BLUE)]

    execute(game, action, dice)
    assert [player_num_resource_cards(game.state, c) for c in (RED, BLUE)] == hands
    assert sum(game.state.resource_freqdeck) == bank


@pytest.mark.parametrize("map_type", ["BASE", "MINI"])
def test_random_number_placement_keeps_red_numbers_apart(map_type):
    for seed in range(100):
        random.seed(seed)
        catan_map = build_map(map_type, number_placement="random")
        reds = [
            set(tile.nodes.values())
            for tile in catan_map.land_tiles.values()
            if tile.number in (6, 8)
        ]
        assert all(len(a & b) < 2 for i, a in enumerate(reds) for b in reds[i + 1 :])
