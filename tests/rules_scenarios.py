"""Positions for checking the official rules (Catan 5th edition Game Rules &
Almanac). tests/test_rules.py checks the engine's results on them, and
tests/jax_rl/test_parity.py checks that the JAX environment matches the engine.

Each scenario returns (game, action, dice): the engine game just before the
action, and the dice to force if the action is a roll.
"""

import networkx as nx

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.decks import freqdeck_subtract
from catanatron.models.enums import (
    CITY,
    KNIGHT,
    RESOURCES,
    SETTLEMENT,
    VICTORY_POINT,
    Action,
    ActionPrompt,
    ActionType,
)
from catanatron.models.map import build_map
from catanatron.models.player import Color, SimplePlayer
from catanatron.state_functions import (
    build_road,
    build_settlement,
    maintain_longest_road,
    player_freqdeck_add,
    player_key,
)

RED, BLUE, WHITE, ORANGE = Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE
SETTLEMENT_COST = [1, 1, 1, 1, 0]
ROAD_COST = [1, 1, 0, 0, 0]


def new_game(colors, seed=0):
    return Game(
        [SimplePlayer(c) for c in colors], seed=seed, catan_map=build_map("BASE")
    )


def place_settlement(game, color, node):
    game.state.board.build_settlement(color, node, initial_build_phase=True)
    build_settlement(game.state, color, node, is_free=True)


def place_road(game, color, edge):
    result = game.state.board.build_road(color, edge)
    build_road(game.state, color, edge, is_free=True)
    maintain_longest_road(game.state, *result)


def place_city(game, color, node):
    """Upgrades color's settlement at node for free (build_city charges the cost)."""
    state = game.state
    state.board.build_city(color, node)
    state.buildings_by_color[color][SETTLEMENT].remove(node)
    state.buildings_by_color[color][CITY].append(node)
    key = player_key(state, color)
    state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] += 1
    state.player_state[f"{key}_CITIES_AVAILABLE"] -= 1
    state.player_state[f"{key}_VICTORY_POINTS"] += 1
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1


def give(game, color, freqdeck):
    player_freqdeck_add(game.state, color, freqdeck)
    game.state.resource_freqdeck = freqdeck_subtract(
        game.state.resource_freqdeck, freqdeck
    )


def start_turn(game, color, rolled=True):
    """Ends initial placement and makes it color's turn."""
    state = game.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.current_player_index = state.current_turn_index = state.colors.index(color)
    for c in state.colors:
        state.player_state[f"{player_key(state, c)}_HAS_ROLLED"] = c == color and rolled
    game.playable_actions = generate_playable_actions(state)


def road_layout():
    """A road path p0..p5 (5 roads) on land with p0 and p5 not adjacent, a node n
    next to p2 off the path, and a node m next to n, far enough from the path for
    a settlement."""
    land = build_map("BASE").land_nodes
    graph = STATIC_GRAPH.subgraph(land)
    for source in sorted(land):
        for target in sorted(land):
            for path in nx.all_simple_paths(graph, source, target, cutoff=5):
                if len(path) != 6 or path[5] in graph[path[0]]:
                    continue
                for n in sorted(set(graph[path[2]]) - set(path)):
                    for m in sorted(set(graph[n]) - set(path)):
                        if not set(graph[m]) & set(path):
                            return path, n, m
    raise AssertionError("no layout found")


def broken_road():
    """RED holds Longest Road with 5 roads. BLUE builds a settlement in the middle,
    leaving RED with 2 + 3: nobody has 5+, so the card is set aside."""
    path, n, m = road_layout()
    game = new_game([RED, BLUE])
    place_settlement(game, RED, path[0])
    for a, b in zip(path, path[1:]):
        place_road(game, RED, (a, b))
    place_settlement(game, BLUE, m)
    place_road(game, BLUE, (m, n))
    place_road(game, BLUE, (n, path[2]))
    give(game, BLUE, SETTLEMENT_COST)
    start_turn(game, BLUE)
    return game, Action(BLUE, ActionType.BUILD_SETTLEMENT, path[2]), None


def road_into_settlement():
    """RED has 4 roads leading to BLUE's settlement and builds the 5th road, which
    ends at that settlement: 5 road pieces, so RED takes Longest Road."""
    path, _, _ = road_layout()
    game = new_game([RED, BLUE])
    place_settlement(game, BLUE, path[5])
    place_settlement(game, RED, path[0])
    for a, b in zip(path[:4], path[1:5]):
        place_road(game, RED, (a, b))
    give(game, RED, ROAD_COST)
    start_turn(game, RED)
    edge = tuple(sorted((path[4], path[5])))
    return game, Action(RED, ActionType.BUILD_ROAD, edge), None


def win_waits_for_own_turn():
    """BLUE has 10 points during RED's turn: a city, a settlement, Largest Army and
    5 victory point cards. RED ends the turn, and BLUE wins as BLUE's turn starts."""
    path, _, m = road_layout()
    game = new_game([RED, BLUE])
    state = game.state
    place_settlement(game, RED, path[0])
    place_settlement(game, BLUE, m)
    place_city(game, BLUE, m)
    place_settlement(game, BLUE, state.board.buildable_node_ids(BLUE, True)[0])
    key = player_key(state, BLUE)
    for card, count in [(VICTORY_POINT, 5), (KNIGHT, 3)]:
        for _ in range(count):
            state.development_listdeck.remove(card)
    state.player_state[f"{key}_VICTORY_POINT_IN_HAND"] = 5
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 5
    state.player_state[f"{key}_PLAYED_KNIGHT"] = 3
    state.player_state[f"{key}_HAS_ARMY"] = True
    state.player_state[f"{key}_VICTORY_POINTS"] += 2
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 2
    start_turn(game, RED)
    return game, Action(RED, ActionType.END_TURN, None), None


def _producing_spot(game):
    """A tile, and one of its nodes no other tile with the same number touches."""
    board = game.state.board
    for tile in board.map.land_tiles.values():
        if tile.resource is None:
            continue
        for node in sorted(tile.nodes.values()):
            numbers = [t.number for t in board.map.adjacent_tiles[node]]
            if numbers.count(tile.number) == 1:
                return tile, node
    raise AssertionError("no spot found")


def _dice(number):
    first = min(6, number - 1)
    return (first, number - first)


def bank_short_for_one_player():
    """RED's city is owed 2 cards of a resource the bank has 1 of. Only RED is
    owed it, so RED gets the last card."""
    game = new_game([RED, BLUE])
    tile, node = _producing_spot(game)
    place_settlement(game, RED, node)
    place_city(game, RED, node)
    _drain_bank(game, tile.resource, keep=1)
    start_turn(game, RED, rolled=False)
    return game, Action(RED, ActionType.ROLL, None), _dice(tile.number)


def bank_short_for_two_players():
    """RED and BLUE are each owed 1 card of a resource the bank has 1 of: nobody
    gets it."""
    game = new_game([RED, BLUE])
    tile, node = _producing_spot(game)
    graph = STATIC_GRAPH
    other = next(
        n for n in sorted(tile.nodes.values()) if n != node and n not in graph[node]
    )
    place_settlement(game, RED, node)
    place_settlement(game, BLUE, other)
    _drain_bank(game, tile.resource, keep=1)
    start_turn(game, RED, rolled=False)
    return game, Action(RED, ActionType.ROLL, None), _dice(tile.number)


def _drain_bank(game, resource, keep):
    """Leaves keep cards of resource in the bank; BLUE holds the rest."""
    index = RESOURCES.index(resource)
    freqdeck = [0, 0, 0, 0, 0]
    freqdeck[index] = game.state.resource_freqdeck[index] - keep
    give(game, BLUE, freqdeck)


SCENARIOS = [
    broken_road,
    road_into_settlement,
    win_waits_for_own_turn,
    bank_short_for_one_player,
    bank_short_for_two_players,
]
