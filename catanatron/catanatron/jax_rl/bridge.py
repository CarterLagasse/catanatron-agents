"""Conversions between the Python engine and the JAX environment.

Used by parity tests and by JaxPlayer, so trained agents play real engine games.
"""

import numpy as np
import jax.numpy as jnp

from catanatron.game import TURNS_LIMIT
from catanatron.jax_rl.env import (
    BUILD_INITIAL_ROAD,
    BUILD_INITIAL_SETTLEMENT,
    DISCARD,
    MOVE_ROBBER,
    NO_ROAD_EVAL,
    PLAY_TURN,
    State,
)
from catanatron.jax_rl.topology import (
    DESERT,
    DEV_DECK,
    MARITIME_PAIRS,
    NUM_DEV_TYPES,
    YOP_PAIRS,
)
from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    RESOURCES,
    SETTLEMENT,
    ActionPrompt,
    ActionType,
)

_PROMPTS = {
    ActionPrompt.BUILD_INITIAL_SETTLEMENT: BUILD_INITIAL_SETTLEMENT,
    ActionPrompt.BUILD_INITIAL_ROAD: BUILD_INITIAL_ROAD,
    ActionPrompt.PLAY_TURN: PLAY_TURN,
    ActionPrompt.DISCARD: DISCARD,
    ActionPrompt.MOVE_ROBBER: MOVE_ROBBER,
}
_RESOURCE_INDEX = {r: i for i, r in enumerate(RESOURCES)}
_YOP_INDEX = {tuple(pair): i for i, pair in enumerate(map(tuple, YOP_PAIRS.tolist()))}
_MARITIME_INDEX = {
    tuple(pair): i for i, pair in enumerate(map(tuple, MARITIME_PAIRS.tolist()))
}


class ActionCodec:
    """Translates engine Actions to JAX action indices."""

    def __init__(self, topo):
        self.topo = topo
        self.edge_index = {tuple(e): i for i, e in enumerate(topo.edges.tolist())}
        self.tile_index = {c: i for i, c in enumerate(topo.tile_coordinates)}

    def encode(self, action):
        topo = self.topo
        kind, value = action.action_type, action.value
        if kind == ActionType.ROLL:
            return topo.ROLL
        if kind == ActionType.END_TURN:
            return topo.END_TURN
        if kind == ActionType.BUY_DEVELOPMENT_CARD:
            return topo.BUY_DEV
        if kind == ActionType.PLAY_KNIGHT_CARD:
            return topo.PLAY_KNIGHT
        if kind == ActionType.PLAY_ROAD_BUILDING:
            return topo.PLAY_ROAD_BUILDING
        if kind == ActionType.DISCARD_RESOURCE:
            return topo.DISCARD + _RESOURCE_INDEX[value]
        if kind == ActionType.BUILD_ROAD:
            return topo.ROAD + self.edge_index[tuple(sorted(value))]
        if kind == ActionType.BUILD_SETTLEMENT:
            return topo.SETTLEMENT + value
        if kind == ActionType.BUILD_CITY:
            return topo.CITY + value
        if kind == ActionType.PLAY_YEAR_OF_PLENTY:
            cards = [_RESOURCE_INDEX[r] for r in value]
            if len(cards) == 1:
                return topo.YOP + len(YOP_PAIRS) + cards[0]
            return topo.YOP + _YOP_INDEX[tuple(cards)]
        if kind == ActionType.PLAY_MONOPOLY:
            return topo.MONOPOLY + _RESOURCE_INDEX[value]
        if kind == ActionType.MOVE_ROBBER:
            coordinate, victim = value
            return (
                topo.ROBBER
                + 2 * self.tile_index[coordinate]
                + (0 if victim is None else 1)
            )
        if kind == ActionType.MARITIME_TRADE:
            give = _RESOURCE_INDEX[value[0]]
            get = _RESOURCE_INDEX[value[-1]]
            return topo.MARITIME + _MARITIME_INDEX[(give, get)]
        raise ValueError(f"Action has no JAX encoding: {action}")


def state_from_game(game, topo, batch_dim=True):
    """Builds a JAX State from a 1v1 engine Game.

    Seats are indices into game.state.colors. The dev deck keeps its true order,
    which is hidden from players because observations only expose its size.
    """
    state = game.state
    board = state.board
    colors = state.colors
    assert len(colors) == 2, "the JAX environment is 1v1 only"
    ps = state.player_state
    catan_map = board.map

    tile_resource = np.zeros(topo.num_tiles, np.int32)
    tile_number = np.zeros(topo.num_tiles, np.int32)
    for i, coordinate in enumerate(topo.tile_coordinates):
        tile = catan_map.land_tiles[coordinate]
        tile_resource[i] = DESERT if tile.resource is None else _RESOURCE_INDEX[tile.resource]
        tile_number[i] = tile.number or 0

    port_resource = np.zeros(topo.num_ports, np.int32)
    for i in range(topo.num_ports):
        resource = catan_map.ports_by_id[i].resource
        port_resource[i] = DESERT if resource is None else _RESOURCE_INDEX[resource]

    node_owner = np.full(topo.num_nodes, -1, np.int32)
    node_level = np.zeros(topo.num_nodes, np.int32)
    for node_id, (color, kind) in board.buildings.items():
        node_owner[node_id] = state.color_to_index[color]
        node_level[node_id] = 1 if kind == SETTLEMENT else 2

    edge_owner = np.full(topo.num_edges, -1, np.int32)
    for i, (a, b) in enumerate(topo.edges.tolist()):
        color = board.roads.get((a, b))
        if color is not None:
            edge_owner[i] = state.color_to_index[color]

    reach = np.zeros((2, topo.num_nodes), bool)
    for seat, color in enumerate(colors):
        for component in board.connected_components[color]:
            reach[seat, list(component)] = True

    def per_seat(key):
        return np.array([ps[f"P{seat}_{key}"] for seat in range(2)], np.int32)

    hand = np.stack([per_seat(f"{r}_IN_HAND") for r in RESOURCES], axis=1)
    dev_hand = np.stack([per_seat(f"{d}_IN_HAND") for d in DEVELOPMENT_CARDS], axis=1)
    dev_played = np.zeros((2, NUM_DEV_TYPES), np.int32)
    dev_owned = np.zeros((2, NUM_DEV_TYPES), bool)
    for i, dev in enumerate(DEVELOPMENT_CARDS[:4]):
        dev_played[:, i] = per_seat(f"PLAYED_{dev}")
        dev_owned[:, i] = per_seat(f"{dev}_OWNED_AT_START").astype(bool)

    # Python pops the deck from the end; JAX draws upward from dev_drawn.
    remaining = [DEVELOPMENT_CARDS.index(c) for c in reversed(state.development_listdeck)]
    drawn = len(DEV_DECK) - len(remaining)
    composition = np.bincount(DEV_DECK, minlength=NUM_DEV_TYPES) - np.bincount(
        remaining, minlength=NUM_DEV_TYPES
    )
    played_out = np.repeat(np.arange(NUM_DEV_TYPES), composition)
    dev_deck = np.concatenate([played_out, remaining]).astype(np.int32)

    last_settlement = np.full(2, -1, np.int32)
    for seat, color in enumerate(colors):
        settlements = state.buildings_by_color[color][SETTLEMENT]
        if settlements:
            last_settlement[seat] = settlements[-1]

    def seat_of(color):
        return -1 if color is None else state.color_to_index[color]

    def largest_army_seat():
        holders = np.flatnonzero(per_seat("HAS_ARMY"))
        return holders[0] if len(holders) else -1

    road_length = np.array(
        [board.road_lengths.get(color, 0) for color in colors], np.int32
    )
    winning = game.winning_color()

    out = State(
        tile_resource=tile_resource,
        tile_number=tile_number,
        port_resource=port_resource,
        node_owner=node_owner,
        node_level=node_level,
        edge_owner=edge_owner,
        robber_tile=np.array(topo.tile_coordinates.index(board.robber_coordinate), np.int32),
        reach=reach,
        hand=hand,
        bank=np.array(state.resource_freqdeck, np.int32),
        dev_hand=dev_hand,
        dev_owned=dev_owned,
        dev_played=dev_played,
        dev_deck=dev_deck,
        dev_drawn=np.array(drawn, np.int32),
        vp_public=per_seat("VICTORY_POINTS"),
        vp_actual=per_seat("ACTUAL_VICTORY_POINTS"),
        roads_left=per_seat("ROADS_AVAILABLE"),
        settlements_left=per_seat("SETTLEMENTS_AVAILABLE"),
        cities_left=per_seat("CITIES_AVAILABLE"),
        last_settlement=last_settlement,
        road_length=road_length,
        road_holder=np.array(seat_of(board.road_color), np.int32),
        road_best=np.array(board.road_length, np.int32),
        army_holder=np.array(largest_army_seat(), np.int32),
        current=np.array(state.current_player_index, np.int32),
        turn_player=np.array(state.current_turn_index, np.int32),
        prompt=np.array(_PROMPTS[state.current_prompt], np.int32),
        has_rolled=per_seat("HAS_ROLLED").astype(bool),
        played_dev=per_seat("HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN").astype(bool),
        road_building=np.array(state.is_road_building, bool),
        free_roads=np.array(state.free_roads_available, np.int32),
        discard_counts=np.array(state.discard_counts, np.int32),
        num_turns=np.array(state.num_turns, np.int32),
        num_plies=np.array(len(state.action_records), np.int32),
        road_eval=np.array(NO_ROAD_EVAL, np.int32),
        road_eval_seat=np.array(0, np.int32),
        winner=np.array(seat_of(winning), np.int32),
        done=np.array(winning is not None or state.num_turns >= TURNS_LIMIT, bool),
    )
    if batch_dim:
        out = State(*[jnp.asarray(x)[None] for x in out])
    return out
