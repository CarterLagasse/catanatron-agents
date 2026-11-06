"""Static board tables and the action layout used by the JAX environment.

The tables are derived from the Python engine's map templates, so node, edge and
tile ids always agree with `catanatron.models.map`. Everything here is plain
NumPy computed once; the jitted code only ever closes over the resulting arrays.

Action space. It is ego-centric, so one network can play any seat:

    ROLL, END_TURN, BUY_DEV, PLAY_KNIGHT, PLAY_ROAD_BUILDING
    DISCARD[resource]
    ROAD[edge], SETTLEMENT[node], CITY[node]
    YEAR_OF_PLENTY[15 pairs, then 5 single cards], MONOPOLY[resource]
    ROBBER[tile, no-steal | steal from the opponent 1..P-1 seats after the mover]
    MARITIME[give, get]: the rate is always the player's best one, since the
    Python move generator only ever offers that rate.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from catanatron.models.board import get_edges
from catanatron.models.enums import RESOURCES
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    BASE_NUMBERS_IN_SPIRAL_ORDER,
    MINI_MAP_TEMPLATE,
    PORT_DIRECTION_TO_NODEREFS,
    TOURNAMENT_MAP,
    build_map,
    initialize_tiles,
)
from catanatron.models.spiral import spiral_land_coordinates
from catanatron.models.tiles import LandTile, Port

NUM_RESOURCES = 5
DESERT = NUM_RESOURCES  # tile_resource value for the desert
ANY_PORT = NUM_RESOURCES  # port_resource value for a 3:1 port

KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT = range(5)
NUM_DEV_TYPES = 5
# Same composition as catanatron.models.decks.starting_devcard_bank
DEV_DECK = np.array(
    [KNIGHT] * 14
    + [YEAR_OF_PLENTY] * 2
    + [ROAD_BUILDING] * 2
    + [MONOPOLY] * 2
    + [VICTORY_POINT] * 5,
    dtype=np.int32,
)

YOP_PAIRS = np.array(
    [(i, j) for i in range(NUM_RESOURCES) for j in range(i, NUM_RESOURCES)],
    dtype=np.int32,
)
MARITIME_PAIRS = np.array(
    [(i, j) for i in range(NUM_RESOURCES) for j in range(NUM_RESOURCES) if i != j],
    dtype=np.int32,
)


def _resource_index(resource):
    return DESERT if resource is None else RESOURCES.index(resource)


def dice_probability(number):
    return 0.0 if number == 0 else (6 - abs(7 - number)) / 36.0


@dataclass(frozen=True)
class Topology:
    map_type: str
    num_nodes: int
    num_edges: int
    num_tiles: int
    num_ports: int
    edges: np.ndarray  # [E, 2] node ids, a < b
    node_neighbors: np.ndarray  # [N, 3] node id, -1 padded
    node_edges: np.ndarray  # [N, 3] edge index, -1 padded
    node_tiles: np.ndarray  # [N, 3] tile index, -1 padded
    tile_nodes: np.ndarray  # [T, 6] node ids
    port_nodes: np.ndarray  # [P, 2] node ids
    tile_coordinates: tuple
    spiral_order: np.ndarray  # [T] tile index in number-placement order
    spiral_numbers: np.ndarray  # numbers placed along the spiral, deserts skipped
    tile_resource_pool: np.ndarray  # [T] resources shuffled for random maps
    port_resource_pool: np.ndarray  # [P] port types shuffled for random maps
    fixed_tile_resource: np.ndarray | None  # set for fixed maps (TOURNAMENT)
    fixed_tile_number: np.ndarray | None
    fixed_port_resource: np.ndarray | None
    num_players: int = 2

    # Action layout
    ROLL: int = 0
    END_TURN: int = 1
    BUY_DEV: int = 2
    PLAY_KNIGHT: int = 3
    PLAY_ROAD_BUILDING: int = 4
    DISCARD: int = 5

    @property
    def ROAD(self):
        return self.DISCARD + NUM_RESOURCES

    @property
    def SETTLEMENT(self):
        return self.ROAD + self.num_edges

    @property
    def CITY(self):
        return self.SETTLEMENT + self.num_nodes

    @property
    def YOP(self):
        return self.CITY + self.num_nodes

    @property
    def MONOPOLY(self):
        return self.YOP + len(YOP_PAIRS) + NUM_RESOURCES

    @property
    def ROBBER(self):
        return self.MONOPOLY + NUM_RESOURCES

    @property
    def MARITIME(self):
        return self.ROBBER + self.num_players * self.num_tiles

    @property
    def num_actions(self):
        return self.MARITIME + len(MARITIME_PAIRS)


def _pad(rows, width, fill=-1):
    out = np.full((len(rows), width), fill, dtype=np.int32)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def _spiral_tile_order(template, tiles):
    start = (2, -2, 0) if template is BASE_MAP_TEMPLATE else (1, -1, 0)
    return [tiles[c].id for c in spiral_land_coordinates(tiles, start)]


def _template_port_order(tiles):
    return [t for t in tiles.values() if isinstance(t, Port)]


@lru_cache(maxsize=None)
def get_topology(map_type="BASE", num_players=2):
    """Builds tables for "BASE", "TOURNAMENT" or "MINI" with 2 to 4 players."""
    assert 2 <= num_players <= 4, "the engine supports 2 to 4 players"
    template = MINI_MAP_TEMPLATE if map_type == "MINI" else BASE_MAP_TEMPLATE
    tiles = initialize_tiles(template)
    land = [t for t in tiles.values() if isinstance(t, LandTile)]
    land_coordinates = tuple(c for c, t in tiles.items() if isinstance(t, LandTile))
    land_nodes = build_map(map_type).land_nodes

    edges = sorted(tuple(sorted(edge)) for edge in get_edges(land_nodes))
    num_nodes = len(land_nodes)
    assert sorted(land_nodes) == list(range(num_nodes)), "node ids must be contiguous"

    neighbor_rows = [[] for _ in range(num_nodes)]
    edge_rows = [[] for _ in range(num_nodes)]
    for index, (a, b) in enumerate(edges):
        neighbor_rows[a].append(b)
        neighbor_rows[b].append(a)
        edge_rows[a].append(index)
        edge_rows[b].append(index)

    tile_rows = [[] for _ in range(num_nodes)]
    for tile in land:
        for node_id in tile.nodes.values():
            tile_rows[node_id].append(tile.id)

    ports = _template_port_order(tiles)
    port_nodes = np.array(
        [
            [port.nodes[ref] for ref in PORT_DIRECTION_TO_NODEREFS[port.direction]]
            for port in ports
        ],
        dtype=np.int32,
    ).reshape(-1, 2)

    spiral = _spiral_tile_order(template, tiles)
    num_numbers = len(template.numbers)

    fixed_resource = fixed_number = fixed_port = None
    if map_type == "TOURNAMENT":
        by_id = TOURNAMENT_MAP.tiles_by_id
        fixed_resource = np.array(
            [_resource_index(by_id[i].resource) for i in range(len(land))],
            dtype=np.int32,
        )
        fixed_number = np.array(
            [by_id[i].number or 0 for i in range(len(land))], dtype=np.int32
        )
        tournament_ports = TOURNAMENT_MAP.ports_by_id
        fixed_port = np.array(
            [_resource_index(tournament_ports[i].resource) for i in range(len(ports))],
            dtype=np.int32,
        )

    return Topology(
        map_type=map_type,
        num_nodes=num_nodes,
        num_edges=len(edges),
        num_tiles=len(land),
        num_ports=len(ports),
        edges=np.array(edges, dtype=np.int32),
        node_neighbors=_pad(neighbor_rows, 3),
        node_edges=_pad(edge_rows, 3),
        node_tiles=_pad(tile_rows, 3),
        tile_nodes=np.array([list(t.nodes.values()) for t in land], dtype=np.int32),
        port_nodes=port_nodes,
        tile_coordinates=land_coordinates,
        spiral_order=np.array(spiral, dtype=np.int32),
        spiral_numbers=np.array(
            BASE_NUMBERS_IN_SPIRAL_ORDER[:num_numbers], dtype=np.int32
        ),
        tile_resource_pool=np.array(
            [_resource_index(r) for r in template.tile_resources], dtype=np.int32
        ),
        port_resource_pool=np.array(
            [_resource_index(r) for r in template.port_resources], dtype=np.int32
        ),
        fixed_tile_resource=fixed_resource,
        fixed_tile_number=fixed_number,
        fixed_port_resource=fixed_port,
        num_players=num_players,
    )
