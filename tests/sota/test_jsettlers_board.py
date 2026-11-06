import random

import pytest

from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap
from catanatron.sota.jsettlers.board import (
    HEX_IDS,
    RESOURCE_BY_ID,
    BoardMapping,
    EmptyBoardError,
    hex_to_cube,
)

# Classic board constants from JSettlers' soc.game.SOCBoard4p
LAND_INDEXES = [5, 6, 7, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 29, 30, 31]
PORT_INDEXES = [0, 2, 8, 21, 32, 35, 33, 22, 9]
PORT_FACINGS = [3, 4, 4, 5, 6, 6, 1, 2, 2]
PORT_EDGES = [0x27, 0x5A, 0x9C, 0xCC, 0xC9, 0xA5, 0x72, 0x42, 0x24]


def random_jsettlers_layout(rng):
    """A layout as sent in BOARDLAYOUT: water 6, desert 0, ports in water hexes."""
    resources = [1] * 3 + [2] * 3 + [3] * 4 + [4] * 4 + [5] * 4 + [0]
    numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
    ports = [0, 0, 0, 0, 1, 2, 3, 4, 5]
    for values in (resources, numbers, ports):
        rng.shuffle(values)
    hexes, dice = [6] * 37, [-1] * 37
    numbers_iter = iter(numbers)
    for index, resource in zip(LAND_INDEXES, resources):
        hexes[index] = resource
        dice[index] = 0 if resource == 0 else next(numbers_iter)
    for index, facing, port in zip(PORT_INDEXES, PORT_FACINGS, ports):
        hexes[index] = 6 + facing if port == 0 else (facing << 4) | port
    return hexes, dice, ports


def edge_nodes(edge):
    """soc.game.SOCBoard.getAdjacentNodesToEdge_arr"""
    if ((edge & 0x0F) + (edge >> 4)) % 2 == 0:
        return edge + 0x01, edge + 0x10
    return edge, edge + 0x11


@pytest.mark.parametrize("seed", range(20))
def test_mapping_matches_jsettlers_board(seed):
    hexes, dice, ports = random_jsettlers_layout(random.Random(seed))
    mapping = BoardMapping(hexes, dice)
    catan_map = mapping.catan_map

    for index in LAND_INDEXES:
        tile = catan_map.land_tiles[hex_to_cube(HEX_IDS[index])]
        assert tile.resource == RESOURCE_BY_ID.get(hexes[index])
        assert (tile.number or 0) == dice[index]

    # Ports land on the same two nodes JSettlers puts them on.
    for edge, port in zip(PORT_EDGES, ports):
        a, b = edge_nodes(edge)
        nodes = {mapping.node_to_id[a], mapping.node_to_id[b]}
        resource = None if port == 0 else RESOURCE_BY_ID[port]
        assert nodes <= catan_map.port_nodes[resource]
        assert mapping.edge_id(edge) == tuple(sorted(nodes))
        assert mapping.js_edge(tuple(sorted(nodes))) == edge


def test_node_and_edge_ids_match_base_map():
    hexes, dice, _ = random_jsettlers_layout(random.Random(0))
    mapping = BoardMapping(hexes, dice)
    base = CatanMap.from_template(BASE_MAP_TEMPLATE)
    assert mapping.catan_map.land_nodes == base.land_nodes
    for coordinate, tile in base.land_tiles.items():
        assert mapping.catan_map.land_tiles[coordinate].nodes == tile.nodes
    assert len(mapping.edge_to_id) == 72


def test_empty_layout_is_rejected():
    with pytest.raises(EmptyBoardError):
        BoardMapping([6] * 37, [-1] * 37)
