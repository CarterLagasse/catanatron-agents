"""Maps the classic JSettlers board (v1 "original" encoding) onto catanatron.

JSettlers hex, node and edge coordinates are bytes 0xRC on two diagonal axes
(see soc.game.SOCBoard). Both games use pointy-top hexes, so the mapping is a
change of axes: moving NE adds 0x02 to a JSettlers hex coordinate and moving
SE adds 0x20, which are catanatron's (1, 0, -1) and (0, -1, 1) cube vectors.

The catanatron map is built over BASE_MAP_TEMPLATE's coordinates in the same
order, so node and edge ids are the same as in any normal catanatron game.
"""

from catanatron.models.coordinate_system import Direction
from catanatron.models.enums import BRICK, ORE, SHEEP, WHEAT, WOOD, NodeRef
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    CatanMap,
    LandTile,
    MapTemplate,
    Port,
    Water,
    initialize_tiles,
)

# soc.game.SOCBoard.numToHexID: hex coordinate of each hexLayout index (rows N to S)
HEX_IDS = [
    0x17, 0x39, 0x5B, 0x7D,
    0x15, 0x37, 0x59, 0x7B, 0x9D,
    0x13, 0x35, 0x57, 0x79, 0x9B, 0xBD,
    0x11, 0x33, 0x55, 0x77, 0x99, 0xBB, 0xDD,
    0x31, 0x53, 0x75, 0x97, 0xB9, 0xDB,
    0x51, 0x73, 0x95, 0xB7, 0xD9,
    0x71, 0x93, 0xB5, 0xD7,
]  # fmt: skip

# soc.game.SOCBoard.HEXNODES: node offsets from a hex, clockwise from north
HEX_NODE_OFFSETS = {
    NodeRef.NORTH: 0x01,
    NodeRef.NORTHEAST: 0x12,
    NodeRef.SOUTHEAST: 0x21,
    NodeRef.SOUTH: 0x10,
    NodeRef.SOUTHWEST: -0x01,
    NodeRef.NORTHWEST: -0x10,
}

# Resource ids (soc.game.SOCResourceConstants) <-> catanatron resources
RESOURCE_BY_ID = {1: BRICK, 2: ORE, 3: SHEEP, 4: WHEAT, 5: WOOD}
ID_BY_RESOURCE = {r: i for i, r in RESOURCE_BY_ID.items()}

# Port facing (SOCBoard.FACING_*): the side of the port hex that touches land
FACING_TO_DIRECTION = {
    1: Direction.NORTHEAST,
    2: Direction.EAST,
    3: Direction.SOUTHEAST,
    4: Direction.SOUTHWEST,
    5: Direction.WEST,
    6: Direction.NORTHWEST,
}

# Hex type values as sent in BOARDLAYOUT(2) (water and desert are remapped on the wire)
SENT_DESERT = 0
SENT_WATER = 6
MISC_PORT_MIN, MISC_PORT_MAX = 7, 12

# Dice numbers as sent in BOARDLAYOUT (v1 message only): index -> number
V1_SENT_NUMBERS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]


def hex_to_cube(hex_id: int):
    dr = (hex_id >> 4) - 7
    dc = (hex_id & 0x0F) - 7
    a, b = dc // 2, dr // 2  # steps NE, steps SE
    return (a, -b, b - a)


def edge_between(node_a: int, node_b: int) -> int:
    """Inverse of SOCBoard.getAdjacentNodesToEdge_arr."""
    lo, hi = sorted((node_a, node_b))
    if hi - lo == 0x11:
        return lo
    if hi - lo == 0x0F:
        return lo - 0x01
    raise ValueError(f"Nodes {lo:#x} and {hi:#x} aren't adjacent")


def decode_hex(value: int):
    """Returns ("land", resource|None) | ("water", None) | ("port", (resource|None, facing))."""
    if value == SENT_WATER:
        return "water", None
    if value == SENT_DESERT:
        return "land", None
    if 1 <= value <= 5:
        return "land", RESOURCE_BY_ID[value]
    if MISC_PORT_MIN <= value <= MISC_PORT_MAX:
        return "port", (None, value - 6)
    if value >= 16:
        return "port", (RESOURCE_BY_ID[value & 0x0F], value >> 4)
    raise ValueError(f"Unknown JSettlers hex value {value}")


class EmptyBoardError(ValueError):
    pass


class BoardMapping:
    """A catanatron map equivalent to a JSettlers board, plus coordinate maps."""

    def __init__(self, hex_layout: list[int], number_layout: list[int]):
        """hex_layout uses the sent encoding; number_layout holds dice numbers (0/-1 = none)."""
        if len(hex_layout) != len(HEX_IDS) or len(number_layout) != len(HEX_IDS):
            raise ValueError("Only the classic 4-player board is supported")
        if all(value == SENT_WATER for value in hex_layout):
            raise EmptyBoardError("Board not laid out yet")

        by_coordinate = {}
        for index, hex_id in enumerate(HEX_IDS):
            by_coordinate[hex_to_cube(hex_id)] = (
                hex_id,
                hex_layout[index],
                number_layout[index],
            )

        topology = {}
        tile_resources, numbers, port_resources = [], [], []
        for coordinate in BASE_MAP_TEMPLATE.topology:
            _, value, number = by_coordinate[coordinate]
            kind, data = decode_hex(value)
            if kind == "land":
                topology[coordinate] = LandTile
                tile_resources.append(data)
                if data is not None:
                    if not 2 <= number <= 12:
                        raise ValueError(
                            f"Land hex at {coordinate} has number {number}"
                        )
                    numbers.append(number)
            elif kind == "water":
                topology[coordinate] = Water
            else:
                resource, facing = data
                topology[coordinate] = (Port, FACING_TO_DIRECTION[facing])
                port_resources.append(resource)

        # initialize_tiles pops from the end of each list as it walks the topology.
        template = MapTemplate(numbers, port_resources, tile_resources, topology)
        tiles = initialize_tiles(
            template,
            shuffled_numbers_param=list(reversed(numbers)),
            shuffled_port_resources_param=list(reversed(port_resources)),
            shuffled_tile_resources_param=list(reversed(tile_resources)),
            number_placement="random",
        )
        self.catan_map = CatanMap.from_tiles(tiles)

        self.hex_to_coordinate = {h: hex_to_cube(h) for h in HEX_IDS}
        self.coordinate_to_hex = {c: h for h, c in self.hex_to_coordinate.items()}

        self.node_to_id = {}
        for hex_id in HEX_IDS:
            tile = tiles[hex_to_cube(hex_id)]
            for noderef, offset in HEX_NODE_OFFSETS.items():
                js_node, node_id = hex_id + offset, tile.nodes[noderef]
                if self.node_to_id.setdefault(js_node, node_id) != node_id:
                    raise AssertionError(f"Inconsistent node mapping at {js_node:#x}")
        self.id_to_node = {n: js for js, n in self.node_to_id.items()}

        self.edge_to_id = {}
        for tile in self.catan_map.land_tiles.values():
            for a, b in tile.edges.values():
                js_edge = edge_between(self.id_to_node[a], self.id_to_node[b])
                self.edge_to_id[js_edge] = tuple(sorted((a, b)))
        self.id_to_edge = {e: js for js, e in self.edge_to_id.items()}

        for port_coordinate, port in self._ports(topology).items():
            land = tiles.get(_step(port_coordinate, port))
            if not isinstance(land, LandTile):
                raise AssertionError(f"Port at {port_coordinate} doesn't face land")

    @staticmethod
    def _ports(topology):
        return {
            c: d for c, t in topology.items() if isinstance(t, tuple) for _, d in [t]
        }

    def edge_id(self, js_edge: int):
        return self.edge_to_id[js_edge]

    def js_edge(self, edge) -> int:
        return self.id_to_edge[tuple(sorted(edge))]


def _step(coordinate, direction):
    from catanatron.models.coordinate_system import UNIT_VECTORS, add

    return add(coordinate, UNIT_VECTORS[direction])


def mapping_from_layout_message(fields: list[str], v2: bool):
    """Parses BOARDLAYOUT (v1) or BOARDLAYOUT2 fields (after the game name).

    Returns (BoardMapping, robber_hex).
    """
    if not v2:
        values = [int(f) for f in fields]
        hexes, sent_numbers, robber = values[:37], values[37:74], values[74]
        numbers = [V1_SENT_NUMBERS[n] if n >= 0 else 0 for n in sent_numbers]
        return BoardMapping(hexes, numbers), robber

    encoding = int(fields[0])
    if encoding != 1:
        raise ValueError(
            f"Board encoding v{encoding} isn't supported (classic 4-player only)"
        )
    parts, i = {}, 1
    while i < len(fields):
        key, value = fields[i], fields[i + 1]
        if value.startswith("["):
            count = int(value[1:])
            parts[key] = [int(v) for v in fields[i + 2 : i + 2 + count]]
            i += 2 + count
        else:
            parts[key] = int(value)
            i += 2
    return BoardMapping(parts["HL"], parts["NL"]), parts["RH"]
