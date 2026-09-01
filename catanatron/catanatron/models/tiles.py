from dataclasses import dataclass
from typing import Union

from catanatron.models.coordinate_system import Direction
from catanatron.models.enums import EdgeRef, FastResource, NodeRef

EdgeId = tuple[int, int]
NodeId = int


@dataclass
class LandTile:
    id: int
    resource: FastResource | None  # None means desert tile
    number: int | None  # None if desert
    nodes: dict[NodeRef, NodeId]  # node_ref => node_id
    edges: dict[EdgeRef, EdgeId]  # edge_ref => edge

    # The id is unique among the tiles, so we can use it as the hash.
    def __hash__(self):
        return self.id


@dataclass
class Port:
    id: int
    resource: FastResource | None  # None means desert tile
    direction: Direction
    nodes: dict[NodeRef, NodeId]  # node_ref => node_id
    edges: dict[EdgeRef, EdgeId]  # edge_ref => edge

    # The id is unique among the tiles, so we can use it as the hash.
    def __hash__(self):
        return self.id


@dataclass(frozen=True)
class Water:
    nodes: dict[NodeRef, int]
    edges: dict[EdgeRef, EdgeId]


Tile = Union[LandTile, Port, Water]
