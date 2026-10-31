"""Ego-centric observations for the graph network.

Everything is from `seat`'s point of view and exposes only what that player can
legally know: the opponent's hand and dev cards appear as counts, and the deck as
its size. Any seat can be encoded at any time, which lets the value of a position
be estimated for a player who is not the one to move.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from catanatron.jax_rl.env import NUM_PROMPTS
from catanatron.jax_rl.topology import (
    ANY_PORT,
    DEV_DECK,
    KNIGHT,
    NUM_DEV_TYPES,
    NUM_RESOURCES,
)

# Index by dice number, in units of probability of being rolled per turn.
DICE_PROBABILITY = jnp.array(
    [0.0, 0.0] + [(6 - abs(7 - n)) / 36.0 for n in range(2, 13)], jnp.float32
)


class Obs(NamedTuple):
    nodes: jnp.ndarray  # [N, NODE_FEATURES]
    edges: jnp.ndarray  # [E, EDGE_FEATURES]
    tiles: jnp.ndarray  # [T, TILE_FEATURES]
    globals: jnp.ndarray  # [GLOBAL_FEATURES]


NODE_FEATURES = 7 + NUM_RESOURCES + 1 + NUM_RESOURCES + 1
EDGE_FEATURES = 3
TILE_FEATURES = NUM_RESOURCES + 1 + 3
GLOBAL_FEATURES = (
    NUM_PROMPTS + 7 + NUM_RESOURCES + 1 + NUM_DEV_TYPES + (NUM_DEV_TYPES - 1) + 2
    + 18 + NUM_RESOURCES + 1 + NUM_RESOURCES + 2
)


def encode(env, s, seat):
    """Observation of a single game, from `seat`'s perspective."""
    other = 1 - seat
    node_owner, node_level = s.node_owner, s.node_level

    # ----- Tiles
    tile_probability = DICE_PROBABILITY[s.tile_number]
    tile_kind = jax.nn.one_hot(s.tile_resource, NUM_RESOURCES + 1, dtype=jnp.float32)
    is_robber = (jnp.arange(env.topo.num_tiles) == s.robber_tile).astype(jnp.float32)
    robber_on_me = (node_owner[env.tile_nodes] == seat).any(axis=1).astype(jnp.float32)
    tiles = jnp.concatenate(
        [
            tile_kind,
            tile_probability[:, None] * 6.0,
            is_robber[:, None],
            robber_on_me[:, None],
        ],
        axis=1,
    )

    # ----- Nodes
    adjacent = env.node_tiles
    pad_kind = jnp.concatenate([tile_kind, jnp.zeros((1, NUM_RESOURCES + 1))], axis=0)
    pad_probability = jnp.concatenate([tile_probability, jnp.zeros(1)])
    adjacent_pad = jnp.where(adjacent < 0, env.topo.num_tiles, adjacent)
    node_production = (
        pad_kind[adjacent_pad][..., :NUM_RESOURCES]
        * pad_probability[adjacent_pad][..., None]
    ).sum(axis=1)  # [N, 5]

    port_one_hot = jax.nn.one_hot(s.port_resource, ANY_PORT + 1, dtype=jnp.float32)
    node_port = jnp.zeros((env.topo.num_nodes, ANY_PORT + 1), jnp.float32)
    if env.topo.num_ports > 0:
        node_port = (
            node_port.at[env.port_nodes[:, 0]]
            .add(port_one_hot)
            .at[env.port_nodes[:, 1]]
            .add(port_one_hot)
        )

    me_here = node_owner == seat
    them_here = node_owner == other
    nodes = jnp.concatenate(
        [
            jnp.stack(
                [
                    me_here & (node_level == 1),
                    me_here & (node_level == 2),
                    them_here & (node_level == 1),
                    them_here & (node_level == 2),
                    env._free_nodes(node_owner),
                    s.reach[seat],
                    s.reach[other],
                ],
                axis=1,
            ).astype(jnp.float32),
            node_production * 6.0,
            node_production.sum(axis=1, keepdims=True) * 2.0,
            node_port,
        ],
        axis=1,
    )

    # ----- Edges
    edges = jnp.stack(
        [
            s.edge_owner == seat,
            s.edge_owner == other,
            env._buildable_edges(s.edge_owner, s.reach[seat]),
        ],
        axis=1,
    ).astype(jnp.float32)

    # ----- Globals
    hand = s.hand[seat].astype(jnp.float32)
    production_of = lambda who: (
        (node_production * (node_level * (node_owner == who))[:, None]).sum(axis=0)
    )
    my_production = production_of(seat)
    their_total_production = production_of(other).sum()
    rates = env.trade_rates(s, seat).astype(jnp.float32)

    globals_ = jnp.concatenate(
        [
            jax.nn.one_hot(s.prompt, NUM_PROMPTS, dtype=jnp.float32),
            jnp.stack(
                [
                    (s.current == seat).astype(jnp.float32),
                    (s.turn_player == seat).astype(jnp.float32),
                    s.has_rolled[s.turn_player].astype(jnp.float32),
                    s.played_dev[seat].astype(jnp.float32),
                    s.road_building.astype(jnp.float32),
                    s.free_roads / 2.0,
                    s.discard_counts[seat] / 5.0,
                ]
            ),
            hand / 5.0,
            hand.sum(keepdims=True) / 10.0,
            s.dev_hand[seat] / 3.0,
            env.can_play_dev(s, seat)[: NUM_DEV_TYPES - 1].astype(jnp.float32),
            jnp.stack(
                [
                    s.hand[other].sum() / 10.0,
                    s.dev_hand[other].sum() / 3.0,
                ]
            ),
            jnp.stack(
                [
                    s.vp_actual[seat] / 10.0,
                    s.vp_public[seat] / 10.0,
                    s.vp_public[other] / 10.0,
                    (s.vp_public[seat] - s.vp_public[other]) / 5.0,
                    s.dev_played[seat, KNIGHT] / 3.0,
                    s.dev_played[other, KNIGHT] / 3.0,
                    s.road_length[seat] / 5.0,
                    s.road_length[other] / 5.0,
                    (s.road_holder == seat).astype(jnp.float32),
                    (s.road_holder == other).astype(jnp.float32),
                    (s.army_holder == seat).astype(jnp.float32),
                    (s.army_holder == other).astype(jnp.float32),
                    s.roads_left[seat] / 15.0,
                    s.settlements_left[seat] / 5.0,
                    s.cities_left[seat] / 4.0,
                    s.roads_left[other] / 15.0,
                    s.settlements_left[other] / 5.0,
                    s.cities_left[other] / 4.0,
                ]
            ).astype(jnp.float32),
            s.bank / 19.0,
            (1.0 - s.dev_drawn / len(DEV_DECK))[None],
            (4.0 - rates) / 2.0,
            jnp.stack(
                [
                    my_production.sum() * 2.0,
                    their_total_production * 2.0,
                ]
            ),
        ]
    ).astype(jnp.float32)

    return Obs(nodes=nodes, edges=edges, tiles=tiles, globals=globals_)


def encode_current(env, s):
    return encode(env, s, s.current)
