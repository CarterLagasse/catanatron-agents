"""Scripted opponents as pure JAX functions from (state, mask) to action preferences.

They give training something meaningful to beat before self-play bootstraps, and a
cheap in-training benchmark. Each returns unmasked logits for one game.
"""

import jax
import jax.numpy as jnp

from catanatron.jax_rl.env import (
    BUILD_INITIAL_ROAD,
    BUILD_INITIAL_SETTLEMENT,
    CITY_COST,
    SETTLEMENT_COST,
)
from catanatron.jax_rl.obs import DICE_PROBABILITY
from catanatron.jax_rl.topology import NUM_RESOURCES

RANDOM, WEIGHTED, HEURISTIC = 0, 1, 2
SCRIPTED_NAMES = {"random": RANDOM, "weighted": WEIGHTED, "heuristic": HEURISTIC}


def random_logits(env, s):
    return jnp.zeros(env.num_actions, jnp.float32)


def weighted_logits(env, s):
    """catanatron.players.weighted_random: cities > settlements > dev cards."""
    topo = env.topo
    logits = jnp.zeros(env.num_actions, jnp.float32)
    logits = logits.at[topo.CITY : topo.CITY + topo.num_nodes].set(jnp.log(10000.0))
    logits = logits.at[topo.SETTLEMENT : topo.SETTLEMENT + topo.num_nodes].set(
        jnp.log(1000.0)
    )
    return logits.at[topo.BUY_DEV].set(jnp.log(100.0))


def _node_values(env, s):
    """Per-node production value with a bonus for resource variety."""
    probability = DICE_PROBABILITY[s.tile_number]
    adjacent = env.node_tiles
    valid = adjacent >= 0
    tile_probability = jnp.where(valid, probability[jnp.maximum(adjacent, 0)], 0.0)
    resources = jnp.where(valid, s.tile_resource[jnp.maximum(adjacent, 0)], 5)
    production = tile_probability.sum(axis=1)
    kinds = jax.nn.one_hot(resources, NUM_RESOURCES + 1)[..., :NUM_RESOURCES]
    variety = (kinds.sum(axis=1) > 0).sum(axis=1)
    return production + 0.03 * variety


def heuristic_logits(env, s):
    """Priority rules: good placements, cities over settlements over dev cards,
    purposeful bank trades, and a robber aimed at the opponent's production."""
    topo = env.topo
    seat = s.current
    other = 1 - seat
    hand = s.hand[seat]
    logits = jnp.zeros(env.num_actions, jnp.float32)

    node_value = _node_values(env, s)
    free_nodes = env._free_nodes(s.node_owner)
    open_value = node_value * free_nodes

    # An edge is worth the best open spot it leads toward, up to two steps away.
    neighbor_best = jnp.where(
        env.node_neighbors >= 0,
        open_value[jnp.maximum(env.node_neighbors, 0)],
        0.0,
    ).max(axis=1)
    node_prospect = jnp.maximum(open_value, 0.6 * neighbor_best)
    edge_prospect = jnp.maximum(
        node_prospect[env.edges[:, 0]] * ~s.reach[seat][env.edges[:, 0]],
        node_prospect[env.edges[:, 1]] * ~s.reach[seat][env.edges[:, 1]],
    )

    initial_road = s.prompt == BUILD_INITIAL_ROAD
    initial_settle = s.prompt == BUILD_INITIAL_SETTLEMENT

    # Building scores. Larger means earlier within a turn.
    settlement_score = jnp.where(initial_settle, 40.0 * node_value, 90.0 + node_value)
    city_score = 100.0 + 100.0 * jnp.where(s.node_owner == seat, node_value, 0.0)
    can_expand = (env._free_nodes(s.node_owner) & s.reach[seat]).any()
    road_base = jnp.where(initial_road, 0.0, jnp.where(can_expand, 20.0, 60.0))
    road_score = road_base + 40.0 * edge_prospect
    logits = logits.at[topo.SETTLEMENT : topo.SETTLEMENT + topo.num_nodes].set(
        settlement_score
    )
    logits = logits.at[topo.CITY : topo.CITY + topo.num_nodes].set(city_score)
    logits = logits.at[topo.ROAD : topo.ROAD + topo.num_edges].set(road_score)
    logits = logits.at[topo.BUY_DEV].set(70.0)
    logits = logits.at[topo.ROLL].set(50.0)
    logits = logits.at[topo.END_TURN].set(1.0)

    # Knights when the robber sits on us, or to contest the army; other dev cards
    # whenever available.
    robber_nodes = env.tile_nodes[s.robber_tile]
    robbed = (s.node_owner[robber_nodes] == seat).any()
    knight_score = jnp.where(robbed | (s.dev_played[seat, 0] >= 1), 80.0, 30.0)
    logits = logits.at[topo.PLAY_KNIGHT].set(knight_score)
    logits = logits.at[topo.PLAY_ROAD_BUILDING].set(65.0)

    # Target what a settlement or city is still missing.
    needs_city = jnp.maximum(CITY_COST - hand, 0)
    needs_settlement = jnp.maximum(SETTLEMENT_COST - hand, 0)
    goal_need = needs_city + needs_settlement
    keep = jnp.maximum(CITY_COST, SETTLEMENT_COST)

    yop_cards = env.yop_cards
    yop_score = 75.0 + 3.0 * (jnp.minimum(yop_cards, goal_need)).sum(axis=1)
    logits = logits.at[topo.YOP : topo.MONOPOLY].set(yop_score)
    # Monopoly on the resource the opponent could hold most of, guessed from
    # their production.
    opponent_levels = (s.node_owner == other) * s.node_level
    tile_levels = opponent_levels[env.tile_nodes].sum(axis=1)
    kinds = jax.nn.one_hot(s.tile_resource, NUM_RESOURCES + 1)[:, :NUM_RESOURCES]
    probability = DICE_PROBABILITY[s.tile_number]
    opponent_income = (tile_levels * probability)[:, None] * kinds
    opponent_income = opponent_income.sum(axis=0)
    logits = logits.at[topo.MONOPOLY : topo.ROBBER].set(55.0 + 40.0 * opponent_income)

    # Robber: hit the opponent's best tile, avoid our own, prefer stealing.
    my_on_tile = (s.node_owner[env.tile_nodes] == seat).any(axis=1)
    robber_value = (tile_levels * probability) * 100.0 - 50.0 * my_on_tile
    robber_scores = jnp.stack([robber_value, robber_value + 2.0], axis=1).ravel()
    logits = logits.at[topo.ROBBER : topo.MARITIME].set(robber_scores)

    # Discard whatever we hold the most of beyond what builds need.
    surplus = hand - keep
    logits = logits.at[topo.DISCARD : topo.DISCARD + NUM_RESOURCES].set(
        surplus.astype(jnp.float32)
    )

    # Bank trades turn true surplus into missing cards.
    rates = env.trade_rates(s, seat)
    give, get = env.maritime_pairs[:, 0], env.maritime_pairs[:, 1]
    spare = hand[give] - rates[give] - keep[give]
    useful = goal_need[get] > 0
    trade_score = jnp.where(useful & (spare >= 0), 85.0 + spare + goal_need[get], -10.0)
    logits = logits.at[topo.MARITIME :].set(trade_score)
    return logits


def scripted_logits(env, s, kind):
    """Dispatches on an integer kind so the choice can vary per game under vmap."""
    return jnp.where(
        kind == RANDOM,
        random_logits(env, s),
        jnp.where(kind == WEIGHTED, weighted_logits(env, s), heuristic_logits(env, s)),
    )
