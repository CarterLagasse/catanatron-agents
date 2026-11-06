import itertools
import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from catanatron.jax_rl.env import (  # noqa: E402
    ROAD_FRONTIER,
    CatanEnv,
    pack_states,
    unpack_states,
)
from catanatron.jax_rl.obs import (  # noqa: E402
    EDGE_FEATURES,
    GLOBAL_FEATURES,
    NODE_FEATURES,
    TILE_FEATURES,
    encode,
)


@pytest.fixture(scope="module")
def env():
    return CatanEnv("BASE")


def _brute_longest(env, owned_edges, reach, blocked=frozenset()):
    """Exhaustive trail search, the reference for the frontier-based one. A trail
    can end at a blocked (enemy) node but not pass through it."""
    adjacency = {}
    for index in owned_edges:
        a, b = env.topo.edges[index].tolist()
        adjacency.setdefault(a, []).append((b, index))
        adjacency.setdefault(b, []).append((a, index))
    max_open = {}

    def walk(node, used, depth):
        max_open[depth] = max_open.get(depth, 0) + 1
        best = depth
        if depth > 0 and node in blocked:
            return best
        for neighbor, edge in adjacency.get(node, []):
            if edge not in used:
                best = max(best, walk(neighbor, used | {edge}, depth + 1))
        return best

    best = max((walk(n, frozenset(), 0) for n in reach), default=0)
    return best, max(max_open.values(), default=0)


def _random_network(env, rng, size):
    start = rng.randrange(env.topo.num_edges)
    owned = {start}
    while len(owned) < size:
        frontier = set()
        for index in owned:
            for node in env.topo.edges[index].tolist():
                frontier.update(e for e in env.topo.node_edges[node].tolist() if e >= 0)
        frontier -= owned
        owned.add(rng.choice(sorted(frontier)))
    return sorted(owned)


def test_longest_road_matches_brute_force(env):
    rng = random.Random(0)
    search = jax.jit(env.longest_road)
    worst_open = 0
    for _ in range(300):
        owned = _random_network(env, rng, rng.randint(1, 15))
        nodes = sorted({n for e in owned for n in env.topo.edges[e].tolist()})
        blocked = (
            set(rng.sample(nodes, rng.randint(0, 2))) if rng.random() < 0.5 else set()
        )
        reach = [n for n in nodes if n not in blocked]

        edge_owner = np.full(env.topo.num_edges, -1, np.int32)
        edge_owner[owned] = 0
        node_owner = np.full(env.topo.num_nodes, -1, np.int32)
        node_owner[list(blocked)] = 1
        reach_mask = np.zeros(env.topo.num_nodes, bool)
        reach_mask[reach] = True

        expected, open_trails = _brute_longest(env, owned, reach, blocked)
        worst_open = max(worst_open, open_trails)
        got = search(
            jnp.asarray(edge_owner), jnp.asarray(node_owner), jnp.asarray(reach_mask), 0
        )
        assert int(got) == expected
    assert worst_open <= ROAD_FRONTIER


def test_longest_road_frontier_bound_on_dense_networks(env):
    """Hexagon-packed networks have the most trails, so they set the frontier size."""
    tile_edges = []
    for tile in env.topo.tile_nodes.tolist():
        nodes = set(tile)
        tile_edges.append(
            [
                i
                for i, (a, b) in enumerate(env.topo.edges.tolist())
                if a in nodes and b in nodes
            ]
        )
    worst = 0
    for first, second, third in itertools.islice(
        itertools.combinations(range(env.topo.num_tiles), 3), 400
    ):
        owned = sorted(set(tile_edges[first] + tile_edges[second] + tile_edges[third]))[
            :15
        ]
        nodes = sorted({n for e in owned for n in env.topo.edges[e].tolist()})
        worst = max(worst, _brute_longest(env, owned, nodes)[1])
    assert worst <= ROAD_FRONTIER


def _random_rollout(env, plies=400, batch=32, seed=0):
    key = jax.random.PRNGKey(seed)
    k_reset, key = jax.random.split(key)
    states = env.batch_reset(k_reset, batch)

    @jax.jit
    def tick(states, key):
        k_action, k_env = jax.random.split(key)
        masks = jax.vmap(env.legal_mask)(states)
        actions = jax.random.categorical(k_action, jnp.where(masks, 0.0, -1e9))
        next_states, info = env.batch_step(k_env, states, actions)
        return next_states, masks, info

    for k in jax.random.split(key, plies):
        states, masks, info = tick(states, k)
        yield states, masks, info


def test_invariants_hold_during_random_play(env):
    for states, masks, _ in _random_rollout(env):
        hand = np.asarray(states.hand)
        bank = np.asarray(states.bank)
        assert (hand >= 0).all()
        assert (bank >= 0).all()
        # Resource cards are conserved between hands and bank
        assert ((hand.sum(axis=1) + bank) == 19).all()
        assert np.asarray(masks.any(axis=1)).all(), "a game has no legal action"
        assert (np.asarray(states.roads_left) >= 0).all()
        assert (np.asarray(states.settlements_left) >= 0).all()
        assert (np.asarray(states.cities_left) >= 0).all()
        dev_total = np.asarray(states.dev_hand).sum(axis=(1, 2)) + np.asarray(
            states.dev_played
        ).sum(axis=(1, 2))
        assert (dev_total == np.asarray(states.dev_drawn)).all()


def test_shaping_is_zero_sum_and_telescopes():
    env = CatanEnv("BASE", shaping_vp=0.05)
    totals = np.zeros(32)
    for states, _, info in _random_rollout(env, plies=300):
        rewards = np.asarray(info.rewards)
        assert np.allclose(rewards[:, 0], -rewards[:, 1])
        totals += rewards[:, 0]
        done = np.asarray(info.done)
        expected = 0.05 * np.asarray(states.vp_actual[:, 0] - states.vp_actual[:, 1])
        totals[done] = 0.0
        assert np.allclose(totals[~done], expected[~done], atol=1e-4)


def test_observation_hides_opponent_information(env):
    states = env.batch_reset(jax.random.PRNGKey(1), 1)
    state = jax.tree_util.tree_map(lambda x: x[0], states)

    def observe(s, seat=0):
        obs = encode(env, s, seat)
        return np.concatenate([np.ravel(x) for x in obs])

    baseline = state._replace(
        hand=state.hand.at[1].set(jnp.array([3, 0, 0, 0, 0])),
        dev_hand=state.dev_hand.at[1].set(jnp.array([1, 0, 0, 0, 0])),
    )
    # Same counts, different identity: the opponent's cards must look identical.
    swapped = baseline._replace(
        hand=baseline.hand.at[1].set(jnp.array([0, 1, 1, 1, 0])),
        dev_hand=baseline.dev_hand.at[1].set(jnp.array([0, 0, 0, 0, 1])),
        dev_deck=baseline.dev_deck[::-1],
        vp_actual=baseline.vp_actual.at[1].add(1),
    )
    assert np.array_equal(observe(baseline), observe(swapped))
    # Our own hand must be visible.
    richer = baseline._replace(hand=baseline.hand.at[0].set(jnp.array([1, 0, 0, 0, 0])))
    assert not np.array_equal(observe(baseline), observe(richer))


def test_observation_shapes(env):
    state = jax.tree_util.tree_map(
        lambda x: x[0], env.batch_reset(jax.random.PRNGKey(0), 1)
    )
    obs = encode(env, state, 0)
    assert obs.nodes.shape == (54, NODE_FEATURES)
    assert obs.edges.shape == (72, EDGE_FEATURES)
    assert obs.tiles.shape == (19, TILE_FEATURES)
    assert obs.globals.shape == (GLOBAL_FEATURES,)


def test_pack_round_trip(env):
    for states, _, _ in _random_rollout(env, plies=150):
        pass
    restored = unpack_states(pack_states(states))
    for original, packed in zip(states, restored):
        assert np.array_equal(np.asarray(original), np.asarray(packed))


def test_mini_map_runs():
    env = CatanEnv("MINI")
    assert env.topo.num_nodes == 24
    for states, masks, _ in _random_rollout(env, plies=60, batch=8):
        assert np.asarray(masks.any(axis=1)).all()


@pytest.fixture(scope="module", params=[3, 4])
def multiplayer_env(request):
    return CatanEnv("BASE", num_players=request.param)


def test_multiplayer_invariants_and_zero_sum_rewards(multiplayer_env):
    env = CatanEnv("BASE", num_players=multiplayer_env.num_players, shaping_vp=0.05)
    players = env.num_players
    winners = []
    for states, masks, info in _random_rollout(env, plies=600):
        hand = np.asarray(states.hand)
        assert hand.shape[1] == players
        assert (hand >= 0).all()
        assert (
            (hand.sum(axis=(1, 2)) + np.asarray(states.bank).sum(axis=1)) == 95
        ).all()
        assert np.asarray(masks.any(axis=1)).all(), "a game has no legal action"
        assert np.allclose(np.asarray(info.rewards).sum(axis=1), 0.0, atol=1e-5)
        assert (np.asarray(states.current) < players).all()
        winners += np.asarray(info.winner)[np.asarray(info.done)].tolist()
    assert winners, "no random game finished"


def test_multiplayer_turns_go_round_the_table(multiplayer_env):
    """Initial placement snakes 0..P-1, P-1..0; then turns go up the seats."""
    env = multiplayer_env
    players = env.num_players
    state = jax.tree_util.tree_map(
        lambda x: x[0], env.batch_reset(jax.random.PRNGKey(3), 1)
    )
    step = jax.jit(env._apply)
    order = []
    rng = random.Random(0)
    chance = jax.tree_util.tree_map(jnp.asarray, (7, 0))
    from catanatron.jax_rl.env import BUILD_INITIAL_SETTLEMENT, PLAY_TURN, Chance

    while int(state.prompt) != PLAY_TURN:
        mask = np.flatnonzero(np.asarray(env.legal_mask(state)))
        if int(state.prompt) == BUILD_INITIAL_SETTLEMENT:
            order.append(int(state.current))
        state = step(state, jnp.asarray(rng.choice(mask)), Chance(*chance))
    assert order == list(range(players)) + list(reversed(range(players)))
    assert int(state.current) == 0


def test_multiplayer_robber_offers_one_steal_per_victim(multiplayer_env):
    env = multiplayer_env
    topo = env.topo
    players = env.num_players
    state = jax.tree_util.tree_map(
        lambda x: x[0], env.batch_reset(jax.random.PRNGKey(4), 1)
    )
    from catanatron.jax_rl.env import MOVE_ROBBER

    tile = 0 if int(state.robber_tile) != 0 else 1
    nodes = topo.tile_nodes[tile]
    owner = state.node_owner.at[nodes[0]].set(1).at[nodes[2]].set(players - 1)
    hand = state.hand.at[1, 0].set(2).at[players - 1, 3].set(1)
    state = state._replace(node_owner=owner, hand=hand, prompt=jnp.asarray(MOVE_ROBBER))
    mask = np.asarray(env.legal_mask(state))
    slots = mask[topo.ROBBER + players * tile : topo.ROBBER + players * (tile + 1)]
    expected = np.zeros(players, bool)
    expected[1] = expected[players - 1] = True  # seat 1 and the last seat, from seat 0
    assert np.array_equal(slots, expected)

    # Stealing from the last seat takes its only card.
    action = topo.ROBBER + players * tile + (players - 1)
    from catanatron.jax_rl.env import Chance

    after = env._apply(
        state, jnp.asarray(action), Chance(jnp.asarray(7), jnp.asarray(3))
    )
    assert int(after.hand[players - 1].sum()) == 0
    assert int(after.hand[0, 3]) == 1
    assert int(after.robber_tile) == tile


def test_multiplayer_observation_shapes_and_hidden_cards(multiplayer_env):
    from catanatron.jax_rl.obs import feature_sizes

    env = multiplayer_env
    players = env.num_players
    state = jax.tree_util.tree_map(
        lambda x: x[0], env.batch_reset(jax.random.PRNGKey(0), 1)
    )
    obs = encode(env, state, 0)
    nodes, edges, tiles, globals_ = feature_sizes(players)
    assert obs.nodes.shape == (54, nodes) and nodes > NODE_FEATURES
    assert obs.edges.shape == (72, edges)
    assert obs.tiles.shape == (19, tiles)
    assert obs.globals.shape == (globals_,)

    def observe(s):
        return np.concatenate([np.ravel(x) for x in encode(env, s, 0)])

    base = state._replace(hand=state.hand.at[2].set(jnp.array([3, 0, 0, 0, 0])))
    same_count = base._replace(hand=base.hand.at[2].set(jnp.array([0, 1, 1, 1, 0])))
    assert np.array_equal(observe(base), observe(same_count))
    more = base._replace(hand=base.hand.at[2].set(jnp.array([4, 0, 0, 0, 0])))
    assert not np.array_equal(observe(base), observe(more))
