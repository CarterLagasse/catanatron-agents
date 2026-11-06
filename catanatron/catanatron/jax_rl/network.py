"""Graph actor-critic, written directly in JAX with parameters as plain dicts.

Boards differ every game, so an MLP on a flat vector would have to learn board
geometry separately at every location. Here weights are shared across nodes, edges
and tiles, messages follow the real adjacency, and each spatial action gets its
logit from the entity it refers to: settlements from nodes, roads from edges, and
the robber from tiles.
"""

import jax
import jax.numpy as jnp

import numpy as np

from catanatron.jax_rl.obs import feature_sizes
from catanatron.jax_rl.topology import MARITIME_PAIRS, NUM_RESOURCES, YOP_PAIRS

MASKED_LOGIT = -1e9


def _dense_init(key, in_dim, out_dim, scale=jnp.sqrt(2.0)):
    weight = jax.nn.initializers.orthogonal(scale)(key, (in_dim, out_dim), jnp.float32)
    return {"w": weight, "b": jnp.zeros(out_dim, jnp.float32)}


def _dense(p, x):
    return x @ p["w"] + p["b"]


class GraphPolicy:
    """`init` builds parameters, `apply` maps one observation to (logits, value)."""

    def __init__(self, env, hidden=64, rounds=3):
        self.topo = topo = env.topo
        self.num_players = env.num_players
        self.feature_sizes = feature_sizes(env.num_players)
        self.hidden = hidden
        self.rounds = rounds
        self.num_actions = topo.num_actions
        self.edges = env.edges
        self.tile_nodes = env.tile_nodes

        # Sentinel row for -1 padding: its embedding is forced to zero.
        self.node_neighbors = env.node_neighbors_pad
        self.node_edges = env.node_edges_pad
        self.node_tiles = jnp.where(env.node_tiles < 0, topo.num_tiles, env.node_tiles)

        # Non-spatial actions: 5 turn actions, discards, YoP, monopoly, maritime.
        num_global = 5 + NUM_RESOURCES + (len(YOP_PAIRS) + NUM_RESOURCES)
        self.num_global_actions = num_global + NUM_RESOURCES + len(MARITIME_PAIRS)
        self._split_turn = 5 + NUM_RESOURCES
        self._split_yop = len(YOP_PAIRS) + NUM_RESOURCES + NUM_RESOURCES

    def init(self, key):
        h = self.hidden
        keys = iter(jax.random.split(key, 9 + 4 * self.rounds))

        def layer(in_dim, out_dim, scale=jnp.sqrt(2.0)):
            return _dense_init(next(keys), in_dim, out_dim, scale)

        node_features, edge_features, tile_features, global_features = (
            self.feature_sizes
        )
        params = {
            "embed_node": layer(node_features, h),
            "embed_edge": layer(edge_features, h),
            "embed_tile": layer(tile_features, h),
            "embed_global": layer(global_features, h),
            "rounds": [
                {
                    "node": layer(h, h),
                    "edge": layer(h, h),
                    "tile": layer(h, h),
                    "global": layer(h, h),
                }
                for _ in range(self.rounds)
            ],
            "head_node": layer(h, 2, 0.01),  # settlement, city
            "head_edge": layer(h, 1, 0.01),
            # robber: no steal, then steal from each opponent
            "head_tile": layer(h, self.num_players, 0.01),
            "head_global": layer(h, self.num_global_actions, 0.01),
            "head_value": layer(h, 1, 1.0),
        }
        return params

    def apply(self, params, obs):
        """Returns unmasked logits [A] and a scalar value in the player's favor."""
        act = jax.nn.relu
        n = act(_dense(params["embed_node"], obs.nodes))
        e = act(_dense(params["embed_edge"], obs.edges))
        t = act(_dense(params["embed_tile"], obs.tiles))
        g = act(_dense(params["embed_global"], obs.globals))

        # Messages are summed into one vector before each dense layer, which costs
        # h^2 per entity instead of the (k * h) * h a concatenation would. On a CPU
        # that is what makes millions of gradient samples an hour possible.
        zero = jnp.zeros((1, self.hidden), jnp.float32)
        for round_params in params["rounds"]:
            n_pad = jnp.concatenate([n, zero])
            e_pad = jnp.concatenate([e, zero])
            t_pad = jnp.concatenate([t, zero])

            node_input = (
                n
                + n_pad[self.node_neighbors].sum(axis=1) / 3.0
                + e_pad[self.node_edges].sum(axis=1) / 3.0
                + t_pad[self.node_tiles].sum(axis=1) / 3.0
                + g
            )
            edge_input = e + n[self.edges].sum(axis=1) / 2.0 + g
            tile_input = t + n[self.tile_nodes].sum(axis=1) / 6.0 + g
            global_input = g + n.mean(axis=0) + e.mean(axis=0) + t.mean(axis=0)

            n = n + _dense(round_params["node"], act(node_input))
            e = e + _dense(round_params["edge"], act(edge_input))
            t = t + _dense(round_params["tile"], act(tile_input))
            g = g + _dense(round_params["global"], act(global_input))

        n, e, t, g = act(n), act(e), act(t), act(g)

        node_logits = _dense(params["head_node"], n)  # [N, 2]
        edge_logits = _dense(params["head_edge"], e)[:, 0]  # [E]
        tile_logits = _dense(params["head_tile"], t).ravel()  # [T * P]
        global_logits = _dense(params["head_global"], g)

        split_a = self._split_turn
        split_b = split_a + self._split_yop
        logits = jnp.concatenate(
            [
                global_logits[:split_a],  # turn actions and discards
                edge_logits,
                node_logits[:, 0],
                node_logits[:, 1],
                global_logits[split_a:split_b],  # year of plenty and monopoly
                tile_logits,
                global_logits[split_b:],  # maritime
            ]
        )
        value = _dense(params["head_value"], g)[0]
        return logits, value


def adapt_params(params, num_players):
    """Widens 1v1 parameters to play `num_players`, as a warm start.

    New inputs get zero weights, so the adapted network first plays as the 1v1 one
    would with every opponent folded into "the opponent". Each extra robber victim
    starts from the 1v1 steal logit.
    """
    if num_players == 2:
        return params
    sizes = dict(zip(("node", "edge", "tile", "global"), feature_sizes(num_players)))
    params = dict(params)
    for name, size in sizes.items():
        layer = dict(params[f"embed_{name}"])
        weight = np.asarray(layer["w"])
        assert weight.shape[0] <= size, "parameters already have more inputs"
        padding = np.zeros((size - weight.shape[0], weight.shape[1]), weight.dtype)
        layer["w"] = np.concatenate([weight, padding])
        params[f"embed_{name}"] = layer
    head = dict(params["head_tile"])
    weight, bias = np.asarray(head["w"]), np.asarray(head["b"])
    assert weight.shape[1] == 2, "robber head is not from a 1v1 network"
    extra = num_players - 2
    head["w"] = np.concatenate([weight, np.repeat(weight[:, 1:], extra, axis=1)], 1)
    head["b"] = np.concatenate([bias, np.repeat(bias[1:], extra)])
    params["head_tile"] = head
    return params


def masked_log_probs(logits, mask):
    return jax.nn.log_softmax(jnp.where(mask, logits, MASKED_LOGIT))


def param_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
