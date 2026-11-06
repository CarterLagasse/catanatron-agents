"""Head-to-head evaluation inside JAX. For real-engine matches see jax_player.py."""

import jax
import jax.numpy as jnp

from catanatron.jax_rl.bots import SCRIPTED_NAMES, scripted_logits
from catanatron.jax_rl.obs import encode_current


def network_policy(policy, params, greedy=False):
    """A policy maps (key, batched states, masks) to actions."""

    def act(key, states, masks, env):
        obs = jax.vmap(lambda s: encode_current(env, s))(states)
        logits, _ = jax.vmap(policy.apply, in_axes=(None, 0))(params, obs)
        logits = jnp.where(masks, logits, -1e9)
        if greedy:
            return jnp.argmax(logits, axis=1)
        return jax.random.categorical(key, logits)

    return act


def scripted_policy(kind):
    def act(key, states, masks, env):
        logits = jax.vmap(lambda s: scripted_logits(env, s, kind))(states)
        return jax.random.categorical(key, jnp.where(masks, logits, -1e9))

    return act


class Arena:
    """Plays a fixed number of plies across a batch and counts finished games.

    Games still running when the plies run out are not counted. Long games are
    therefore slightly under-represented; raise `ticks` to reduce that.
    """

    def __init__(self, env, policy, num_envs=256, ticks=800):
        self.env = env
        self.policy = policy
        self.num_envs = num_envs
        self.ticks = ticks
        self._vs_network = jax.jit(self._match_network)
        self._vs_scripted = jax.jit(self._match_scripted, static_argnames=("greedy",))
        self._scripted_vs = jax.jit(self._match_scripted_first)

    def _random_seats(self, key, shape):
        if self.env.num_players == 2:  # the 1v1 draws, so old results reproduce
            return jax.random.bernoulli(key, 0.5, shape).astype(jnp.int32)
        return jax.random.randint(key, shape, 0, self.env.num_players)

    def _play(self, key, act_a, act_b):
        """Policy A takes a random seat in each game and B plays every other seat.
        Returns counters from A's side."""
        env = self.env
        k_reset, k_seat, k_loop = jax.random.split(key, 3)
        states = env.batch_reset(k_reset, self.num_envs)
        seat_a = self._random_seats(k_seat, (self.num_envs,))

        def tick(carry, key):
            states, seat_a = carry
            k_a, k_b, k_step, k_seat = jax.random.split(key, 4)
            masks = jax.vmap(env.legal_mask)(states)
            actions = jnp.where(
                states.current == seat_a,
                act_a(k_a, states, masks, env),
                act_b(k_b, states, masks, env),
            )
            states, info = env.batch_step(k_step, states, actions)
            won = info.done & (info.winner == seat_a)
            lost = info.done & (info.winner >= 0) & (info.winner != seat_a)
            drew = info.done & (info.winner < 0)
            new_seats = self._random_seats(k_seat, seat_a.shape)
            seat_a = jnp.where(info.done, new_seats, seat_a)
            counts = jnp.stack([won.sum(), lost.sum(), drew.sum()])
            return (states, seat_a), counts

        _, counts = jax.lax.scan(
            tick, (states, seat_a), jax.random.split(k_loop, self.ticks)
        )
        return counts.sum(axis=0)

    def _match_network(self, key, params_a, params_b):
        return self._play(
            key,
            network_policy(self.policy, params_a),
            network_policy(self.policy, params_b),
        )

    def _match_scripted(self, key, params, kind, greedy=False):
        return self._play(
            key,
            network_policy(self.policy, params, greedy=greedy),
            scripted_policy(kind),
        )

    def _match_scripted_first(self, key, kind, params):
        return self._play(
            key, scripted_policy(kind), network_policy(self.policy, params)
        )

    def _summary(self, counts):
        """payoff is A's mean reward: +1 a win, -1/(P-1) a loss, so 0 for equals."""
        wins, losses, draws = (int(x) for x in counts)
        games = wins + losses + draws
        loss = 1.0 / (self.env.num_players - 1)
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": games,
            "win_rate": wins / games if games else 0.0,
            "payoff": (wins - loss * losses) / games if games else 0.0,
        }

    def against_scripted(self, key, params, name, greedy=False):
        kind = jnp.asarray(SCRIPTED_NAMES[name], jnp.int32)
        return self._summary(self._vs_scripted(key, params, kind, greedy=greedy))

    def against_network(self, key, params_a, params_b):
        return self._summary(self._vs_network(key, params_a, params_b))

    def scripted_against(self, key, name, params):
        """The scripted bot in one seat, the network in every other seat."""
        kind = jnp.asarray(SCRIPTED_NAMES[name], jnp.int32)
        return self._summary(self._scripted_vs(key, kind, params))
