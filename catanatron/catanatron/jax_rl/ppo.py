"""PPO with masked actions, fully jitted: rollout, GAE and updates run inside XLA.

Games are stepped one ply at a time across the whole batch. Catan is not strictly
alternating (a player makes many decisions per turn, and may discard on the other
player's turn), so a tick's reward cannot be paired with the next tick's value.
`compute_gae` instead follows each seat's own decision chain within each game.

Each game belongs to a group:
    self:     both seats play the current network and both generate training data
    pool:     the learner plays one random seat against a frozen network
    scripted: the learner plays one random seat against a scripted bot

Train:
    python -m catanatron.jax_rl.ppo --run-dir runs/ppo --timesteps 50000000
"""

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from catanatron.jax_rl import checkpoint
from catanatron.jax_rl.bots import HEURISTIC, RANDOM, WEIGHTED, scripted_logits
from catanatron.jax_rl.env import CatanEnv, pack_states, unpack_states
from catanatron.jax_rl.evaluate import Arena
from catanatron.jax_rl.network import GraphPolicy, masked_log_probs
from catanatron.jax_rl.obs import encode, encode_current

# opponent_code values for one game
OPPONENT_SELF, OPPONENT_POOL = -1, 3  # scripted kinds use bots.RANDOM..HEURISTIC


@dataclass(frozen=True)
class PPOConfig:
    map_type: str = "BASE"
    vps_to_win: int = 10
    num_envs: int = 256
    rollout_ticks: int = 128
    # Group sizes as fractions of num_envs. Whatever remains is self-play.
    pool_fraction: float = 0.25
    scripted_fraction: float = 0.15
    pool_segments: int = 4  # distinct frozen opponents per phase; one forward pass each
    heuristic_share: float = 0.7  # of scripted games; the rest split weighted / random
    # PPO
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-4
    gamma: float = 0.997  # per own decision
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    final_entropy_coef: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Simulating is about 3x cheaper than a gradient sample on CPU, so fresh data
    # beats reusing it. Backprop memory scales with minibatch size.
    update_epochs: int = 1
    num_minibatches: int = 32
    # VP-difference shaping per point, on top of +1/-1 for the result
    shaping_vp: float = 0.05
    # Network. Width drives speed: cost grows with hidden^2.
    hidden: int = 48
    rounds: int = 3
    # Loop
    total_timesteps: int = 50_000_000
    phase_iterations: int = 10  # iterations between re-drawing frozen opponents
    snapshot_every: int = 50  # iterations between adding self to the pool
    pool_capacity: int = 12
    eval_every: int = 50
    eval_envs: int = 128
    eval_ticks: int = 600
    seed: int = 0

    @property
    def num_pool(self):
        if self.pool_fraction <= 0:
            return 0
        per_segment = max(1, round(self.num_envs * self.pool_fraction / self.pool_segments))
        return per_segment * self.pool_segments

    @property
    def num_scripted(self):
        return int(round(self.num_envs * self.scripted_fraction))

    @property
    def num_self(self):
        return self.num_envs - self.num_pool - self.num_scripted

    @property
    def expected_learner_fraction(self):
        return (self.num_self + 0.5 * (self.num_pool + self.num_scripted)) / self.num_envs


class Transition(NamedTuple):
    state: object  # packed State, from which observations are rebuilt
    seat: jnp.ndarray
    is_learner: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    rewards: jnp.ndarray  # [2]
    done: jnp.ndarray


class Carry(NamedTuple):
    states: object
    learner_seat: jnp.ndarray  # [B], meaningless for self-play games


class Trainer:
    """Compiled training step for one configuration."""

    def __init__(self, config):
        self.config = config
        self.env = CatanEnv(
            config.map_type, vps_to_win=config.vps_to_win, shaping_vp=config.shaping_vp
        )
        self.policy = GraphPolicy(self.env, config.hidden, config.rounds)

        num_train = int(
            config.num_envs * config.rollout_ticks * config.expected_learner_fraction * 0.98
        )
        self.minibatch_size = num_train // config.num_minibatches
        self.num_train = self.minibatch_size * config.num_minibatches

        iterations = max(self.iterations, 1)
        updates = iterations * config.update_epochs * config.num_minibatches
        self.schedule = optax.linear_schedule(
            config.learning_rate, config.final_learning_rate, updates
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(self.schedule, eps=1e-5),
        )
        self.iteration_fn = jax.jit(self._iteration)

    @property
    def iterations(self):
        c = self.config
        return c.total_timesteps // (c.num_envs * c.rollout_ticks)

    def init_params(self, key):
        return self.policy.init(key)

    def init_carry(self, key):
        k_states, k_seats = jax.random.split(key)
        return Carry(
            states=self.env.batch_reset(k_states, self.config.num_envs),
            learner_seat=jax.random.bernoulli(
                k_seats, 0.5, (self.config.num_envs,)
            ).astype(jnp.int32),
        )

    def scripted_codes(self, key):
        """Opponent codes for the fixed self / pool / scripted game layout."""
        c = self.config
        share = c.heuristic_share
        rest = (1.0 - share) / 2.0
        kinds = jax.random.choice(
            key,
            jnp.array([RANDOM, WEIGHTED, HEURISTIC]),
            (c.num_scripted,),
            p=jnp.array([rest, rest, share]),
        )
        return jnp.concatenate(
            [
                jnp.full(c.num_self, OPPONENT_SELF, jnp.int32),
                jnp.full(c.num_pool, OPPONENT_POOL, jnp.int32),
                kinds.astype(jnp.int32),
            ]
        )

    # ===== Rollout
    def _rollout(self, params, carry, opponent_codes, pool_params, pool_kinds, key):
        c = self.config
        env = self.env
        policy = self.policy
        mask_fn = jax.vmap(env.legal_mask)
        obs_fn = jax.vmap(lambda s: encode_current(env, s))
        apply_fn = jax.vmap(policy.apply, in_axes=(None, 0))
        is_self = opponent_codes == OPPONENT_SELF
        per_segment = c.num_pool // max(c.pool_segments, 1)

        def pool_actions(key, states, obs, masks):
            """Each segment's games share one frozen network, so memory use and cost
            scale with the batch and not with batch x parameters."""
            first = c.num_self

            def segment(params_k, kind, k, seg_obs, seg_states, seg_masks):
                logits, _ = jax.vmap(policy.apply, in_axes=(None, 0))(params_k, seg_obs)
                # Segments can also hold a scripted population member (PSRO).
                bot = jax.vmap(lambda s: scripted_logits(env, s, jnp.maximum(kind, 0)))(
                    seg_states
                )
                logits = jnp.where(kind < 0, logits, bot)
                return jax.random.categorical(
                    k, jnp.where(seg_masks, logits, -1e9)
                )

            def carve(x):
                y = x[first : first + c.num_pool]
                return y.reshape((c.pool_segments, per_segment) + x.shape[1:])

            actions = jax.vmap(segment)(
                pool_params,
                pool_kinds,
                jax.random.split(key, c.pool_segments),
                jax.tree_util.tree_map(carve, obs),
                jax.tree_util.tree_map(carve, states),
                carve(masks),
            )
            return actions.ravel()

        def tick(carry, key):
            k_learner, k_bot, k_pool, k_env, k_seats = jax.random.split(key, 5)
            states = carry.states
            masks = mask_fn(states)
            obs = obs_fn(states)
            logits, values = apply_fn(params, obs)
            log_probs = masked_log_probs(logits, masks)
            actions = jax.random.categorical(k_learner, log_probs)

            seats = states.current
            is_learner = is_self | (seats == carry.learner_seat)

            if c.num_scripted > 0:
                start = c.num_self + c.num_pool
                scripted_states = jax.tree_util.tree_map(lambda x: x[start:], states)
                bot_logits = jax.vmap(lambda s, kind: scripted_logits(env, s, kind))(
                    scripted_states, opponent_codes[start:]
                )
                bot_actions = jax.random.categorical(
                    k_bot, jnp.where(masks[start:], bot_logits, -1e9)
                )
                other = actions.at[start:].set(bot_actions)
            else:
                other = actions
            if c.num_pool > 0:
                other = other.at[c.num_self : c.num_self + c.num_pool].set(
                    pool_actions(k_pool, states, obs, masks)
                )
            final_actions = jnp.where(is_learner, actions, other)

            taken_log_prob = jnp.take_along_axis(
                log_probs, final_actions[:, None], axis=1
            )[:, 0]
            next_states, info = env.batch_step(k_env, states, final_actions)
            # Each new game gets a fresh random seat for the learner.
            new_seats = jax.random.bernoulli(
                k_seats, 0.5, carry.learner_seat.shape
            ).astype(jnp.int32)
            learner_seat = jnp.where(info.done, new_seats, carry.learner_seat)

            transition = Transition(
                state=pack_states(states),
                seat=seats.astype(jnp.int8),
                is_learner=is_learner,
                action=final_actions.astype(jnp.int16),
                log_prob=taken_log_prob,
                value=values,
                rewards=info.rewards,
                done=info.done,
            )
            metrics = (
                info.done,
                info.winner,
                carry.learner_seat,
                info.plies,
            )
            return Carry(next_states, learner_seat), (transition, metrics)

        keys = jax.random.split(key, c.rollout_ticks)
        carry, (transitions, metrics) = jax.lax.scan(tick, carry, keys)

        # Bootstrap: value of the final positions from both seats' perspectives.
        def seat_values(state):
            return jax.vmap(
                lambda seat: policy.apply(params, encode(env, state, seat))[1]
            )(jnp.arange(2))

        last_values = jax.vmap(seat_values)(carry.states)
        return carry, transitions, last_values, metrics

    # ===== GAE
    def compute_gae(self, transitions, last_values):
        """Generalized advantage estimation over each seat's own decision chain.

        Rewards received between two decisions of a seat, including on the other
        player's ticks, accrue to that seat's earlier decision. A game's end cuts
        every chain for that game.
        """
        c = self.config

        def backward(carry, t):
            next_value, next_advantage, pending = carry  # each [B, 2]
            alive = (1.0 - t.done.astype(jnp.float32))[:, None]
            next_value = next_value * alive
            next_advantage = next_advantage * alive
            pending = pending * alive + t.rewards

            batch = jnp.arange(t.seat.shape[0])
            seat = t.seat.astype(jnp.int32)
            delta = (
                pending[batch, seat]
                + c.gamma * next_value[batch, seat]
                - t.value
            )
            advantage = delta + c.gamma * c.gae_lambda * next_advantage[batch, seat]

            learner = t.is_learner
            next_value = next_value.at[batch, seat].set(
                jnp.where(learner, t.value, next_value[batch, seat])
            )
            next_advantage = next_advantage.at[batch, seat].set(
                jnp.where(learner, advantage, next_advantage[batch, seat])
            )
            pending = pending.at[batch, seat].set(
                jnp.where(learner, 0.0, pending[batch, seat])
            )
            return (next_value, next_advantage, pending), advantage

        init = (last_values, jnp.zeros_like(last_values), jnp.zeros_like(last_values))
        _, advantages = jax.lax.scan(backward, init, transitions, reverse=True)
        return advantages, advantages + transitions.value

    # ===== Update
    def _loss(self, params, batch, entropy_coef):
        c = self.config
        env = self.env
        states = unpack_states(batch["state"])
        obs = jax.vmap(lambda s: encode_current(env, s))(states)
        masks = jax.vmap(env.legal_mask)(states)
        logits, values = jax.vmap(self.policy.apply, in_axes=(None, 0))(params, obs)
        log_probs = masked_log_probs(logits, masks)
        actions = batch["action"].astype(jnp.int32)
        new_log_prob = jnp.take_along_axis(log_probs, actions[:, None], axis=1)[:, 0]

        probs = jnp.exp(log_probs)
        entropy = -(probs * jnp.where(masks, log_probs, 0.0)).sum(axis=1)

        weight = batch["weight"]
        norm = jnp.maximum(weight.sum(), 1.0)
        adv = batch["advantage"]
        adv_mean = (adv * weight).sum() / norm
        adv_std = jnp.sqrt((((adv - adv_mean) ** 2) * weight).sum() / norm) + 1e-8
        adv = (adv - adv_mean) / adv_std

        ratio = jnp.exp(new_log_prob - batch["log_prob"])
        unclipped = adv * ratio
        clipped = adv * jnp.clip(ratio, 1 - c.clip_range, 1 + c.clip_range)
        policy_loss = -(jnp.minimum(unclipped, clipped) * weight).sum() / norm

        value_clipped = batch["value"] + jnp.clip(
            values - batch["value"], -c.clip_range, c.clip_range
        )
        value_loss = (
            jnp.maximum(
                (values - batch["return"]) ** 2,
                (value_clipped - batch["return"]) ** 2,
            )
            * weight
        ).sum() / norm
        mean_entropy = (entropy * weight).sum() / norm

        loss = policy_loss + c.value_coef * value_loss - entropy_coef * mean_entropy
        approx_kl = ((ratio - 1.0 - jnp.log(ratio)) * weight).sum() / norm
        return loss, (policy_loss, value_loss, mean_entropy, approx_kl)

    def _update(self, params, opt_state, transitions, advantages, returns, entropy_coef, key):
        c = self.config

        # Keep learner decisions, in random order. Slots past the true count are
        # opponent ticks and get zero weight, which keeps array shapes static.
        def flatten(x):
            return x.reshape((c.rollout_ticks * c.num_envs,) + x.shape[2:])

        flat_learner = flatten(transitions.is_learner)
        k_order, k_epochs = jax.random.split(key)
        noise = jax.random.uniform(k_order, flat_learner.shape)
        order = jnp.argsort(jnp.where(flat_learner, noise, 2.0 + noise))[: self.num_train]
        dataset = {
            "state": jax.tree_util.tree_map(lambda x: flatten(x)[order], transitions.state),
            "action": flatten(transitions.action)[order],
            "log_prob": flatten(transitions.log_prob)[order],
            "value": flatten(transitions.value)[order],
            "advantage": flatten(advantages)[order],
            "return": flatten(returns)[order],
            "weight": flat_learner[order].astype(jnp.float32),
        }

        grad_fn = jax.value_and_grad(self._loss, has_aux=True)

        def minibatch(carry, indices):
            params, opt_state = carry
            batch = jax.tree_util.tree_map(lambda x: x[indices], dataset)
            (_, aux), grads = grad_fn(params, batch, entropy_coef)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), aux

        def epoch(carry, key):
            permutation = jax.random.permutation(key, self.num_train)
            batches = permutation.reshape(c.num_minibatches, self.minibatch_size)
            return jax.lax.scan(minibatch, carry, batches)

        (params, opt_state), aux = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(k_epochs, c.update_epochs)
        )
        policy_loss, value_loss, entropy, approx_kl = jax.tree_util.tree_map(jnp.mean, aux)
        y = dataset["return"]
        weight = dataset["weight"]
        norm = jnp.maximum(weight.sum(), 1.0)
        residual = dataset["return"] - dataset["value"]
        explained = 1.0 - _weighted_var(residual, weight, norm) / (
            _weighted_var(y, weight, norm) + 1e-8
        )
        return params, opt_state, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "explained_variance": explained,
        }

    def _iteration(
        self,
        params,
        opt_state,
        carry,
        opponent_codes,
        pool_params,
        pool_kinds,
        progress,
        key,
    ):
        c = self.config
        k_rollout, k_update = jax.random.split(key)
        carry, transitions, last_values, rollout_metrics = self._rollout(
            params, carry, opponent_codes, pool_params, pool_kinds, k_rollout
        )
        advantages, returns = self.compute_gae(transitions, last_values)
        entropy_coef = c.entropy_coef + progress * (c.final_entropy_coef - c.entropy_coef)
        params, opt_state, metrics = self._update(
            params, opt_state, transitions, advantages, returns, entropy_coef, k_update
        )

        done, winner, learner_seat, plies = rollout_metrics
        groups = {
            "self": opponent_codes == OPPONENT_SELF,
            "pool": opponent_codes == OPPONENT_POOL,
            "random": opponent_codes == RANDOM,
            "weighted": opponent_codes == WEIGHTED,
            "heuristic": opponent_codes == HEURISTIC,
        }
        won = (winner == learner_seat) & done
        lost = (winner >= 0) & (winner != learner_seat) & done
        for name, group in groups.items():
            games = (done & group).sum()
            metrics[f"{name}_games"] = games
            metrics[f"{name}_wins"] = (won & group).sum()
            metrics[f"{name}_losses"] = (lost & group).sum()
        metrics["games"] = done.sum()
        metrics["game_plies"] = (plies * done).sum()
        metrics["learner_fraction"] = transitions.is_learner.mean()
        metrics["reward_abs"] = jnp.abs(transitions.rewards).mean()
        return params, opt_state, carry, metrics

    def run_iteration(
        self, params, opt_state, carry, opponent_codes, pool_params, pool_kinds, progress, key
    ):
        return self.iteration_fn(
            params,
            opt_state,
            carry,
            opponent_codes,
            pool_params,
            pool_kinds,
            jnp.asarray(progress, jnp.float32),
            key,
        )

    def stack_segments(self, members, pool_key_params):
        """Stacks (kind, params) members into per-segment pytrees.

        kind is -1 for a network, otherwise a scripted bot kind. Bots still carry
        pool_key_params as a shape-compatible placeholder.
        """
        kinds = jnp.array([m[0] for m in members], jnp.int32)
        trees = [pool_key_params if m[1] is None else m[1] for m in members]
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)
        return stacked, kinds


def _weighted_var(x, weight, norm):
    mean = (x * weight).sum() / norm
    return (((x - mean) ** 2) * weight).sum() / norm


def reset_pool_games(trainer, carry, key):
    """Restarts pool games so a new frozen opponent never inherits a midgame."""
    c = trainer.config
    fresh = trainer.env.batch_reset(key, c.num_pool)
    start, stop = c.num_self, c.num_self + c.num_pool
    states = jax.tree_util.tree_map(
        lambda old, new: old.at[start:stop].set(new), carry.states, fresh
    )
    return carry._replace(states=states)


def sample_pool(trainer, pool, params, rng):
    """Draws one frozen opponent per segment, favoring recent snapshots."""
    c = trainer.config
    if not pool:
        members = [(-1, params)] * c.pool_segments
    else:
        weights = np.linspace(0.5, 1.5, len(pool))
        weights /= weights.sum()
        picks = rng.choice(len(pool), c.pool_segments, p=weights)
        members = [(-1, pool[i]) for i in picks]
    return trainer.stack_segments(members, params)


def _save_resume_state(run_dir, **state):
    """Everything needed to continue a run. Games in progress are simply restarted."""
    state = jax.tree_util.tree_map(np.asarray, state)
    temporary = run_dir / "train_state.tmp.pkl"
    with temporary.open("wb") as f:
        pickle.dump(state, f)
    checkpoint.replace_file(temporary, run_dir / "train_state.pkl")


def _load_resume_state(trainer, run_dir, log):
    """Loads train_state.pkl, or falls back to latest.npz plus metrics.jsonl.

    The fallback recovers the network and the schedule position, but starts with
    empty Adam moments and a league holding only the current network.
    """
    path = run_dir / "train_state.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    params = checkpoint.load_params(run_dir / "latest.npz")[0]
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    evaluated = [r for r in rows if "eval_heuristic_win" in r]
    last = evaluated[-1]
    config = trainer.config
    updates = last["iteration"] * config.update_epochs * config.num_minibatches
    opt_state = optax.tree_utils.tree_set(
        trainer.optimizer.init(params), count=jnp.asarray(updates, jnp.int32)
    )
    log(f"No train_state.pkl: resuming from latest.npz at iteration {last['iteration']}")
    return {
        "params": params,
        "opt_state": opt_state,
        "pool": [params],
        "iteration": last["iteration"],
        "steps": last["steps"],
        "seconds": last["seconds"],
        "best_score": max(r["eval_heuristic_win"] for r in evaluated),
        "key": np.asarray(jax.random.PRNGKey(config.seed + last["iteration"])),
    }


def train(config, run_dir, log=print, init_params=None, resume=False):
    """Self-play with a league of past selves. Saves `best.npz` and `latest.npz`,
    plus `train_state.pkl` at each evaluation so a run can continue with resume=True.

    Each evaluation also keeps `snapshots/it_XXXXX.npz`. `best.npz` is picked from
    a few hundred noisy games, so compare snapshots with tournament.py before
    trusting it. It is the same network as `latest.npz` when the last evaluation wins.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    trainer = Trainer(config)
    arena = Arena(trainer.env, trainer.policy, config.eval_envs, config.eval_ticks)
    rng = np.random.default_rng(config.seed)
    key = jax.random.PRNGKey(config.seed)
    key, k_params, k_carry, k_codes = jax.random.split(key, 4)

    params = init_params if init_params is not None else trainer.init_params(k_params)
    opt_state = trainer.optimizer.init(params)
    carry = trainer.init_carry(k_carry)
    codes = trainer.scripted_codes(k_codes)

    pool = [params]
    pool_params, pool_kinds = sample_pool(trainer, pool, params, rng)

    log_path = run_dir / "metrics.jsonl"
    best_score = -np.inf
    steps = 0
    first_iteration = 1
    elapsed_before = 0.0
    if resume:
        saved = _load_resume_state(trainer, run_dir, log)
        params = saved["params"]
        opt_state = saved["opt_state"]
        pool = saved["pool"]
        best_score = float(saved["best_score"])
        steps = int(saved["steps"])
        first_iteration = int(saved["iteration"]) + 1
        elapsed_before = float(saved["seconds"])
        key = jnp.asarray(saved["key"])
        pool_params, pool_kinds = sample_pool(trainer, pool, params, rng)
        # Iterations after the saved state are about to be replayed.
        rows = log_path.read_text().splitlines()
        kept = [r for r in rows if json.loads(r)["iteration"] < first_iteration]
        log_path.write_text("".join(r + "\n" for r in kept))
    start = time.time() - elapsed_before
    log(
        f"PPO: {trainer.iterations} iterations of {config.num_envs}x{config.rollout_ticks} ticks, "
        f"self={config.num_self} pool={config.num_pool} scripted={config.num_scripted}"
    )

    with log_path.open("a") as metrics_file:
        for iteration in range(first_iteration, trainer.iterations + 1):
            progress = (iteration - 1) / max(trainer.iterations - 1, 1)
            key, k_iter, k_reset, k_codes = jax.random.split(key, 4)
            params, opt_state, carry, metrics = trainer.run_iteration(
                params, opt_state, carry, codes, pool_params, pool_kinds, progress, k_iter
            )
            steps += config.num_envs * config.rollout_ticks

            if iteration % config.snapshot_every == 0:
                pool.append(jax.tree_util.tree_map(np.asarray, params))
                pool = pool[-config.pool_capacity :]
            if iteration % config.phase_iterations == 0:
                codes = trainer.scripted_codes(k_codes)
                if config.num_pool > 0:
                    pool_params, pool_kinds = sample_pool(trainer, pool, params, rng)
                    carry = reset_pool_games(trainer, carry, k_reset)

            row = {k: float(v) for k, v in metrics.items()}
            row.update(iteration=iteration, steps=steps, seconds=time.time() - start)

            should_eval = iteration % config.eval_every == 0 or iteration == trainer.iterations
            if should_eval:
                key, k_eval = jax.random.split(key)
                score = {}
                for name in ("random", "weighted", "heuristic"):
                    k_eval, k_match = jax.random.split(k_eval)
                    result = arena.against_scripted(k_match, params, name)
                    score[name] = result
                    row[f"eval_{name}_win"] = result["win_rate"]
                checkpoint.save_params(run_dir / "latest.npz", params, config.hidden, config.rounds, config.map_type)
                checkpoint.save_params(run_dir / "snapshots" / f"it_{iteration:05d}.npz", params, config.hidden, config.rounds, config.map_type)
                if score["heuristic"]["win_rate"] >= best_score:
                    best_score = score["heuristic"]["win_rate"]
                    checkpoint.save_params(run_dir / "best.npz", params, config.hidden, config.rounds, config.map_type)

            # The row goes first so saved state never gets ahead of the metrics file.
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            if should_eval:
                _save_resume_state(
                    run_dir,
                    params=params,
                    opt_state=opt_state,
                    pool=pool,
                    iteration=iteration,
                    steps=steps,
                    seconds=row["seconds"],
                    best_score=best_score,
                    key=key,
                )
            if should_eval or iteration % 5 == 0 or iteration == 1:
                log(_format_row(row))

    checkpoint.save_params(run_dir / "latest.npz", params, config.hidden, config.rounds, config.map_type)
    return params


def _rate(row, name):
    games = row.get(f"{name}_games", 0)
    return row.get(f"{name}_wins", 0) / games if games else float("nan")


def _format_row(row):
    rate = row["steps"] / max(row["seconds"], 1e-6)
    parts = [
        f"it {row['iteration']:5d}",
        f"steps {row['steps'] / 1e6:7.2f}M",
        f"{rate:6.0f}/s",
        f"games {row['games']:4.0f}",
        f"plies {row['game_plies'] / max(row['games'], 1):4.0f}",
        f"pi {row['policy_loss']:+.3f}",
        f"v {row['value_loss']:.3f}",
        f"ent {row['entropy']:.2f}",
        f"kl {row['approx_kl']:.4f}",
        f"ev {row['explained_variance']:+.2f}",
        f"W-heur {_rate(row, 'heuristic'):.2f}",
        f"W-pool {_rate(row, 'pool'):.2f}",
    ]
    for name in ("random", "weighted", "heuristic"):
        key = f"eval_{name}_win"
        if key in row:
            parts.append(f"EVAL {name} {row[key]:.3f}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", default="runs/ppo")
    parser.add_argument("--map-type", default="BASE", choices=["BASE", "MINI", "TOURNAMENT"])
    parser.add_argument("--timesteps", type=int, default=PPOConfig.total_timesteps)
    parser.add_argument("--num-envs", type=int, default=PPOConfig.num_envs)
    parser.add_argument("--rollout-ticks", type=int, default=PPOConfig.rollout_ticks)
    parser.add_argument("--hidden", type=int, default=PPOConfig.hidden)
    parser.add_argument("--rounds", type=int, default=PPOConfig.rounds)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--update-epochs", type=int, default=PPOConfig.update_epochs)
    parser.add_argument("--entropy-coef", type=float, default=PPOConfig.entropy_coef)
    parser.add_argument("--shaping-vp", type=float, default=PPOConfig.shaping_vp)
    parser.add_argument("--pool-fraction", type=float, default=PPOConfig.pool_fraction)
    parser.add_argument("--scripted-fraction", type=float, default=PPOConfig.scripted_fraction)
    parser.add_argument("--eval-every", type=int, default=PPOConfig.eval_every)
    parser.add_argument("--eval-envs", type=int, default=PPOConfig.eval_envs)
    parser.add_argument("--init-from", default=None, help="checkpoint to warm start from")
    parser.add_argument("--resume", action="store_true", help="continue the run in --run-dir")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = PPOConfig(
        map_type=args.map_type,
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        rollout_ticks=args.rollout_ticks,
        hidden=args.hidden,
        rounds=args.rounds,
        learning_rate=args.learning_rate,
        final_learning_rate=args.learning_rate / 10,
        update_epochs=args.update_epochs,
        entropy_coef=args.entropy_coef,
        shaping_vp=args.shaping_vp,
        pool_fraction=args.pool_fraction,
        scripted_fraction=args.scripted_fraction,
        eval_every=args.eval_every,
        eval_envs=args.eval_envs,
        seed=args.seed,
    )
    init_params = None
    if args.init_from:
        init_params = checkpoint.load_params(args.init_from)[0]
    train(config, args.run_dir, init_params=init_params, resume=args.resume)


if __name__ == "__main__":
    main()
