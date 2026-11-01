"""Measures throughput, so optimizations can be judged by numbers.

    python -m catanatron.jax_rl.benchmark --num-envs 256 --hidden 48

Other work on the machine skews results, so compare runs made under similar load.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from catanatron.jax_rl.env import CatanEnv
from catanatron.jax_rl.network import GraphPolicy
from catanatron.jax_rl.obs import encode_current
from catanatron.jax_rl.ppo import PPOConfig, Trainer, sample_pool


def _timed(fn, repeats=5):
    """Best time of several calls, after one call to compile."""
    jax.block_until_ready(fn())
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - started)
    return best


def measure(num_envs, hidden, rounds, iterations=2):
    env = CatanEnv("BASE")
    policy = GraphPolicy(env, hidden, rounds)
    key = jax.random.PRNGKey(0)
    params = policy.init(key)
    states = env.batch_reset(key, num_envs)

    masks = jax.vmap(env.legal_mask)(states)
    obs = jax.vmap(lambda s: encode_current(env, s))(states)
    actions = jnp.argmax(masks, axis=1)

    def loss(p):
        logits, values = jax.vmap(policy.apply, in_axes=(None, 0))(p, obs)
        return (logits**2).mean() + (values**2).mean()

    step = jax.jit(env.batch_step)
    mask = jax.jit(jax.vmap(env.legal_mask))
    encode = jax.jit(jax.vmap(lambda s: encode_current(env, s)))
    forward = jax.jit(jax.vmap(policy.apply, in_axes=(None, 0)))
    gradient = jax.jit(jax.grad(loss))

    results = {
        "step_us": _timed(lambda: step(key, states, actions)) / num_envs * 1e6,
        "mask_us": _timed(lambda: mask(states)) / num_envs * 1e6,
        "encode_us": _timed(lambda: encode(states)) / num_envs * 1e6,
        "forward_us": _timed(lambda: forward(params, obs)) / num_envs * 1e6,
        "gradient_us": _timed(lambda: gradient(params)) / num_envs * 1e6,
    }

    config = PPOConfig(
        num_envs=num_envs,
        hidden=hidden,
        rounds=rounds,
        total_timesteps=num_envs * PPOConfig.rollout_ticks * (iterations + 1),
    )
    trainer = Trainer(config)
    opt_state = trainer.optimizer.init(params)
    carry = trainer.init_carry(key)
    codes = trainer.scripted_codes(key)
    pool_params, pool_kinds = sample_pool(
        trainer, [params], params, np.random.default_rng(0)
    )

    def iterate(params, opt_state, carry, seed):
        return trainer.run_iteration(
            params,
            opt_state,
            carry,
            codes,
            pool_params,
            pool_kinds,
            0.0,
            jax.random.PRNGKey(seed),
        )

    params, opt_state, carry, _ = jax.block_until_ready(
        iterate(params, opt_state, carry, 0)
    )
    started = time.perf_counter()
    for seed in range(1, iterations + 1):
        params, opt_state, carry, _ = jax.block_until_ready(
            iterate(params, opt_state, carry, seed)
        )
    elapsed = time.perf_counter() - started
    results["training_ticks_per_s"] = (
        iterations * num_envs * config.rollout_ticks / elapsed
    )
    results["device"] = str(jax.devices()[0])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-envs", type=int, default=PPOConfig.num_envs)
    parser.add_argument("--hidden", type=int, default=PPOConfig.hidden)
    parser.add_argument("--rounds", type=int, default=PPOConfig.rounds)
    args = parser.parse_args()
    for name, value in measure(args.num_envs, args.hidden, args.rounds).items():
        print(
            f"{name}: {value:.1f}" if isinstance(value, float) else f"{name}: {value}"
        )


if __name__ == "__main__":
    main()
