"""Adapters to train custom PPO Actor-Critic agents directly with CatanatronEnv.

Provides:
- mask_fn / catan_mask_fn: action masking for SB3-Contrib MaskablePPO
- make_masked_env / make_catan_env: env factory helpers (single + vectorized)
- Pure PyTorch PPO-compatible helpers for from-scratch training

Example (SB3):
    from catanatron.gym.adapters import make_masked_env
    env = make_masked_env(representation="vector")
    from catanatron.gym.policies import CatanMaskablePolicy
    from sb3_contrib.ppo_mask import MaskablePPO
    model = MaskablePPO(CatanMaskablePolicy, env, verbose=1)
    model.learn(1_000_000)

Example (pure PyTorch rollout):
    from catanatron.gym.adapters import CatanGymAdapter
    adapter = CatanGymAdapter(config={"representation": "vector"})
    obs, info = adapter.reset()
    mask = adapter.action_mask()  # bool ndarray for logits masking
"""

import gymnasium as gym
import numpy as np

import catanatron.gym  # noqa: F401  # register env


def catan_mask_fn(env) -> np.ndarray:
    """Mask function for SB3-Contrib ActionMasker.

    Returns bool array of length action_space.n where True == valid.
    Uses env.unwrapped.get_valid_actions() at catanatron/gym/envs/catanatron_env.py:108
    """
    valid = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid] = True
    return mask


# Backcompat alias
mask_fn = catan_mask_fn


def make_catan_env(config=None, **kwargs):
    """Create a raw CatanatronEnv (no masking wrapper).

    Args:
        config: dict passed to CatanatronEnv (see catanatron_env.py:53)
            keys: map_type, vps_to_win, enemies, reward_function, representation
        **kwargs: forwarded to gymnasium.make

    Returns:
        gym.Env: CatanatronEnv (has .get_valid_actions() and .action_masks())
    """
    return gym.make("catanatron/Catanatron-v0", config=config or {}, **kwargs)


def make_masked_env(config=None, use_action_masker=True, **kwargs):
    """Create a masked env ready for MaskablePPO.

    Args:
        config: same as make_catan_env
        use_action_masker: if True wraps with sb3_contrib ActionMasker using
            catan_mask_fn. If False, returns raw env which already implements
            action_masks() at catanatron_env.py:120 (SB3 will call it directly
            when use_masks=True is passed to predict/learn).
        **kwargs: forwarded to gymnasium.make

    Returns:
        gym.Env: masked env
    """
    env = make_catan_env(config=config, **kwargs)
    # Wrap with Monitor for rollout/ep_rew_mean logging (one-line logger)
    try:
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(env)
    except Exception:
        pass
    if use_action_masker:
        try:
            from sb3_contrib.common.wrappers import ActionMasker

            env = ActionMasker(env, catan_mask_fn)
        except ImportError as e:
            raise ImportError(
                "sb3-contrib required for ActionMasker. "
                "Install with: pip install sb3-contrib stable-baselines3"
            ) from e
    return env


def make_vec_env(n_envs=4, config=None, use_action_masker=True, seed=None):
    """Create a vectorized masked env (DummyVecEnv) for faster PPO training.

    Args:
        n_envs: number of parallel envs
        config: forwarded to make_masked_env
        use_action_masker: whether to wrap each sub-env
        seed: optional base seed

    Returns:
        VecEnv: SB3 DummyVecEnv wrapping n copies
    """
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        raise ImportError("stable-baselines3 required for make_vec_env") from e

    def _make_one(rank):
        def _init():
            env = make_masked_env(config=config, use_action_masker=use_action_masker)
            if seed is not None:
                env.reset(seed=seed + rank)
            return env

        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])


class CatanGymAdapter:
    """Lightweight adapter for pure-PyTorch PPO loops (no SB3 dependency).

    Wraps CatanatronEnv and exposes tensor-friendly helpers:
    - observation as dict or ndarray (matching representation)
    - action_mask() for logits masking (set invalid logits to -1e8)
    - step() returning (obs, reward, terminated, truncated, info)

    Useful if you implement PPO from scratch and want to avoid SB3.

    Example:
        adapter = CatanGymAdapter(config={"representation": "mixed"})
        obs, info = adapter.reset(seed=0)
        # obs["board"].shape == (channels, 21, 11), obs["numeric"].shape == (N,)
        mask = adapter.action_mask()  # shape (action_space.n,)
        logits = policy(obs)  # your network
        logits[~mask] = -1e8
        action = torch.distributions.Categorical(logits=logits).sample()
        obs, reward, terminated, truncated, info = adapter.step(int(action))
    """

    def __init__(self, config=None, **kwargs):
        self.env = make_catan_env(config=config, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def get_valid_actions(self):
        return self.env.unwrapped.get_valid_actions()

    def action_mask(self) -> np.ndarray:
        """Bool mask for current state (True == valid)."""
        valid = self.get_valid_actions()
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[valid] = True
        return mask

    def action_masks(self):
        """Alias for SB3 compatibility (catanatron_env.py:120)."""
        return self.action_mask().tolist()

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Delegate to underlying env (e.g., game, render)
        return getattr(self.env, name)
