# Catanatron Gymnasium Environment

For reinforcement learning purposes, we provide an Open AI Gym / Gymnasium environment. To use:

```
pip install catanatron[gym]
```

Make your training loop, ensuring to respect `env.get_valid_actions()`.

```python
import random
import gymnasium
import catanatron.gym

env = gymnasium.make('catanatron.gym/Catanatron-v0')
observation, info = env.reset()
for _ in range(1000):
  # your agent here (this takes random actions)
  action = random.choice(info['valid_actions'])

  observation, reward, terminated, truncated, info = env.step(action)
  done = terminated or truncated
  if done:
      observation, info = env.reset()
env.close()
```

For `action` documentation see [here](https://catanatron.readthedocs.io/en/latest/catanatron.gym.envs.html#catanatron.gym.envs.catanatron.gym.CatanatronEnv.action_space).

For `observation` documentation see [here](https://catanatron.readthedocs.io/en/latest/catanatron.gym.envs.html#catanatron.gym.envs.catanatron.gym.CatanatronEnv.observation_space).

You can access `env.game.state` and build your own "observation" (features) vector as well.

## Stable-Baselines3 Example

Catanatron works well with SB3, and better with the Maskable models of the [SB3 Contrib](https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html) repo. Here a small example of how it may work.

```python
import gymnasium
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import catanatron.gym


def mask_fn(env) -> np.ndarray:
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1

    return np.array([bool(i) for i in mask])


# Init Environment and Model
env = gymnasium.make("catanatron/Catanatron-v0")
env = ActionMasker(env, mask_fn)  # Wrap to enable masking
model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

# Train
model.learn(total_timesteps=10_000)
```

## Configuration

You can also configure what map to use, how many vps to win, among other variables in the environment,
with the `config` keyword argument. See source for details.

```python
import gymnasium
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
import catanatron.gym


def my_reward_function(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 100
    elif winning_color is None:
        return 0
    else:
        return -100


# 3-player catan on a "Mini" map (7 tiles) until 6 points.
env = gymnasium.make(
    "catanatron.gym/Catanatron-v0",
    config={
        "map_type": "MINI",
        "vps_to_win": 6,
        "enemies": [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.ORANGE),
        ],
        "reward_function": my_reward_function,
        "representation": "mixed",
    },
)
```

## Custom PPO Actor-Critic (new architecture)

For training with a new architecture directly on the environment, use the
built-in adapters and policies at `catanatron/catanatron/gym/adapters.py:1` and
`catanatron/catanatron/gym/policies/actor_critic.py:1`:

```python
# One-liner with helpers (handles action masking + representation)
from catanatron.gym.adapters import make_masked_env, make_vec_env
from catanatron.gym.policies import CatanMaskablePolicy
from sb3_contrib.ppo_mask import MaskablePPO

# vector: Box(614,) MLP | mixed: Dict{board:(C,21,11), numeric} CNN+MLP
env = make_masked_env(config={"representation": "vector"})
# or 4 parallel envs: env = make_vec_env(n_envs=4, config={"representation": "mixed"})

model = MaskablePPO(
    CatanMaskablePolicy,
    env,
    verbose=1,
    policy_kwargs=dict(
        features_extractor_kwargs=dict(features_dim=256),  # CNN/MLP width
        net_arch=[256, 256],  # shared trunk; try dict(pi=[128], vf=[128])
    ),
)
model.learn(total_timesteps=1_000_000)
model.save("ppo_catan_custom")
```

Or run the scripted example:

```bash
pip install catanatron[gym] sb3-contrib stable-baselines3 torch
python examples/train_ppo_custom.py --representation vector --timesteps 100000
python examples/train_ppo_custom.py --representation mixed --timesteps 500000 --n-envs 4
```

For pure PyTorch without SB3, see `catanatron/catanatron/gym/policies/pure_ppo.py:1`
`PurePpoActorCritic` + `catanatron/gym/adapters.py:133` `CatanGymAdapter`:

```python
from catanatron.gym.adapters import CatanGymAdapter
from catanatron.gym.policies.pure_ppo import PurePpoActorCritic

adapter = CatanGymAdapter(config={"representation": "vector"})
policy = PurePpoActorCritic(adapter.observation_space, adapter.action_space.n)
mask = adapter.action_mask()  # bool array for logits masking
logits, value = policy.forward(obs, mask)  # invalid logits -> -1e8
```

See `examples/train_ppo_custom.py:1` and `examples/train_ppo_pure.py:1` for full loops.

### Appendix

This project was created with:

```bash
copier copy https://github.com/Farama-Foundation/gymnasium-env-template.git "path/to/directory"
```

