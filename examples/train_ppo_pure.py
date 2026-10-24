"""Pure-PyTorch PPO training loop using CatanGymAdapter (no SB3).

Demonstrates how to train PurePpoActorCritic at
catanatron/catanatron/gym/policies/pure_ppo.py directly with the env
without SB3.

Run:
    python examples/train_ppo_pure.py --representation vector --timesteps 50000

This is a minimal rollout + PPO update sketch (not full sb3 feature parity).
For production, prefer examples/train_ppo_custom.py (SB3 MaskablePPO).
"""

import argparse
import numpy as np
import torch
import torch.optim as optim

from catanatron.gym.adapters import CatanGymAdapter
from catanatron.gym.policies.pure_ppo import PurePpoActorCritic


def collect_rollout(adapter, policy, n_steps=2048):
    """Collect n_steps transitions."""
    obs_list, masks_list, actions_list = [], [], []
    rewards_list, dones_list, values_list, logps_list = [], [], [], []

    obs, info = adapter.reset(seed=0)
    mask = adapter.action_mask()

    for _ in range(n_steps):
        action, logp, value = policy.act(obs, mask)

        # Store
        obs_list.append(obs if isinstance(obs, dict) else obs.copy())
        masks_list.append(mask.copy())
        actions_list.append(action)
        values_list.append(value)
        logps_list.append(logp)

        next_obs, reward, terminated, truncated, info = adapter.step(action)
        done = terminated or truncated
        rewards_list.append(reward)
        dones_list.append(done)

        if done:
            next_obs, info = adapter.reset()
        obs = next_obs
        mask = adapter.action_mask()

    return {
        "obs": obs_list,
        "masks": np.array(masks_list),
        "actions": np.array(actions_list),
        "rewards": np.array(rewards_list, dtype=np.float32),
        "dones": np.array(dones_list, dtype=np.float32),
        "values": np.array(values_list, dtype=np.float32),
        "logps": np.array(logps_list, dtype=np.float32),
    }


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0):
    """Compute GAE advantages and returns."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0
    for t in reversed(range(n)):
        next_nonterminal = 1.0 - dones[t]
        next_value = values[t + 1] if t + 1 < n else last_value
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_nonterminal * last_gae
    returns = advantages + values
    return advantages, returns


def train_pure(representation="vector", timesteps=50000, n_steps=2048, lr=3e-4, n_epochs=4):
    adapter = CatanGymAdapter(config={"representation": representation})
    act_dim = adapter.action_space.n

    policy = PurePpoActorCritic(adapter.observation_space, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    steps_done = 0
    iteration = 0
    while steps_done < timesteps:
        rollout = collect_rollout(adapter, policy, n_steps=n_steps)
        # Estimate last value for GAE bootstrap
        obs, _ = adapter.reset()  # dummy; better to use last obs but keep simple
        _, last_value = policy.forward(rollout["obs"][-1], rollout["masks"][-1])
        last_value = float(last_value.detach().cpu().numpy().mean())

        advantages, returns = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"], last_value=last_value
        )
        # Normalize advantages
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Convert to tensors for minibatch updates
        # For vector obs, stack; for mixed, handle dict
        if representation == "mixed":
            # Not implemented for brevity — use SB3 path for mixed
            raise NotImplementedError("Pure PPO mixed not batched here; use train_ppo_custom.py")
        else:
            obs_batch = torch.from_numpy(np.stack(rollout["obs"])).float()
            masks_batch = torch.from_numpy(rollout["masks"])
            actions_batch = torch.from_numpy(rollout["actions"]).long()
            old_logps_batch = torch.from_numpy(rollout["logps"]).float()
            returns_batch = torch.from_numpy(returns).float()
            adv_batch = torch.from_numpy(advantages).float()

        # PPO minibatch updates
        clip_range = 0.2
        batch_size = 64
        indices = np.arange(n_steps)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps, batch_size):
                mb = indices[start : start + batch_size]
                logp, entropy, values = policy.evaluate(
                    obs_batch[mb].numpy(), actions_batch[mb], masks_batch[mb].numpy()
                )
                # SB3 evaluate expects numpy, but our pure does torch; adapt:
                # Re-evaluate via torch forward for simplicity
                logits, v = policy.forward(obs_batch[mb].numpy(), masks_batch[mb].numpy())
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions_batch[mb])
                entropy = dist.entropy()
                v = v

                ratio = torch.exp(logp - old_logps_batch[mb])
                pg_loss1 = adv_batch[mb] * ratio
                pg_loss2 = adv_batch[mb] * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                vf_loss = ((v - returns_batch[mb]) ** 2).mean()
                ent_loss = -entropy.mean()

                loss = pg_loss + 0.5 * vf_loss + 0.01 * ent_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        steps_done += n_steps
        iteration += 1
        mean_reward = rollout["rewards"].mean()
        print(f"Iter {iteration} steps {steps_done}/{timesteps} mean_reward {mean_reward:.3f} loss {loss.item():.3f}")

    torch.save(policy.state_dict(), "pure_ppo_catan.pt")
    print("Saved pure_ppo_catan.pt")
    adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=["vector", "mixed"], default="vector")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--n-steps", type=int, default=2048)
    args = parser.parse_args()
    train_pure(args.representation, args.timesteps, args.n_steps)
