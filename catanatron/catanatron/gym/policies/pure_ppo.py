"""Pure PyTorch PPO Actor-Critic (no SB3 dependency) for CatanatronEnv.

Use this if you want to train without sb3-contrib and control PPO loop yourself.
Works directly with CatanGymAdapter at catanatron/catanatron/gym/adapters.py
and handles action masking by setting invalid logits to -1e8.

Architecture mirrors sb3 policy but standalone:
- vector: MLP 256->256 -> shared trunk -> pi head (Discrete) + vf head (1)
- mixed:  CNN(board) + MLP(numeric) -> fusion -> shared trunk -> pi/vf

Example training loop sketch:
    from catanatron.gym.adapters import CatanGymAdapter
    from catanatron.gym.policies.pure_ppo import PurePpoActorCritic
    import torch, numpy as np

    adapter = CatanGymAdapter(config={"representation": "vector"})
    obs_space = adapter.observation_space
    act_dim = adapter.action_space.n
    policy = PurePpoActorCritic(obs_space, act_dim, features_dim=256)

    obs, info = adapter.reset(seed=0)
    mask = adapter.action_mask()
    action, logp, value = policy.act(obs, mask)  # sample
    next_obs, reward, terminated, truncated, info = adapter.step(int(action))
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Categorical


def _get_obs_dim(observation_space):
    if isinstance(observation_space, spaces.Box):
        return int(observation_space.shape[0])
    elif isinstance(observation_space, spaces.Dict):
        board_dim = int(np.prod(observation_space.spaces["board"].shape))
        num_dim = int(observation_space.spaces["numeric"].shape[0])
        return board_dim + num_dim
    else:
        raise ValueError(f"Unsupported space: {observation_space}")


class PurePpoActorCritic(nn.Module):
    """Standalone Actor-Critic for PPO.

    Args:
        observation_space: from CatanatronEnv (Box or Dict)
        action_dim: len(get_action_array()) ~327
        features_dim: trunk output dim
        net_arch: shared trunk hidden dims e.g. [256,256] or dict(pi=[128], vf=[128])
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        features_dim: int = 256,
        net_arch: list | dict = None,
        activation_fn=nn.ReLU,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_dim = action_dim
        self.is_mixed = isinstance(observation_space, spaces.Dict)

        if net_arch is None:
            net_arch = [256, 256]

        # Feature extractor
        if self.is_mixed:
            board_space = observation_space.spaces["board"]
            numeric_space = observation_space.spaces["numeric"]
            in_ch = int(board_space.shape[0])
            n_num = int(numeric_space.shape[0])

            self.cnn = nn.Sequential(
                nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
                activation_fn(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                activation_fn(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, in_ch, 21, 11)
                cnn_out = self.cnn(dummy).shape[1]
            self.cnn_fc = nn.Sequential(nn.Linear(cnn_out, 128), activation_fn())
            self.num_mlp = nn.Sequential(
                nn.Linear(n_num, 64), activation_fn(), nn.Linear(64, 64), activation_fn()
            )
            self.fusion = nn.Sequential(nn.Linear(128 + 64, features_dim), activation_fn())
            trunk_input = features_dim
        else:
            n_input = int(observation_space.shape[0])
            self.vector_mlp = nn.Sequential(
                nn.Linear(n_input, 256),
                activation_fn(),
                nn.Linear(256, 256),
                activation_fn(),
                nn.Linear(256, features_dim),
                activation_fn(),
            )
            trunk_input = features_dim

        # Shared or separate trunk
        if isinstance(net_arch, dict):
            pi_arch = net_arch.get("pi", [128])
            vf_arch = net_arch.get("vf", [128])
            self.shared = False
            self.pi_trunk = self._make_mlp(trunk_input, pi_arch, activation_fn)
            self.vf_trunk = self._make_mlp(trunk_input, vf_arch, activation_fn)
            pi_out = pi_arch[-1] if pi_arch else trunk_input
            vf_out = vf_arch[-1] if vf_arch else trunk_input
        else:
            self.shared = True
            self.shared_trunk = self._make_mlp(trunk_input, net_arch, activation_fn)
            pi_out = net_arch[-1] if net_arch else trunk_input
            vf_out = pi_out

        self.policy_head = nn.Linear(pi_out, action_dim)
        self.value_head = nn.Linear(vf_out, 1)

        # Orthogonal init like SB3
        self._init_weights()

    def _make_mlp(self, in_dim, arch, act):
        layers = []
        prev = in_dim
        for h in arch:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            prev = h
        return nn.Sequential(*layers) if layers else nn.Identity()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _extract(self, obs):
        """obs: ndarray or dict of ndarrays -> Tensor batch."""
        if self.is_mixed:
            # obs is dict of np arrays or tensors, batch or single
            board = obs["board"]
            numeric = obs["numeric"]
            if isinstance(board, np.ndarray):
                board = torch.from_numpy(board)
                numeric = torch.from_numpy(numeric)
            # Ensure batch dim
            if board.dim() == 3:
                board = board.unsqueeze(0)
            if numeric.dim() == 1:
                numeric = numeric.unsqueeze(0)
            board = board.float()
            numeric = numeric.float()
            cnn_f = self.cnn_fc(self.cnn(board))
            num_f = self.num_mlp(numeric)
            return self.fusion(torch.cat([cnn_f, num_f], dim=1))
        else:
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            obs = obs.float()
            return self.vector_mlp(obs)

    def forward(self, obs, action_mask=None):
        """Returns logits (masked) and value."""
        feat = self._extract(obs)
        if self.shared:
            feat = self.shared_trunk(feat)
            pi_feat = vf_feat = feat
        else:
            pi_feat = self.pi_trunk(feat)
            vf_feat = self.vf_trunk(feat)

        logits = self.policy_head(pi_feat)
        value = self.value_head(vf_feat).squeeze(-1)

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(logits.device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            # Mask invalid to large negative
            logits = torch.where(
                action_mask.bool(), logits, torch.full_like(logits, -1e8)
            )
        return logits, value

    @torch.no_grad()
    def act(self, obs, action_mask=None):
        """Sample action, return (action, log_prob, value)."""
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # Squeeze batch dim for single obs
        if action.numel() == 1:
            return int(action.item()), float(log_prob.item()), float(value.item())
        return action, log_prob, value

    def evaluate(self, obs, actions, action_mask=None):
        """For PPO update: returns log_prob, entropy, value."""
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
