"""SB3-Contrib Maskable PPO Actor-Critic for CatanatronEnv.

Supports both representations at catanatron/catanatron/gym/envs/catanatron_env.py:62:
- vector: Box(len(features)) ~615 dims (BASE 2p). Uses MLP extractor.
- mixed: Dict{board: Box(channels,21,11), numeric: Box} (board_tensor_features.py:27).
  Uses CNN branch for board + MLP branch for numeric, concat fusion.

Usage:
    from catanatron.gym.adapters import make_masked_env
    from catanatron.gym.policies import CatanMaskablePolicy
    from sb3_contrib.ppo_mask import MaskablePPO

    env = make_masked_env(config={"representation": "vector"})
    model = MaskablePPO(CatanMaskablePolicy, env, verbose=1,
                        policy_kwargs=dict(net_arch=[256,256], activation_fn=torch.nn.ReLU))
    model.learn(1_000_000)
    model.save("ppo_catan_vector")

For mixed (CNN):
    env = make_masked_env(config={"representation": "mixed"})
    model = MaskablePPO(CatanMaskablePolicy, env, verbose=1,
                        policy_kwargs=dict(
                            features_extractor_kwargs=dict(features_dim=256, board_channels=None),
                            net_arch=[dict(pi=[128], vf=[128])]))
"""

import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium import spaces

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    BaseFeaturesExtractor = object  # type: ignore
    MaskableActorCriticPolicy = object  # type: ignore


# ---------------------------------------------------------------------------
# Feature Extractors
# ---------------------------------------------------------------------------
class CatanVectorExtractor(BaseFeaturesExtractor):
    """MLP extractor for vector obs at catanatron_env.py:102 Box(HIGH)."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input = int(observation_space.shape[0])
        self.mlp = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class CatanMixedExtractor(BaseFeaturesExtractor):
    """CNN+MLP extractor for Dict obs at catanatron_env.py:93 board+numeric."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        board_channels: int | None = None,
        cnn_channels: tuple = (32, 64),
    ):
        super().__init__(observation_space, features_dim)

        # Infer shapes from spaces
        board_space = observation_space.spaces["board"]  # Box(channels,21,11)
        numeric_space = observation_space.spaces["numeric"]  # Box(n_numeric)
        in_channels = (
            int(board_space.shape[0]) if board_channels is None else board_channels
        )
        n_numeric = int(numeric_space.shape[0])

        # Board CNN branch: (C,21,11) -> conv -> flatten
        # Keep spatial dims small: use padding=1, kernel=3
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size: 21*11*cnn_channels[1] = 231*64=14784
        # Add a bottleneck linear to reduce
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 21, 11)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 128),
            nn.ReLU(),
        )

        # Numeric MLP branch
        self.numeric_mlp = nn.Sequential(
            nn.Linear(n_numeric, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        # SB3 passes Dict[str, Tensor] with keys "board", "numeric"
        board = observations["board"]
        numeric = observations["numeric"]
        # Ensure float
        if board.dtype != torch.float32:
            board = board.float()
        if numeric.dtype != torch.float32:
            numeric = numeric.float()

        cnn_feat = self.cnn_fc(self.cnn(board))
        num_feat = self.numeric_mlp(numeric)
        fused = torch.cat([cnn_feat, num_feat], dim=1)
        return self.fusion(fused)


class CatanFeaturesExtractor(BaseFeaturesExtractor):
    """Auto-select extractor based on observation_space type.

    - Box -> CatanVectorExtractor
    - Dict -> CatanMixedExtractor
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256, **kwargs):
        super().__init__(observation_space, features_dim)
        if isinstance(observation_space, spaces.Dict):
            self.extractor = CatanMixedExtractor(
                observation_space, features_dim=features_dim, **kwargs
            )
        elif isinstance(observation_space, spaces.Box):
            self.extractor = CatanVectorExtractor(
                observation_space, features_dim=features_dim
            )
        else:
            raise ValueError(f"Unsupported obs space: {observation_space}")

    def forward(self, observations) -> torch.Tensor:
        return self.extractor(observations)


# ---------------------------------------------------------------------------
# Maskable PPO Policies
# ---------------------------------------------------------------------------
if _SB3_AVAILABLE:

    class CatanMaskablePolicy(MaskableActorCriticPolicy):
        """Maskable PPO policy that auto-picks extractor for Catanatron.

        Extends sb3_contrib MaskableActorCriticPolicy at
        sb3_contrib/common/maskable/policies.py. Handles action masking
        via env.action_masks() at catanatron_env.py:120 or via
        ActionMasker wrapper at adapters.py:catan_mask_fn.

        Customize via policy_kwargs:
            policy_kwargs=dict(
                net_arch=[256,256],  # shared MLP
                # or separate: net_arch=dict(pi=[128], vf=[128])
                activation_fn=nn.ReLU,
                features_extractor_class=CatanFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                normalize_images=False,  # we handle board tensor already normalized 0-1
            )
        """

        def __init__(self, *args, **kwargs):
            # Default to our auto extractor if not specified
            kwargs.setdefault("features_extractor_class", CatanFeaturesExtractor)
            kwargs.setdefault("features_extractor_kwargs", dict(features_dim=256))
            # SB3 default net_arch is [64,64]; we use wider for Catan
            kwargs.setdefault("net_arch", [256, 256])
            super().__init__(*args, **kwargs)

    # Alias for convenience (same class, explicit MLP name)
    class CatanMlpPolicy(CatanMaskablePolicy):
        """Alias for CatanMaskablePolicy — kept for SB3 naming parity."""

        pass

else:

    class CatanMaskablePolicy:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "sb3-contrib and stable-baselines3 required for CatanMaskablePolicy. "
                "Install: pip install sb3-contrib stable-baselines3 gymnasium torch"
            )

    class CatanMlpPolicy(CatanMaskablePolicy):  # type: ignore
        pass
