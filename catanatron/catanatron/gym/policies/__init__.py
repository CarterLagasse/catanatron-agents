"""Custom PPO Actor-Critic policies for Catanatron.

Exports:
- CatanVectorExtractor / CatanMixedExtractor / CatanFeaturesExtractor (auto)
- CatanMaskablePolicy / CatanMlpPolicy (SB3-Contrib MaskablePPO)
- PurePpoActorCritic (pure PyTorch, no SB3)
"""

try:
    from catanatron.gym.policies.actor_critic import (
        CatanFeaturesExtractor,
        CatanMaskablePolicy,
        CatanMlpPolicy,
        CatanMixedExtractor,
        CatanVectorExtractor,
    )
except ImportError:
    # Allow import of pure policy without SB3
    CatanFeaturesExtractor = None
    CatanMaskablePolicy = None
    CatanMlpPolicy = None
    CatanMixedExtractor = None
    CatanVectorExtractor = None

try:
    from catanatron.gym.policies.pure_ppo import PurePpoActorCritic
except ImportError:
    PurePpoActorCritic = None

__all__ = [
    "CatanFeaturesExtractor",
    "CatanVectorExtractor",
    "CatanMixedExtractor",
    "CatanMaskablePolicy",
    "CatanMlpPolicy",
    "PurePpoActorCritic",
]
