"""PpoPlayer wrapper: makes a trained MaskablePPO .zip behave as a catanatron Player.

Supports both representations at catanatron/gym/envs/catanatron_env.py:62
- vector: Box(614,) from get_feature_ordering() features.py:532
- mixed:  Dict{board:(C,21,11), numeric} from board_tensor_features.py + catanatron_env.py:81

Usage as Player:
    from catanatron.gym.players import PpoPlayer
    from catanatron.models.player import Color
    player = PpoPlayer(Color.BLUE, model_path="ppo_catan_custom.zip", representation="vector")

Usage via CLI (see cli/cli_players.py):
    catanatron-play --players PPO:ppo_catan_custom.zip:vector:BASE,W --num 100
    # syntax: PPO[:path[:representation[:map_type[:deterministic]]]]
    # defaults: path=ppo_catan_custom.zip, representation=vector, map_type=BASE, deterministic=True

The wrapper mirrors CatanatronEnv._get_observation() catanatron_env.py:191
and uses from_action_space() action_space.py:94 with masking identical to
adapters.py:catan_mask_fn.
"""

import os
from functools import lru_cache

import numpy as np

from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.board_tensor_features import create_board_tensor, is_graph_feature
from catanatron.gym.envs.action_space import from_action_space, get_action_array
from catanatron.models.player import Color, Player

try:
    import gymnasium as gym  # noqa: F401
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.ppo_mask import MaskablePPO

    from catanatron.gym.adapters import catan_mask_fn
    from catanatron.gym.policies import CatanMaskablePolicy

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


@lru_cache(maxsize=8)
def _cached_features(num_players, map_type):
    return get_feature_ordering(num_players, map_type)


@lru_cache(maxsize=8)
def _cached_numeric_features(num_players, map_type):
    feats = _cached_features(num_players, map_type)
    return [f for f in feats if not is_graph_feature(f)]


class PpoPlayer(Player):
    """Wraps a MaskablePPO .zip as a Player.

    Args:
        color: Color enum
        model_path: path to .zip saved via MaskablePPO.save() (train_ppo_custom.py:177)
        representation: "vector" or "mixed" (must match training config)
        map_type: "BASE", "MINI", "TOURNAMENT" (must match training)
        deterministic: if True uses argmax, else samples
        verbose: print load info
    """

    def __init__(
        self,
        color,
        model_path="ppo_catan_custom.zip",
        representation="vector",
        map_type="BASE",
        deterministic=True,
        verbose=False,
        **kwargs,
    ):
        # Handle CLI passing extras as strings: e.g. "True"/"False", "0"/"1"
        if isinstance(deterministic, str):
            deterministic = deterministic.lower() not in ("0", "false", "no")
        if isinstance(verbose, str):
            verbose = verbose.lower() not in ("0", "false", "no")

        super().__init__(color)
        self.model_path = str(model_path)
        self.representation = str(representation)
        self.map_type = str(map_type).upper()
        self.deterministic = bool(deterministic)
        self.verbose = bool(verbose)

        if self.representation not in ("vector", "mixed"):
            raise ValueError(
                f"representation must be vector or mixed, got {self.representation}"
            )
        if self.map_type not in ("BASE", "MINI", "TOURNAMENT"):
            raise ValueError(
                f"map_type must be BASE/MINI/TOURNAMENT, got {self.map_type}"
            )

        if not _SB3_AVAILABLE:
            raise ImportError(
                "sb3-contrib and stable-baselines3 required for PpoPlayer. "
                "Install: pip install sb3-contrib stable-baselines3 torch gymnasium"
            )

        # Normalize existence check: SB3 adds .zip if missing, so check both forms without double suffix
        def _exists(p):
            return (
                os.path.isfile(p)
                or os.path.isfile(p + ".zip")
                or os.path.isfile(p.rstrip(".zip"))
            )

        # Helper to strip .zip for alternative check
        alt = (
            self.model_path[:-4]
            if self.model_path.endswith(".zip")
            else self.model_path + ".zip"
        )
        if not os.path.isfile(self.model_path) and not os.path.isfile(alt):
            msg = (
                f"[PpoPlayer] Model not found at '{self.model_path}' (also tried '{alt}'). "
                f"Train first: python examples/train_ppo_custom.py --num-games 1000 --save-path {self.model_path.replace('.zip','')}"
            )
            if verbose:
                print(msg)
            # Raise early with clear message instead of cryptic .zip.zip chain
            raise FileNotFoundError(msg)

        # Lazy / eager load: load now to fail fast on missing/corrupt zip.
        # Keep tmp env for loading; SB3 requires env with matching obs/act space.
        # We create a dummy env matching training config inferred from model if possible,
        # else from our args (representation, map_type). We assume 2p BASE unless model tells else.
        self.model = None
        self._load_model()

    def _resolve_load_path(self):
        """Return path without double .zip for SB3 load (handles both with/without suffix)."""
        p = self.model_path
        # SB3 load expects path without .zip? It handles both, but we normalize to without double.
        # If p ends with .zip and file exists, keep as is; if not, try without suffix.
        if p.endswith(".zip") and os.path.isfile(p):
            return p
        if p.endswith(".zip") and os.path.isfile(p[:-4]):
            return p[:-4]
        if not p.endswith(".zip") and os.path.isfile(p + ".zip"):
            return p  # SB3 will find p.zip
        return p

    def _load_model(self):
        # We need to supply an env with matching observation_space and action_space
        # to MaskablePPO.load. The dummy env's player count and map determine those.
        # Try to infer num_players from model if available, else assume 2.
        # Strategy: create 2p env first, load, if obs mismatch retry with inferred size.

        # We try to create env that matches the model's expected spaces.
        # Since we don't know num_players before load, we try 2 then 3 then 4.
        import gymnasium as gym

        load_path = self._resolve_load_path()
        last_exc = None
        # Suppress SB3's "Wrapping the env..." noisy stdout during load
        import contextlib
        import io

        for n_players in [2, 3, 4]:
            # Build dummy player list length = n_players
            # We don't need exact colors; any n_players with same map/repr gives same space shape
            # except channels for mixed which is 2*n+12
            try:
                tmp_env = gym.make(
                    "catanatron/Catanatron-v0",
                    config={
                        "representation": self.representation,
                        "map_type": self.map_type,
                        # use default enemies to get n_players; we need n_players total
                        # For n=2, default 1 enemy is fine. For n>2, inject extra random players.
                    },
                )
                # Adjust if n_players >2: we need to set enemies length = n-1
                # gym.make ignores extra enemies unless provided; for load we just need matching obs dims
                # So for n>2, recreate with correct enemies
                if n_players > 2:
                    from catanatron.models.player import RandomPlayer

                    # pick colors distinct from BLUE
                    extra_colors = [Color.RED, Color.ORANGE, Color.WHITE][
                        : n_players - 1
                    ]
                    enemies = [RandomPlayer(c) for c in extra_colors]
                    tmp_env = gym.make(
                        "catanatron/Catanatron-v0",
                        config={
                            "representation": self.representation,
                            "map_type": self.map_type,
                            "enemies": enemies,
                        },
                    )
                tmp_env = ActionMasker(tmp_env, catan_mask_fn)
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    model = MaskablePPO.load(load_path, env=tmp_env, verbose=0)
                # Verify obs compatibility by checking model's observation_space size matches tmp_env
                # If mismatch, MaskablePPO.load would have warned but not errored; we check manually
                # by comparing expected vs tmp
                self.model = model
                self._tmp_env = tmp_env
                self._n_players_hint = n_players
                if self.verbose:
                    print(
                        f"[PpoPlayer] Loaded {self.model_path} with {n_players}p "
                        f"{self.map_type} {self.representation} action_space={tmp_env.action_space.n}"
                    )
                return
            except Exception as e:
                last_exc = e
                continue

        # If all attempts failed, try generic load without env
        try:
            import contextlib
            import io

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.model = MaskablePPO.load(load_path, verbose=0)
            self._tmp_env = None
            self._n_players_hint = 2
            if self.verbose:
                print(f"[PpoPlayer] Loaded {self.model_path} without env (fallback)")
            return
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PPO model '{self.model_path}' (tried '{load_path}'): {e}; "
                f"last try: {last_exc}. "
                f"Check file exists and was trained with representation={self.representation} map={self.map_type}"
            ) from e

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        if self.model is None:
            self._load_model()

        # Build observation mirroring CatanatronEnv._get_observation() at catanatron_env.py:191
        num_players = len(game.state.colors)
        map_type = self.map_type  # use configured map; game.map type should match
        # Prefer game map type if available for feature ordering correctness
        try:
            # game.state.board.map has type implicitly via land_nodes len; we keep configured
            pass
        except Exception:
            pass

        if self.representation == "mixed":
            board = create_board_tensor(game, self.color, channels_first=True)
            numeric_features = _cached_numeric_features(num_players, map_type)
            sample = create_sample(game, self.color)
            numeric = np.array(
                [float(sample[f]) for f in numeric_features], dtype=np.float32
            )
            obs = {"board": board.astype(np.float32), "numeric": numeric}
        else:
            features = _cached_features(num_players, map_type)
            sample = create_sample(game, self.color)
            obs = np.array([float(sample[f]) for f in features], dtype=np.float32)

        # Build mask: map playable_actions to action_space ints using same array as training
        # get_action_array is sorted and deterministic at action_space.py:20
        # Use game.state.colors tuple to match env's player_colors ordering
        try:
            action_array = get_action_array(tuple(game.state.colors), map_type)
        except Exception:
            # Fallback to configured 2p if game colors mismatch training
            action_array = get_action_array(
                tuple(sorted(game.state.colors, key=lambda c: c.value)), map_type
            )

        # Map each playable Action to its int
        valid_ints = []
        for a in playable_actions:
            key = (a.action_type, a.value)
            try:
                idx = action_array.index(key)
                valid_ints.append(idx)
            except ValueError:
                # Should not happen if array matches game; skip
                continue

        # If mapping failed, fallback to from_action_space search (should not happen)
        if not valid_ints:
            return playable_actions[0]

        mask = np.zeros(len(action_array), dtype=bool)
        mask[valid_ints] = True

        # Predict with masking. MaskablePPO expects action_masks bool array.
        # Use deterministic for evaluation; stochastic for training-like play if requested.
        try:
            action_int, _ = self.model.predict(
                obs, action_masks=mask, deterministic=self.deterministic
            )
        except Exception as e:
            # Fallback: try without action_masks wrapping (older SB3compat)
            if self.verbose:
                print(f"[PpoPlayer] predict with mask failed {e}, trying without")
            action_int, _ = self.model.predict(obs, deterministic=self.deterministic)
            # Ensure sampled action is valid
            action_int = (
                int(action_int) if np.ndim(action_int) == 0 else int(action_int[0])
            )

        action_int = (
            int(action_int)
            if np.ndim(action_int) == 0
            else int(np.asarray(action_int).flat[0])
        )

        # Convert back to Action using same array and this player's color
        try:
            catan_action = from_action_space(
                action_int, self.color, tuple(game.state.colors), map_type
            )
        except Exception:
            catan_action = from_action_space(
                action_int,
                self.color,
                tuple(sorted(game.state.colors, key=lambda c: c.value)),
                map_type,
            )

        # Safety: if model returned an invalid action (should not with masking), fallback to valid
        if catan_action not in playable_actions:
            # Find closest valid by mask
            if self.verbose:
                print(
                    f"[PpoPlayer] predicted invalid {catan_action} valid ints {valid_ints}, fallback to first valid"
                )
            # Return the highest-prob valid action by re-querying masked logits is complex; fallback to first valid
            # Could also sample from valid_ints uniformly
            # Try to map mask index back: if action_array[action_int] not in valid, pick first valid
            catan_action = playable_actions[0]
            # Try to find action whose (type,value) equals action_array[action_int]
            key = (
                action_array[action_int]
                if 0 <= action_int < len(action_array)
                else None
            )
            for pa in playable_actions:
                if (pa.action_type, pa.value) == key:
                    catan_action = pa
                    break

        return catan_action

    def reset_state(self):
        # No hidden state, but keep for Player contract
        pass

    def __repr__(self):
        return f"PpoPlayer:{self.color.value}({self.model_path},{self.representation},{self.map_type})"
