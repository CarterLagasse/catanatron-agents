"""Train PPO with self-play + mixed opponent pool.

Trains MaskablePPO (CatanMaskablePolicy) vs sampled enemy each episode:
- random: RandomPlayer
- weighted: WeightedRandomPlayer (cities>settlements>devs)
- value: ValueFunctionPlayer (base_fn)
- alphabeta: AlphaBetaPlayer depth=2 (MCTS alternative, fast enough)
- self: PpoPlayer snapshot of current policy (league/self-play)

Install:
    pip install catanatron[gym]
    pip install sb3-contrib stable-baselines3 torch

Run (sensible default mix — edit later after lit review):
    python examples/train_ppo_selfplay.py --num-games 1000 --representation vector --n-envs 4
    python examples/train_ppo_selfplay.py --opponents random,weighted,value,alphabeta,self --num-games 5000 --n-envs 4 --self-update-every 200 --map-type BASE
    # Legacy timesteps:
    python examples/train_ppo_selfplay.py --timesteps 200000 --representation mixed --n-envs 4

Notes:
- 1 timestep = 1 P0 decision (not 1 game). StopAfterGamesCallback budgets by games.
- Self pool: saves checkpoint to ppo_agents/ppo_training/<run>/iter_<games>.zip every --self-update-every games (league size --self-pool-size FIFO).
- Per-episode opponent resampling via ResampleWrapper (keeps 1v1 BLUE vs RED, action space fixed).
- For now representation/mixed both supported; map_type BASE/MINI/TOURNAMENT.
"""

import argparse
import datetime
import os
import random
import sys
import time
from pathlib import Path

import gymnasium as gym

import catanatron.gym  # register env
from catanatron.gym.adapters import catan_mask_fn
from catanatron.gym.policies import CatanMaskablePolicy


OPPONENT_CHOICES = ["random", "weighted", "value", "alphabeta", "self"]


def build_enemy(spec, color, map_type="BASE", snapshot_path=None, representation="vector", ab_depth=2):
    """Create a Player enemy for given spec."""
    spec = spec.lower()
    if spec == "random":
        from catanatron.models.player import RandomPlayer

        return RandomPlayer(color)
    if spec == "weighted":
        from catanatron.players.weighted_random import WeightedRandomPlayer

        return WeightedRandomPlayer(color)
    if spec == "value":
        from catanatron.players.value import ValueFunctionPlayer

        # base_fn by default
        return ValueFunctionPlayer(color)
    if spec == "alphabeta":
        from catanatron.players.minimax import AlphaBetaPlayer

        return AlphaBetaPlayer(color, depth=int(ab_depth))
    if spec == "self":
        if snapshot_path is None:
            from catanatron.models.player import RandomPlayer

            return RandomPlayer(color)
        from catanatron.gym.players import PpoPlayer

        # PpoPlayer handles deterministic + representation/map must match training
        return PpoPlayer(
            color,
            model_path=snapshot_path,
            representation=representation,
            map_type=map_type,
            deterministic=True,
            verbose=False,
        )
    raise ValueError(f"Unknown opponent spec: {spec}")


class OpponentSampler:
    """Samples (spec, snapshot_path) for each episode.

    If spec==self but pool empty, falls back to non-self spec.
    """

    def __init__(self, specs, weights, snapshot_pool):
        self.specs = list(specs)
        norm = sum(weights)
        self.weights = [w / norm for w in weights] if norm > 0 else [1 / len(weights)] * len(weights)
        self.pool = snapshot_pool  # list, mutated externally

    def __call__(self):
        spec = random.choices(self.specs, weights=self.weights, k=1)[0]
        if spec == "self":
            if not self.pool:
                # fallback to any non-self
                non_self = [s for s in self.specs if s != "self"]
                if not non_self:
                    return "random", None
                spec = random.choice(non_self)
                return spec, None
            path = random.choice(self.pool)
            return spec, path
        return spec, None


class ResampleWrapper(gym.Wrapper):
    """gym.Wrapper that resamples opponent on each reset().

    Keeps 1v1 (BLUE vs single RED enemy) so action_space stays fixed.
    Wraps an ActionMasker(CatanatronEnv) and mutates underlying CatanatronEnv.enemies.
    """

    def __init__(self, env, sampler, base_config, ab_depth=2):
        super().__init__(env)
        self.sampler = sampler
        self.base_config = dict(base_config)
        self.ab_depth = ab_depth
        self.spec_history = []

    def reset(self, **kwargs):
        from catanatron.models.player import Color

        spec, snapshot_path = self.sampler()
        self.spec_history.append(spec)
        try:
            enemy = build_enemy(
                spec,
                Color.RED,
                map_type=self.base_config.get("map_type", "BASE"),
                snapshot_path=snapshot_path,
                representation=self.base_config.get("representation", "vector"),
                ab_depth=self.ab_depth,
            )
        except Exception as e:
            # Fallback to random on failure (e.g. snapshot corrupt, AB timeout)
            print(f"[ResampleWrapper] build_enemy {spec} failed {e}, fallback random", file=sys.stderr)
            from catanatron.models.player import RandomPlayer

            enemy = RandomPlayer(Color.RED)
            spec = "random"

        # Mutate underlying CatanatronEnv (unwrap ActionMasker)
        try:
            catan_env = self.env.unwrapped  # CatanatronEnv
            catan_env.enemies = [enemy]
            catan_env.players = [catan_env.p0] + catan_env.enemies
        except Exception:
            pass
        return self.env.reset(**kwargs)

    def action_masks(self):
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()
        return self.env.unwrapped.action_masks()


def make_selfplay_env(config, sampler, seed=None, ab_depth=2):
    """Create single resampling masked env."""
    import gymnasium as gym
    from sb3_contrib.common.wrappers import ActionMasker

    # initial sample for construction (so initial space correct)
    spec, spath = sampler()
    from catanatron.models.player import Color

    try:
        enemy = build_enemy(spec, Color.RED, map_type=config.get("map_type", "BASE"), snapshot_path=spath, representation=config.get("representation", "vector"), ab_depth=ab_depth)
    except Exception:
        from catanatron.models.player import RandomPlayer

        enemy = RandomPlayer(Color.RED)
    cfg = dict(config)
    cfg["enemies"] = [enemy]
    env = gym.make("catanatron/Catanatron-v0", config=cfg)
    if seed is not None:
        # will be reset with seed later
        pass
    try:
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(env)
    except Exception:
        pass
    env = ActionMasker(env, catan_mask_fn)
    env = ResampleWrapper(env, sampler, config, ab_depth=ab_depth)
    return env


def make_selfplay_vec_env(n_envs, config, sampler, seed=None, ab_depth=2):
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        raise ImportError("stable-baselines3 required for make_selfplay_vec_env") from e

    def _make_one(rank):
        def _init():
            env = make_selfplay_env(config, sampler, seed=(seed + rank) if seed is not None else None, ab_depth=ab_depth)
            if seed is not None:
                env.reset(seed=seed + rank)
            return env

        return _init

    return DummyVecEnv([_make_one(i) for i in range(n_envs)])


def make_unified_callback(target_games, snapshot_pool, save_path, checkpoint_dir, update_every, pool_size, verbose, start_time, total_games_hint):
    """Single callback: counts games, saves snapshots, logs one-line per iteration, stops at target."""
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as e:
        raise ImportError("stable-baselines3 required") from e
    import numpy as np

    class _Unified(BaseCallback):
        def __init__(self, target, pool, save_path, ckpt_dir, every, max_pool, verbose, start, total_hint):
            super().__init__(verbose)
            self.target_games = int(target) if target is not None else None
            self.pool = pool
            self.save_path = save_path
            self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
            self.every = int(every) if every else None
            self.max_pool = int(max_pool) if max_pool else 5
            self.start_time = start
            self.total_hint = total_hint
            self.games_done = 0
            self.last_n_updates = -1
            self.ep_rewards = None
            self.ep_rew_mean = 0.0
            self.completed_rews = []
            if self.ckpt_dir is not None:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        def _fetch(self, key, default=0.0):
            # Try model logger
            try:
                v = self.model.logger.name_to_value.get(key, None)
                if v is not None:
                    return v
            except Exception:
                pass
            try:
                v = self.logger.name_to_value.get(key, None)
                if v is not None:
                    return v
            except Exception:
                pass
            return default

        def _on_step(self) -> bool:
            # track episode reward mean manually
            try:
                rews = self.locals.get("rewards")
                dones_raw = self.locals.get("dones")
                if rews is not None and dones_raw is not None:
                    import numpy as _np

                    if self.ep_rewards is None:
                        self.ep_rewards = _np.zeros_like(_np.asarray(rews, dtype=float), dtype=float)
                    rews_arr = _np.asarray(rews, dtype=float)
                    dones_arr = _np.asarray(dones_raw, dtype=bool)
                    if rews_arr.shape == ():
                        rews_arr = _np.array([rews_arr])
                        dones_arr = _np.array([dones_arr])
                    self.ep_rewards += rews_arr
                    for i, d in enumerate(dones_arr):
                        if d:
                            self.completed_rews.append(float(self.ep_rewards[i]))
                            self.ep_rewards[i] = 0.0
                    if self.completed_rews:
                        last = self.completed_rews[-100:]
                        self.ep_rew_mean = float(sum(last) / len(last))
            except Exception:
                pass
            dones = self.locals.get("dones")
            if dones is None:
                dones = self.locals.get("done")
                if dones is None:
                    dones = np.array([False])
                else:
                    dones = np.array([dones])
            try:
                count = int(np.sum(dones))
            except Exception:
                count = int(bool(dones))
            if count:
                self.games_done += count
                # snapshot -> ppo_agents/ppo_training/<run>/iter_<games>.zip
                if self.pool is not None and self.every and self.ckpt_dir is not None and self.games_done % self.every == 0:
                    try:
                        base = str(self.ckpt_dir / f"iter_{self.games_done}")
                        self.model.save(base)
                        actual = base + ".zip"
                        self.pool.append(actual if os.path.isfile(actual) else base)
                        while len(self.pool) > self.max_pool:
                            self.pool.pop(0)
                        if self.verbose:
                            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            elapsed = int(time.time() - self.start_time)
                            print(f"{ts} | {elapsed//60:02d}:{elapsed%60:02d} | iter_{self.games_done} -> {self.pool[-1]} pool={len(self.pool)}", flush=True)
                    except Exception as e:
                        print(f"[self-play] snapshot save failed {e}", file=sys.stderr)

            # one-line log per PPO update (when _n_updates increased)
            try:
                n_updates = getattr(self.model, "_n_updates", 0) or 0
            except Exception:
                n_updates = 0
            if n_updates != self.last_n_updates and n_updates > 0:
                self.last_n_updates = n_updates
                # only log after at least one rollout (avoid first empty)
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elapsed = int(time.time() - self.start_time)
                elapsed_s = f"{elapsed//60:02d}:{elapsed%60:02d}"
                # iter = total_timesteps // 2048 (constant per iteration)
                try:
                    it = int(self.num_timesteps // 2048)
                except Exception:
                    it = n_updates
                rew = self.ep_rew_mean if self.completed_rews else self._fetch("rollout/ep_rew_mean", 0.0)
                kl = self._fetch("train/approx_kl", 0.0)
                lr = self._fetch("train/learning_rate", 0.0)
                # fallback: model learning_rate attribute
                if lr == 0.0:
                    try:
                        lr = float(self.model.learning_rate if isinstance(self.model.learning_rate, float) else 3e-4)
                    except Exception:
                        lr = 3e-4
                loss = self._fetch("train/loss", 0.0)
                pg = self._fetch("train/policy_gradient_loss", 0.0)
                vf = self._fetch("train/value_loss", 0.0)
                ent = self._fetch("train/entropy_loss", 0.0)
                ev = self._fetch("train/explained_variance", 0.0)
                total_str = f"/{self.target_games}" if self.target_games else ""
                if self.verbose:
                    print(f"{ts} | {elapsed_s} | game {self.games_done}{total_str} | iter {it} | reward {rew:.3f} | kl {kl:.3f} | lr {lr:.2e} | loss {loss:.3f} | policy_loss {pg:.3f} | value_loss {vf:.3f} | ent {ent:.3f} | ev {ev:.3f}", flush=True)
            if self.target_games is not None and self.games_done >= self.target_games:
                if self.verbose:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{ts} | stopping: reached {self.games_done} games (target {self.target_games}) at timestep {self.num_timesteps}", flush=True)
                return False
            return True

    total_hint = target_games if target_games is not None else total_games_hint
    return _Unified(target_games, snapshot_pool, save_path, checkpoint_dir, update_every, pool_size, verbose, start_time, total_hint)


# Backwards compat shims (keep old names but delegate to unified for single-opponent scripts)
def make_stop_after_games_callback(target_games, verbose=1):
    return make_unified_callback(target_games, None, None, None, None, 5, verbose, time.time(), target_games)


def make_self_snapshot_callback(snapshot_pool, save_path, checkpoint_dir, update_every=200, pool_size=5, verbose=1):
    return make_unified_callback(None, snapshot_pool, save_path, checkpoint_dir, update_every, pool_size, verbose, time.time(), None)


def train_selfplay(
    representation="vector",
    map_type="BASE",
    opponents=None,
    opponent_weights=None,
    timesteps=None,
    num_games=None,
    n_envs=4,
    seed=0,
    save_path="ppo_catan_selfplay",
    checkpoint_dir=None,
    ab_depth=2,
    self_update_every=200,
    self_pool_size=5,
    tensorboard_log=None,
    verbose=1,
):
    try:
        from sb3_contrib.ppo_mask import MaskablePPO
        from stable_baselines3.common.callbacks import CallbackList
    except ImportError as e:
        raise ImportError("sb3-contrib and stable-baselines3 required: pip install sb3-contrib stable-baselines3 torch") from e

    if opponents is None:
        # default mix: good coverage for lit-review later — easy to tune after
        opponents = ["random", "weighted", "value", "alphabeta", "self"]
    if opponent_weights is None:
        # default weights: emphasize fast opponents, lower self/alphabeta due to cost
        # random 30%, weighted 25%, value 20%, alphabeta 10%, self 15%
        default_w = {"random": 0.30, "weighted": 0.25, "value": 0.20, "alphabeta": 0.10, "self": 0.15}
        opponent_weights = [default_w.get(o, 1.0) for o in opponents]
    if len(opponents) != len(opponent_weights):
        raise ValueError("opponents and opponent_weights length mismatch")
    # normalize later in sampler

    if num_games is not None and timesteps is not None:
        raise ValueError("Specify only one of --num-games or --timesteps")
    if num_games is None and timesteps is None:
        num_games = 1000

    if checkpoint_dir is None:
        # Default: ppo_agents/ppo_training/<save_path_stem>  (separate folder per run)
        stem = Path(save_path).stem
        checkpoint_dir = str(Path("ppo_agents") / "ppo_training" / stem)
    snapshot_pool = []

    config = {"representation": representation, "map_type": map_type}
    sampler = OpponentSampler(opponents, opponent_weights, snapshot_pool)

    print(f"[self-play] opponents={list(zip(opponents, opponent_weights))} representation={representation} map={map_type} ab_depth={ab_depth}")
    print(f"  pool_size={self_pool_size} update_every={self_update_every} snapshots->{checkpoint_dir}")

    if n_envs > 1:
        env = make_selfplay_vec_env(n_envs=n_envs, config=config, sampler=sampler, seed=seed, ab_depth=ab_depth)
    else:
        env = make_selfplay_env(config, sampler, seed=seed, ab_depth=ab_depth)
        # SB3 expects vec env? MaskablePPO works with single but we wrap to DummyVecEnv style via env itself
        # To keep callback dones shape consistent, we ensure env is not vec but callback handles both
        # Alternatively wrap to DummyVecEnv(1)
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],
    )
    # Suppress SB3 Box logger; we use one-line logger
    model = MaskablePPO(
        CatanMaskablePolicy,
        env,
        verbose=0,
        seed=seed,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048 // max(n_envs, 1),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # one-line unified callback (games + snapshot + log)
    start_time = time.time()
    ts0 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if num_games is not None:
        avg_len_upper = 800
        total_timesteps = int(num_games * avg_len_upper)
        unified = make_unified_callback(num_games, snapshot_pool if "self" in opponents else None, save_path, checkpoint_dir, self_update_every, self_pool_size, verbose, start_time, num_games)
        if verbose:
            print(f"{ts0} | 00:00 | game 0/{num_games} | start self-play training opponents={opponents} rep={representation} map={map_type} n_envs={n_envs}", flush=True)
            print(f"{ts0} | training until {num_games} games (~{num_games*500} decisions; upper bound {total_timesteps})", flush=True)
        model.learn(total_timesteps=total_timesteps, callback=unified)
    else:
        assert timesteps is not None
        total_games_hint = int(timesteps / 500) if timesteps else None
        unified = make_unified_callback(None, snapshot_pool if "self" in opponents else None, save_path, checkpoint_dir, self_update_every, self_pool_size, verbose, start_time, total_games_hint)
        if verbose:
            print(f"{ts0} | 00:00 | game 0 | start self-play training opponents={opponents} rep={representation} map={map_type} n_envs={n_envs}", flush=True)
            print(f"{ts0} | training for {timesteps} timesteps (~{timesteps/500:.0f} games)", flush=True)
        model.learn(total_timesteps=timesteps, callback=unified)

    model.save(save_path)
    print(f"Saved model to {save_path}.zip (snapshots in {checkpoint_dir})")
    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with self-play + mixed opponents (AlphaBeta for MCTS)")
    parser.add_argument("--representation", choices=["vector", "mixed"], default="vector")
    parser.add_argument("--map-type", choices=["BASE", "MINI", "TOURNAMENT"], default="BASE")
    parser.add_argument("--opponents", type=str, default="random,weighted,value,alphabeta,self", help="comma list of opponent specs: random,weighted,value,alphabeta,self")
    parser.add_argument("--opponent-weights", type=str, default=None, help="comma list of weights same order as --opponents (e.g. 0.3,0.25,0.2,0.1,0.15)")
    parser.add_argument("--ab-depth", type=int, default=2, help="AlphaBeta depth for alphabeta opponent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--timesteps", type=int, default=None, help="budget by timesteps (1 step=1 P0 decision)")
    group.add_argument("--num-games", type=int, default=None, help="budget by games (episodes)")
    parser.add_argument("--n-envs", type=int, default=4, help="vectorized envs (games counted across all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="ppo_catan_selfplay")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="dir for self snapshots (default ppo_agents/ppo_training/<save_path_stem>)")
    parser.add_argument("--self-update-every", type=int, default=200, help="games between self snapshots")
    parser.add_argument("--self-pool-size", type=int, default=5, help="FIFO cap for self pool")
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    for o in opponents:
        if o not in OPPONENT_CHOICES:
            raise ValueError(f"Unknown opponent {o}, choices {OPPONENT_CHOICES}")
    weights = None
    if args.opponent_weights is not None:
        weights = [float(x.strip()) for x in args.opponent_weights.split(",") if x.strip()]
        if len(weights) != len(opponents):
            raise ValueError("--opponent-weights must have same length as --opponents")

    train_selfplay(
        representation=args.representation,
        map_type=args.map_type,
        opponents=opponents,
        opponent_weights=weights,
        timesteps=args.timesteps,
        num_games=args.num_games,
        n_envs=args.n_envs,
        seed=args.seed,
        save_path=args.save_path,
        checkpoint_dir=args.checkpoint_dir,
        ab_depth=args.ab_depth,
        self_update_every=args.self_update_every,
        self_pool_size=args.self_pool_size,
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
    )
