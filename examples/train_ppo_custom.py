"""Train custom PPO Actor-Critic directly with CatanatronEnv.

Perspective: 1 timestep = 1 P0 decision (one Action from playable_actions),
             NOT 1 game. A game is many timesteps (roll, build, trade, END_TURN,
             each is a timestep; enemies are auto-played inside step()).

             Use --num-games to budget training by games (episodes). Internally
             this stops after that many games have terminated (winning_color or
             TURNS_LIMIT or >10 invalid actions).

Install:
    pip install catanatron[gym]
    pip install sb3-contrib stable-baselines3 torch

Run (by games — recommended):
    python examples/train_ppo_custom.py --representation vector --num-games 1000
    python examples/train_ppo_custom.py --representation mixed --num-games 5000 --n-envs 4
    python examples/train_ppo_custom.py --representation vector --num-games 50 --eval  # eval 50 games

Run (by timesteps — legacy):
    python examples/train_ppo_custom.py --representation vector --timesteps 100000
    python examples/train_ppo_custom.py --representation mixed --timesteps 500000 --n-envs 4

Architecture (see catanatron/catanatron/gym/policies/actor_critic.py):
- vector: MLP 256->256 -> pi/vf heads (Discrete ~327, Value 1), masking via
  sb3_contrib ActionMasker + catanatron/gym/adapters.py:catan_mask_fn
  (or env.action_masks() at gym/envs/catanatron_env.py:120)
- mixed: CNN board( C,21,11 )->32->64 ->128 + MLP numeric->64 -> fusion 256 -> pi/vf
"""

import argparse
import datetime
import time

import catanatron.gym  # register env
from catanatron.gym.adapters import catan_mask_fn, make_masked_env, make_vec_env
from catanatron.gym.policies import CatanMaskablePolicy


class StopAfterGamesCallback:
    """SB3 callback that stops training after N games + one-line log.

    One-line format: timestamp | elapsed | game done/target | iter | reward | kl | lr | loss | policy_loss | value_loss | ent | ev
    verbose=0 suppresses line, verbose=1 prints one line per PPO iteration (no SB3 Box).
    """

    def __new__(cls, target_games, verbose=0):
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as e:
            raise ImportError("stable-baselines3 required for num_games callback") from e
        import numpy as np

        class _Callback(BaseCallback):
            def __init__(self, target, verbose=0):
                super().__init__(verbose)
                self.target_games = int(target) if target is not None else None
                self.games_done = 0
                self.start_time = time.time()
                self.last_n_updates = -1
                self.ep_rewards = None
                self.ep_rew_mean = 0.0
                self.completed_rews = []

            def _fetch(self, key, default=0.0):
                for src in (getattr(self.model, "logger", None), getattr(self, "logger", None)):
                    try:
                        if src and hasattr(src, "name_to_value"):
                            v = src.name_to_value.get(key, None)
                            if v is not None:
                                return v
                    except Exception:
                        pass
                return default

            def _on_step(self) -> bool:
                # track episode reward mean manually (Monitor not reliable with ActionMasker)
                try:
                    rews = self.locals.get("rewards")
                    dones_raw = self.locals.get("dones")
                    if rews is not None and dones_raw is not None:
                        import numpy as _np

                        if self.ep_rewards is None:
                            self.ep_rewards = _np.zeros_like(rews, dtype=float)
                        # rews may be array shape (n_envs,)
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
                            # mean of last 100
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
                self.games_done += count

                # one-line log per PPO update
                try:
                    n_updates = getattr(self.model, "_n_updates", 0) or 0
                except Exception:
                    n_updates = 0
                if n_updates != self.last_n_updates and n_updates > 0:
                    self.last_n_updates = n_updates
                    if self.verbose:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        elapsed = int(time.time() - self.start_time)
                        elapsed_s = f"{elapsed//60:02d}:{elapsed%60:02d}"
                        try:
                            it = int(self.num_timesteps // 2048)
                        except Exception:
                            it = n_updates
                        rew = self.ep_rew_mean if self.completed_rews else self._fetch("rollout/ep_rew_mean", 0.0)
                        kl = self._fetch("train/approx_kl", 0.0)
                        lr = self._fetch("train/learning_rate", 0.0)
                        if lr == 0.0:
                            try:
                                lr = float(self.model.learning_rate) if isinstance(self.model.learning_rate, float) else 3e-4
                            except Exception:
                                lr = 3e-4
                        loss = self._fetch("train/loss", 0.0)
                        pg = self._fetch("train/policy_gradient_loss", 0.0)
                        vf = self._fetch("train/value_loss", 0.0)
                        ent = self._fetch("train/entropy_loss", 0.0)
                        ev = self._fetch("train/explained_variance", 0.0)
                        total_str = f"/{self.target_games}" if self.target_games else ""
                        print(f"{ts} | {elapsed_s} | game {self.games_done}{total_str} | iter {it} | reward {rew:.3f} | kl {kl:.3f} | lr {lr:.2e} | loss {loss:.3f} | policy_loss {pg:.3f} | value_loss {vf:.3f} | ent {ent:.3f} | ev {ev:.3f}", flush=True)
                if self.target_games is not None and self.games_done >= self.target_games:
                    if self.verbose:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{ts} | stopping: reached {self.games_done} games (target {self.target_games}) at timestep {self.num_timesteps}", flush=True)
                    return False
                return True

        return _Callback(target_games, verbose=verbose)


def train(
    representation="vector",
    timesteps=None,
    num_games=None,
    n_envs=1,
    seed=0,
    save_path="ppo_catan_custom",
    tensorboard_log=None,
    verbose=1,
):
    try:
        from sb3_contrib.ppo_mask import MaskablePPO
    except ImportError as e:
        raise ImportError(
            "sb3-contrib required: pip install sb3-contrib stable-baselines3"
        ) from e

    if num_games is not None and timesteps is not None:
        raise ValueError("Specify only one of --num-games or --timesteps")
    if num_games is None and timesteps is None:
        # Default to games for intuitive budgeting
        num_games = 1000

    config = {"representation": representation}

    if n_envs > 1:
        env = make_vec_env(n_envs=n_envs, config=config, seed=seed)
    else:
        env = make_masked_env(config=config)

    # Policy kwargs: new architecture tunable here
    # features_extractor_kwargs controls CNN/MLP extractor at
    # catanatron/catanatron/gym/policies/actor_critic.py:30
    # net_arch is shared trunk after extractor (SB3 style).
    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],  # try dict(pi=[128], vf=[128]) for separate
    )

    # Suppress SB3 Box logger; we use one-line logger
    model = MaskablePPO(
        CatanMaskablePolicy,
        env,
        verbose=0,
        seed=seed,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        # PPO hyperparams (SB3 defaults tuned for Catan)
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

    # --- decide total_timesteps + callback based on num_games vs timesteps ---
    if num_games is not None:
        avg_len_upper = 800
        total_timesteps = int(num_games * avg_len_upper)
        callback = StopAfterGamesCallback(num_games, verbose=verbose)
        if verbose:
            ts0 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts0} | 00:00 | game 0/{num_games} | start training rep={representation} n_envs={n_envs} up to {total_timesteps} steps", flush=True)
        model.learn(total_timesteps=total_timesteps, callback=callback)
    else:
        assert timesteps is not None
        # timesteps path still logs via generic callback (games counted but no target)
        callback = StopAfterGamesCallback(None, verbose=verbose) if verbose else None
        if verbose:
            ts0 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts0} | 00:00 | game 0 | start training {timesteps} timesteps rep={representation} n_envs={n_envs}", flush=True)
        model.learn(total_timesteps=timesteps, callback=callback)

    model.save(save_path)
    print(f"Saved model to {save_path}.zip")
    env.close()
    return model


def evaluate(model_path="ppo_catan_custom", representation="vector", n_episodes=10):
    """Quick eval vs RandomPlayer using trained policy. n_episodes = num_games."""
    import gymnasium as gym
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.ppo_mask import MaskablePPO

    env = gym.make("catanatron/Catanatron-v0", config={"representation": representation})
    env = ActionMasker(env, catan_mask_fn)
    model = MaskablePPO.load(model_path, env=env)

    wins = 0
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            # Use masking: SB3 automatically uses env.action_masks()
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        # reward at terminal is simple_reward (1 if BLUE won)
        winning = env.unwrapped.game.winning_color()
        from catanatron.models.player import Color

        if winning == Color.BLUE:
            wins += 1
        print(f"Game {ep+1}/{n_episodes}: {'WIN' if winning==Color.BLUE else 'LOSS'} winning={winning}")

    print(f"Win rate over {n_episodes} games: {wins}/{n_episodes} = {wins/n_episodes:.1%}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom PPO on CatanatronEnv (budget by games)")
    parser.add_argument("--representation", choices=["vector", "mixed"], default="vector")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--timesteps", type=int, default=None, help="budget by timesteps (legacy, 1 step=1 P0 decision)")
    group.add_argument("--num-games", type=int, default=None, help="budget by games (episodes)")
    parser.add_argument("--n-envs", type=int, default=1, help="vectorized envs (games counted across all envs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="ppo_catan_custom")
    parser.add_argument("--tensorboard-log", type=str, default=None)
    parser.add_argument("--eval", action="store_true", help="eval saved model instead of train")
    parser.add_argument("--eval-episodes", type=int, default=10, help="games to evaluate (alias --num-games for eval)")
    parser.add_argument("--num-eval-games", type=int, default=None, help="alias for --eval-episodes")
    args = parser.parse_args()

    if args.eval:
        n_eval = args.num_eval_games if args.num_eval_games is not None else args.eval_episodes
        # also allow --num-games to mean eval games
        if args.num_games is not None:
            n_eval = args.num_games
        evaluate(args.save_path, args.representation, n_eval)
    else:
        train(
            representation=args.representation,
            timesteps=args.timesteps,
            num_games=args.num_games,
            n_envs=args.n_envs,
            seed=args.seed,
            save_path=args.save_path,
            tensorboard_log=args.tensorboard_log,
        )
