"""Policy-Space Response Oracles for 2- to 4-player Catan, in pure JAX.

Each iteration:
    1. Solve the empirical game: find the Nash mixture over the current population.
    2. Train a best response against that mixture, warm started from the previous one.
    3. Add the response to the population and fill in its row of the payoff matrix.

payoffs[i, j] is member i's mean reward in one seat with member j in every other
seat (+1 a win, -1/(P-1) a loss). In 1v1 the game is symmetric and zero-sum once
seats are randomized, so the matrix is antisymmetric and one mixture describes both
players. With more players payoffs[i, j] and payoffs[j, i] are separate matches, and
the meta-game is the antisymmetric part (payoffs - payoffs.T) / 2: i ranks above j
when i does better against a table of j than j does against a table of i. In 1v1
that is the payoff matrix itself. Responses train against tables drawn from the
mixture, one member per table. The BR's score against the mixture estimates how
exploitable the mixture still is.

Train:
    python -m catanatron.jax_rl.psro --run-dir runs/psro --iterations 12
    python -m catanatron.jax_rl.psro --run-dir runs/psro4 --num-players 4 \
        --init-from runs/gpu/ppo/selected.npz
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from catanatron.jax_rl import checkpoint
from catanatron.jax_rl.bots import SCRIPTED_NAMES
from catanatron.jax_rl.evaluate import Arena
from catanatron.jax_rl.ppo import PPOConfig, Trainer, load_init_params, reset_pool_games


@partial(jax.jit, static_argnames=("iterations",))
def _regret_matching(payoffs, iterations):
    """Regret matching+ in self-play. The average strategy converges to Nash."""
    size = payoffs.shape[0]

    def update(carry, _):
        regrets, average = carry
        positive = jnp.maximum(regrets, 0.0)
        total = positive.sum()
        strategy = jnp.where(
            total > 0, positive / jnp.maximum(total, 1e-12), 1.0 / size
        )
        values = payoffs @ strategy
        regrets = jnp.maximum(regrets + values - strategy @ values, 0.0)
        return (regrets, average + strategy), None

    init = (jnp.zeros(size), jnp.zeros(size))
    (_, average), _ = jax.lax.scan(update, init, None, length=iterations)
    return average / average.sum()


def solve_meta(payoffs, iterations=50_000, floor=1e-3):
    """Approximate Nash of a symmetric zero-sum game, with dust weights removed."""
    payoffs = jnp.asarray(payoffs, jnp.float32)
    strategy = np.asarray(_regret_matching(payoffs, iterations), np.float64)
    strategy[strategy < floor] = 0.0
    strategy /= strategy.sum()
    return strategy


def exploitability(payoffs, strategy):
    """Payoff of the best pure response in the population against the strategy."""
    return float(np.max(np.asarray(payoffs) @ np.asarray(strategy)))


@dataclass(frozen=True)
class PSROConfig:
    map_type: str = "BASE"
    num_players: int = 2
    iterations: int = 12
    # Timesteps of each best response. The first starts from nothing, so it gets more.
    first_response_timesteps: int = 8_000_000
    response_timesteps: int = 4_000_000
    num_envs: int = 256
    rollout_ticks: int = 128
    hidden: int = 48
    rounds: int = 3
    phase_iterations: int = 5
    learning_rate: float = 5e-4
    # Share of each response's games played against itself. Pure PSRO uses 0; a
    # little self-play keeps the response from forgetting general skill while it
    # specializes against a narrow mixture.
    self_fraction: float = 0.2
    pool_segments: int = 4
    scripted_population: tuple = ("random", "weighted", "heuristic")
    eval_envs: int = 128
    eval_ticks: int = 800
    seed: int = 0


class Population:
    """Members are scripted bots (params None) or networks. Holds the payoff matrix."""

    def __init__(self):
        self.names = []
        self.kinds = []  # -1 for a network, otherwise bots.SCRIPTED_NAMES value
        self.params = []
        self.payoffs = np.zeros((0, 0), np.float64)

    def __len__(self):
        return len(self.names)

    def meta_payoffs(self):
        """The antisymmetric game the meta-solver works on (see the module docs)."""
        return (self.payoffs - self.payoffs.T) / 2.0

    def add(self, name, kind, params, row, column=None):
        """row[i] is the new member's payoff against a table of member i, and
        column[i] member i's payoff against a table of the new member. In 1v1 the
        column is -row and can be left out."""
        size = len(self)
        column = -np.asarray(row) if column is None else np.asarray(column)
        payoffs = np.zeros((size + 1, size + 1), np.float64)
        payoffs[:size, :size] = self.payoffs
        payoffs[size, :size] = row
        payoffs[:size, size] = column
        self.payoffs = payoffs
        self.names.append(name)
        self.kinds.append(kind)
        self.params.append(params)

    def save(self, run_dir, config, strategy, history):
        run_dir = Path(run_dir)
        data = {
            "names": self.names,
            "kinds": self.kinds,
            "payoffs": self.payoffs.tolist(),
            "strategy": list(map(float, strategy)),
            "history": history,
            "hidden": config.hidden,
            "rounds": config.rounds,
            "map_type": config.map_type,
            "num_players": config.num_players,
        }
        temporary = run_dir / "population.tmp.json"
        temporary.write_text(json.dumps(data, indent=1))
        checkpoint.replace_file(temporary, run_dir / "population.json")


def _bot_name(kind):
    return next(n for n, v in SCRIPTED_NAMES.items() if v == kind)


def _match(arena, kind_a, params_a, kind_b, params_b, key):
    """Payoff of A in one seat against B in every other seat."""
    if kind_a < 0 and kind_b < 0:
        return arena.against_network(key, params_a, params_b)["payoff"]
    if kind_a < 0:
        return arena.against_scripted(key, params_a, _bot_name(kind_b))["payoff"]
    if kind_b < 0:
        if arena.env.num_players == 2:  # zero-sum: reuse the network's match
            return -arena.against_scripted(key, params_b, _bot_name(kind_a))["payoff"]
        return arena.scripted_against(key, _bot_name(kind_a), params_b)["payoff"]
    return _scripted_vs_scripted(arena, kind_a, kind_b, key)


def evaluate_member(arena, population, kind, params, key):
    """Payoff of a new member against a table of each member already in the
    population. With more than 2 players, also returns each member's payoff
    against a table of the new member; in 1v1 that column is None."""
    multiplayer = arena.env.num_players > 2
    row, column = [], []
    for member_kind, member_params in zip(population.kinds, population.params):
        if multiplayer:
            key, k_row, k_column = jax.random.split(key, 3)
            column.append(
                _match(arena, member_kind, member_params, kind, params, k_column)
            )
        else:  # the 1v1 key sequence, so old runs reproduce
            key, k_row = jax.random.split(key)
        row.append(_match(arena, kind, params, member_kind, member_params, k_row))
    return row, (column if multiplayer else None)


def _scripted_vs_scripted(arena, kind_a, kind_b, key):
    from catanatron.jax_rl.evaluate import scripted_policy

    counts = jax.jit(
        lambda k: arena._play(k, scripted_policy(kind_a), scripted_policy(kind_b))
    )(key)
    return arena._summary(counts)["payoff"]


def _response_config(config, timesteps, seed):
    pool_fraction = 1.0 - config.self_fraction
    return PPOConfig(
        map_type=config.map_type,
        num_players=config.num_players,
        num_envs=config.num_envs,
        rollout_ticks=config.rollout_ticks,
        pool_fraction=pool_fraction,
        scripted_fraction=0.0,
        pool_segments=config.pool_segments,
        hidden=config.hidden,
        rounds=config.rounds,
        total_timesteps=timesteps,
        learning_rate=config.learning_rate,
        final_learning_rate=config.learning_rate / 4,
        phase_iterations=config.phase_iterations,
        seed=seed,
    )


def train_response(trainer, population, strategy, params, key, rng, log, label):
    """PPO against opponents sampled from the meta-strategy. Trainer's compiled step
    is reused across PSRO iterations, so only the first one pays for compilation."""
    config = trainer.config
    key, k_carry = jax.random.split(key)
    opt_state = trainer.optimizer.init(params)
    carry = trainer.init_carry(k_carry)
    # Every non-self game plays a sampled member through the pool segments.
    codes = trainer.scripted_codes(key)

    def sample_segments():
        picks = rng.choice(len(population), config.pool_segments, p=strategy)
        members = [(population.kinds[i], population.params[i]) for i in picks]
        return trainer.stack_segments(members, params)

    pool_params, pool_kinds = sample_segments()
    start = time.time()
    window = {"wins": 0.0, "losses": 0.0, "games": 0.0}
    iterations = trainer.iterations
    for iteration in range(1, iterations + 1):
        progress = (iteration - 1) / max(iterations - 1, 1)
        key, k_iter, k_reset = jax.random.split(key, 3)
        params, opt_state, carry, metrics = trainer.run_iteration(
            params, opt_state, carry, codes, pool_params, pool_kinds, progress, k_iter
        )
        window["wins"] += float(metrics["pool_wins"])
        window["losses"] += float(metrics["pool_losses"])
        window["games"] += float(metrics["pool_games"])
        if iteration % config.phase_iterations == 0:
            pool_params, pool_kinds = sample_segments()
            carry = reset_pool_games(trainer, carry, k_reset)
        if iteration % 25 == 0 or iteration == iterations:
            games = max(window["games"], 1.0)
            rate = (
                iteration
                * config.num_envs
                * config.rollout_ticks
                / (time.time() - start)
            )
            log(
                f"  {label} it {iteration}/{iterations} {rate:.0f}/s "
                f"win vs mixture {window['wins'] / games:.3f} over {games:.0f} games "
                f"| ent {float(metrics['entropy']):.2f} ev {float(metrics['explained_variance']):+.2f}"
            )
            window = {"wins": 0.0, "losses": 0.0, "games": 0.0}
    return params, key


def _restore(population, run_dir):
    """Reloads a saved run. A response interrupted midway is simply trained again."""
    data = json.loads((run_dir / "population.json").read_text())
    population.names = data["names"]
    population.kinds = data["kinds"]
    population.payoffs = np.array(data["payoffs"], np.float64)
    population.params = [
        checkpoint.load_params(run_dir / f"{name}.npz")[0] if kind < 0 else None
        for name, kind in zip(population.names, population.kinds)
    ]
    return data["history"]


def train(config, run_dir, log=print, init_params=None, resume=False, max_hours=None):
    """Args:
    init_params: a trained network, e.g. from PPO. It joins the population as
        "init" and warm starts the first response, so PSRO builds on it.
    resume: continue from population.json in run_dir, if there is one.
    max_hours: don't start a response unless it should finish within this time.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "psro_config.json").write_text(json.dumps(asdict(config), indent=2))
    launched = time.time()

    rng = np.random.default_rng(config.seed)
    key = jax.random.PRNGKey(config.seed)

    first = _response_config(config, config.first_response_timesteps, config.seed)
    later = _response_config(config, config.response_timesteps, config.seed)
    first_trainer = Trainer(first)
    # Same step function layout, shorter learning-rate schedule.
    later_trainer = Trainer(later)
    arena = Arena(
        first_trainer.env, first_trainer.policy, config.eval_envs, config.eval_ticks
    )

    population = Population()
    key, k_params = jax.random.split(key)
    params = (
        init_params if init_params is not None else first_trainer.init_params(k_params)
    )
    if resume and (run_dir / "population.json").exists():
        history = _restore(population, run_dir)
        params = next(p for p in reversed(population.params) if p is not None)
        key = jax.random.fold_in(key, len(history))
        rng = np.random.default_rng(config.seed + len(history))
        log(f"Resuming with {len(population)} members, {len(history)} responses done")
    else:
        history = []
        for name in config.scripted_population:
            key, k_eval = jax.random.split(key)
            kind = SCRIPTED_NAMES[name]
            row, column = evaluate_member(arena, population, kind, None, k_eval)
            population.add(name, kind, None, row, column)
        if init_params is not None:
            key, k_eval = jax.random.split(key)
            frozen = jax.tree_util.tree_map(np.asarray, init_params)
            row, column = evaluate_member(arena, population, -1, frozen, k_eval)
            population.add("init", -1, frozen, row, column)
            checkpoint.save_params(
                run_dir / "init.npz",
                frozen,
                config.hidden,
                config.rounds,
                config.map_type,
                config.num_players,
            )
            log(f"init payoffs {np.round(row, 2).tolist()}")
        population.save(run_dir, config, solve_meta(population.meta_payoffs()), history)

    for iteration in range(len(history), config.iterations):
        if max_hours is not None and history:
            needed = max(h["seconds"] for h in history)
            if time.time() - launched + needed > max_hours * 3600:
                log(f"PSRO: stopping before response {iteration}, time budget reached")
                break
        started = time.time()
        strategy = solve_meta(population.meta_payoffs())
        mixture = {
            n: round(float(w), 3) for n, w in zip(population.names, strategy) if w > 0
        }
        log(f"PSRO {iteration}: meta-strategy {mixture}")

        trainer = first_trainer if iteration == 0 else later_trainer
        key, k_train, k_eval = jax.random.split(key, 3)
        params, _ = train_response(
            trainer, population, strategy, params, k_train, rng, log, f"BR {iteration}"
        )
        frozen = jax.tree_util.tree_map(np.asarray, params)

        row, column = evaluate_member(arena, population, -1, frozen, k_eval)
        response_value = float(np.dot(row, strategy))
        name = f"br_{iteration:02d}"
        population.add(name, -1, frozen, row, column)
        checkpoint.save_params(
            run_dir / f"{name}.npz",
            frozen,
            config.hidden,
            config.rounds,
            config.map_type,
            config.num_players,
        )

        strategy = solve_meta(population.meta_payoffs())
        history.append(
            {
                "iteration": iteration,
                # > 0 means the old mixture could still be exploited.
                "response_vs_previous_mixture": response_value,
                "exploitability_within_population": exploitability(
                    population.meta_payoffs(), strategy
                ),
                "seconds": time.time() - started,
            }
        )
        population.save(run_dir, config, strategy, history)
        log(
            f"PSRO {iteration}: {name} payoff vs previous mixture {response_value:+.3f}, "
            f"row {np.round(row, 2).tolist()}, took {history[-1]['seconds'] / 60:.1f} min"
        )

    return population


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", default="runs/psro")
    parser.add_argument(
        "--map-type", default="BASE", choices=["BASE", "MINI", "TOURNAMENT"]
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=PSROConfig.num_players,
        choices=[2, 3, 4],
        help="players per game; 4 matches the JSettlers benchmark",
    )
    parser.add_argument("--iterations", type=int, default=PSROConfig.iterations)
    parser.add_argument(
        "--first-response-timesteps",
        type=int,
        default=PSROConfig.first_response_timesteps,
    )
    parser.add_argument(
        "--response-timesteps", type=int, default=PSROConfig.response_timesteps
    )
    parser.add_argument("--num-envs", type=int, default=PSROConfig.num_envs)
    parser.add_argument("--hidden", type=int, default=PSROConfig.hidden)
    parser.add_argument("--rounds", type=int, default=PSROConfig.rounds)
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=PSROConfig.eval_envs,
        help="more means less noisy payoffs",
    )
    parser.add_argument(
        "--eval-ticks",
        type=int,
        default=PSROConfig.eval_ticks,
        help="plies per payoff match; games still running at the end are not counted, "
        "so raise it for longer games such as 4-player ones",
    )
    parser.add_argument(
        "--scripted-population",
        default=",".join(PSROConfig.scripted_population),
        help="scripted bots in the population. Leave out any that weak networks cannot beat, "
        "or the mixture ends up playing the bot and not a network",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="checkpoint to warm start from; a 1v1 one is widened for more players",
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue the run in --run-dir"
    )
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = replace(
        PSROConfig(),
        map_type=args.map_type,
        num_players=args.num_players,
        iterations=args.iterations,
        first_response_timesteps=args.first_response_timesteps,
        response_timesteps=args.response_timesteps,
        num_envs=args.num_envs,
        hidden=args.hidden,
        rounds=args.rounds,
        eval_envs=args.eval_envs,
        eval_ticks=args.eval_ticks,
        scripted_population=tuple(args.scripted_population.split(",")),
        seed=args.seed,
    )
    init_params = (
        load_init_params(args.init_from, config.num_players) if args.init_from else None
    )
    train(
        config,
        args.run_dir,
        init_params=init_params,
        resume=args.resume,
        max_hours=args.max_hours,
    )


if __name__ == "__main__":
    main()
