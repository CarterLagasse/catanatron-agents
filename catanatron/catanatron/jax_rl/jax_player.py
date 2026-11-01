"""Plays trained JAX agents in the real Python engine.

JaxPlayer works anywhere a Player does, including the CLI:
    catanatron-play --players=JAX:runs/ppo/best.npz,F --num=100
A PSRO run directory plays its Nash mixture, drawing one member per game:
    catanatron-play --players=JAX:runs/psro,F --num=100

Benchmark against the engine's bots, using several processes:
    python -m catanatron.jax_rl.jax_player --agent runs/ppo/best.npz --opponents R,W,F,AB:2 --games 200
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from catanatron.models.player import Color, Player


class _Model:
    """A loaded agent and its compiled functions."""

    def __init__(self, path):
        import jax
        import jax.numpy as jnp

        from catanatron.jax_rl import checkpoint
        from catanatron.jax_rl.bots import scripted_logits
        from catanatron.jax_rl.bridge import ActionCodec, state_from_game
        from catanatron.jax_rl.env import CatanEnv
        from catanatron.jax_rl.network import GraphPolicy
        from catanatron.jax_rl.obs import encode_current

        members, meta = _load_members(Path(path), checkpoint)
        self.members = [params for _, params in members]
        self.member_kinds = [kind for kind, _ in members]
        self.weights = np.array(meta["weights"], np.float64)
        self.env = env = CatanEnv(meta["map_type"])
        self.codec = ActionCodec(env.topo)
        policy = GraphPolicy(env, meta["hidden"], meta["rounds"])

        def network_logits(params, state):
            return policy.apply(params, encode_current(env, state))[0]

        network = jax.jit(network_logits)
        scripted = jax.jit(lambda state, kind: scripted_logits(env, state, kind))

        def logits(game, active):
            batched = state_from_game(game, env.topo)
            state = jax.tree_util.tree_map(lambda x: x[0], batched)
            kind = self.member_kinds[active]
            if kind < 0:
                result = network(self.members[active], state)
            else:
                result = scripted(state, jnp.asarray(kind))
            return np.asarray(result, np.float64)

        self.logits = logits


_MODELS = {}


def _model(path):
    """Compiling takes seconds, and the web server rebuilds players from the database
    on every request, so models are shared per process and never pickled."""
    if path not in _MODELS:
        _MODELS[path] = _Model(path)
    return _MODELS[path]


class JaxPlayer(Player):
    """Args:
    path: .npz checkpoint, or a PSRO run directory
    greedy: take the most likely action instead of sampling ("0"/"1" from the CLI)
    """

    def __init__(self, color, path="runs/ppo/best.npz", greedy=False, is_bot=True):
        super().__init__(color, is_bot)
        self.path = str(path)
        self.greedy = str(greedy).lower() not in ("0", "false", "no", "")
        self._game_id = None
        self._active = 0
        self._rng = np.random.default_rng()
        _model(self.path)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        model = _model(self.path)
        if self._game_id != game.id:
            self._game_id = game.id
            self._active = self._rng.choice(len(model.members), p=model.weights)

        logits = model.logits(game, self._active)
        choices = [model.codec.encode(a) for a in playable_actions]
        scores = logits[choices]
        if self.greedy:
            return playable_actions[int(scores.argmax())]
        scores = scores - scores.max()
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum()
        return playable_actions[self._rng.choice(len(choices), p=probabilities)]

    def __repr__(self):
        return f"JaxPlayer:{self.color.value}({self.path})"


def _load_members(path, checkpoint):
    """Returns ([(kind, params)], meta with weights)."""
    if path.is_dir():
        data = json.loads((path / "population.json").read_text())
        members = []
        weights = []
        for name, kind, weight in zip(data["names"], data["kinds"], data["strategy"]):
            if weight <= 0:
                continue
            params = None
            if kind < 0:
                params = checkpoint.load_params(path / f"{name}.npz")[0]
            members.append((kind, params))
            weights.append(weight)
        total = sum(weights)
        return members, {
            "weights": [w / total for w in weights],
            "hidden": data["hidden"],
            "rounds": data["rounds"],
            "map_type": data["map_type"],
        }
    params, meta = checkpoint.load_params(path)
    meta["weights"] = [1.0]
    return [(-1, params)], meta


def _build_agent(spec, greedy):
    """A checkpoint or PSRO directory, or bot:<CLI code> for a control match."""
    from catanatron.cli.cli_players import parse_cli_string

    if spec.startswith("bot:"):
        agent = parse_cli_string(spec[len("bot:") :])[0]
        agent.color = Color.BLUE
        return agent
    return JaxPlayer(Color.BLUE, spec, greedy=greedy)


def _play_games(args):
    """Returns one record per game. Everything random follows from the game's seed,
    so two agents given the same seeds face the same boards and can be paired."""
    agent_path, greedy, opponent_spec, seeds, map_type = args
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from catanatron.cli.cli_players import parse_cli_string
    from catanatron.game import Game
    from catanatron.models.map import build_map

    agent = _build_agent(agent_path, greedy)
    opponent = parse_cli_string(opponent_spec)[0]
    opponent.color = Color.RED
    records = []
    for game_seed in seeds:
        random.seed(game_seed)
        for offset, player in enumerate((agent, opponent)):
            if isinstance(player, JaxPlayer):
                player._rng = np.random.default_rng([game_seed, offset])
                player._game_id = None
        game = Game([agent, opponent], seed=game_seed, catan_map=build_map(map_type))
        winner = game.play()
        seat = game.state.color_to_index[Color.BLUE]
        records.append(
            {
                "seed": game_seed,
                "result": "draw" if winner is None else ("win" if winner == Color.BLUE else "loss"),
                "seat": seat,
                "vps": game.state.player_state[f"P{seat}_ACTUAL_VICTORY_POINTS"],
                "member": int(agent._active) if isinstance(agent, JaxPlayer) else 0,
            }
        )
    return records


def wilson_interval(successes, total, z=1.96):
    """Stays meaningful near 0% and 100%, where the normal approximation gives +/- 0."""
    if total == 0:
        return 0.0, 1.0
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(center - spread, 0.0), min(center + spread, 1.0)


def summarize(records):
    """Win rate with its uncertainty, plus the split by seat."""
    total = len(records)
    wins = sum(r["result"] == "win" for r in records)
    losses = sum(r["result"] == "loss" for r in records)
    low, high = wilson_interval(wins, total)
    seats = {}
    for seat in (0, 1):
        played = [r for r in records if r["seat"] == seat]
        seats[f"seat{seat}_games"] = len(played)
        seats[f"seat{seat}_win_rate"] = (
            sum(r["result"] == "win" for r in played) / len(played) if played else 0.0
        )
    return {
        "games": total,
        "wins": wins,
        "losses": losses,
        "draws": total - wins - losses,
        "win_rate": wins / total if total else 0.0,
        "ci_low": low,
        "ci_high": high,
        "mean_vps": sum(r["vps"] for r in records) / total if total else 0.0,
        **seats,
    }


def play_match(
    agent, opponent, games, pool, greedy=False, map_type="BASE", seed=0, records_path=None, chunk=25
):
    """Plays seeds seed..seed+games-1. With records_path, finished games are kept
    in a JSON-lines file and a rerun plays only the seeds still missing."""
    records = []
    if records_path is not None and Path(records_path).exists():
        records = [json.loads(line) for line in Path(records_path).read_text().splitlines()]
    finished = {r["seed"] for r in records}
    missing = [s for s in range(seed, seed + games) if s not in finished]
    jobs = [
        (agent, greedy, opponent, missing[i : i + chunk], map_type)
        for i in range(0, len(missing), chunk)
    ]
    for batch in pool.imap_unordered(_play_games, jobs):
        records.extend(batch)
        if records_path is not None:
            with Path(records_path).open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in batch)
    records.sort(key=lambda r: r["seed"])
    return records


def benchmark(agent_path, opponents, games, workers, greedy=False, map_type="BASE", seed=0):
    """Returns {opponent: result}, playing `games` real engine games per opponent."""
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return {
            spec: summarize(play_match(agent_path, spec, games, pool, greedy, map_type, seed))
            for spec in opponents
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--agent", required=True)
    parser.add_argument("--opponents", default="R,W,F")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--map-type", default="BASE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="write results to this JSON file")
    args = parser.parse_args()
    results = benchmark(
        args.agent,
        args.opponents.split(","),
        args.games,
        args.workers,
        args.greedy,
        args.map_type,
        args.seed,
    )
    for spec, r in results.items():
        print(
            f"{args.agent} vs {spec}: {r['wins']}W {r['losses']}L {r['draws']}D of {r['games']} "
            f"= {r['win_rate']:.3f} (95% CI {r['ci_low']:.3f}-{r['ci_high']:.3f}), "
            f"mean VPs {r['mean_vps']:.1f}"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
