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

from catanatron.models.player import Color, Player


class JaxPlayer(Player):
    """Args:
    path: .npz checkpoint, or a PSRO run directory
    greedy: take the most likely action instead of sampling ("0"/"1" from the CLI)
    """

    def __init__(self, color, path="runs/ppo/best.npz", greedy=False, is_bot=True):
        super().__init__(color, is_bot)
        import jax
        import jax.numpy as jnp
        import numpy as np

        from catanatron.jax_rl import checkpoint
        from catanatron.jax_rl.bots import scripted_logits
        from catanatron.jax_rl.bridge import ActionCodec, state_from_game
        from catanatron.jax_rl.env import CatanEnv
        from catanatron.jax_rl.network import GraphPolicy
        from catanatron.jax_rl.obs import encode_current

        self.path = str(path)
        self.greedy = str(greedy).lower() not in ("0", "false", "no", "")
        self._np = np
        self._state_from_game = state_from_game

        members, meta = _load_members(Path(self.path), checkpoint)
        self.members = [params for _, params in members]
        self.member_kinds = [kind for kind, _ in members]
        self.weights = np.array(meta["weights"], np.float64)
        env = CatanEnv(meta["map_type"])
        self.env = env
        self.codec = ActionCodec(env.topo)
        policy = GraphPolicy(env, meta["hidden"], meta["rounds"])

        def network_logits(params, state):
            return policy.apply(params, encode_current(env, state))[0]

        self._network = jax.jit(network_logits)
        self._scripted = jax.jit(lambda state, kind: scripted_logits(env, state, kind))
        self._unbatched = lambda state: jax.tree_util.tree_map(lambda x: x[0], state)
        self._jnp = jnp
        self._game_id = None
        self._active = 0
        self._rng = np.random.default_rng()

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        if self._game_id != game.id:
            self._game_id = game.id
            self._active = self._rng.choice(len(self.members), p=self.weights)

        state = self._unbatched(self._state_from_game(game, self.env.topo))
        kind = self.member_kinds[self._active]
        if kind < 0:
            logits = self._network(self.members[self._active], state)
        else:
            logits = self._scripted(state, self._jnp.asarray(kind))
        logits = self._np.asarray(logits, self._np.float64)

        choices = [self.codec.encode(a) for a in playable_actions]
        scores = logits[choices]
        if self.greedy:
            return playable_actions[int(scores.argmax())]
        scores = scores - scores.max()
        probabilities = self._np.exp(scores)
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


def _play_games(args):
    agent_path, greedy, opponent_spec, games, seed, map_type = args
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from catanatron.cli.cli_players import parse_cli_string
    from catanatron.game import Game
    from catanatron.models.map import build_map

    agent = JaxPlayer(Color.BLUE, agent_path, greedy=greedy)
    opponent = parse_cli_string(opponent_spec)[0]
    opponent.color = Color.RED
    wins = losses = draws = 0
    vp_total = 0
    for i in range(games):
        game_seed = seed + i
        random.seed(game_seed)
        game = Game([agent, opponent], seed=game_seed, catan_map=build_map(map_type))
        winner = game.play()
        if winner == Color.BLUE:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
        key = f"P{game.state.color_to_index[Color.BLUE]}_ACTUAL_VICTORY_POINTS"
        vp_total += game.state.player_state[key]
    return wins, losses, draws, vp_total


def benchmark(agent_path, opponents, games, workers, greedy=False, map_type="BASE", seed=0):
    """Returns {opponent: result}, playing `games` real engine games per opponent."""
    import multiprocessing

    results = {}
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        for spec in opponents:
            base = seed
            share = math.ceil(games / workers)
            jobs = [
                (agent_path, greedy, spec, share, base + w * share, map_type)
                for w in range(workers)
            ]
            out = pool.map(_play_games, jobs)
            wins, losses, draws, vp_total = (sum(x) for x in zip(*out))
            total = wins + losses + draws
            rate = wins / total
            results[spec] = {
                "games": total,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": rate,
                "ci95": 1.96 * math.sqrt(rate * (1 - rate) / total),
                "mean_vps": vp_total / total,
            }
    return results


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
            f"= {r['win_rate']:.3f} +/- {r['ci95']:.3f}, mean VPs {r['mean_vps']:.1f}"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
