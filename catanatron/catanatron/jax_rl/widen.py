"""Turns a 1v1 agent into one that plays 3- or 4-player games, without training.

New inputs start with zero weights, so the widened agent plays as the 1v1 one would
with every opponent treated as "the opponent" (see network.adapt_params). It is a
baseline for multiplayer training, and the usual way to start it:

    python -m catanatron.jax_rl.widen runs/gpu/psro runs/widened/psro4 --num-players 4
    catanatron-jsettlers --player=JAX:runs/widened/psro4:1 --num=100
"""

import argparse
import json
import shutil
from pathlib import Path

from catanatron.jax_rl import checkpoint
from catanatron.jax_rl.network import adapt_params


def widen_checkpoint(source, target, num_players):
    params, meta = checkpoint.load_params(source)
    if meta["num_players"] != 2:
        raise ValueError(f"{source} is not a 1v1 checkpoint")
    checkpoint.save_params(
        target,
        adapt_params(params, num_players),
        meta["hidden"],
        meta["rounds"],
        meta["map_type"],
        num_players,
    )


def widen(source, target, num_players):
    """Widens a checkpoint, or every network of a PSRO run directory."""
    source, target = Path(source), Path(target)
    if not source.is_dir():
        widen_checkpoint(source, target, num_players)
        return
    data = json.loads((source / "population.json").read_text())
    if data.get("num_players", 2) != 2:
        raise ValueError(f"{source} is not a 1v1 PSRO run")
    target.mkdir(parents=True, exist_ok=True)
    for name, kind in zip(data["names"], data["kinds"]):
        if kind < 0:
            widen_checkpoint(
                source / f"{name}.npz", target / f"{name}.npz", num_players
            )
    # The mixture is kept as it is: its weights were solved for 1v1 play.
    data["num_players"] = num_players
    data["widened_from"] = str(source)
    (target / "population.json").write_text(json.dumps(data, indent=1))
    if (source / "psro_config.json").exists():
        shutil.copy(source / "psro_config.json", target / "psro_config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="1v1 .npz checkpoint or PSRO run directory")
    parser.add_argument("target")
    parser.add_argument("--num-players", type=int, default=4, choices=[3, 4])
    args = parser.parse_args()
    widen(args.source, args.target, args.num_players)
    print(f"Wrote {args.target}")


if __name__ == "__main__":
    main()
