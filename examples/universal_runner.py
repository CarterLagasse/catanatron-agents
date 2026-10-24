"""Universal runner: any agent can play any agent.

Supports all built-in Catanatron agents (R,W,F,AB,M,G,VP,SAB) plus
PPO .zip trained via examples/train_ppo_custom.py (PPO).

Examples:
  # PPO vs built-ins (universal, any vs any)
  python examples/universal_runner.py --players PPO:ppo_catan_custom.zip,W --num 100
  python examples/universal_runner.py --players PPO:ppo_catan_custom.zip:vector:BASE:1,F --num 50 --seed 0
  python examples/universal_runner.py --players W,W --num 100
  python examples/universal_runner.py --players PPO:ppo_catan_custom.zip,R --num 20 --verbose

  # Also works with CLI raw string or catanatron-play equivalent:
  catanatron-play --players PPO:ppo_catan_custom.zip:vector:BASE,W --num 100

  # 4 players, any mix:
  python examples/universal_runner.py --players PPO:ppo_catan_custom.zip,W,F,R --num 50
  python examples/universal_runner.py --players AB:2,W,M:50,R --num 20

PPO syntax:
  PPO
  PPO:path
  PPO:path:representation   (vector or mixed, must match training)
  PPO:path:representation:map  (BASE/MINI/TOURNAMENT)
  PPO:path:representation:map:deterministic (1/0 or True/False)

  Defaults: path=ppo_catan_custom.zip, representation=vector, map=BASE, deterministic=1

Seating order = order in --players string.
"""

import argparse
import sys

from catanatron.cli.cli_players import parse_cli_string, player_help_table
from catanatron.cli.play import GameConfigOptions, play_batch
from catanatron.models.player import Color


def parse_args():
    p = argparse.ArgumentParser(
        description="Universal runner: any agent vs any agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Player codes: R,W,F,VP,G,M,AB,SAB,PPO,H  (see --help-players)",
    )
    p.add_argument(
        "--players",
        default="PPO:ppo_catan_custom.zip,W",
        help='Comma-separated player specs e.g. "PPO:ppo_catan_custom.zip:vector,W" (seating order matters)',
    )
    p.add_argument("--num", type=int, default=20, help="Number of games to play")
    p.add_argument("--seed", type=int, default=None, help="Optional (not yet wired to all players, informational)")
    p.add_argument("--map", type=str, default=None, help="Map override (BASE/MINI/TOURNAMENT), else per args or default BASE")
    p.add_argument("--vps", type=int, default=None, help="VPs to win override")
    p.add_argument("--verbose", action="store_true", help="Verbose per-game table")
    p.add_argument("--quiet", action="store_true", help="Quiet mode (no tables)")
    p.add_argument("--help-players", action="store_true", help="Show player codes and exit")
    p.add_argument("--code", type=str, default=None, help="Path to custom Players file to import (like catanatron-play --code)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.help_players:
        from rich.console import Console

        Console().print(player_help_table())
        # also print PPO hint if available
        print("\nPPO extra: PPO[:path[:rep[:map[:det]]]] e.g. PPO:ppo_catan_custom.zip:vector:BASE:1")
        sys.exit(0)

    if args.code:
        import importlib.util
        import os

        abspath = os.path.abspath(args.code)
        spec = importlib.util.spec_from_file_location("module.name", abspath)
        if spec is not None and spec.loader is not None:
            user_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_module)

    # Parse players via same logic as catanatron-play
    try:
        players = parse_cli_string(args.players)
    except Exception as e:
        print(f"Failed to parse --players='{args.players}': {e}", file=sys.stderr)
        print("Use --help-players to see codes. PPO syntax: PPO:path:rep:map:det", file=sys.stderr)
        sys.exit(2)

    if not players:
        print("No players parsed from --players", file=sys.stderr)
        sys.exit(2)

    # Validate PPO files exist early with hint
    for p in players:
        if type(p).__name__ == "PpoPlayer":
            from pathlib import Path

            path = getattr(p, "model_path", "")
            if not Path(path).exists() and not Path(path + ".zip").exists():
                print(
                    f"Warning: PpoPlayer {p.color} model not found at '{path}'. "
                    f"Train first: python examples/train_ppo_custom.py --num-games 1000",
                    file=sys.stderr,
                )

    # Game config overrides
    game_config_kwargs = {}
    if args.map is not None:
        game_config_kwargs["map_type"] = args.map.upper()
    if args.vps is not None:
        game_config_kwargs["vps_to_win"] = args.vps

    game_config = GameConfigOptions(**game_config_kwargs) if game_config_kwargs else None

    quiet = args.quiet or not args.verbose
    # play_batch handles tables; when quiet=False it prints Last 10 Games + summaries
    # We force quiet logic: if --verbose show tables, else quiet
    if args.verbose:
        quiet = False

    print(f"Universal runner: {len(players)} players, {args.num} games")
    print(f"  Seating: {', '.join(str(p) + f'({p.color.value})' for p in players)}")
    print(f"  Spec: --players {args.players}")

    wins, results, games = play_batch(
        args.num, players, game_config=game_config, quiet=quiet
    )

    # When quiet, print summary ourselves
    if quiet:
        print("\n--- Results (quiet) ---")
        for color, w in wins.items():
            print(f"  {color.value}: {w} wins ({w/args.num:.1%})")
        # Also print win rate for PPO if present
        for p in players:
            if type(p).__name__ == "PpoPlayer":
                w = wins.get(p.color, 0)
                print(f"  PpoPlayer {p.color.value} vs field: {w}/{args.num} = {w/args.num:.1%}")


if __name__ == "__main__":
    main()
