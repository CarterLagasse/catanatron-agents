"""catanatron-jsettlers: play a catanatron Player against JSettlers2's robots.

Examples:
    catanatron-jsettlers --player=AB:2 --num=50
    catanatron-jsettlers --player=HM:gpt --num=100 --parallel=4
    catanatron-jsettlers --player=JAX:runs/gpu/psro:1 --players=2 --num=100 --fully-observable
"""

import importlib.util
import logging
import math
import os
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from catanatron.cli.cli_players import parse_cli_string
from catanatron.sota.jsettlers.client import BridgeClient
from catanatron.sota.jsettlers.server import JSettlersServer
from catanatron.sota.jsettlers.starter import GameStarter


def wilson_interval(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    phat = wins / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


@click.command()
@click.option(
    "--player",
    default="F",
    show_default=True,
    help="Catanatron player to test, as a catanatron-play code (e.g. AB:2, HM:gpt, JAX:path).",
)
@click.option(
    "--players",
    default=4,
    show_default=True,
    type=click.IntRange(2, 4),
    help="Seats per game: the catanatron player plus N-1 robots. Use 2 for 1v1 agents such as JAX.",
)
@click.option("-n", "--num", default=10, show_default=True, help="Games to play.")
@click.option("--parallel", default=1, show_default=True, help="Games played at once.")
@click.option(
    "--robots",
    default=3,
    show_default=True,
    help="Built-in JSettlers robots to start; 30% (rounded down) are 'fast', the rest 'smart'. "
    "3 gives three smart opponents.",
)
@click.option(
    "--fully-observable",
    is_flag=True,
    help="Show all hands and dev cards to every player (JSettlers option PLAY_FO), "
    "like catanatron's own games. Off by default: hidden information, as in real Catan.",
)
@click.option("--port", default=8880, show_default=True)
@click.option(
    "--turn-timeout",
    default=300,
    show_default=True,
    help="Seconds the bridge may spend on one turn before the server ends it. "
    "A game with no activity for twice this long is abandoned and replaced.",
)
@click.option(
    "--pause-percent",
    default=1,
    show_default=True,
    help="JSettlers robots' pauses between actions, as a percent of their human-paced default.",
)
@click.option(
    "--check-tracking",
    is_flag=True,
    help="Test the bridge: have the server send true hands each turn and report "
    "where the bridge's tracked hands disagree. The player never sees them.",
)
@click.option(
    "--code",
    default=None,
    help="Python file registering custom players (as in catanatron-play).",
)
@click.option(
    "--connect",
    default=None,
    help="host:port of an already running JSettlers server instead of starting one (needs --cookie). "
    "It must allow the debug user and set jsettlers.bots.botgames.total != 0 and percent3p=25.",
)
@click.option(
    "--cookie", default=None, help="Robot cookie of the server given by --connect."
)
@click.option(
    "--log-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Write the server log and the bridge's message traffic here.",
)
def main(
    player,
    players,
    num,
    parallel,
    robots,
    fully_observable,
    port,
    turn_timeout,
    pause_percent,
    check_tracking,
    code,
    connect,
    cookie,
    log_dir,
):
    """Plays a catanatron Player against the JSettlers2 robots on a JSettlers server."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if code:
        spec = importlib.util.spec_from_file_location(
            "module.name", os.path.abspath(code)
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def player_factory():
        return parse_cli_string(player)[0]

    sample = player_factory()
    console = Console()
    console.print(
        f"Playing {sample} against {players - 1} JSettlers robot(s) per game, {num} games"
    )

    log_path = None
    traffic = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / "jsettlers-server.log"
        traffic = open(Path(log_dir) / "bridge-traffic.log", "w", encoding="utf-8")

    results = []
    abandoned = []
    start = time.time()

    def on_result(result):
        results.append(result)
        if result.won:
            outcome = "[green]WON[/green]"
        else:
            outcome = f"lost to {result.names[result.winner_pn]}"
        wins = sum(r.won for r in results)
        console.print(
            f"game {len(results):>3}/{num}  seat {result.our_pn}  "
            f"VP {result.scores[result.our_pn]:>2}  scores {result.scores}  "
            f"{outcome}  ({wins}/{len(results)} won)"
        )

    def play(host, server_port, server_cookie):
        client = BridgeClient(
            host,
            server_port,
            server_cookie,
            player_factory,
            on_result=on_result,
            max_games=num,
            traffic_log=traffic,
        )
        client.connect()
        options = [f"PL={players}"] + (["PLAY_FO=t"] if fully_observable else [])
        starter = GameStarter(
            host,
            server_port,
            total=num,
            parallel=parallel,
            options=",".join(options),
            players=players,
            stall_timeout=2 * turn_timeout,
            on_abandon=client.abandon,
        )
        starter.start()
        try:
            client.run(should_stop=lambda: starter.error is not None)
        finally:
            starter.stop()
            abandoned.extend(starter.abandoned)
        if starter.error is not None:
            raise starter.error

    try:
        if connect:
            if not cookie:
                raise click.UsageError("--connect needs --cookie")
            host, _, server_port = connect.rpartition(":")
            play(host or "127.0.0.1", int(server_port), cookie)
        else:
            server = JSettlersServer(
                port=port,
                robots=robots,
                fast_pause_percent=pause_percent,
                turn_timeout=turn_timeout,
                check_tracking=check_tracking,
                log_path=log_path,
            )
            with server:
                play("127.0.0.1", port, server.cookie)
    finally:
        if traffic:
            traffic.close()

    print_summary(console, sample, results, time.time() - start, players, abandoned)


def print_summary(console, sample, results, seconds, players=4, abandoned=()):
    if abandoned:
        console.print(
            f"[yellow]{len(abandoned)} games stalled or ended without a winner; "
            "they were abandoned, not counted, and replaced:[/yellow]"
        )
        for game, reason in abandoned:
            console.print(f"  {game} {reason}")
    n = len(results)
    if n == 0:
        console.print("[red]No games finished.[/red]")
        return
    wins = sum(r.won for r in results)
    low, high = wilson_interval(wins, n)
    our_vps = [r.scores[r.our_pn] for r in results]
    bot_wins = {}
    for r in results:
        if r.winner_pn is not None and not r.won:
            bot_wins[r.names[r.winner_pn]] = bot_wins.get(r.names[r.winner_pn], 0) + 1

    table = Table(title=f"{sample} vs JSettlers robots")
    table.add_column("GAMES", justify="right")
    table.add_column("WINS", justify="right")
    table.add_column("WIN RATE (95% CI)", justify="right")
    table.add_column("AVG VP", justify="right")
    table.add_column("AVG TURNS", justify="right")
    table.add_column("SEC/DECISION", justify="right")
    decisions = sum(r.decisions for r in results)
    table.add_row(
        str(n),
        str(wins),
        f"{wins / n:.1%} ({low:.1%}-{high:.1%})",
        f"{sum(our_vps) / n:.2f}",
        f"{sum(r.turns for r in results) / n:.1f}",
        f"{sum(r.decide_seconds for r in results) / max(1, decisions):.3f}",
    )
    console.print(table)
    console.print(
        f"An equally strong player wins {1 / players:.0%} of {players}-player games. "
        f"Took {seconds / 60:.1f} min."
    )
    if bot_wins:
        console.print(
            "Robot wins: " + ", ".join(f"{k}: {v}" for k, v in sorted(bot_wins.items()))
        )
    rejected = sum(r.rejected_requests for r in results)
    if rejected:
        console.print(
            f"[yellow]{rejected} requests were declined by the server (see warnings).[/yellow]"
        )
    checks = sum(r.tracking_checks for r in results)
    if checks:
        errors = [e for r in results for e in r.tracking_errors]
        console.print(
            f"Hand tracking: {len(errors)} mismatches in {checks} server checks"
        )
        for error in errors[:10]:
            console.print(f"  {error}")


if __name__ == "__main__":
    main()
