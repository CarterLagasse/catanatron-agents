from collections import namedtuple

from rich.table import Table

from catanatron.models.player import Color, HumanPlayer, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

# PPO trained via catanatron/gym (optional, requires sb3-contrib)
try:
    from catanatron.gym.players import PpoPlayer  # noqa: F401

    _PPO_AVAILABLE = True
except ImportError:
    _PPO_AVAILABLE = False
    PpoPlayer = None  # type: ignore

# Player must have a CODE, NAME, DESCRIPTION, CLASS.
CliPlayer = namedtuple("CliPlayer", ["code", "name", "description", "import_fn"])
CLI_PLAYERS = [
    CliPlayer(
        "H", "HumanPlayer", "Human player, uses input() to get action.", HumanPlayer
    ),
    CliPlayer("R", "RandomPlayer", "Chooses actions at random.", RandomPlayer),
    CliPlayer(
        "W",
        "WeightedRandomPlayer",
        "Like RandomPlayer, but favors buying cities, settlements, and dev cards when possible.",
        WeightedRandomPlayer,
    ),
    CliPlayer(
        "VP",
        "VictoryPointPlayer",
        "Chooses randomly from actions that increase victory points immediately if possible, else at random.",
        VictoryPointPlayer,
    ),
    CliPlayer(
        "G",
        "GreedyPlayoutsPlayer",
        "For each action, will play N random 'playouts'. "
        + "Takes the action that led to best winning percent. "
        + "First param is NUM_PLAYOUTS",
        GreedyPlayoutsPlayer,
    ),
    CliPlayer(
        "M",
        "MCTSPlayer",
        "Decides according to the MCTS algorithm. First param is NUM_SIMULATIONS.",
        MCTSPlayer,
    ),
    CliPlayer(
        "F",
        "ValueFunctionPlayer",
        "Chooses the action that leads to the most immediate reward, based on a hand-crafted value function.",
        ValueFunctionPlayer,
    ),
    CliPlayer(
        "AB",
        "AlphaBetaPlayer",
        "Implements alpha-beta algorithm. That is, looks ahead a couple "
        + "levels deep evaluating leafs with hand-crafted value function. "
        + "Params are DEPTH, PRUNNING",
        AlphaBetaPlayer,
    ),
    CliPlayer(
        "SAB",
        "SameTurnAlphaBetaPlayer",
        "AlphaBeta but searches only within turn",
        SameTurnAlphaBetaPlayer,
    ),
]

# Append PPO if available (code PPO, params: path, representation, map_type, deterministic)
if _PPO_AVAILABLE:

    def _ppo_import_fn(color, *args):
        # args: model_path, representation, map_type, deterministic
        # Called as PPO, PPO:path, PPO:path:vector, PPO:path:vector:BASE:0
        path = args[0] if len(args) >= 1 and args[0] else "ppo_catan_custom.zip"
        rep = args[1] if len(args) >= 2 and args[1] else "vector"
        map_t = args[2] if len(args) >= 3 and args[2] else "BASE"
        det = args[3] if len(args) >= 4 else True
        return PpoPlayer(
            color,
            model_path=path,
            representation=rep,
            map_type=map_t,
            deterministic=det,
        )

    CLI_PLAYERS.append(
        CliPlayer(
            "PPO",
            "PpoPlayer",
            "PPO trained via gym MaskablePPO .zip. Params: PATH, REPRESENTATION(vector/mixed), MAP(BASE/MINI/TOURNAMENT), DETERMINISTIC(1/0). "
            "E.g. PPO:ppo_catan_custom.zip:vector:BASE:1",
            _ppo_import_fn,
        )
    )


def parse_cli_string(player_string):
    players = []
    player_keys = player_string.split(",")
    colors = [c for c in Color]
    for i, key in enumerate(player_keys):
        parts = key.split(":")
        code = parts[0]
        for cli_player in CLI_PLAYERS:
            if cli_player.code == code:
                params = [colors[i]] + parts[1:]
                player = cli_player.import_fn(*params)
                players.append(player)
                break
    return players


def register_cli_player(code, player_class):
    CLI_PLAYERS.append(
        CliPlayer(
            code,
            player_class.__name__,
            player_class.__doc__,
            player_class,
        ),
    )


CUSTOM_ACCUMULATORS = []


def register_cli_accumulator(accumulator_class):
    CUSTOM_ACCUMULATORS.append(accumulator_class)


def player_help_table():
    table = Table(title="Player Legend")
    table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
    table.add_column("PLAYER")
    table.add_column("DESCRIPTION")
    for player in CLI_PLAYERS:
        table.add_row(player.code, player.name, player.description)
    return table
