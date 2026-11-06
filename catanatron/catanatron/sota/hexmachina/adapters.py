"""The adapters surface the HexMachina players were evolved against.

The evolved players do `from .adapters import ...`. Their original adapters.py
imported from an older catanatron layout (catanatron_experimental.*,
catanatron_gym.*); this maps the same names onto this repo's modules. It is the
union of the two adapter versions in the paper's repo: the one shipped with
supplementary/Evolutions/* and the one in agents/best_player/.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from catanatron.features import build_production_features
from catanatron.game import Game
from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    RESOURCES,
    SETTLEMENT,
    Action,
    ActionType,
)
from catanatron.models.map import number_probability
from catanatron.models.player import Color, Player
from catanatron.players.minimax import (
    AlphaBetaPlayer,
    DebugActionNode,
    DebugStateNode,
    SameTurnAlphaBetaPlayer,
)
from catanatron.players.tree_search_utils import (
    DETERMINISTIC_ACTIONS,
    execute_deterministic,
    execute_spectrum,
    expand_spectrum,
    list_prunned_actions,
    prune_robber_actions,
)
from catanatron.players.value import (
    DEFAULT_WEIGHTS,
    ValueFunctionPlayer,
    base_fn,
    contender_fn,
    get_value_fn,
    value_production,
)
from catanatron.state_functions import (
    get_dev_cards_in_hand,
    get_enemy_colors,
    get_player_buildings,
    get_player_freqdeck,
)


def copy_game(game: Game) -> Game:
    return game.copy()


def render_debug_tree(node: DebugStateNode) -> str:
    raise NotImplementedError("render_debug_tree needs graphviz; not ported")


# --- Names only the agents/best_player adapter had ---


def current_color(game: Game) -> Color:
    return game.state.current_color()


def playable_actions(game: Game) -> List[Action]:
    return list(game.playable_actions)


def is_initial_build_phase(game: Game) -> bool:
    return bool(game.state.is_initial_build_phase)


def colors(game: Game) -> List[Color]:
    return list(game.state.colors)


def board(game: Game) -> Any:
    return game.state.board


def execute(game: Game, action: Action, *, validate: bool = False) -> None:
    game.execute(action, validate_action=validate)


def winning_color(game: Game) -> Optional[Color]:
    return game.winning_color()


def make_action(color: Color, action_type: ActionType, value: Any = None) -> Action:
    return Action(color, action_type, value)


def chance_children_for_action(game: Game, action: Action) -> List[Tuple[Game, float]]:
    return execute_spectrum(game, action)


def chance_children(
    game: Game, actions: Iterable[Action]
) -> Dict[Action, List[Tuple[Game, float]]]:
    return expand_spectrum(game, list(actions))


def pruned_actions(game: Game) -> List[Action]:
    return list_prunned_actions(game)


def make_value_fn(
    builder_name: str, params=DEFAULT_WEIGHTS, custom_value_fn=None
) -> Callable[[Game, Color], float]:
    return get_value_fn(builder_name, params, custom_value_fn)


def production_features_sampler(include_variety: bool = True):
    return build_production_features(include_variety)


def player_build_nodes(game: Game, color: Color, building_type: Any) -> Iterable[int]:
    return get_player_buildings(game.state, color, building_type)


def dev_cards_in_hand(game: Game, color: Color, card: Any) -> int:
    return get_dev_cards_in_hand(game.state, color, card)


def enemy_colors_of(game: Game, my_color: Color) -> Iterable[Color]:
    return get_enemy_colors(game.state.colors, my_color)


def opponent_freqdeck(game: Game, color: Color) -> Any:
    return get_player_freqdeck(game.state, color)


def p_roll(total: int) -> float:
    return number_probability(total)


def state_action_count(game: Game) -> int:
    return len(game.state.action_records)


def num_turns(game: Game) -> int:
    return getattr(game.state, "num_turns", 0)
