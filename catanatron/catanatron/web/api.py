import json
import logging
import re
import traceback
from pathlib import Path

from flask import Blueprint, Response, abort, jsonify, request

from catanatron.game import Game
from catanatron.json import GameEncoder, action_from_json
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.web.mcts_analysis import GameAnalyzer
from catanatron.web.models import get_game_state, upsert_game_state

bp = Blueprint("api", __name__, url_prefix="/api")
VALID_MAP_TEMPLATES = {"BASE", "MINI", "TOURNAMENT"}

# ppo_agents folder at repo root (mounted as /app/ppo_agents in docker)
PPO_AGENTS_ROOT = Path("ppo_agents")


def _list_ppo_agents():
    """List all .zip agents under ppo_agents (recursive, relative posix paths)."""
    if not PPO_AGENTS_ROOT.exists():
        return []
    agents = []
    for p in sorted(PPO_AGENTS_ROOT.rglob("*.zip")):
        # return posix relative to repo root, e.g. ppo_agents/ppo_training/first_agent/iter_1200.zip
        try:
            rel = p.as_posix()
            # ensure it starts with ppo_agents/
            if not rel.startswith("ppo_agents/"):
                rel = f"ppo_agents/{p.name}"
            agents.append(rel)
        except Exception:
            continue
    return agents


def player_factory(player_key):
    raw = player_key[0]
    color = player_key[1]
    # Handle PPO agents: "PPO" or "PPO:<path>[:rep[:map[:det]]]"
    if isinstance(raw, str) and raw.startswith("PPO"):
        # Split PPO prefix
        parts = raw.split(":")
        # parts[0] is "PPO"
        path = None
        if len(parts) >= 2 and parts[1]:
            # Re-join remaining parts that are part of path if path contains colons (unlikely for posix)
            # For PPO:path:rep:map:det, path is parts[1], rep parts[2], etc.
            # Detect if parts[1] looks like a file path (contains / or endswith .zip)
            path = parts[1]
            # If path does not contain "/" and next parts look like rep/map, keep as is
        else:
            # No explicit path: pick most recent agent if available, else fallback
            agents = _list_ppo_agents()
            if agents:
                path = sorted(agents)[-1]
            else:
                path = "ppo_agents/ppo_catan_custom.zip"
        # Determine rep/map/det if provided as extra colon parts and not part of path
        rep = "vector"
        map_t = "BASE"
        det = True
        # Heuristic: if len(parts) >=3, check if parts[2] is rep
        if len(parts) >= 3 and parts[2] in ("vector", "mixed"):
            rep = parts[2]
            if len(parts) >= 4 and parts[3] in ("BASE", "MINI", "TOURNAMENT"):
                map_t = parts[3]
                if len(parts) >= 5:
                    det = parts[4].lower() not in ("0", "false", "no")
            # path already parts[1]
        elif len(parts) >= 3 and parts[2] in ("BASE", "MINI", "TOURNAMENT"):
            map_t = parts[2]
        # If path is just filename without dir, assume inside ppo_agents
        if path and not path.startswith("ppo_agents/"):
            # Allow bare filename like "iter_1200.zip" -> search under ppo_agents
            cand = PPO_AGENTS_ROOT / path
            if cand.exists():
                path = cand.as_posix()
            else:
                # try find recursively
                found = None
                for p in PPO_AGENTS_ROOT.rglob(path):
                    found = p.as_posix()
                    break
                if found:
                    path = found
        try:
            from catanatron.gym.players import PpoPlayer

            return PpoPlayer(color, model_path=path, representation=rep, map_type=map_t, deterministic=det)
        except Exception as e:
            logging.warning(f"Failed to load PpoPlayer {path}: {e}, fallback to RandomPlayer")
            return RandomPlayer(color)
    if raw == "CATANATRON":
        return AlphaBetaPlayer(color, 2, True)
    elif raw == "WEIGHTED_RANDOM":
        return WeightedRandomPlayer(color)
    elif raw == "RANDOM":
        return RandomPlayer(color)
    elif raw == "HUMAN":
        return ValueFunctionPlayer(color, is_bot=False)
    else:
        raise ValueError(f"Invalid player key: {raw}")


@bp.route("/games", methods=("POST",))
def post_game_endpoint():
    if not request.is_json or request.json is None or "players" not in request.json:
        abort(400, description="Missing or invalid JSON body: 'players' key required")

    player_keys = request.json["players"]
    if not isinstance(player_keys, list) or not 2 <= len(player_keys) <= 4:
        abort(400, description="'players' must be a list with 2 to 4 entries")

    map_template = request.json.get("map_template", "BASE")
    if map_template not in VALID_MAP_TEMPLATES:
        abort(
            400,
            description="'map_template' must be one of BASE, MINI, or TOURNAMENT",
        )

    discard_limit = request.json.get("discard_limit", 7)
    if not isinstance(discard_limit, int) or not 5 <= discard_limit <= 20:
        abort(400, description="'discard_limit' must be an integer between 5 and 20")

    vps_to_win = request.json.get("vps_to_win", 10)
    if not isinstance(vps_to_win, int) or not 3 <= vps_to_win <= 20:
        abort(400, description="'vps_to_win' must be an integer between 3 and 20")

    friendly_robber = request.json.get("friendly_robber", False)
    if not isinstance(friendly_robber, bool):
        abort(400, description="'friendly_robber' must be a boolean")

    players = list(map(player_factory, zip(player_keys, Color)))
    catan_map = build_map(map_template)

    game = Game(
        players=players,
        discard_limit=discard_limit,
        friendly_robber=friendly_robber,
        vps_to_win=vps_to_win,
        catan_map=catan_map,
    )
    upsert_game_state(game)
    return jsonify({"game_id": game.id})


@bp.route("/games/<string:game_id>/states/<string:state_index>", methods=("GET",))
def get_game_endpoint(game_id, state_index):
    parsed_state_index = _parse_state_index(state_index)
    game = get_game_state(game_id, parsed_state_index)
    if game is None:
        abort(404, description="Resource not found")

    payload = json.dumps(game, cls=GameEncoder)
    return Response(
        response=payload,
        status=200,
        mimetype="application/json",
    )


@bp.route("/games/<string:game_id>/actions", methods=["POST"])
def post_action_endpoint(game_id):
    game = get_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    if game.winning_color() is not None:
        return Response(
            response=json.dumps(game, cls=GameEncoder),
            status=200,
            mimetype="application/json",
        )

    # TODO: remove `or body_is_empty` when fully implement actions in FE
    body_is_empty = (not request.data) or request.json is None or request.json == {}
    if game.state.current_player().is_bot or body_is_empty:
        game.play_tick()
        upsert_game_state(game)
    else:
        action = action_from_json(request.json)
        game.execute(action)
        upsert_game_state(game)

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route("/stress-test", methods=["GET"])
def stress_test_endpoint():
    players = [
        AlphaBetaPlayer(Color.RED, 2, True),
        AlphaBetaPlayer(Color.BLUE, 2, True),
        AlphaBetaPlayer(Color.ORANGE, 2, True),
        AlphaBetaPlayer(Color.WHITE, 2, True),
    ]
    game = Game(players=players)
    game.play_tick()
    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route(
    "/games/<string:game_id>/states/<string:state_index>/mcts-analysis", methods=["GET"]
)
def mcts_analysis_endpoint(game_id, state_index):
    """Get MCTS analysis for specific game state."""
    logging.info(f"MCTS analysis request for game {game_id} at state {state_index}")

    # Convert 'latest' to None for consistency with get_game_state
    parsed_state_index = _parse_state_index(state_index)
    try:
        game = get_game_state(game_id, parsed_state_index)
        if game is None:
            logging.error(
                f"Game/state not found: {game_id}/{state_index}"
            )  # Use original state_index for logging
            abort(404, description="Game state not found")

        analyzer = GameAnalyzer(num_simulations=100)
        probabilities = analyzer.analyze_win_probabilities(game)

        logging.info(f"Analysis successful. Probabilities: {probabilities}")
        return Response(
            response=json.dumps(
                {
                    "success": True,
                    "probabilities": probabilities,
                    "state_index": (
                        parsed_state_index
                        if parsed_state_index is not None
                        else len(game.state.action_records)
                    ),
                }
            ),
            status=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.error(f"Error in MCTS analysis endpoint: {e!s}")
        logging.error(traceback.format_exc())
        return Response(
            response=json.dumps(
                {"success": False, "error": str(e), "trace": traceback.format_exc()}
            ),
            status=500,
            mimetype="application/json",
        )


@bp.route("/ppo-agents", methods=["GET"])
def list_ppo_agents_endpoint():
    """List all available PPO agents in ppo_agents folder."""
    try:
        agents = _list_ppo_agents()
        return jsonify({"agents": agents})
    except Exception as e:
        logging.error(f"Error listing ppo agents: {e}")
        return jsonify({"agents": [], "error": str(e)})


def _parse_state_index(state_index_str: str):
    """Helper function to parse and validate state_index."""
    if state_index_str == "latest":
        return None
    try:
        return int(state_index_str)
    except ValueError:
        abort(
            400,
            description="Invalid state_index format. state_index must be an integer or 'latest'.",
        )


# ===== Debugging Routes
# @app.route(
#     "/games/<string:game_id>/players/<int:player_index>/features", methods=["GET"]
# )
# def get_game_feature_vector(game_id, player_index):
#     game = get_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     return create_sample(game, game.state.colors[player_index])


# @app.route("/games/<string:game_id>/value-function", methods=["GET"])
# def get_game_value_function(game_id):
#     game = get_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     # model = tf.keras.models.load_model("data/models/mcts-rep-a")
#     model2 = tf.keras.models.load_model("data/models/mcts-rep-b")
#     feature_ordering = get_feature_ordering()
#     indices = [feature_ordering.index(f) for f in NUMERIC_FEATURES]
#     data = {}
#     for color in game.state.colors:
#         sample = create_sample_vector(game, color)
#         # scores = model.call(tf.convert_to_tensor([sample]))

#         inputs1 = [create_board_tensor(game, color)]
#         inputs2 = [[float(sample[i]) for i in indices]]
#         scores2 = model2.call(
#             [tf.convert_to_tensor(inputs1), tf.convert_to_tensor(inputs2)]
#         )
#         data[color.value] = float(scores2.numpy()[0][0])

#     return data
