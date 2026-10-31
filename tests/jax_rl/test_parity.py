"""Replays engine games through the JAX environment, move by move.

The JAX state is never re-synced from the engine: it runs from its own previous
state, with only the dice and stolen card forced to what the engine rolled.
"""

import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from catanatron.game import Game  # noqa: E402
from catanatron.jax_rl.bridge import ActionCodec, state_from_game  # noqa: E402
from catanatron.jax_rl.env import CatanEnv, Chance  # noqa: E402
from catanatron.models.enums import RESOURCES, ActionType  # noqa: E402
from catanatron.models.map import build_map  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402

# Fields the rules depend on. Left out: dev_deck, whose drawn prefix is unknowable
# from Python, and last_settlement, which only matters during initial placement and
# is covered there by the legal-move comparison.
COMPARED = [
    "tile_resource",
    "tile_number",
    "port_resource",
    "node_owner",
    "node_level",
    "edge_owner",
    "robber_tile",
    "reach",
    "hand",
    "bank",
    "dev_hand",
    "dev_owned",
    "dev_played",
    "dev_drawn",
    "vp_public",
    "vp_actual",
    "roads_left",
    "settlements_left",
    "cities_left",
    "road_length",
    "road_holder",
    "road_best",
    "army_holder",
    "current",
    "turn_player",
    "prompt",
    "has_rolled",
    "played_dev",
    "road_building",
    "free_roads",
    "discard_counts",
    "num_turns",
    "winner",
    "done",
]


class Replayer:
    def __init__(self, map_type):
        self.map_type = map_type
        self.env = CatanEnv(map_type=map_type)
        self.codec = ActionCodec(self.env.topo)
        self.step = jax.jit(self.env.apply_chance)
        self.mask = jax.jit(jax.vmap(self.env.legal_mask))

    def compare_state(self, game, state, context):
        expected = state_from_game(game, self.env.topo)
        for name in COMPARED:
            got = np.asarray(getattr(state, name))
            want = np.asarray(getattr(expected, name))
            assert np.array_equal(got, want), (
                f"{context}: field {name} differs\n jax   {got}\n python {want}"
            )

    def compare_mask(self, game, state, context):
        mask = np.asarray(self.mask(state))[0]
        expected = sorted(self.codec.encode(a) for a in game.playable_actions)
        got = np.flatnonzero(mask).tolist()
        assert got == sorted(set(expected)), (
            f"{context}: legal actions differ\n jax only   {set(got) - set(expected)}"
            f"\n python only {set(expected) - set(got)}"
        )

    def play(self, seed, choose):
        random.seed(seed)
        game = Game(
            [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
            seed=seed,
            catan_map=build_map(self.map_type),
        )
        state = state_from_game(game, self.env.topo)
        self.compare_mask(game, state, f"seed {seed} start")

        ply = 0
        while game.winning_color() is None and game.state.num_turns < 1000:
            action = choose(game)
            index = self.codec.encode(action)
            record = game.execute(action)

            dice, stolen = 7, 0
            if action.action_type == ActionType.ROLL:
                dice = sum(record.result)
            elif (
                action.action_type == ActionType.MOVE_ROBBER
                and record.result is not None
            ):
                stolen = RESOURCES.index(record.result)
            chance = Chance(jnp.array([dice]), jnp.array([stolen]))
            state = self.step(state, jnp.array([index]), chance)

            ply += 1
            context = f"seed {seed} ply {ply} after {action}"
            self.compare_state(game, state, context)
            if game.winning_color() is None:
                self.compare_mask(game, state, context)
        return game, ply


@pytest.fixture(scope="module")
def base_replayer():
    return Replayer("BASE")


def random_choice(game):
    return random.choice(game.playable_actions)


def builder_choice(game):
    """Prefers building, reaching states random play rarely sees: longest road
    fights, cut roads, depleted bank and deck, and running out of pieces."""
    weights = {
        ActionType.BUILD_CITY: 200,
        ActionType.BUILD_SETTLEMENT: 200,
        ActionType.BUILD_ROAD: 100,
        ActionType.BUY_DEVELOPMENT_CARD: 50,
        ActionType.PLAY_KNIGHT_CARD: 50,
        ActionType.PLAY_ROAD_BUILDING: 50,
        ActionType.PLAY_YEAR_OF_PLENTY: 50,
        ActionType.PLAY_MONOPOLY: 50,
        ActionType.MARITIME_TRADE: 5,
    }
    actions = game.playable_actions
    return random.choices(
        actions, [weights.get(a.action_type, 1) for a in actions], k=1
    )[0]


@pytest.mark.parametrize("seed", range(10))
def test_random_games_match(base_replayer, seed):
    base_replayer.play(seed, random_choice)


@pytest.mark.parametrize("seed", range(100, 110))
def test_builder_games_match(base_replayer, seed):
    base_replayer.play(seed, builder_choice)


@pytest.mark.parametrize("map_type", ["MINI", "TOURNAMENT"])
def test_other_maps_match(map_type):
    replayer = Replayer(map_type)
    for seed in range(3):
        replayer.play(seed, builder_choice)


def test_action_space_size():
    # 5 + 5 discards + 72 roads + 54 settlements + 54 cities + 20 YoP + 5 monopoly
    # + 38 robber + 20 maritime
    assert CatanEnv("BASE").num_actions == 273
