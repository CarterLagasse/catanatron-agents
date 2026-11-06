"""Replays engine games through the JAX environment, move by move.

The JAX state is never re-synced from the engine: it runs from its own previous
state, with only the dice and stolen card forced to what the engine rolled.
"""

import itertools
import random

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from catanatron.game import Game  # noqa: E402
from catanatron.jax_rl.bridge import ActionCodec, state_from_game  # noqa: E402
from catanatron.jax_rl.env import ROAD_EVAL_CUT, CatanEnv, Chance  # noqa: E402
from catanatron.models.board import Board  # noqa: E402
from catanatron.models.enums import RESOURCES, ActionRecord, ActionType  # noqa: E402
from catanatron.models.map import build_map  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402
from tests import rules_scenarios  # noqa: E402

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


COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


class Replayer:
    def __init__(self, map_type, num_players=2):
        self.map_type = map_type
        self.num_players = num_players
        self.env = CatanEnv(map_type=map_type, num_players=num_players)
        self.codec = ActionCodec(self.env.topo)
        self.step = jax.jit(self.env.apply_chance)
        self.mask = jax.jit(jax.vmap(self.env.legal_mask))

    def compare_state(self, game, state, context):
        expected = state_from_game(game, self.env.topo)
        for name in COMPARED:
            got = np.asarray(getattr(state, name))
            want = np.asarray(getattr(expected, name))
            assert np.array_equal(
                got, want
            ), f"{context}: field {name} differs\n jax   {got}\n python {want}"

    def compare_mask(self, game, state, context):
        mask = np.asarray(self.mask(state))[0]
        expected = sorted(
            self.codec.encode(a, game.state.color_to_index)
            for a in game.playable_actions
        )
        got = np.flatnonzero(mask).tolist()
        assert got == sorted(set(expected)), (
            f"{context}: legal actions differ\n jax only   {set(got) - set(expected)}"
            f"\n python only {set(expected) - set(got)}"
        )

    def play(self, seed, choose):
        random.seed(seed)
        game = Game(
            [RandomPlayer(color) for color in COLORS[: self.num_players]],
            seed=seed,
            catan_map=build_map(self.map_type),
        )
        state = state_from_game(game, self.env.topo)
        self.compare_mask(game, state, f"seed {seed} start")

        ply = 0
        while game.winning_color() is None and game.state.num_turns < 1000:
            action = choose(game)
            index = self.codec.encode(action, game.state.color_to_index)
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


@pytest.fixture(scope="module", params=[3, 4])
def multiplayer_replayer(request):
    return Replayer("BASE", request.param)


@pytest.mark.parametrize("seed", range(6))
def test_multiplayer_random_games_match(multiplayer_replayer, seed):
    multiplayer_replayer.play(seed, random_choice)


@pytest.mark.parametrize("seed", range(200, 212))
def test_multiplayer_builder_games_match(multiplayer_replayer, seed):
    multiplayer_replayer.play(seed, builder_choice)


@pytest.mark.parametrize(
    "scenario", rules_scenarios.SCENARIOS, ids=lambda scenario: scenario.__name__
)
def test_rule_scenarios_match(base_replayer, scenario):
    """The positions tests/test_rules.py checks the engine's rules on."""
    game, action, dice = scenario()
    state = state_from_game(game, base_replayer.env.topo)
    base_replayer.compare_mask(game, state, f"{scenario.__name__} before")

    index = base_replayer.codec.encode(action, game.state.color_to_index)
    game.execute(action, action_record=ActionRecord(action, dice) if dice else None)
    chance = Chance(jnp.array([sum(dice) if dice else 7]), jnp.array([0]))
    state = base_replayer.step(state, jnp.array([index]), chance)
    base_replayer.compare_state(game, state, scenario.__name__)


def test_longest_road_after_a_cut_matches_the_engine():
    """Every holder and road lengths for 3 players, against Board's special case."""
    env = CatanEnv("BASE", num_players=3)
    apply = jax.jit(env._apply_road_result)
    base = env.reset(jax.random.PRNGKey(0))
    colors = COLORS[:3]
    for lengths in itertools.product(range(3, 8), repeat=3):
        for holder in range(-1, 3):
            board = Board(build_map("BASE"))
            board.road_lengths.update(zip(colors, lengths))
            board.road_color = colors[holder] if holder >= 0 else None
            board._reassign_longest_road_after_cut()
            want = colors.index(board.road_color) if board.road_color else -1

            for cut_seat in range(3):
                s = base._replace(
                    road_length=jnp.array(lengths, jnp.int32),
                    road_holder=jnp.array(holder, jnp.int32),
                    road_eval=jnp.array(ROAD_EVAL_CUT, base.road_eval.dtype),
                    road_eval_seat=jnp.array(cut_seat, jnp.int32),
                )
                out = apply(s, jnp.array(lengths[cut_seat], jnp.int32))
                context = f"lengths {lengths} holder {holder} cut {cut_seat}"
                assert int(out.road_holder) == want, context
                assert int(out.road_best) == board.road_length, context
                delta = np.asarray(out.vp_actual - s.vp_actual)
                expected = np.zeros(3, int)
                if want != holder:
                    expected[want] += 2 if want >= 0 else 0
                    expected[holder] -= 2 if holder >= 0 else 0
                assert delta.tolist() == expected.tolist(), context


def test_action_space_size():
    # 5 + 5 discards + 72 roads + 54 settlements + 54 cities + 20 YoP + 5 monopoly
    # + 38 robber + 20 maritime
    assert CatanEnv("BASE").num_actions == 273
    # One more robber slot per tile for each extra opponent
    assert CatanEnv("BASE", num_players=4).num_actions == 273 + 2 * 19
