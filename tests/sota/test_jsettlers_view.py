import random

from catanatron.models.enums import (
    BRICK,
    ORE,
    SHEEP,
    WHEAT,
    WOOD,
    ActionPrompt,
    ActionType,
)
from catanatron.models.player import Color, RandomPlayer
from catanatron.sota.jsettlers import protocol as p
from catanatron.sota.jsettlers.board import HEX_IDS, V1_SENT_NUMBERS
from catanatron.sota.jsettlers.view import (
    PLAY1,
    START1A,
    START1B,
    START2A,
    START2B,
    GameView,
    Hand,
)
from tests.sota.test_jsettlers_board import random_jsettlers_layout

GAME = "g"
US = 2


def feed(view, *texts):
    for text in texts:
        view.apply(p.parse(text))


def started_view():
    hexes, dice, _ = random_jsettlers_layout(random.Random(1))
    sent_numbers = [V1_SENT_NUMBERS.index(n) if n > 0 else -1 for n in dice]
    desert = HEX_IDS[hexes.index(0)]
    layout = ",".join(str(v) for v in hexes + sent_numbers + [desert])

    view = GameView(GAME, US, Color.RED)
    feed(view, *(f"1012|{GAME},bot{pn},{pn},true" for pn in range(4)))
    feed(
        view, f"1014|{GAME},{layout}", f"1096|{GAME}|3|0|4|0", f"1025|{GAME},{START1A}"
    )
    return view


def test_hand_known_counts_stay_lower_bounds():
    hand = Hand()
    hand.gain(SHEEP, 2)
    hand.gain(WOOD, 1)
    hand.lose_unknown(1)  # a steal of unknown type: could have been either
    assert hand.total() == 2
    assert hand.known[SHEEP] == 1 and hand.known[WOOD] == 0
    assert hand.unknown == 1

    hand.lose(SHEEP, 2)  # a known-type loss beyond what we knew uses the unknown card
    assert hand.total() == 0


def test_initial_placement_prompts():
    view = started_view()
    agent = RandomPlayer(Color.RED)
    feed(view, f"1026|{GAME},{US},{START1A}")

    game = view.snapshot(agent, ActionPrompt.BUILD_INITIAL_SETTLEMENT)
    assert game.state.current_color() == Color.RED
    assert len(game.playable_actions) == 54

    node = game.playable_actions[0].value
    js_node = view.mapping.id_to_node[node]
    feed(view, f"1009|{GAME},{US},1,{js_node}", f"1025|{GAME},{START1B}")
    game = view.snapshot(agent, ActionPrompt.BUILD_INITIAL_ROAD)
    assert game.playable_actions
    assert all(node in a.value for a in game.playable_actions)


def test_second_settlement_collects_resources():
    view = started_view()
    feed(view, f"1026|{GAME},{US},{START2A}")
    js_node = next(
        js
        for js, node in view.mapping.node_to_id.items()
        if len(view.mapping.catan_map.adjacent_tiles[node]) == 3
        and all(t.resource for t in view.mapping.catan_map.adjacent_tiles[node])
    )
    feed(view, f"1009|{GAME},{US},1,{js_node}")
    assert view.us.hand.total() == 3


def test_cancelled_second_settlement_returns_its_resources():
    """When a robot gets stuck placing its second road, the server ends its turn
    and takes back the second settlement along with the resources it gave."""
    view = started_view()
    robot = 0
    feed(view, f"1026|{GAME},{robot},{START2A}")
    js_node = next(
        js
        for js, node in view.mapping.node_to_id.items()
        if len(view.mapping.catan_map.adjacent_tiles[node]) == 3
        and all(t.resource for t in view.mapping.catan_map.adjacent_tiles[node])
    )
    feed(view, f"1009|{GAME},{robot},1,{js_node}", f"1025|{GAME},{START2B}")
    assert view.players[robot].hand.total() == 3

    feed(view, f"1044|{GAME},1")
    assert view.players[robot].hand.total() == 0
    assert not view.buildings[robot]["SETTLEMENT"]


def test_dice_resources_and_snapshot_hands():
    view = started_view()
    feed(
        view,
        f"1026|{GAME},{US},15",
        f"1028|{GAME},8",
        # 2 players gain: p2 has 3 cards after +2 ore(2) +1 wood(5); p1 has 1 after +1 clay(1)
        f"1092|{GAME}|2|{US}|3|2|2|1|5|0|1|1|1|1",
        f"1025|{GAME},{PLAY1}",
    )
    assert view.us.hand.known[ORE] == 2 and view.us.hand.known[WOOD] == 1
    assert view.players[1].hand.known[BRICK] == 1

    game = view.snapshot(RandomPlayer(Color.RED), ActionPrompt.PLAY_TURN)
    key = f"P{game.state.color_to_index[Color.RED]}"
    assert game.state.player_state[f"{key}_ORE_IN_HAND"] == 2
    assert game.state.player_state[f"{key}_HAS_ROLLED"]
    assert ActionType.END_TURN in {a.action_type for a in game.playable_actions}


def test_final_scores_one_per_seat():
    view = started_view()
    feed(view, f"1061|{GAME},10,7,0,0,false,true,false,false")
    assert view.final_scores == [10, 7, 0, 0]


def test_full_set_clears_unknown_cards():
    view = started_view()
    view.us.hand.unknown = 3
    feed(view, f"1086|{GAME}|{US}|100|1|0|2|0|3|1|4|2|5|0")
    assert view.us.hand.unknown == 0
    assert view.us.hand.known[SHEEP] == 1 and view.us.hand.known[WHEAT] == 2
