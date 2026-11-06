from catanatron import Color, Game, RandomPlayer
from catanatron.sota.hexmachina import player as hexmachina

# Mimics the evolved files: relative adapters import, hardcoded seat color, chatty prints.
FAKE_FOO_PLAYER = """
from .adapters import Player, Color, copy_game, base_fn

class FooPlayer(Player):
    def __init__(self, name=None):
        super().__init__(Color.BLUE, name)
        self.value_fn = base_fn()
        print("FooPlayer.__init__")

    def decide(self, game, playable_actions):
        print("FooPlayer.decide", len(playable_actions))
        return max(playable_actions, key=lambda a: self._score(game, a))

    def _score(self, game, action):
        copy = copy_game(game)
        copy.execute(action, validate_action=False)
        return self.value_fn(copy, self.color)
"""


def test_evolved_player_plays_as_its_seat_color(tmp_path, monkeypatch, capsys):
    path = tmp_path / "fake_foo_player.py"
    path.write_text(FAKE_FOO_PLAYER)
    monkeypatch.setattr(hexmachina, "variant_path", lambda variant: path)

    player = hexmachina.HexMachinaPlayer(Color.RED, variant="fake")
    assert player.color == Color.RED
    assert player._impl.color == Color.RED
    assert repr(player) == "HexMachinaPlayer:RED(variant=fake)"

    player.color = Color.WHITE  # benchmarks reassign colors after construction
    game = Game([player, RandomPlayer(Color.BLUE)])
    game.play()
    assert game.winning_color() is not None
    assert player._impl.color == Color.WHITE
    assert "FooPlayer" not in capsys.readouterr().out  # silenced unless VERBOSE


def test_adapters_expose_what_the_evolved_players_import():
    from catanatron.sota.hexmachina import adapters

    names = {
        # supplementary/Evolutions/* players
        "Game", "Player", "Color", "copy_game", "execute_deterministic", "execute_spectrum",
        "expand_spectrum", "list_prunned_actions", "prune_robber_actions", "base_fn",
        "contender_fn", "value_production", "get_value_fn", "DEFAULT_WEIGHTS", "ActionType",
        # agents/best_player
        "Action", "execute", "make_value_fn", "chance_children_for_action", "chance_children",
        "opponent_freqdeck", "player_build_nodes", "dev_cards_in_hand",
    }  # fmt: skip
    assert all(hasattr(adapters, name) for name in names)
