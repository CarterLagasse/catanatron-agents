"""
This is to allow an API like:

from catanatron import Game, Player, Color, Accumulator
"""

from catanatron.game import Game, GameAccumulator
from catanatron.models.enums import (
    BRICK,
    ORE,
    RESOURCES,
    SHEEP,
    WHEAT,
    WOOD,
    Action,
    ActionType,
)
from catanatron.models.player import Color, Player, RandomPlayer
