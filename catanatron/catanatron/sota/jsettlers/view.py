"""Tracks a JSettlers game from server messages and snapshots it as a catanatron Game.

The snapshot is what the catanatron Player decides on. JSettlers hides some
information that catanatron's State expects to be concrete, so the snapshot
fills it in:
- Opponents' resource cards of unknown type (from steals between other players
  or discards) are assigned to the resources their buildings produce most.
- Opponents' unplayed dev cards are sampled from the cards not yet seen,
  excluding victory point cards.
Players see their own hand exactly, as in JSettlers.
"""

import random
from collections import Counter, defaultdict

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import Board
from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    KNIGHT,
    MONOPOLY,
    RESOURCES,
    ROAD,
    ROAD_BUILDING,
    SETTLEMENT,
    VICTORY_POINT,
    YEAR_OF_PLENTY,
    ActionPrompt,
)
from catanatron.models.player import Color, Player
from catanatron.state import State
from catanatron.sota.jsettlers import protocol as p
from catanatron.sota.jsettlers.board import (
    RESOURCE_BY_ID,
    EmptyBoardError,
    mapping_from_layout_message,
)

# soc.game.SOCGame states
START1A, START1B, START2A, START2B = 5, 6, 10, 11
ROLL_OR_CARD, PLAY1 = 15, 20
PLACING_ROBBER = 33
PLACING_FREE_ROAD1, PLACING_FREE_ROAD2 = 40, 41
WAITING_FOR_DISCARDS = 50
WAITING_FOR_ROB_CHOOSE_PLAYER = 51
WAITING_FOR_DISCOVERY = 52
WAITING_FOR_MONOPOLY = 53
OVER = 1000

# soc.game.SOCPlayingPiece types
PIECE_ROAD, PIECE_SETTLEMENT, PIECE_CITY = 0, 1, 2

# soc.game.SOCDevCardConstants (v2.0.00+ numbering)
DEV_BY_ID = {
    1: ROAD_BUILDING,
    2: YEAR_OF_PLENTY,
    3: MONOPOLY,
    4: VICTORY_POINT,
    5: VICTORY_POINT,
    6: VICTORY_POINT,
    7: VICTORY_POINT,
    8: VICTORY_POINT,
    9: KNIGHT,
}
ID_BY_DEV = {ROAD_BUILDING: 1, YEAR_OF_PLENTY: 2, MONOPOLY: 3, KNIGHT: 9}
FULL_DEV_DECK = Counter(
    {KNIGHT: 14, VICTORY_POINT: 5, ROAD_BUILDING: 2, YEAR_OF_PLENTY: 2, MONOPOLY: 2}
)

# soc.message.SOCPlayerElement
PE_SET, PE_GAIN, PE_LOSE = 100, 101, 102
PE_UNKNOWN_RESOURCE = 6
PE_RESOURCE_COUNT = 17
PE_PLAYED_DEV_CARD_FLAG = 19

# soc.message.SOCGameElements
GE_DEV_CARD_COUNT, GE_FIRST_PLAYER, GE_CURRENT_PLAYER = 2, 3, 4
GE_LARGEST_ARMY_PLAYER, GE_LONGEST_ROAD_PLAYER = 5, 6

# soc.message.SOCDevCardAction
DC_DRAW, DC_PLAY, DC_ADD_NEW, DC_ADD_OLD, DC_CANNOT_PLAY = 0, 1, 2, 3, 4

SIMPLEACTION_DEVCARD_BOUGHT = 1

SEAT_COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


class Hand:
    """One player's resources as far as we know them: typed counts plus unknown cards.

    known[r] is a lower bound on the player's r cards; unknown is the rest of
    the hand, which the server tells us the size of.
    """

    def __init__(self):
        self.known = {r: 0 for r in RESOURCES}
        self.unknown = 0

    def total(self):
        return sum(self.known.values()) + self.unknown

    def gain(self, resource, n):
        if resource is None:
            self.unknown += n
        else:
            self.known[resource] += n

    def lose(self, resource, n):
        if resource is None:
            self.lose_unknown(n)
            return
        taken = min(self.known[resource], n)
        self.known[resource] -= taken
        # Cards we thought were other types (or unknown) must have been this one.
        self.unknown = max(0, self.unknown - (n - taken))

    def lose_unknown(self, n):
        # Lost n cards of unknown type (stolen or discarded): any typed card
        # may be among them, so each typed count can drop by up to n.
        remaining = max(0, self.total() - n)
        self.known = {r: max(0, k - n) for r, k in self.known.items()}
        self.unknown = remaining - sum(self.known.values())

    def set(self, resource, n):
        if resource is None:
            self.unknown = n
        else:
            self.known[resource] = n

    def set_total(self, n):
        if self.total() == n:
            return
        typed = sum(self.known.values())
        if typed <= n:
            self.unknown = n - typed
        else:
            self.known = {r: 0 for r in RESOURCES}
            self.unknown = n


class PlayerView:
    def __init__(self, color):
        self.color = color
        self.hand = Hand()
        # Unplayed dev cards whose type we know (always ours; everyone's if fully
        # observable): playable this turn, or bought this turn
        self.dev_old = Counter()
        self.dev_new = Counter()
        self.dev_hidden = 0  # unplayed cards of unknown type
        self.played = Counter()
        self.played_dev_this_turn = False
        self.seated = False
        self.name = None


class GameView:
    def __init__(self, name, our_pn, our_color):
        self.name = name
        self.our_pn = our_pn
        colors = [c for c in SEAT_COLORS if c != our_color]
        self.players = []
        for pn in range(4):
            color = our_color if pn == our_pn else colors.pop(0)
            self.players.append(PlayerView(color))

        self.state = 0
        self.current = -1
        self.first = None
        self.mapping = None
        self.robber_hex = None
        self.board = None
        self.pieces = []  # (pn, piece_type, coord, is_initial) in placement order
        self.longest_road_pn = -1
        self.largest_army_pn = -1
        self.dev_deck_count = sum(FULL_DEV_DECK.values())
        self.has_rolled = False
        self.num_turns = 0
        self.final_scores = None
        self.revealed_vp = Counter()
        self.rng = random.Random()

    @property
    def us(self) -> PlayerView:
        return self.players[self.our_pn]

    def color_of(self, pn):
        return self.players[pn].color

    def seated_pns(self):
        return [pn for pn, pl in enumerate(self.players) if pl.seated]

    # --- Message handling ------------------------------------------------

    def apply(self, msg: p.Message):
        handler = _HANDLERS.get(msg.type)
        if handler is not None:
            handler(self, msg)

    def _sitdown(self, msg):
        pn = int(msg.fields[2])
        self.players[pn].seated = True
        self.players[pn].name = msg.fields[1]

    def _board_layout(self, msg):
        # On joining, the server sends an empty all-water layout; the real one comes at game start.
        try:
            mapping, robber_hex = mapping_from_layout_message(
                msg.fields[1:], v2=(msg.type == p.BOARDLAYOUT2)
            )
        except EmptyBoardError:
            return
        self.mapping, self.robber_hex = mapping, robber_hex
        self._rebuild_board()

    def _game_state(self, msg):
        if len(msg.fields) > 1 and msg.fields[1]:
            self.state = int(msg.fields[1])

    def _turn(self, msg):
        self.current = int(msg.fields[1])
        if len(msg.fields) > 2:
            self.state = int(msg.fields[2])
        self.has_rolled = False
        for pl in self.players:
            pl.played_dev_this_turn = False
        current = self.players[self.current]
        current.dev_old += current.dev_new
        current.dev_new = Counter()
        if self.state >= ROLL_OR_CARD:
            self.num_turns += 1

    def _set_current(self, msg):
        self.current = int(msg.fields[1])

    def _first_player(self, msg):
        self.first = int(msg.fields[1])

    def _game_elements(self, msg):
        values = msg.ints()
        for etype, value in zip(values[::2], values[1::2]):
            if etype == GE_DEV_CARD_COUNT:
                self.dev_deck_count = value
            elif etype == GE_FIRST_PLAYER:
                self.first = value
            elif etype == GE_CURRENT_PLAYER:
                self.current = value
            elif etype == GE_LARGEST_ARMY_PLAYER:
                self.largest_army_pn = value
            elif etype == GE_LONGEST_ROAD_PLAYER:
                self.longest_road_pn = value

    def _longest_road(self, msg):
        self.longest_road_pn = int(msg.fields[1])

    def _largest_army(self, msg):
        self.largest_army_pn = int(msg.fields[1])

    def _put_piece(self, msg):
        pn, piece_type, coord = (int(f) for f in msg.fields[1:4])
        self.pieces.append((pn, piece_type, coord, self.state < ROLL_OR_CARD))
        self._place(*self.pieces[-1])
        if piece_type == PIECE_SETTLEMENT and self.state == START2A:
            # The second initial settlement collects its resources; the server
            # doesn't announce these, clients are expected to apply the rule.
            self._starting_resources(pn, coord, +1)

    def _starting_resources(self, pn, coord, sign):
        hand = self.players[pn].hand
        node = self.mapping.node_to_id[coord]
        for tile in self.mapping.catan_map.adjacent_tiles[node]:
            if tile.resource is not None:
                if sign > 0:
                    hand.gain(tile.resource, 1)
                else:
                    hand.lose(tile.resource, 1)

    def _cancel_build(self, msg):
        # Only initial settlements can be taken back (the player then re-places it).
        if int(msg.fields[1]) != PIECE_SETTLEMENT or self.state not in (
            START1B,
            START2B,
        ):
            return
        for i in range(len(self.pieces) - 1, -1, -1):
            if (
                self.pieces[i][0] == self.current
                and self.pieces[i][1] == PIECE_SETTLEMENT
            ):
                coord = self.pieces[i][2]
                del self.pieces[i]
                if self.state == START2B:
                    # The server also takes back the second settlement's
                    # resources (e.g. when it ends a stuck robot's turn).
                    self._starting_resources(self.current, coord, -1)
                break
        self._rebuild_board()

    def _rebuild_board(self):
        self.board = Board(self.mapping.catan_map)
        self.buildings = {
            pn: {SETTLEMENT: [], CITY: [], ROAD: []} for pn in range(len(self.players))
        }
        self.board.robber_coordinate = self.mapping.hex_to_coordinate[self.robber_hex]
        for piece in self.pieces:
            self._place(*piece)

    def _place(self, pn, piece_type, coord, is_initial):
        color = self.color_of(pn)
        buildings = self.buildings[pn]
        if piece_type == PIECE_ROAD:
            edge = self.mapping.edge_id(coord)
            self.board.build_road(color, edge)
            buildings[ROAD].append(edge)
        elif piece_type == PIECE_SETTLEMENT:
            node = self.mapping.node_to_id[coord]
            self.board.build_settlement(color, node, initial_build_phase=is_initial)
            buildings[SETTLEMENT].append(node)
        elif piece_type == PIECE_CITY:
            node = self.mapping.node_to_id[coord]
            self.board.build_city(color, node)
            buildings[SETTLEMENT].remove(node)
            buildings[CITY].append(node)

    def _player_element(self, msg):
        pn, action, etype, amount = (int(f) for f in msg.fields[1:5])
        self._element(pn, action, etype, amount)

    def _player_elements(self, msg):
        values = msg.ints()
        pn, action = values[0], values[1]
        etypes = values[2::2]
        for etype, amount in zip(etypes, values[3::2]):
            self._element(pn, action, etype, amount)
        if action == PE_SET and set(range(1, 6)) <= set(etypes):
            self.players[pn].hand.unknown = 0  # the whole hand was given

    def _element(self, pn, action, etype, amount):
        pl = self.players[pn]
        if 1 <= etype <= 6:
            resource = RESOURCE_BY_ID.get(etype)  # None for UNKNOWN_RESOURCE
            if action == PE_SET:
                pl.hand.set(resource, amount)
            elif action == PE_GAIN:
                pl.hand.gain(resource, amount)
            elif action == PE_LOSE:
                pl.hand.lose(resource, amount)
        elif etype == PE_RESOURCE_COUNT and action == PE_SET:
            pl.hand.set_total(amount)
        elif etype == PE_PLAYED_DEV_CARD_FLAG and action == PE_SET:
            pl.played_dev_this_turn = amount != 0

    def _resource_count(self, msg):
        pn, count = int(msg.fields[1]), int(msg.fields[2])
        self.players[pn].hand.set_total(count)

    def _dice_result(self, msg):
        self.has_rolled = True

    def _dice_result_resources(self, msg):
        values = msg.ints()
        i = 1
        for _ in range(values[0]):
            pn, total = values[i], values[i + 1]
            i += 2
            while i < len(values) and values[i] != 0:
                amount, rtype = values[i], values[i + 1]
                self.players[pn].hand.gain(RESOURCE_BY_ID.get(rtype), amount)
                i += 2
            i += 1  # skip the 0 that ends this player's list
            self.players[pn].hand.set_total(total)

    def _discard(self, msg):
        if not msg.fields[1].startswith("p"):
            return  # only the client->server form omits the player
        pn = int(msg.fields[1][1:])
        amounts = [int(f) for f in msg.fields[2:8]]
        hand = self.players[pn].hand
        for rtype, amount in enumerate(amounts[:5], start=1):
            if amount:
                hand.lose(RESOURCE_BY_ID[rtype], amount)
        if amounts[5]:
            hand.lose_unknown(amounts[5])

    def _move_robber(self, msg):
        coord = int(msg.fields[2])
        if coord > 0:
            self.robber_hex = coord

    def _robbery_result(self, msg):
        perp, victim = int(msg.fields[1]), int(msg.fields[2])
        kind = msg.fields[3]
        if kind == "R":
            rtype, amount, is_gain_lose = (
                int(msg.fields[4]),
                int(msg.fields[5]),
                msg.fields[6],
            )
            if is_gain_lose != "T":
                return
            resource = RESOURCE_BY_ID.get(rtype)
            self.players[victim].hand.lose(resource, amount)
            self.players[perp].hand.gain(resource, amount)
        elif kind == "S":
            # S, then (type, amount) pairs, then T/F
            fields, i, stolen = msg.fields, 4, []
            while i + 1 < len(fields) and fields[i] not in ("T", "F"):
                stolen.append((int(fields[i]), int(fields[i + 1])))
                i += 2
            if i < len(fields) and fields[i] != "T":
                return
            for rtype, amount in stolen:
                resource = RESOURCE_BY_ID.get(rtype)
                self.players[victim].hand.lose(resource, amount)
                self.players[perp].hand.gain(resource, amount)

    def _dev_card_action(self, msg):
        pn, action = int(msg.fields[1]), int(msg.fields[2])
        card_ids = [int(f) for f in msg.fields[3:]]
        if pn < 0:
            return
        if self.state == OVER:
            # The server reveals everyone's victory point cards at the end.
            self.revealed_vp[pn] += sum(
                DEV_BY_ID.get(c) == VICTORY_POINT for c in card_ids
            )
            return
        pl = self.players[pn]
        for card_id in card_ids:
            card = DEV_BY_ID.get(card_id)  # None: type hidden from us
            if action in (DC_DRAW, DC_ADD_NEW, DC_ADD_OLD):
                if card is None:
                    pl.dev_hidden += 1
                elif action == DC_ADD_OLD:
                    pl.dev_old[card] += 1
                else:
                    pl.dev_new[card] += 1
            elif action == DC_PLAY:
                pl.played[card] += 1
                if pl.dev_old[card] > 0:
                    pl.dev_old[card] -= 1
                elif pl.dev_new[card] > 0:
                    pl.dev_new[card] -= 1
                else:
                    pl.dev_hidden = max(0, pl.dev_hidden - 1)

    def _simple_action(self, msg):
        if int(msg.fields[2]) == SIMPLEACTION_DEVCARD_BOUGHT:
            self.dev_deck_count = int(msg.fields[3])

    def _bank_trade(self, msg):
        values = [int(f) for f in msg.fields[1:]]
        give, get, pn = values[0:5], values[5:10], values[10]
        hand = self.players[pn].hand
        for rtype in range(1, 6):
            hand.lose(RESOURCE_BY_ID[rtype], give[rtype - 1])
            hand.gain(RESOURCE_BY_ID[rtype], get[rtype - 1])

    def _accept_offer(self, msg):
        values = [int(f) for f in msg.fields[1:]]
        accepting, offering = values[0], values[1]
        if len(values) < 12:
            return
        to_accepting, to_offering = values[2:7], values[7:12]
        for rtype in range(1, 6):
            resource = RESOURCE_BY_ID[rtype]
            self.players[offering].hand.lose(resource, to_accepting[rtype - 1])
            self.players[accepting].hand.gain(resource, to_accepting[rtype - 1])
            self.players[accepting].hand.lose(resource, to_offering[rtype - 1])
            self.players[offering].hand.gain(resource, to_offering[rtype - 1])

    def _pick_resources(self, msg):
        values = [int(f) for f in msg.fields[1:]]
        if len(values) < 7:
            return
        pn = values[5]
        for rtype in range(1, 6):
            self.players[pn].hand.gain(RESOURCE_BY_ID[rtype], values[rtype - 1])

    def _game_stats(self, msg):
        # One score per seat of the game (2 to 4), then one is-robot flag per seat
        n = (len(msg.fields) - 1) // 2
        self.final_scores = [int(f) for f in msg.fields[1 : 1 + n]]

    # --- Snapshot --------------------------------------------------------

    def turn_order(self):
        first = self.first if self.first is not None else 0
        seated = self.seated_pns()
        return [pn for pn in sorted(seated, key=lambda pn: (pn - first) % 4)]

    def snapshot(self, agent: Player, prompt: ActionPrompt, discard_count=0) -> Game:
        order = self.turn_order()
        colors = tuple(self.color_of(pn) for pn in order)
        players = [
            agent if pn == self.our_pn else _StandIn(self.color_of(pn)) for pn in order
        ]

        state = State([], None, initialize=False)
        state.players = players
        state.colors = colors
        state.color_to_index = {color: i for i, color in enumerate(colors)}
        state.discard_limit = 7
        state.friendly_robber = False

        state.board = self.board.copy()
        state.board.robber_coordinate = self.mapping.hex_to_coordinate[self.robber_hex]
        road_color = (
            self.color_of(self.longest_road_pn) if self.longest_road_pn >= 0 else None
        )
        state.board.road_color = road_color
        # The length to take the card from its holder, or (no holder) to beat everyone
        state.board.road_length = (
            state.board.road_lengths[road_color]
            if road_color
            else max(state.board.road_lengths.values(), default=0)
        )

        hands, dev_hands = self._determinize()
        state.player_state = {}
        held = Counter()
        for i, pn in enumerate(order):
            self._fill_player_state(
                state.player_state, f"P{i}", pn, hands[pn], dev_hands[pn]
            )
            held.update(hands[pn])
        state.resource_freqdeck = [max(0, 19 - held[r]) for r in RESOURCES]
        state.development_listdeck = self._remaining_dev_deck(dev_hands)

        state.buildings_by_color = {
            self.color_of(pn): defaultdict(
                list, {k: list(v) for k, v in self.buildings[pn].items()}
            )
            for pn in order
        }
        state.action_records = []
        state.num_turns = self.num_turns
        state.current_turn_index = state.color_to_index[self.color_of(self.current)]
        state.current_player_index = state.color_to_index[agent.color]
        state.current_prompt = prompt
        state.is_initial_build_phase = self.state < ROLL_OR_CARD
        state.is_discarding = prompt == ActionPrompt.DISCARD
        state.discard_counts = [0] * len(colors)
        if state.is_discarding:
            state.discard_counts[state.current_player_index] = discard_count
        state.is_moving_knight = prompt == ActionPrompt.MOVE_ROBBER
        state.is_road_building = self.state in (PLACING_FREE_ROAD1, PLACING_FREE_ROAD2)
        state.free_roads_available = {PLACING_FREE_ROAD1: 2, PLACING_FREE_ROAD2: 1}.get(
            self.state, 0
        )
        state.is_resolving_trade = False
        state.current_trade = (0,) * 11
        state.acceptees = tuple(False for _ in colors)

        game = Game(players=[], initialize=False)
        game.seed = 0
        game.id = f"jsettlers:{self.name}"
        game.vps_to_win = 10
        game.friendly_robber = False
        game.state = state
        game.playable_actions = generate_playable_actions(state)
        return game

    def public_vps(self, pn):
        b = self.buildings[pn]
        has_road = pn == self.longest_road_pn
        has_army = pn == self.largest_army_pn
        return len(b[SETTLEMENT]) + 2 * len(b[CITY]) + 2 * has_road + 2 * has_army

    def _fill_player_state(self, ps, key, pn, hand, dev_hand):
        pl = self.players[pn]
        b = self.buildings[pn]
        has_road = pn == self.longest_road_pn
        has_army = pn == self.largest_army_pn
        public_vps = self.public_vps(pn)
        ps[f"{key}_VICTORY_POINTS"] = public_vps
        ps[f"{key}_ACTUAL_VICTORY_POINTS"] = public_vps + dev_hand[VICTORY_POINT]
        ps[f"{key}_ROADS_AVAILABLE"] = 15 - len(b[ROAD])
        ps[f"{key}_SETTLEMENTS_AVAILABLE"] = 5 - len(b[SETTLEMENT])
        ps[f"{key}_CITIES_AVAILABLE"] = 4 - len(b[CITY])
        ps[f"{key}_HAS_ROAD"] = has_road
        ps[f"{key}_HAS_ARMY"] = has_army
        ps[f"{key}_HAS_ROLLED"] = pn == self.current and self.has_rolled
        ps[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = pl.played_dev_this_turn
        ps[f"{key}_LONGEST_ROAD_LENGTH"] = self.board.road_lengths[pl.color]
        for resource in RESOURCES:
            ps[f"{key}_{resource}_IN_HAND"] = hand[resource]
        for card in DEVELOPMENT_CARDS:
            ps[f"{key}_{card}_IN_HAND"] = dev_hand[card]
            ps[f"{key}_PLAYED_{card}"] = pl.played[card]
        for card in (KNIGHT, MONOPOLY, YEAR_OF_PLENTY, ROAD_BUILDING):
            # Cards bought this turn can't be played yet
            owned_at_start = dev_hand[card] - pl.dev_new[card] > 0
            ps[f"{key}_{card}_OWNED_AT_START"] = owned_at_start

    def _determinize(self):
        """Concrete resource and dev card hands for every player."""
        hands, dev_hands = {}, {}
        unseen = FULL_DEV_DECK.copy()
        for pl in self.players:
            unseen.subtract(pl.played)
            unseen.subtract(pl.dev_old + pl.dev_new)

        for pn, pl in enumerate(self.players):
            hand = Counter(pl.hand.known)
            if pl.hand.unknown:
                hand.update(self._guess_unknown(pn, pl.hand.unknown))
            hands[pn] = hand
            dev_hands[pn] = pl.dev_old + pl.dev_new
            dev_hands[pn].update(self._sample_devs(unseen, pl.dev_hidden))
        return hands, dev_hands

    def _guess_unknown(self, pn, n):
        # Spread unknown cards over the resources this player produces most.
        weights = Counter({r: 0.05 for r in RESOURCES})
        production = self.mapping.catan_map.node_production
        for node in self.buildings[pn][SETTLEMENT]:
            weights.update(production[node])
        for node in self.buildings[pn][CITY]:
            weights.update({r: 2 * v for r, v in production[node].items()})
        assigned = Counter()
        for _ in range(n):
            resource = max(RESOURCES, key=lambda r: weights[r] / (1 + assigned[r]))
            assigned[resource] += 1
        return assigned

    def _sample_devs(self, unseen: Counter, n):
        hand = Counter()
        for _ in range(n):
            pool = [c for c in unseen.elements() if c != VICTORY_POINT] or list(
                unseen.elements()
            )
            card = self.rng.choice(pool) if pool else KNIGHT
            hand[card] += 1
            unseen[card] -= 1
        return hand

    def _remaining_dev_deck(self, dev_hands):
        deck = FULL_DEV_DECK.copy()
        for pn, pl in enumerate(self.players):
            deck.subtract(pl.played)
            deck.subtract(dev_hands[pn])
        cards = list((+deck).elements())
        self.rng.shuffle(cards)
        if len(cards) > self.dev_deck_count:
            cards = cards[: self.dev_deck_count]
        while len(cards) < self.dev_deck_count:
            cards.append(KNIGHT)
        return cards


class _StandIn(Player):
    """Opponent seat in snapshots. Only used if the agent runs playouts."""

    rng = random.Random()

    def decide(self, game, playable_actions):
        return self.rng.choice(playable_actions)


_HANDLERS = {
    p.SITDOWN: GameView._sitdown,
    p.BOARDLAYOUT: GameView._board_layout,
    p.BOARDLAYOUT2: GameView._board_layout,
    p.GAMESTATE: GameView._game_state,
    p.STARTGAME: GameView._game_state,
    p.TURN: GameView._turn,
    p.SETTURN: GameView._set_current,
    p.FIRSTPLAYER: GameView._first_player,
    p.GAMEELEMENTS: GameView._game_elements,
    p.LONGESTROAD: GameView._longest_road,
    p.LARGESTARMY: GameView._largest_army,
    p.PUTPIECE: GameView._put_piece,
    p.CANCELBUILDREQUEST: GameView._cancel_build,
    p.PLAYERELEMENT: GameView._player_element,
    p.PLAYERELEMENTS: GameView._player_elements,
    p.RESOURCECOUNT: GameView._resource_count,
    p.DICERESULT: GameView._dice_result,
    p.DICERESULTRESOURCES: GameView._dice_result_resources,
    p.DISCARD: GameView._discard,
    p.MOVEROBBER: GameView._move_robber,
    p.ROBBERYRESULT: GameView._robbery_result,
    p.DEVCARDACTION: GameView._dev_card_action,
    p.SIMPLEACTION: GameView._simple_action,
    p.BANKTRADE: GameView._bank_trade,
    p.ACCEPTOFFER: GameView._accept_offer,
    p.PICKRESOURCES: GameView._pick_resources,
    p.GAMESTATS: GameView._game_stats,
}
