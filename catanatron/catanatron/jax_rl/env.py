"""Pure-JAX implementation of 2- to 4-player Catan: jittable, vmappable, no Python state.

Rules follow the Python engine (`catanatron.apply_action`, `models.actions`,
`models.board`) move for move, including its quirks; tests/jax_rl/test_parity.py
replays real games through both and compares them after every action. Domestic
trades are not modelled, matching the gym environment, whose action space has none.

Seats are abstract: seat 0 moves first and play goes up the seat numbers. Mapping
agents to seats is up to callers. P below is the number of players.

The step function is branchless because `cond`/`switch` become `select` under
vmap, which would run every action handler for every game anyway. The one
expensive operation, the longest-road search, is deferred: `_apply` only flags
it and `batch_step` runs the search on just the games that need it.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from catanatron.jax_rl.topology import (
    ANY_PORT,
    DESERT,
    DEV_DECK,
    KNIGHT,
    MARITIME_PAIRS,
    MONOPOLY,
    NUM_DEV_TYPES,
    NUM_RESOURCES,
    ROAD_BUILDING,
    VICTORY_POINT,
    YEAR_OF_PLENTY,
    YOP_PAIRS,
    get_topology,
)

# Prompts
BUILD_INITIAL_SETTLEMENT, BUILD_INITIAL_ROAD, PLAY_TURN, DISCARD, MOVE_ROBBER = range(5)
NUM_PROMPTS = 5

ROAD_COST = jnp.array([1, 1, 0, 0, 0], dtype=jnp.int32)
SETTLEMENT_COST = jnp.array([1, 1, 1, 1, 0], dtype=jnp.int32)
CITY_COST = jnp.array([0, 0, 0, 2, 3], dtype=jnp.int32)
DEV_COST = jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32)

DISCARD_LIMIT = 7
MAX_ROADS = 15
# Upper bound on simultaneously open trails in the longest-road search. A player
# has at most 15 roads on a degree-3 graph; tests/jax_rl/test_env.py checks this
# against a brute-force search on adversarial road networks.
ROAD_FRONTIER = 256
# How many longest-road searches run per pass over a batch.
ROAD_SEARCH_BATCH = 32

NO_ROAD_EVAL, ROAD_EVAL_BUILT, ROAD_EVAL_CUT = 0, 1, 2


class State(NamedTuple):
    # Board, fixed for the whole game
    tile_resource: jnp.ndarray  # [T] 0-4, DESERT
    tile_number: jnp.ndarray  # [T] dice number, 0 for desert
    port_resource: jnp.ndarray  # [P] 0-4, ANY_PORT
    # Board, dynamic
    node_owner: jnp.ndarray  # [N] -1 or seat
    node_level: jnp.ndarray  # [N] 0 none, 1 settlement, 2 city
    edge_owner: jnp.ndarray  # [E] -1 or seat
    robber_tile: jnp.ndarray
    reach: jnp.ndarray  # [P, N] union of Board.connected_components per seat
    # Cards
    hand: jnp.ndarray  # [P, 5]
    bank: jnp.ndarray  # [5]
    dev_hand: jnp.ndarray  # [P, 5]
    dev_owned: jnp.ndarray  # [P, 5] *_OWNED_AT_START
    dev_played: jnp.ndarray  # [P, 5]
    dev_deck: jnp.ndarray  # [25] shuffled
    dev_drawn: jnp.ndarray
    # Scores and pieces
    vp_public: jnp.ndarray  # [P]
    vp_actual: jnp.ndarray  # [P]
    roads_left: jnp.ndarray  # [P]
    settlements_left: jnp.ndarray  # [P]
    cities_left: jnp.ndarray  # [P]
    last_settlement: jnp.ndarray  # [P]
    road_length: jnp.ndarray  # [P] Board.road_lengths
    road_holder: jnp.ndarray  # -1 or seat
    road_best: jnp.ndarray  # Board.road_length
    army_holder: jnp.ndarray  # -1 or seat
    # Turn structure
    current: jnp.ndarray  # seat that must decide now
    turn_player: jnp.ndarray
    prompt: jnp.ndarray
    has_rolled: jnp.ndarray  # [P]
    played_dev: jnp.ndarray  # [P]
    road_building: jnp.ndarray
    free_roads: jnp.ndarray
    discard_counts: jnp.ndarray  # [P]
    num_turns: jnp.ndarray
    num_plies: jnp.ndarray
    # Pending longest-road search, see batch_step
    road_eval: jnp.ndarray
    road_eval_seat: jnp.ndarray
    # Outcome
    winner: jnp.ndarray  # -1 or seat
    done: jnp.ndarray


class Chance(NamedTuple):
    """Outcome of the random events a step could need."""

    dice: jnp.ndarray  # sum of two dice
    stolen: jnp.ndarray  # resource taken if this action steals, from its victim


class StepInfo(NamedTuple):
    done: jnp.ndarray
    winner: jnp.ndarray
    plies: jnp.ndarray
    rewards: jnp.ndarray  # [P], per seat, for the action just taken


class CatanEnv:
    """Holds static tables. All methods are pure functions of their arguments."""

    def __init__(
        self,
        map_type="BASE",
        vps_to_win=10,
        num_players=2,
        turns_limit=1000,
        max_plies=6000,
        shaping_vp=0.0,
    ):
        self.topo = topo = get_topology(map_type, num_players)
        self.map_type = map_type
        self.num_players = num_players
        self.vps_to_win = vps_to_win
        self.turns_limit = turns_limit
        # The Python engine has no ply limit, but untrained policies can trade with
        # the bank forever inside one turn. Hitting this counts as a draw.
        self.max_plies = max_plies
        self.shaping_vp = shaping_vp
        self.num_actions = topo.num_actions

        self.edges = jnp.asarray(topo.edges)
        self.tile_nodes = jnp.asarray(topo.tile_nodes)
        self.port_nodes = jnp.asarray(topo.port_nodes)
        self.node_tiles = jnp.asarray(topo.node_tiles)
        # -1 padding is remapped to a sentinel slot one past the end
        self.node_neighbors = jnp.asarray(topo.node_neighbors)
        self.node_edges = jnp.asarray(topo.node_edges)
        self.node_neighbors_pad = jnp.where(
            self.node_neighbors < 0, topo.num_nodes, self.node_neighbors
        )
        self.node_edges_pad = jnp.where(
            self.node_edges < 0, topo.num_edges, self.node_edges
        )

        self.yop_pairs = jnp.asarray(YOP_PAIRS)
        self.maritime_pairs = jnp.asarray(MARITIME_PAIRS)
        self.yop_cards = jax.nn.one_hot(YOP_PAIRS, NUM_RESOURCES, dtype=jnp.int32).sum(
            1
        )
        self.yop_cards = jnp.concatenate(
            [self.yop_cards, jnp.eye(NUM_RESOURCES, dtype=jnp.int32)]
        )
        # Which YOP pairs involve each resource
        self.yop_involves = (self.yop_cards[: len(YOP_PAIRS)] > 0).T

    # ===== Reset
    def reset(self, key):
        topo = self.topo
        k_tiles, k_ports, k_deck = jax.random.split(key, 3)
        if topo.fixed_tile_resource is not None:
            tile_resource = jnp.asarray(topo.fixed_tile_resource)
            tile_number = jnp.asarray(topo.fixed_tile_number)
            port_resource = jnp.asarray(topo.fixed_port_resource)
        else:
            tile_resource = jax.random.permutation(
                k_tiles, jnp.asarray(topo.tile_resource_pool)
            )
            port_resource = jax.random.permutation(
                k_ports, jnp.asarray(topo.port_resource_pool)
            )
            # Numbers follow the official spiral, skipping the desert.
            order = jnp.asarray(topo.spiral_order)
            is_land = tile_resource[order] != DESERT
            position = jnp.cumsum(is_land) - 1
            numbers = jnp.asarray(topo.spiral_numbers)[jnp.maximum(position, 0)]
            tile_number = jnp.zeros(topo.num_tiles, jnp.int32)
            tile_number = tile_number.at[order].set(jnp.where(is_land, numbers, 0))

        players = self.num_players
        zeros_p = jnp.zeros(players, jnp.int32)
        scalar0 = jnp.zeros((), jnp.int32)
        return State(
            tile_resource=tile_resource,
            tile_number=tile_number,
            port_resource=port_resource,
            node_owner=jnp.full(topo.num_nodes, -1, jnp.int32),
            node_level=jnp.zeros(topo.num_nodes, jnp.int32),
            edge_owner=jnp.full(topo.num_edges, -1, jnp.int32),
            robber_tile=jnp.argmax(tile_resource == DESERT).astype(jnp.int32),
            reach=jnp.zeros((players, topo.num_nodes), bool),
            hand=jnp.zeros((players, NUM_RESOURCES), jnp.int32),
            bank=jnp.full(NUM_RESOURCES, 19, jnp.int32),
            dev_hand=jnp.zeros((players, NUM_DEV_TYPES), jnp.int32),
            dev_owned=jnp.zeros((players, NUM_DEV_TYPES), bool),
            dev_played=jnp.zeros((players, NUM_DEV_TYPES), jnp.int32),
            dev_deck=jax.random.permutation(k_deck, jnp.asarray(DEV_DECK)),
            dev_drawn=scalar0,
            vp_public=zeros_p,
            vp_actual=zeros_p,
            roads_left=jnp.full(players, MAX_ROADS, jnp.int32),
            settlements_left=jnp.full(players, 5, jnp.int32),
            cities_left=jnp.full(players, 4, jnp.int32),
            last_settlement=jnp.full(players, -1, jnp.int32),
            road_length=zeros_p,
            road_holder=jnp.full((), -1, jnp.int32),
            road_best=scalar0,
            army_holder=jnp.full((), -1, jnp.int32),
            current=scalar0,
            turn_player=scalar0,
            prompt=jnp.full((), BUILD_INITIAL_SETTLEMENT, jnp.int32),
            has_rolled=jnp.zeros(players, bool),
            played_dev=jnp.zeros(players, bool),
            road_building=jnp.zeros((), bool),
            free_roads=scalar0,
            discard_counts=zeros_p,
            num_turns=scalar0,
            num_plies=scalar0,
            road_eval=jnp.full((), NO_ROAD_EVAL, jnp.int32),
            road_eval_seat=scalar0,
            winner=jnp.full((), -1, jnp.int32),
            done=jnp.zeros((), bool),
        )

    # ===== Board queries
    def _free_nodes(self, node_owner):
        """Board.board_buildable_ids: empty and no building next door."""
        occupied = node_owner >= 0
        occupied_pad = jnp.concatenate([occupied, jnp.zeros(1, bool)])
        return ~occupied & ~occupied_pad[self.node_neighbors_pad].any(axis=1)

    def _buildable_edges(self, edge_owner, reach_seat):
        touches = reach_seat[self.edges[:, 0]] | reach_seat[self.edges[:, 1]]
        return (edge_owner < 0) & touches

    def trade_rates(self, s, seat):
        owned = (s.node_owner[self.port_nodes] == seat).any(axis=1)
        three = (owned & (s.port_resource == ANY_PORT)).any()
        two = jax.vmap(lambda r: (owned & (s.port_resource == r)).any())(
            jnp.arange(NUM_RESOURCES)
        )
        return jnp.where(two, 2, jnp.where(three, 3, 4))

    def can_play_dev(self, s, seat):
        return (~s.played_dev[seat]) & (s.dev_hand[seat] >= 1) & s.dev_owned[seat]

    def opponents(self, seat):
        """[P - 1] seats after `seat` in turn order, which are the robber's victim slots."""
        return (seat + jnp.arange(1, self.num_players)) % self.num_players

    def robber_victims(self, s, seat):
        """[T, P - 1]: whether the robber on each tile could steal from each opponent."""
        opponents = self.opponents(seat)
        on_tile = (s.node_owner[self.tile_nodes][:, :, None] == opponents).any(axis=1)
        return on_tile & (s.hand[opponents].sum(axis=1) >= 1)

    # ===== Move generation
    def legal_mask(self, s):
        """models.actions.generate_playable_actions as a boolean action mask."""
        topo = self.topo
        seat = s.current
        hand = s.hand[seat]

        initial_settle = s.prompt == BUILD_INITIAL_SETTLEMENT
        initial_road = s.prompt == BUILD_INITIAL_ROAD
        play = s.prompt == PLAY_TURN
        discarding = s.prompt == DISCARD
        robbing = s.prompt == MOVE_ROBBER

        normal = play & ~s.road_building
        rolled = s.has_rolled[seat]
        main = normal & rolled

        playable = self.can_play_dev(s, seat)
        free_nodes = self._free_nodes(s.node_owner)
        buildable = self._buildable_edges(s.edge_owner, s.reach[seat])
        has_road_piece = s.roads_left[seat] > 0

        # Roads
        at_last = (self.edges == s.last_settlement[seat]).any(axis=1)
        afford_road = (hand >= ROAD_COST).all()
        roads = (
            (initial_road & buildable & at_last)
            | (play & s.road_building & has_road_piece & buildable)
            | (main & afford_road & has_road_piece & buildable)
        )

        # Settlements and cities
        afford_settlement = (hand >= SETTLEMENT_COST).all()
        settlements = (initial_settle & free_nodes) | (
            main
            & afford_settlement
            & (s.settlements_left[seat] > 0)
            & s.reach[seat]
            & free_nodes
        )
        afford_city = (hand >= CITY_COST).all()
        cities = (
            main
            & afford_city
            & (s.cities_left[seat] > 0)
            & (s.node_owner == seat)
            & (s.node_level == 1)
        )

        # Year of plenty
        pair_ok = (s.bank >= self.yop_cards[: len(YOP_PAIRS)]).all(axis=1)
        single_ok = (s.bank >= 1) & (self.yop_involves & ~pair_ok).any(axis=1)
        can_yop = normal & playable[YEAR_OF_PLENTY]

        # Robber: a tile offers one steal per opponent with a building there and
        # cards in hand, or else a move without stealing.
        can_steal = self.robber_victims(s, seat)
        may_move = robbing & (jnp.arange(topo.num_tiles) != s.robber_tile)

        # Maritime
        rates = self.trade_rates(s, seat)
        give, get = self.maritime_pairs[:, 0], self.maritime_pairs[:, 1]
        maritime = main & (hand[give] >= rates[give]) & (s.bank[get] > 0)

        mask = jnp.concatenate(
            [
                jnp.stack(
                    [
                        normal & ~rolled,  # ROLL
                        main,  # END_TURN
                        main & (hand >= DEV_COST).all() & (s.dev_drawn < len(DEV_DECK)),
                        normal & playable[KNIGHT],
                        normal
                        & playable[ROAD_BUILDING]
                        & has_road_piece
                        & buildable.any(),
                    ]
                ),
                discarding & (s.discard_counts[seat] > 0) & (hand > 0),
                roads,
                settlements,
                cities,
                can_yop & pair_ok,
                can_yop & single_ok,
                jnp.broadcast_to(normal & playable[MONOPOLY], (NUM_RESOURCES,)),
                jnp.concatenate(
                    [
                        (may_move & ~can_steal.any(axis=1))[:, None],
                        may_move[:, None] & can_steal,
                    ],
                    axis=1,
                ).ravel(),
                maritime,
            ]
        )
        return mask

    # ===== Dynamics
    def robber_victim(self, s, action):
        """Seat a robber action steals from; the next seat for any other action."""
        topo = self.topo
        slot = action - topo.ROBBER
        is_robber = (slot >= 0) & (slot < self.num_players * topo.num_tiles)
        offset = jnp.where(is_robber, slot % self.num_players, 1)
        return (s.current + jnp.maximum(offset, 1)) % self.num_players

    def sample_chance(self, key, s, action):
        k_a, k_b, k_steal = jax.random.split(key, 3)
        dice = jax.random.randint(k_a, (), 1, 7) + jax.random.randint(k_b, (), 1, 7)
        victim_hand = s.hand[self.robber_victim(s, action)]
        # log(0) = -inf is never drawn by categorical
        stolen = jax.random.categorical(
            k_steal, jnp.log(victim_hand.astype(jnp.float32))
        )
        return Chance(dice=dice, stolen=stolen)

    def _apply(self, s, action, chance):
        """Applies one legal action. Longest road is resolved later."""
        topo = self.topo
        players = self.num_players
        seat = s.current
        seats = jnp.arange(players)
        next_seat = (seat + 1) % players

        def in_range(start, count):
            return (action >= start) & (action < start + count)

        is_roll = action == topo.ROLL
        is_end = action == topo.END_TURN
        is_buy = action == topo.BUY_DEV
        is_knight = action == topo.PLAY_KNIGHT
        is_rb = action == topo.PLAY_ROAD_BUILDING
        is_discard = in_range(topo.DISCARD, NUM_RESOURCES)
        is_road = in_range(topo.ROAD, topo.num_edges)
        is_settlement = in_range(topo.SETTLEMENT, topo.num_nodes)
        is_city = in_range(topo.CITY, topo.num_nodes)
        is_yop = in_range(topo.YOP, self.yop_cards.shape[0])
        is_monopoly = in_range(topo.MONOPOLY, NUM_RESOURCES)
        is_robber = in_range(topo.ROBBER, players * topo.num_tiles)
        is_maritime = in_range(topo.MARITIME, len(MARITIME_PAIRS))

        resource = jnp.clip(
            action - jnp.where(is_discard, topo.DISCARD, topo.MONOPOLY), 0, 4
        )
        edge = jnp.clip(action - topo.ROAD, 0, topo.num_edges - 1)
        settlement_node = jnp.clip(action - topo.SETTLEMENT, 0, topo.num_nodes - 1)
        city_node = jnp.clip(action - topo.CITY, 0, topo.num_nodes - 1)
        yop_index = jnp.clip(action - topo.YOP, 0, self.yop_cards.shape[0] - 1)
        robber_index = jnp.clip(action - topo.ROBBER, 0, players * topo.num_tiles - 1)
        trade = self.maritime_pairs[jnp.clip(action - topo.MARITIME, 0, 19)]

        initial_settle = s.prompt == BUILD_INITIAL_SETTLEMENT
        initial_road = s.prompt == BUILD_INITIAL_ROAD
        is_initial = initial_settle | initial_road

        hand = s.hand
        bank = s.bank
        dev_hand = s.dev_hand
        dev_owned = s.dev_owned
        dev_played = s.dev_played
        vp_public = s.vp_public
        vp_actual = s.vp_actual
        current = s.current
        turn_player = s.turn_player
        prompt = s.prompt
        num_turns = s.num_turns

        # ----- Roll
        rolled_seven = is_roll & (chance.dice == 7)
        hand_sizes = hand.sum(axis=1)
        over_limit = hand_sizes > DISCARD_LIMIT
        seven_discards = jnp.where(over_limit, hand_sizes // 2, 0)
        producing = (s.tile_number == chance.dice) & (
            jnp.arange(topo.num_tiles) != s.robber_tile
        )
        node_levels = s.node_level * (s.node_owner == seats[:, None])  # [P, N]
        tile_levels = node_levels[:, self.tile_nodes].sum(axis=2)  # [P, T]
        by_resource = jax.nn.one_hot(
            s.tile_resource, NUM_RESOURCES + 1, dtype=jnp.int32
        )[
            :, :NUM_RESOURCES
        ]  # [T, 5]
        payout = (tile_levels * producing) @ by_resource  # [P, 5]
        # A resource the bank cannot fully pay out is paid to nobody, unless only
        # 1 player is owed it: they get what is left (yield_resources).
        payable = payout.sum(axis=0) <= bank
        sole_claimant = (payout > 0).sum(axis=0) == 1
        payout = jnp.where(payable, payout, jnp.minimum(payout, bank) * sole_claimant)
        payout = payout * (is_roll & ~rolled_seven)
        hand = hand + payout
        bank = bank - payout.sum(axis=0)
        has_rolled = s.has_rolled.at[seat].set(s.has_rolled[seat] | is_roll)

        discard_counts = jnp.where(rolled_seven, seven_discards, s.discard_counts)
        anyone_discards = rolled_seven & over_limit.any()
        current = jnp.where(anyone_discards, jnp.argmax(over_limit), current)
        prompt = jnp.where(anyone_discards, DISCARD, prompt)
        prompt = jnp.where(rolled_seven & ~anyone_discards, MOVE_ROBBER, prompt)

        # ----- Discard
        one_card = jax.nn.one_hot(resource, NUM_RESOURCES, dtype=jnp.int32)
        hand = hand.at[seat].add(-one_card * is_discard)
        bank = bank + one_card * is_discard
        discard_counts = discard_counts.at[seat].add(-is_discard.astype(jnp.int32))
        finished_own = is_discard & (discard_counts[seat] == 0)
        # Seats are asked in seat order (not from the roller), as in apply_discard.
        later = (seats > seat) & (discard_counts > 0)
        next_discarder = finished_own & later.any()
        all_discarded = finished_own & ~next_discarder
        current = jnp.where(next_discarder, jnp.argmax(later), current)
        current = jnp.where(all_discarded, turn_player, current)
        prompt = jnp.where(all_discarded, MOVE_ROBBER, prompt)
        discard_counts = jnp.where(
            all_discarded, jnp.zeros(players, jnp.int32), discard_counts
        )

        # ----- Robber
        robber_tile = jnp.where(is_robber, robber_index // players, s.robber_tile)
        steals = is_robber & (robber_index % players != 0)
        victim = (seat + robber_index % players) % players
        stolen = jax.nn.one_hot(chance.stolen, NUM_RESOURCES, dtype=jnp.int32) * steals
        hand = hand.at[victim].add(-stolen).at[seat].add(stolen)
        prompt = jnp.where(is_robber, PLAY_TURN, prompt)

        # ----- Development cards played
        plays_dev = (
            jax.nn.one_hot(KNIGHT, NUM_DEV_TYPES, dtype=jnp.int32) * is_knight
            + jax.nn.one_hot(ROAD_BUILDING, NUM_DEV_TYPES, dtype=jnp.int32) * is_rb
            + jax.nn.one_hot(YEAR_OF_PLENTY, NUM_DEV_TYPES, dtype=jnp.int32) * is_yop
            + jax.nn.one_hot(MONOPOLY, NUM_DEV_TYPES, dtype=jnp.int32) * is_monopoly
        )
        dev_hand = dev_hand.at[seat].add(-plays_dev)
        dev_played = dev_played.at[seat].add(plays_dev)
        played_dev = s.played_dev.at[seat].set(
            s.played_dev[seat] | (plays_dev.sum() > 0)
        )

        prompt = jnp.where(is_knight, MOVE_ROBBER, prompt)

        # Largest army
        knights = dev_played[seat, KNIGHT]
        holder = s.army_holder
        holder_knights = s.dev_played[jnp.clip(holder, 0, players - 1), KNIGHT]
        takes_army = (
            is_knight
            & (knights >= 3)
            & ((holder < 0) | ((holder != seat) & (holder_knights < knights)))
        )
        army_delta = 2 * takes_army * (seats == seat) - 2 * (
            takes_army & (holder >= 0)
        ) * (seats == holder)
        vp_public = vp_public + army_delta
        vp_actual = vp_actual + army_delta
        army_holder = jnp.where(takes_army, seat, holder)

        # Year of plenty
        yop_cards = self.yop_cards[yop_index] * is_yop
        hand = hand.at[seat].add(yop_cards)
        bank = bank - yop_cards

        # Monopoly takes from every opponent.
        taken = one_card * s.hand * ((seats != seat) & is_monopoly)[:, None]  # [P, 5]
        hand = (hand - taken).at[seat].add(taken.sum(axis=0))

        # ----- Buy development card
        drawn_card = s.dev_deck[jnp.clip(s.dev_drawn, 0, len(DEV_DECK) - 1)]
        dev_hand = dev_hand.at[seat, drawn_card].add(is_buy.astype(jnp.int32))
        dev_drawn = s.dev_drawn + is_buy
        hand = hand.at[seat].add(-DEV_COST * is_buy)
        bank = bank + DEV_COST * is_buy
        vp_actual = vp_actual.at[seat].add(is_buy & (drawn_card == VICTORY_POINT))

        # ----- Maritime trade
        rates = self.trade_rates(s, seat)
        give, get = trade[0], trade[1]
        offered = jax.nn.one_hot(give, NUM_RESOURCES, dtype=jnp.int32) * rates[give]
        asked = jax.nn.one_hot(get, NUM_RESOURCES, dtype=jnp.int32)
        hand = hand.at[seat].add((asked - offered) * is_maritime)
        bank = bank + (offered - asked) * is_maritime

        # ----- City
        node_owner = s.node_owner
        node_level = s.node_level.at[city_node].add(is_city.astype(jnp.int32))
        hand = hand.at[seat].add(-CITY_COST * is_city)
        bank = bank + CITY_COST * is_city
        settlements_left = s.settlements_left.at[seat].add(is_city.astype(jnp.int32))
        cities_left = s.cities_left.at[seat].add(-is_city.astype(jnp.int32))
        vp_public = vp_public.at[seat].add(is_city.astype(jnp.int32))
        vp_actual = vp_actual.at[seat].add(is_city.astype(jnp.int32))

        # ----- Settlement
        node_owner = node_owner.at[settlement_node].set(
            jnp.where(is_settlement, seat, node_owner[settlement_node])
        )
        node_level = node_level.at[settlement_node].add(is_settlement.astype(jnp.int32))
        settlements_left = settlements_left.at[seat].add(
            -is_settlement.astype(jnp.int32)
        )
        vp_public = vp_public.at[seat].add(is_settlement.astype(jnp.int32))
        vp_actual = vp_actual.at[seat].add(is_settlement.astype(jnp.int32))
        last_settlement = s.last_settlement.at[seat].set(
            jnp.where(is_settlement, settlement_node, s.last_settlement[seat])
        )
        paid_settlement = is_settlement & ~is_initial
        hand = hand.at[seat].add(-SETTLEMENT_COST * paid_settlement)
        bank = bank + SETTLEMENT_COST * paid_settlement
        prompt = jnp.where(is_settlement & is_initial, BUILD_INITIAL_ROAD, prompt)

        # The second initial settlement yields its adjacent tiles.
        second_house = is_settlement & is_initial & (settlements_left[seat] == 3)
        adjacent = self.node_tiles[settlement_node]
        adjacent_resources = jnp.where(adjacent >= 0, s.tile_resource[adjacent], DESERT)
        start_cards = (
            jax.nn.one_hot(adjacent_resources, NUM_RESOURCES + 1, dtype=jnp.int32).sum(
                axis=0
            )[:NUM_RESOURCES]
        ) * second_house
        hand = hand.at[seat].add(start_cards)
        bank = bank - start_cards

        # Building on a node may cut opponents' roads (Board.build_settlement).
        reach = s.reach
        reach = reach.at[seat, settlement_node].set(
            reach[seat, settlement_node] | (is_settlement & is_initial)
        )
        edge_owner_pad = jnp.concatenate([s.edge_owner, jnp.full(1, -1, jnp.int32)])
        incident = edge_owner_pad[self.node_edges_pad[settlement_node]]
        # [P] roads of each player at the node. The builder's own road is one of the
        # node's three, so at most one opponent can be cut through.
        incident_by_seat = (incident == seats[:, None]).sum(axis=1) * (seats != seat)
        cuts_end = paid_settlement & (incident_by_seat == 1)
        cut_through_by_seat = paid_settlement & (incident_by_seat == 2)
        cuts_through = cut_through_by_seat.any()
        cut_seat = jnp.where(cuts_through, jnp.argmax(cut_through_by_seat), next_seat)
        reach = reach.at[:, settlement_node].set(reach[:, settlement_node] & ~cuts_end)
        # Splitting a component re-walks both sides from the cut, as Board.dfs_walk
        # does: it follows the cut player's roads and records nodes with someone
        # else's building without walking past them. That records the new settlement
        # too, which keeps it in the cut player's components.
        cut_roads = s.edge_owner == cut_seat
        walk_stops = (node_owner >= 0) & (node_owner != cut_seat)
        road_ends = self.edges[:, 0], self.edges[:, 1]
        starts = (
            jnp.zeros(topo.num_nodes + 1, bool)
            .at[self.node_neighbors_pad[settlement_node]]
            .max(incident == cut_seat)[: topo.num_nodes]
        )

        def walk_step(_, visited):
            passable = visited & ~walk_stops
            a, b = road_ends
            return (
                visited.at[b]
                .max(cut_roads & passable[a])
                .at[a]
                .max(cut_roads & passable[b])
            )

        walked = jax.lax.fori_loop(0, MAX_ROADS, walk_step, starts)
        reach = reach.at[cut_seat].set(reach[cut_seat] | (walked & cuts_through))

        # ----- Road
        edge_a, edge_b = self.edges[edge, 0], self.edges[edge, 1]
        edge_owner = s.edge_owner.at[edge].set(
            jnp.where(is_road, seat, s.edge_owner[edge])
        )
        enemy_a = (node_owner[edge_a] >= 0) & (node_owner[edge_a] != seat)
        enemy_b = (node_owner[edge_b] >= 0) & (node_owner[edge_b] != seat)
        reach = reach.at[seat, edge_a].set(reach[seat, edge_a] | (is_road & ~enemy_a))
        reach = reach.at[seat, edge_b].set(reach[seat, edge_b] | (is_road & ~enemy_b))
        roads_left = s.roads_left.at[seat].add(-is_road.astype(jnp.int32))

        free_road = is_road & s.road_building & (s.free_roads > 0) & ~is_initial
        paid_road = is_road & ~is_initial & ~free_road
        hand = hand.at[seat].add(-ROAD_COST * paid_road)
        bank = bank + ROAD_COST * paid_road

        free_roads = jnp.where(is_rb, 2, s.free_roads - free_road)
        possible_after = (roads_left[seat] > 0) & self._buildable_edges(
            edge_owner, reach[seat]
        ).any()
        stops_building = free_road & ((free_roads == 0) | ~possible_after)
        road_building = (s.road_building | is_rb) & ~stops_building
        free_roads = jnp.where(stops_building, 0, free_roads)

        # Initial-placement snake order (apply_build_road)
        placed_road = is_road & initial_road
        total_settlements = 5 * players - settlements_left.sum()
        forward = placed_road & (total_settlements < players)
        turnaround = placed_road & (total_settlements == players)
        finished = placed_road & (total_settlements == 2 * players)
        backward = placed_road & ~forward & ~turnaround & ~finished
        moves_on = forward | backward
        snake_seat = jnp.where(forward, next_seat, (seat - 1) % players)
        current = jnp.where(moves_on, snake_seat, current)
        turn_player = jnp.where(moves_on, snake_seat, turn_player)
        num_turns = num_turns + moves_on
        prompt = jnp.where(
            placed_road,
            jnp.where(finished, PLAY_TURN, BUILD_INITIAL_SETTLEMENT),
            prompt,
        )

        # ----- End turn
        owned_now = (dev_hand[seat] > 0).at[VICTORY_POINT].set(False)
        dev_owned = dev_owned.at[seat].set(
            jnp.where(is_end, owned_now, dev_owned[seat])
        )
        played_dev = played_dev.at[seat].set(played_dev[seat] & ~is_end)
        has_rolled = has_rolled.at[seat].set(has_rolled[seat] & ~is_end)
        current = jnp.where(is_end, next_seat, current)
        turn_player = jnp.where(is_end, next_seat, turn_player)
        num_turns = num_turns + is_end
        prompt = jnp.where(is_end, PLAY_TURN, prompt)

        road_eval = jnp.where(
            is_road,
            ROAD_EVAL_BUILT,
            jnp.where(cuts_through, ROAD_EVAL_CUT, NO_ROAD_EVAL),
        )
        road_eval_seat = jnp.where(is_road, seat, cut_seat)

        return s._replace(
            node_owner=node_owner,
            node_level=node_level,
            edge_owner=edge_owner,
            robber_tile=robber_tile,
            reach=reach,
            hand=hand,
            bank=bank,
            dev_hand=dev_hand,
            dev_owned=dev_owned,
            dev_played=dev_played,
            dev_drawn=dev_drawn,
            vp_public=vp_public,
            vp_actual=vp_actual,
            roads_left=roads_left,
            settlements_left=settlements_left,
            cities_left=cities_left,
            last_settlement=last_settlement,
            army_holder=army_holder,
            current=current,
            turn_player=turn_player,
            prompt=prompt,
            has_rolled=has_rolled,
            played_dev=played_dev,
            road_building=road_building,
            free_roads=free_roads,
            discard_counts=discard_counts,
            num_turns=num_turns,
            num_plies=s.num_plies + 1,
            road_eval=road_eval,
            road_eval_seat=road_eval_seat,
        )

    # ===== Longest road
    def longest_road(self, edge_owner, node_owner, reach_seat, seat):
        """Length of the longest trail, as Board's longest_acyclic_path: walks only
        friendly roads, starts from any reach node, and ends at an enemy node (its
        road counts, but the trail can't pass through it)."""
        own = edge_owner == seat
        bits = jnp.where(own, 1 << (jnp.cumsum(own) - 1), 0)
        bits = jnp.concatenate([bits, jnp.zeros(1, jnp.int32)])
        blocked = (node_owner >= 0) & (node_owner != seat)
        blocked = jnp.concatenate([blocked, jnp.ones(1, bool)])

        node = jnp.nonzero(reach_seat, size=ROAD_FRONTIER, fill_value=0)[0]
        alive = reach_seat[node] & (jnp.arange(ROAD_FRONTIER) < reach_seat.sum())
        used = jnp.zeros(ROAD_FRONTIER, jnp.int32)

        def expand(frontier):
            node, used, alive, _, length = frontier
            step_bits = bits[self.node_edges_pad[node]]  # [F, 3]
            neighbors = self.node_neighbors_pad[node]  # [F, 3]
            # All trails in the frontier have the same length: 0 at their starts
            passable = alive & (~blocked[node] | (length == 0))
            ok = (
                passable[:, None]
                & (step_bits != 0)
                & ((used[:, None] & step_bits) == 0)
            )
            flat_ok = ok.ravel()
            index = jnp.nonzero(flat_ok, size=ROAD_FRONTIER, fill_value=0)[0]
            count = flat_ok.sum()
            new_alive = flat_ok[index] & (jnp.arange(ROAD_FRONTIER) < count)
            new_node = neighbors.ravel()[index]
            new_used = (used[:, None] | step_bits).ravel()[index]
            return new_node, new_used, new_alive, count, length + (count > 0)

        def not_finished(frontier):
            _, _, _, count, length = frontier
            return (count > 0) & (length < MAX_ROADS)

        start = (
            node,
            used,
            alive,
            jnp.maximum(alive.sum(), 1),
            jnp.zeros((), jnp.int32),
        )
        return jax.lax.while_loop(not_finished, expand, start)[4]

    def _apply_road_result(self, s, length):
        """Board.build_road / build_settlement bookkeeping plus maintain_longest_road."""
        seat = s.road_eval_seat
        built = s.road_eval == ROAD_EVAL_BUILT
        cut = s.road_eval == ROAD_EVAL_CUT
        previous = s.road_holder

        road_length = s.road_length.at[seat].set(
            jnp.where(built, jnp.maximum(s.road_length[seat], length), length)
        )
        takes = built & (length >= 5) & (length > s.road_best)
        holder = jnp.where(takes, seat, previous)
        best = jnp.where(takes, length, s.road_best)
        # After a cut (Board._reassign_longest_road_after_cut): the holder keeps the
        # card while tied for longest, else the only leader with 5+ takes it, else
        # it is set aside.
        last = self.num_players - 1
        longest = road_length.max()
        leaders = road_length == longest
        holder_leads = (previous >= 0) & leaders[jnp.clip(previous, 0, last)]
        after_cut = jnp.where(
            longest < 5,
            -1,
            jnp.where(
                holder_leads,
                previous,
                jnp.where(leaders.sum() == 1, jnp.argmax(road_length), -1),
            ),
        )
        holder = jnp.where(cut, after_cut, holder)
        best = jnp.where(cut, longest, best)

        changed = holder != previous
        delta = jnp.zeros(self.num_players, jnp.int32)
        delta = delta.at[jnp.clip(holder, 0, last)].add(2 * (changed & (holder >= 0)))
        delta = delta.at[jnp.clip(previous, 0, last)].add(
            -2 * (changed & (previous >= 0))
        )
        return s._replace(
            road_length=road_length,
            road_holder=holder,
            road_best=best,
            vp_public=s.vp_public + delta,
            vp_actual=s.vp_actual + delta,
            road_eval=jnp.full_like(s.road_eval, NO_ROAD_EVAL),
        )

    def _resolve_roads(self, states):
        """Runs pending longest-road searches on a batch, a few games at a time."""
        batch = states.road_eval.shape[0]
        count = min(ROAD_SEARCH_BATCH, batch)

        def search(s):
            return self.longest_road(
                s.edge_owner, s.node_owner, s.reach[s.road_eval_seat], s.road_eval_seat
            )

        def body(states):
            pending = states.road_eval != NO_ROAD_EVAL
            index = jnp.nonzero(pending, size=count, fill_value=batch)[0]
            valid = index < batch
            index = jnp.where(valid, index, 0)
            subset = jax.tree_util.tree_map(lambda x: x[index], states)
            lengths = jax.vmap(search)(subset)
            updated = jax.vmap(self._apply_road_result)(subset, lengths)
            # Invalid slots alias index 0: keep whatever is already there.
            write = jnp.where(valid, index, batch)

            def merge(old, new):
                return old.at[write].set(new, mode="drop")

            return jax.tree_util.tree_map(merge, states, updated)

        def has_pending(states):
            return (states.road_eval != NO_ROAD_EVAL).any()

        return jax.lax.while_loop(has_pending, body, states)

    # ===== Outcome
    def _finalize(self, s):
        # As Game.winning_color: only the player whose turn it is can win.
        reached = s.vp_actual[s.turn_player] >= self.vps_to_win
        winner = jnp.where(reached, s.turn_player, -1)
        done = (
            (winner >= 0)
            | (s.num_turns >= self.turns_limit)
            | (s.num_plies >= self.max_plies)
        )
        return s._replace(winner=winner, done=done)

    def potentials(self, s):
        """[P] potentials for optional potential-based reward shaping: each seat's
        VP lead over the average opponent. They sum to zero."""
        vp = s.vp_actual.astype(jnp.float32)
        others = (vp.sum() - vp) / (self.num_players - 1)
        return self.shaping_vp * (vp - others)

    def potential(self, s):
        """Seat 0's potential."""
        return self.potentials(s)[0]

    def terminal_rewards(self, winner):
        """[P] rewards for a result: +1 for the winner, and the losers share -1, so
        rewards sum to zero and equal players expect 0. A draw gives everyone 0."""
        seats = jnp.arange(self.num_players)
        loss = -1.0 / (self.num_players - 1)
        return jnp.where(winner < 0, 0.0, jnp.where(seats == winner, 1.0, loss))

    # ===== Batched API
    def apply_chance(self, states, actions, chances):
        """One ply on a batch with explicit chance outcomes. Does not auto-reset."""
        states = jax.vmap(self._apply)(states, actions, chances)
        states = self._resolve_roads(states)
        return jax.vmap(self._finalize)(states)

    def batch_reset(self, key, batch):
        return jax.vmap(self.reset)(jax.random.split(key, batch))

    def batch_step(self, key, states, actions):
        """One ply per game. Finished games restart immediately.

        Returns the next states and a StepInfo describing the plies just played.
        """
        batch = states.done.shape[0]
        k_chance, k_reset = jax.random.split(key)
        chances = jax.vmap(self.sample_chance)(
            jax.random.split(k_chance, batch), states, actions
        )
        before = jax.vmap(self.potentials)(states)
        played = self.apply_chance(states, actions, chances)
        after = jax.vmap(self.potentials)(played)

        terminal = jax.vmap(self.terminal_rewards)(played.winner)
        info = StepInfo(
            done=played.done,
            winner=played.winner,
            plies=played.num_plies,
            rewards=terminal + (after - before),
        )

        fresh = self.batch_reset(k_reset, batch)

        def pick(new, old):
            shape = (batch,) + (1,) * (old.ndim - 1)
            return jnp.where(played.done.reshape(shape), new, old)

        return jax.tree_util.tree_map(pick, fresh, played), info


def pack_states(states):
    """Narrow integer types so rollouts can store raw states instead of observations."""

    def narrow(path, x):
        name = path[-1].name
        if x.dtype == bool:
            return x
        return x.astype(jnp.int16 if name in ("num_turns", "num_plies") else jnp.int8)

    return jax.tree_util.tree_map_with_path(narrow, states)


def unpack_states(states):
    def widen(x):
        return x if x.dtype == bool else x.astype(jnp.int32)

    return jax.tree_util.tree_map(widen, states)
