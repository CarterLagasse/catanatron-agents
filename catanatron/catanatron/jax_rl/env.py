"""Pure-JAX implementation of 1v1 Catan: jittable, vmappable, no Python state.

Rules follow the Python engine (`catanatron.apply_action`, `models.actions`,
`models.board`) move for move, including its quirks; tests/jax_rl/test_parity.py
replays real games through both and compares them after every action. Domestic
trades are not modelled, matching the gym environment, whose action space has none.

Seats are abstract: seat 0 moves first. Mapping agents to seats is up to callers.

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
    reach: jnp.ndarray  # [2, N] union of Board.connected_components per seat
    # Cards
    hand: jnp.ndarray  # [2, 5]
    bank: jnp.ndarray  # [5]
    dev_hand: jnp.ndarray  # [2, 5]
    dev_owned: jnp.ndarray  # [2, 5] *_OWNED_AT_START
    dev_played: jnp.ndarray  # [2, 5]
    dev_deck: jnp.ndarray  # [25] shuffled
    dev_drawn: jnp.ndarray
    # Scores and pieces
    vp_public: jnp.ndarray  # [2]
    vp_actual: jnp.ndarray  # [2]
    roads_left: jnp.ndarray  # [2]
    settlements_left: jnp.ndarray  # [2]
    cities_left: jnp.ndarray  # [2]
    last_settlement: jnp.ndarray  # [2]
    road_length: jnp.ndarray  # [2] Board.road_lengths
    road_holder: jnp.ndarray  # -1 or seat
    road_best: jnp.ndarray  # Board.road_length
    army_holder: jnp.ndarray  # -1 or seat
    # Turn structure
    current: jnp.ndarray  # seat that must decide now
    turn_player: jnp.ndarray
    prompt: jnp.ndarray
    has_rolled: jnp.ndarray  # [2]
    played_dev: jnp.ndarray  # [2]
    road_building: jnp.ndarray
    free_roads: jnp.ndarray
    discard_counts: jnp.ndarray  # [2]
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
    stolen: jnp.ndarray  # resource taken if this action steals


class StepInfo(NamedTuple):
    done: jnp.ndarray
    winner: jnp.ndarray
    plies: jnp.ndarray
    rewards: jnp.ndarray  # [2], per seat, for the action just taken


class CatanEnv:
    """Holds static tables. All methods are pure functions of their arguments."""

    def __init__(
        self,
        map_type="BASE",
        vps_to_win=10,
        turns_limit=1000,
        max_plies=6000,
        shaping_vp=0.0,
    ):
        self.topo = topo = get_topology(map_type)
        self.map_type = map_type
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

        zeros2 = jnp.zeros(2, jnp.int32)
        scalar0 = jnp.zeros((), jnp.int32)
        return State(
            tile_resource=tile_resource,
            tile_number=tile_number,
            port_resource=port_resource,
            node_owner=jnp.full(topo.num_nodes, -1, jnp.int32),
            node_level=jnp.zeros(topo.num_nodes, jnp.int32),
            edge_owner=jnp.full(topo.num_edges, -1, jnp.int32),
            robber_tile=jnp.argmax(tile_resource == DESERT).astype(jnp.int32),
            reach=jnp.zeros((2, topo.num_nodes), bool),
            hand=jnp.zeros((2, NUM_RESOURCES), jnp.int32),
            bank=jnp.full(NUM_RESOURCES, 19, jnp.int32),
            dev_hand=jnp.zeros((2, NUM_DEV_TYPES), jnp.int32),
            dev_owned=jnp.zeros((2, NUM_DEV_TYPES), bool),
            dev_played=jnp.zeros((2, NUM_DEV_TYPES), jnp.int32),
            dev_deck=jax.random.permutation(k_deck, jnp.asarray(DEV_DECK)),
            dev_drawn=scalar0,
            vp_public=zeros2,
            vp_actual=zeros2,
            roads_left=jnp.full(2, MAX_ROADS, jnp.int32),
            settlements_left=jnp.full(2, 5, jnp.int32),
            cities_left=jnp.full(2, 4, jnp.int32),
            last_settlement=jnp.full(2, -1, jnp.int32),
            road_length=zeros2,
            road_holder=jnp.full((), -1, jnp.int32),
            road_best=scalar0,
            army_holder=jnp.full((), -1, jnp.int32),
            current=scalar0,
            turn_player=scalar0,
            prompt=jnp.full((), BUILD_INITIAL_SETTLEMENT, jnp.int32),
            has_rolled=jnp.zeros(2, bool),
            played_dev=jnp.zeros(2, bool),
            road_building=jnp.zeros((), bool),
            free_roads=scalar0,
            discard_counts=zeros2,
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

    # ===== Move generation
    def legal_mask(self, s):
        """models.actions.generate_playable_actions as a boolean action mask."""
        topo = self.topo
        seat = s.current
        other = 1 - seat
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

        # Robber: a tile offers steal if an opponent building with cards is there.
        enemy_on_tile = (s.node_owner[self.tile_nodes] == other).any(axis=1)
        can_steal = enemy_on_tile & (s.hand[other].sum() >= 1)
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
                jnp.stack(
                    [may_move & ~can_steal, may_move & can_steal], axis=1
                ).ravel(),
                maritime,
            ]
        )
        return mask

    # ===== Dynamics
    def sample_chance(self, key, s):
        k_a, k_b, k_steal = jax.random.split(key, 3)
        dice = jax.random.randint(k_a, (), 1, 7) + jax.random.randint(k_b, (), 1, 7)
        victim_hand = s.hand[1 - s.current]
        # log(0) = -inf is never drawn by categorical
        stolen = jax.random.categorical(
            k_steal, jnp.log(victim_hand.astype(jnp.float32))
        )
        return Chance(dice=dice, stolen=stolen)

    def _apply(self, s, action, chance):
        """Applies one legal action. Longest road is resolved later."""
        topo = self.topo
        seat = s.current
        other = 1 - seat

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
        is_robber = in_range(topo.ROBBER, 2 * topo.num_tiles)
        is_maritime = in_range(topo.MARITIME, len(MARITIME_PAIRS))

        resource = jnp.clip(
            action - jnp.where(is_discard, topo.DISCARD, topo.MONOPOLY), 0, 4
        )
        edge = jnp.clip(action - topo.ROAD, 0, topo.num_edges - 1)
        settlement_node = jnp.clip(action - topo.SETTLEMENT, 0, topo.num_nodes - 1)
        city_node = jnp.clip(action - topo.CITY, 0, topo.num_nodes - 1)
        yop_index = jnp.clip(action - topo.YOP, 0, self.yop_cards.shape[0] - 1)
        robber_index = jnp.clip(action - topo.ROBBER, 0, 2 * topo.num_tiles - 1)
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
        tile_levels = jnp.stack(
            [
                (s.node_level * (s.node_owner == 0))[self.tile_nodes].sum(axis=1),
                (s.node_level * (s.node_owner == 1))[self.tile_nodes].sum(axis=1),
            ]
        )  # [2, T]
        by_resource = jax.nn.one_hot(
            s.tile_resource, NUM_RESOURCES + 1, dtype=jnp.int32
        )[
            :, :NUM_RESOURCES
        ]  # [T, 5]
        payout = (tile_levels * producing) @ by_resource  # [2, 5]
        # A resource the bank cannot fully pay out is paid to nobody.
        payable = payout.sum(axis=0) <= bank
        payout = payout * payable * (is_roll & ~rolled_seven)
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
        # Seats are asked in order, so only seat 0 can hand over to another seat.
        next_discarder = finished_own & (seat == 0) & (discard_counts[1] > 0)
        all_discarded = finished_own & ~next_discarder
        current = jnp.where(next_discarder, 1, current)
        current = jnp.where(all_discarded, turn_player, current)
        prompt = jnp.where(all_discarded, MOVE_ROBBER, prompt)
        discard_counts = jnp.where(
            all_discarded, jnp.zeros(2, jnp.int32), discard_counts
        )

        # ----- Robber
        robber_tile = jnp.where(is_robber, robber_index // 2, s.robber_tile)
        steals = is_robber & (robber_index % 2 == 1)
        stolen = jax.nn.one_hot(chance.stolen, NUM_RESOURCES, dtype=jnp.int32) * steals
        hand = hand.at[other].add(-stolen).at[seat].add(stolen)
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
        takes_army = (
            is_knight
            & (knights >= 3)
            & (
                (holder < 0)
                | ((holder != seat) & (s.dev_played[other, KNIGHT] < knights))
            )
        )
        army_delta = jnp.zeros(2, jnp.int32).at[seat].set(2 * takes_army)
        army_delta = army_delta.at[other].set(-2 * (takes_army & (holder == other)))
        vp_public = vp_public + army_delta
        vp_actual = vp_actual + army_delta
        army_holder = jnp.where(takes_army, seat, holder)

        # Year of plenty
        yop_cards = self.yop_cards[yop_index] * is_yop
        hand = hand.at[seat].add(yop_cards)
        bank = bank - yop_cards

        # Monopoly
        taken = one_card * s.hand[other] * is_monopoly
        hand = hand.at[other].add(-taken).at[seat].add(taken)

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

        # Building on a node may cut the opponent's road (Board.build_settlement).
        reach = s.reach
        reach = reach.at[seat, settlement_node].set(
            reach[seat, settlement_node] | (is_settlement & is_initial)
        )
        edge_owner_pad = jnp.concatenate([s.edge_owner, jnp.full(1, -1, jnp.int32)])
        incident = edge_owner_pad[self.node_edges_pad[settlement_node]]
        enemy_incident = (incident == other).sum()
        cuts_end = paid_settlement & (enemy_incident == 1)
        cuts_through = paid_settlement & (enemy_incident == 2)
        reach = reach.at[other, settlement_node].set(
            reach[other, settlement_node] & ~cuts_end
        )
        # Splitting a component re-walks it. As in Board.dfs_walk the walk records
        # the cutting node too, which keeps it in the cut player's component.
        enemy_roads = s.edge_owner == other
        walked = (
            jnp.zeros(topo.num_nodes, bool)
            .at[self.edges[:, 0]]
            .max(enemy_roads)
            .at[self.edges[:, 1]]
            .max(enemy_roads)
        )
        reach = reach.at[other].set(reach[other] | (walked & cuts_through))

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
        total_settlements = 10 - settlements_left.sum()
        forward = placed_road & (total_settlements < 2)
        turnaround = placed_road & (total_settlements == 2)
        finished = placed_road & (total_settlements == 4)
        backward = placed_road & ~forward & ~turnaround & ~finished
        moves_on = forward | backward
        current = jnp.where(moves_on, other, current)
        turn_player = jnp.where(moves_on, other, turn_player)
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
        current = jnp.where(is_end, other, current)
        turn_player = jnp.where(is_end, other, turn_player)
        num_turns = num_turns + is_end
        prompt = jnp.where(is_end, PLAY_TURN, prompt)

        road_eval = jnp.where(
            is_road,
            ROAD_EVAL_BUILT,
            jnp.where(cuts_through, ROAD_EVAL_CUT, NO_ROAD_EVAL),
        )
        road_eval_seat = jnp.where(is_road, seat, other)

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
        friendly roads, cannot enter an enemy node, starts from any reach node."""
        topo = self.topo
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
            ok = (
                alive[:, None]
                & (step_bits != 0)
                & ((used[:, None] & step_bits) == 0)
                & ~blocked[neighbors]
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
        # A cut re-awards the title to the longest network with no 5-road minimum.
        # On a tie the seat that built first wins, which is always seat 0.
        holder = jnp.where(cut, jnp.argmax(road_length), holder)
        best = jnp.where(cut, road_length.max(), best)

        changed = (holder >= 0) & (holder != previous)
        delta = jnp.zeros(2, jnp.int32)
        delta = delta.at[jnp.clip(holder, 0, 1)].add(2 * changed)
        delta = delta.at[jnp.clip(previous, 0, 1)].add(-2 * (changed & (previous >= 0)))
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
        # Game.winning_color scans seats in order, so the later seat wins a tie.
        winner = jnp.where(
            s.vp_actual[1] >= self.vps_to_win,
            1,
            jnp.where(s.vp_actual[0] >= self.vps_to_win, 0, -1),
        )
        done = (
            (winner >= 0)
            | (s.num_turns >= self.turns_limit)
            | (s.num_plies >= self.max_plies)
        )
        return s._replace(winner=winner, done=done)

    def potential(self, s):
        """Seat-0 potential for optional potential-based reward shaping."""
        return self.shaping_vp * (s.vp_actual[0] - s.vp_actual[1]).astype(jnp.float32)

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
            jax.random.split(k_chance, batch), states
        )
        before = jax.vmap(self.potential)(states)
        played = self.apply_chance(states, actions, chances)
        after = jax.vmap(self.potential)(played)

        terminal = jnp.where(
            played.winner == 0, 1.0, jnp.where(played.winner == 1, -1.0, 0.0)
        )
        seat0_reward = terminal + (after - before)
        info = StepInfo(
            done=played.done,
            winner=played.winner,
            plies=played.num_plies,
            rewards=jnp.stack([seat0_reward, -seat0_reward], axis=1),
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
