"""Connects a catanatron Player to a JSettlers server as a third-party robot.

The server asks the bot to join games (BOTJOINGAMEREQUEST). For each game a
GameController keeps a GameView up to date and, whenever the server is waiting
on us, asks the Player to decide on a snapshot and sends the matching request.
The Player never trades with other players: offers to it are rejected.
"""

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from catanatron.models.enums import RESOURCES, Action, ActionPrompt, ActionType
from catanatron.sota.jsettlers import protocol as p
from catanatron.sota.jsettlers.board import ID_BY_RESOURCE, RESOURCE_BY_ID
from catanatron.sota.jsettlers.view import (
    ID_BY_DEV,
    OVER,
    PIECE_CITY,
    PIECE_ROAD,
    PIECE_SETTLEMENT,
    PLACING_FREE_ROAD1,
    PLACING_FREE_ROAD2,
    PLACING_ROBBER,
    PLAY1,
    ROLL_OR_CARD,
    START1A,
    START1B,
    START2A,
    START2B,
    WAITING_FOR_DISCARDS,
    WAITING_FOR_DISCOVERY,
    WAITING_FOR_MONOPOLY,
    DC_CANNOT_PLAY,
    GameView,
)
from catanatron.models.enums import (
    KNIGHT,
    MONOPOLY,
    ROAD_BUILDING,
    VICTORY_POINT,
    YEAR_OF_PLENTY,
)

log = logging.getLogger(__name__)

# The server allows any number of bank trades and builds per turn; stop runaway loops.
MAX_ACTIONS_PER_TURN = 60

DECLINE_REASON_LOCATION = 4  # SOCDeclinePlayerRequest.REASON_LOCATION

RBCLASS = "catanatron.sota.jsettlers.Bridge"


@dataclass
class GameResult:
    game: str
    our_pn: int
    scores: list[int]
    names: list[str]
    turns: int
    winner_pn: int | None
    decisions: int
    decide_seconds: float
    rejected_requests: int = 0
    tracking_checks: int = 0
    tracking_errors: list[str] = field(default_factory=list)

    @property
    def won(self):
        return self.winner_pn == self.our_pn


class GameController:
    def __init__(self, client, game_name, our_pn, player):
        self.client = client
        self.name = game_name
        self.player = player
        self.view = GameView(game_name, our_pn, player.color)
        self.awaiting = False
        self.last_action = None
        self.banned = set()
        self.banned_types = set()
        self.pending_discard = 0
        self.choose_options = None
        self.planned_victim = None
        self.planned_pick = None
        self.planned_monopoly = None
        self.actions_this_turn = 0
        self.decisions = 0
        self.decide_seconds = 0.0
        self.rejected = 0
        self.tracking_checks = 0
        self.tracking_errors = []
        self.result = None

    @property
    def our_pn(self):
        return self.view.our_pn

    def send(self, msg_type, *fields):
        self.client.send(p.encode(msg_type, self.name, *fields))

    # --- Incoming -------------------------------------------------------

    def handle(self, msg: p.Message):
        before = self.view.state
        self.view.apply(msg)
        t = msg.type

        if t == p.MAKEOFFER:
            offered_to = msg.fields[2:6]
            if self.our_pn < len(offered_to) and offered_to[self.our_pn] == "true":
                self.send(p.REJECTOFFER, self.our_pn)
        elif t == p.DISCARDREQUEST:
            self.pending_discard = int(msg.fields[1])
        elif t == p.CHOOSEPLAYERREQUEST:
            flags = [f for f in msg.fields[1:] if f != "NONE"]
            self.choose_options = [pn for pn, f in enumerate(flags) if f == "true"]
        elif t in (p.GAMESTATE, p.TURN, p.STARTGAME):
            self.awaiting = False
            if t == p.TURN:
                self.actions_this_turn = 0
            if t == p.TURN or self.view.state != before:
                self.banned.clear()
                self.banned_types.clear()
        elif t == p.BANKTRADE and int(msg.fields[-1]) == self.our_pn:
            self.awaiting = False
        elif t == p.DECLINEPLAYERREQUEST:
            location_only = int(msg.fields[2]) == DECLINE_REASON_LOCATION
            self._declined(msg, whole_type=not location_only)
        elif t == p.DEVCARDACTION and int(msg.fields[2]) == DC_CANNOT_PLAY:
            self._declined(msg, whole_type=True)
        elif (
            t == p.REJECTOFFER
            and self.awaiting
            and self.last_action is not None
            and self.last_action.action_type == ActionType.MARITIME_TRADE
        ):
            self._declined(msg, whole_type=False)  # how the server refuses a bank trade
        elif t == p.BOTGAMEDATACHECK:
            self._check_tracking(msg.ints())
        elif t == p.GAMESTATS:
            self._finish()

        if self.result is None:
            self.maybe_act()

    def _check_tracking(self, values):
        """Compares our view of everyone's hands with the server's true amounts.

        Only sent when the server runs with jsettlers.debug.bots.datacheck.rsrc=Y.
        Used to test the bridge; never used for decisions.
        """
        if values[0] != 1:  # TYPE_RESOURCE_AMOUNTS
            return
        self.tracking_checks += 1
        for i in range(1, len(values) - 5, 6):
            pn, amounts = values[i], values[i + 1 : i + 6]
            actual = {
                RESOURCE_BY_ID[rtype]: n for rtype, n in enumerate(amounts, start=1)
            }
            hand = self.view.players[pn].hand
            if pn == self.our_pn:
                ok = hand.known == actual and hand.unknown == 0
            else:
                ok = hand.total() == sum(amounts) and all(
                    hand.known[r] <= actual[r] for r in RESOURCES
                )
            if not ok:
                self.tracking_errors.append(
                    f"turn {self.view.num_turns} p{pn}: server {actual}, "
                    f"tracked {hand.known} + {hand.unknown} unknown"
                )

    def _declined(self, msg, whole_type):
        """The server refused our last request: don't offer it to the player again this turn.

        Unless only the location was wrong, the whole action type is likely
        impossible right now (e.g. our tracked hand is off), so drop all of it.
        """
        self.rejected += 1
        log.warning(
            "%s: server declined %s (%s)", self.name, self.last_action, msg.fields
        )
        if self.last_action is not None:
            if whole_type:
                self.banned_types.add(self.last_action.action_type)
            else:
                self.banned.add(self.last_action)
        self.awaiting = False

    def _finish(self):
        v = self.view
        scores = v.final_scores or []
        winner = None
        if scores:
            best = max(scores)
            if best >= 10:
                winner = scores.index(best)
        if v.state == OVER and v.current >= 0 and scores and scores[v.current] >= 10:
            winner = v.current
        # Our tracked buildings, longest road and largest army, plus VP cards
        # (known when drawn, or revealed at the end), should add up to the
        # server's final scores.
        for pn in v.seated_pns():
            pl = v.players[pn]
            known_vp = (pl.dev_old + pl.dev_new)[VICTORY_POINT]
            tracked = v.public_vps(pn) + max(v.revealed_vp[pn], known_vp)
            if pn < len(scores) and tracked != scores[pn]:
                self.tracking_errors.append(
                    f"final VP p{pn}: server {scores[pn]}, tracked {tracked}"
                )
        self.result = GameResult(
            game=self.name,
            our_pn=self.our_pn,
            scores=scores,
            names=[pl.name for pl in v.players],
            turns=v.num_turns,
            winner_pn=winner,
            decisions=self.decisions,
            decide_seconds=self.decide_seconds,
            rejected_requests=self.rejected,
            tracking_checks=self.tracking_checks,
            tracking_errors=self.tracking_errors,
        )
        self.client.send(p.encode(p.LEAVEGAME, self.client.nickname, "-", self.name))
        self.client.game_finished(self)

    # --- Deciding -------------------------------------------------------

    def maybe_act(self):
        v = self.view
        if self.awaiting or v.mapping is None or v.state == OVER:
            return
        if self.pending_discard and v.state == WAITING_FOR_DISCARDS:
            return self._discard()
        if self.choose_options is not None:
            return self._choose_victim()
        if v.current != self.our_pn:
            return

        s = v.state
        if s in (START1A, START2A):
            self._decide_and_send(ActionPrompt.BUILD_INITIAL_SETTLEMENT)
        elif s in (START1B, START2B):
            self._decide_and_send(ActionPrompt.BUILD_INITIAL_ROAD)
        elif s in (ROLL_OR_CARD, PLAY1, PLACING_FREE_ROAD1, PLACING_FREE_ROAD2):
            self._decide_and_send(ActionPrompt.PLAY_TURN)
        elif s == PLACING_ROBBER:
            self._decide_and_send(ActionPrompt.MOVE_ROBBER)
        elif s == WAITING_FOR_DISCOVERY:
            self._pick_resources()
        elif s == WAITING_FOR_MONOPOLY:
            self._pick_monopoly()

    def _decide(self, prompt, discard_count=0):
        game = self.view.snapshot(self.player, prompt, discard_count)
        actions = [
            a
            for a in game.playable_actions
            if a not in self.banned and a.action_type not in self.banned_types
        ]
        if (
            prompt == ActionPrompt.PLAY_TURN
            and self.actions_this_turn >= MAX_ACTIONS_PER_TURN
        ):
            actions = [
                a
                for a in actions
                if a.action_type in (ActionType.END_TURN, ActionType.ROLL)
            ]
        if not actions:
            return self._fallback(game)
        game.playable_actions = actions
        start = time.perf_counter()
        action = self.player.decide(game, actions)
        self.decide_seconds += time.perf_counter() - start
        self.decisions += 1
        if action not in actions:
            log.warning("%s: %s returned unplayable %s", self.name, self.player, action)
            action = actions[0]
        return action

    def _fallback(self, game):
        # Everything catanatron offered was declined by the server (rules mismatch).
        color = self.player.color
        if self.view.state == ROLL_OR_CARD:
            return Action(color, ActionType.ROLL, None)
        if self.view.state in (PLAY1, PLACING_FREE_ROAD1, PLACING_FREE_ROAD2):
            return Action(color, ActionType.END_TURN, None)
        options = game.playable_actions
        if not options:
            raise RuntimeError(
                f"{self.name}: no action possible in state {self.view.state}"
            )
        return options[0]

    def _decide_and_send(self, prompt):
        action = self._decide(prompt)
        self.last_action = action
        self.actions_this_turn += 1
        self._send_action(action)

    def _send_action(self, action):
        at, value = action.action_type, action.value
        mapping = self.view.mapping
        pn = self.our_pn
        if at == ActionType.ROLL:
            self.send(p.ROLLDICE)
        elif at == ActionType.END_TURN:
            self.send(p.ENDTURN)
        elif at == ActionType.BUILD_SETTLEMENT:
            self.send(p.PUTPIECE, pn, PIECE_SETTLEMENT, mapping.id_to_node[value])
        elif at == ActionType.BUILD_CITY:
            self.send(p.PUTPIECE, pn, PIECE_CITY, mapping.id_to_node[value])
        elif at == ActionType.BUILD_ROAD:
            self.send(p.PUTPIECE, pn, PIECE_ROAD, mapping.js_edge(value))
        elif at == ActionType.BUY_DEVELOPMENT_CARD:
            self.send(p.BUYDEVCARDREQUEST)
        elif at == ActionType.PLAY_KNIGHT_CARD:
            self.send(p.PLAYDEVCARDREQUEST, ID_BY_DEV[KNIGHT])
        elif at == ActionType.PLAY_ROAD_BUILDING:
            self.send(p.PLAYDEVCARDREQUEST, ID_BY_DEV[ROAD_BUILDING])
        elif at == ActionType.PLAY_YEAR_OF_PLENTY:
            self.planned_pick = list(value)
            self.send(p.PLAYDEVCARDREQUEST, ID_BY_DEV[YEAR_OF_PLENTY])
        elif at == ActionType.PLAY_MONOPOLY:
            self.planned_monopoly = value
            self.send(p.PLAYDEVCARDREQUEST, ID_BY_DEV[MONOPOLY])
        elif at == ActionType.MARITIME_TRADE:
            give = Counter(r for r in value[:4] if r is not None)
            get = Counter([value[4]])
            self.send(p.BANKTRADE, *_resource_counts(give), *_resource_counts(get))
        elif at == ActionType.MOVE_ROBBER:
            coordinate, victim_color = value
            self.planned_victim = next(
                (
                    i
                    for i, pl in enumerate(self.view.players)
                    if pl.color == victim_color
                ),
                None,
            )
            self.send(p.MOVEROBBER, pn, mapping.coordinate_to_hex[coordinate])
        else:
            raise ValueError(f"Can't send {action} to JSettlers")
        self.awaiting = True

    def _discard(self):
        n = self.pending_discard
        hand = self.view.us.hand
        chosen = Counter()
        try:
            for k in range(n):
                action = self._decide(ActionPrompt.DISCARD, discard_count=n - k)
                chosen[action.value] += 1
                hand.known[action.value] -= 1
        finally:
            for resource, count in chosen.items():
                hand.known[resource] += count
        self.pending_discard = 0
        self.last_action = None
        self.send(p.DISCARD, *_resource_counts(chosen), 0)
        self.awaiting = True

    def _choose_victim(self):
        options, self.choose_options = self.choose_options, None
        choice = self.planned_victim if self.planned_victim in options else options[0]
        self.send(p.CHOOSEPLAYER, choice)
        self.awaiting = True

    def _pick_resources(self):
        picks = self.planned_pick or []
        self.planned_pick = None
        if not picks:  # shouldn't happen: we only play the card with a choice made
            hand = self.view.us.hand.known
            picks = [min(RESOURCES, key=lambda r: hand[r])]
        while len(picks) < 2:
            picks.append(picks[0])
        self.send(p.PICKRESOURCES, *_resource_counts(Counter(picks)))
        self.awaiting = True

    def _pick_monopoly(self):
        resource = self.planned_monopoly or RESOURCES[0]
        self.planned_monopoly = None
        self.send(p.PICKRESOURCETYPE, ID_BY_RESOURCE[resource])
        self.awaiting = True


def _resource_counts(counter):
    """Amounts in JSettlers order: clay, ore, sheep, wheat, wood."""
    by_id = {ID_BY_RESOURCE[r]: n for r, n in counter.items()}
    return [by_id.get(i, 0) for i in range(1, 6)]


class BridgeClient:
    """A third-party JSettlers robot that plays catanatron Players.

    Args:
        player_factory: called with no args for each new game; returns the
            catanatron Player that will play that game.
        on_result: called with a GameResult as each game ends with a winner.
            Games that end without one (the robots failed) are not counted.
        max_games: disconnect after this many games (None: until the server closes).
    """

    POLL_SECONDS = 2.0  # how often run() checks should_stop and abandon requests

    def __init__(
        self,
        host,
        port,
        cookie,
        player_factory,
        nickname="catanatron",
        on_result=None,
        max_games=None,
        traffic_log=None,
    ):
        self.host, self.port, self.cookie = host, port, cookie
        self.player_factory = player_factory
        self.nickname = nickname
        self.on_result = on_result
        self.max_games = max_games
        self.traffic_log = traffic_log
        self.games: dict[str, GameController] = {}
        self.seat_requests: dict[str, int] = {}
        self.results: list[GameResult] = []
        self.unfinished: list[GameResult] = []  # games that ended without a winner
        self._to_abandon = deque()
        self.conn = None

    def send(self, text):
        if self.traffic_log:
            self.traffic_log.write(f"OUT {text!r}\n")
        self.conn.send(text)

    def connect(self):
        """Logs in as a robot; returns once the server has accepted us."""
        self.conn = p.Connection(self.host, self.port, timeout=30)
        self.send(
            p.encode(
                p.VERSION,
                p.VERSION_NUMBER,
                p.VERSION_STRING,
                p.VERSION_BUILD,
                p.EMPTYSTR,
            )
        )
        self.send(p.encode(p.IMAROBOT, self.nickname, self.cookie, RBCLASS))
        while True:
            msg = self._recv()
            if msg.type == p.UPDATEROBOTPARAMS:  # sent once the robot is accepted
                return
            self.dispatch(msg)

    def run(self, should_stop=lambda: False):
        """Plays games until max_games finish, the server disconnects, or should_stop()."""
        try:
            while self.max_games is None or len(self.results) < self.max_games:
                if should_stop():
                    break
                while self._to_abandon:
                    self._leave(self._to_abandon.popleft())
                try:
                    msg = self._recv(timeout=self.POLL_SECONDS)
                except ConnectionError:
                    break
                if msg is not None:
                    self.dispatch(msg)
        finally:
            self.conn.close()
        return self.results

    def abandon(self, game):
        """Leaves a game without a result. Safe to call from another thread."""
        self._to_abandon.append(game)

    def _leave(self, game):
        if self.games.pop(game, None) is not None:
            self.send(p.encode(p.LEAVEGAME, self.nickname, "-", game))

    def _recv(self, timeout=None):
        text = self.conn.recv(timeout)
        if text is None:
            return None
        if self.traffic_log:
            self.traffic_log.write(f"IN  {text!r}\n")
        return p.parse(text)

    def dispatch(self, msg: p.Message):
        t = msg.type
        if t == p.SERVERPING:
            self.send(p.encode(p.SERVERPING, *msg.fields))
        elif t == p.REJECTCONNECTION:
            raise ConnectionError(f"JSettlers server rejected the bridge: {msg.fields}")
        elif t == p.STATUSMESSAGE:
            log.info("server status: %s", msg.fields)
        elif t == p.BOTJOINGAMEREQUEST:
            game, pn = msg.fields[0], int(msg.fields[1])
            self.seat_requests[game] = pn
            self.games[game] = GameController(self, game, pn, self.player_factory())
            self.send(p.encode(p.JOINGAME, self.nickname, p.EMPTYSTR, p.EMPTYSTR, game))
        elif t == p.GAMEMEMBERS:
            pn = self.seat_requests.pop(msg.game, None)
            if pn is not None:
                self.send(p.encode(p.SITDOWN, msg.game, p.EMPTYSTR, pn, "true"))
        elif t in (p.ROBOTDISMISS, p.DELETEGAME):
            if msg.game in self.games and t == p.ROBOTDISMISS:
                self.send(p.encode(p.LEAVEGAME, self.nickname, "-", msg.game))
            self.games.pop(msg.game, None)
        else:
            controller = self.games.get(msg.game)
            if controller is not None:
                try:
                    controller.handle(msg)
                except Exception:
                    log.exception("%s: error handling %s", msg.game, msg)
                    raise

    def game_finished(self, controller: GameController):
        self.games.pop(controller.name, None)
        if controller.result.winner_pn is None:
            log.warning(
                "%s ended without a winner %s; not counted",
                controller.name,
                controller.result.scores,
            )
            self.unfinished.append(controller.result)
            return
        self.results.append(controller.result)
        if self.on_result:
            self.on_result(controller.result)
