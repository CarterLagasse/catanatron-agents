"""Starts robot-only games on a JSettlers server, a few at a time.

Connects as the server's "debug" user (needs jsettlers.allow.debug=Y), creates
each game and starts it with the *STARTBOTGAME* debug command, which fills every
seat with robots (one of them the bridge, per jsettlers.bots.percent3p). The
starter stays in each game as an observer and leaves when it's over.

A game that stalls (no messages for stall_timeout seconds), is deleted, or ends
without a winner is abandoned and replaced by a new one. The JSettlers robots
occasionally leave a game or never join one, and such a game never finishes.
"""

import logging
import threading
import time

from catanatron.sota.jsettlers import protocol as p

log = logging.getLogger(__name__)

NEWGAMEWITHOPTIONSREQUEST = 1078
GAMETEXTMSG = 1010
TEXT_SEP = "\0"  # SOCGameTextMsg separates its fields with NUL
WINNING_VP = 10


class GameStarter(threading.Thread):
    """Plays total games to the end, at most parallel at a time.

    Args:
        stall_timeout: seconds without any message after which a game is abandoned.
        max_abandoned: raise once more games than this were abandoned (None: 5%
            of total, at least 5), since the robots are then likely broken.
        on_abandon: called with each abandoned game's name, from this thread.
    """

    def __init__(
        self,
        host,
        port,
        total,
        parallel=1,
        options="-",
        players=4,
        prefix="cat",
        stall_timeout=600.0,
        max_abandoned=None,
        on_abandon=None,
    ):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.total = total
        self.parallel = max(1, parallel)
        self.options = options
        self.players = players
        self.prefix = prefix
        self.stall_timeout = stall_timeout
        self.max_abandoned = (
            max(5, total // 20) if max_abandoned is None else max_abandoned
        )
        self.on_abandon = on_abandon
        self.started = 0
        self.running = set()
        self.last_seen = {}
        self.finished = 0
        self.abandoned = []  # (game, reason)
        self.error = None
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self._run()
        except Exception as e:  # surfaced by the CLI
            if not self._stop.is_set():
                self.error = e
                log.exception("game starter failed")

    def _run(self):
        conn = p.Connection(self.host, self.port, timeout=30)
        self.conn = conn
        try:
            conn.send(
                p.encode(
                    p.VERSION,
                    p.VERSION_NUMBER,
                    p.VERSION_STRING,
                    p.VERSION_BUILD,
                    p.EMPTYSTR,
                )
            )
            self._fill()
            check_every = min(5.0, self.stall_timeout / 4)
            next_check = time.monotonic() + check_every
            while not self._stop.is_set() and (
                self.running or self.finished < self.total
            ):
                if time.monotonic() >= next_check:
                    self._abandon_stalled()
                    next_check = time.monotonic() + check_every
                text = conn.recv(timeout=check_every)
                if text is None:
                    continue
                msg = p.parse(text)
                if msg.game in self.running:
                    self.last_seen[msg.game] = time.monotonic()
                if msg.type == p.SERVERPING:
                    conn.send(p.encode(p.SERVERPING, *msg.fields))
                elif msg.type == p.JOINGAMEAUTH and msg.game in self.running:
                    conn.send(
                        f"{GAMETEXTMSG}|{msg.game}{TEXT_SEP}debug{TEXT_SEP}*STARTBOTGAME* {self.players}"
                    )
                elif msg.type == p.STATUSMESSAGE:
                    log.info("server status: %s", msg.fields)
                elif msg.type == p.GAMESTATS and msg.game in self.running:
                    # One score per seat, then one is-robot flag per seat
                    n = (len(msg.fields) - 1) // 2
                    scores = [int(f) for f in msg.fields[1 : 1 + n]]
                    if max(scores, default=0) < WINNING_VP:
                        self._abandon(msg.game, f"ended without a winner {scores}")
                    else:
                        self._leave(msg.game)
                        self.finished += 1
                        self._fill()
                elif msg.type == p.DELETEGAME and msg.game in self.running:
                    self._abandon(msg.game, "was deleted by the server")
        finally:
            conn.close()

    def _leave(self, game):
        self.running.discard(game)
        self.last_seen.pop(game, None)
        self.conn.send(p.encode(p.LEAVEGAME, "debug", "-", game))

    def _abandon_stalled(self):
        now = time.monotonic()
        for game in sorted(self.running):
            if now - self.last_seen[game] > self.stall_timeout:
                self._abandon(game, f"had no activity for {self.stall_timeout:.0f}s")

    def _abandon(self, game, reason):
        """Gives up on a game without counting it, and starts another in its place."""
        self._leave(game)
        self.abandoned.append((game, reason))
        log.warning("%s %s; abandoned it and started another game", game, reason)
        if self.on_abandon:
            self.on_abandon(game)
        if len(self.abandoned) > self.max_abandoned:
            raise RuntimeError(
                f"Abandoned {len(self.abandoned)} games that stalled or ended "
                "without a winner; the JSettlers robots may be broken"
            )
        self._fill()

    def _fill(self):
        while (
            len(self.running) < self.parallel
            and self.finished + len(self.running) < self.total
        ):
            self.started += 1
            name = f"{self.prefix}-{self.started}"
            self.running.add(name)
            self.last_seen[name] = time.monotonic()
            self.conn.send(
                p.encode(
                    NEWGAMEWITHOPTIONSREQUEST,
                    "debug",
                    p.EMPTYSTR,
                    p.EMPTYSTR,
                    name,
                    self.options,
                )
            )
