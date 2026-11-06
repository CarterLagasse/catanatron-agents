import socket
import struct
import threading

import pytest

from catanatron.sota.jsettlers import protocol as p
from catanatron.sota.jsettlers.client import BridgeClient, GameResult
from catanatron.sota.jsettlers.starter import GameStarter


class Peer:
    """The server side of a connection, speaking JSettlers' framing."""

    def __init__(self, sock):
        sock.settimeout(5)
        self.sock = sock
        self.file = sock.makefile("rb")

    def send(self, text):
        data = text.encode("utf-8")
        self.sock.sendall(struct.pack(">H", len(data)) + data)

    def recv(self):
        (length,) = struct.unpack(">H", self.file.read(2))
        return self.file.read(length).decode("utf-8")

    def close(self):
        self.file.close()
        self.sock.close()


@pytest.fixture
def server():
    listener = socket.create_server(("127.0.0.1", 0))
    yield listener
    listener.close()


def _accept(listener):
    sock, _ = listener.accept()
    return Peer(sock)


def test_recv_reassembles_split_messages_and_times_out(server):
    conn = p.Connection("127.0.0.1", server.getsockname()[1])
    peer = _accept(server)
    assert conn.recv(timeout=0.05) is None

    framed = [struct.pack(">H", len(m)) + m for m in (b"1|a", b"2|bb")]
    data = b"".join(framed)
    peer.sock.sendall(data[:1])
    assert conn.recv(timeout=0.05) is None
    peer.sock.sendall(data[1:])
    assert conn.recv() == "1|a"
    assert conn.recv(timeout=1) == "2|bb"

    peer.close()
    with pytest.raises(ConnectionError):
        conn.recv(timeout=1)
    conn.close()


def test_starter_replaces_games_that_stall_or_end_without_a_winner(server):
    abandoned = []
    starter = GameStarter(
        "127.0.0.1",
        server.getsockname()[1],
        total=1,
        stall_timeout=0.4,
        on_abandon=abandoned.append,
    )
    starter.start()
    peer = _accept(server)
    assert peer.recv().startswith(f"{p.VERSION}|")
    assert ",cat-1," in peer.recv()  # new game request

    # The robots never start cat-1: once it stalls, it's replaced by cat-2.
    assert peer.recv() == f"{p.LEAVEGAME}|debug,-,cat-1"
    assert ",cat-2," in peer.recv()

    # cat-2 ends at the start, without a winner: replaced by cat-3.
    peer.send(f"{p.GAMESTATS}|cat-2,2,2,3,2,true,true,true,true")
    assert peer.recv() == f"{p.LEAVEGAME}|debug,-,cat-2"
    assert ",cat-3," in peer.recv()

    peer.send(f"{p.GAMESTATS}|cat-3,10,5,4,7,true,true,true,true")
    assert peer.recv() == f"{p.LEAVEGAME}|debug,-,cat-3"
    starter.join(5)
    assert not starter.is_alive()
    assert starter.error is None
    assert starter.finished == 1
    assert [game for game, _ in starter.abandoned] == ["cat-1", "cat-2"] == abandoned


def test_starter_gives_up_when_too_many_games_fail(server):
    starter = GameStarter(
        "127.0.0.1",
        server.getsockname()[1],
        total=3,
        stall_timeout=0.2,
        max_abandoned=1,
    )
    starter.start()
    peer = _accept(server)  # never answers: every game stalls
    starter.join(5)
    peer.close()
    assert not starter.is_alive()
    assert isinstance(starter.error, RuntimeError)
    assert len(starter.abandoned) == 2


class FakeConnection:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)

    def recv(self, timeout=None):
        return None

    def close(self):
        pass


def _result(winner_pn):
    return GameResult(
        game="g",
        our_pn=0,
        scores=[2, 2, 2, 2],
        names=["catanatron", "robot 1", "robot 2", "robot 3"],
        turns=1,
        winner_pn=winner_pn,
        decisions=1,
        decide_seconds=0.0,
    )


def test_bridge_does_not_count_games_without_a_winner():
    seen = []
    client = BridgeClient("127.0.0.1", 0, "cookie", None, on_result=seen.append)
    client.games["g"] = controller = type("C", (), {"name": "g"})()

    controller.result = _result(winner_pn=None)
    client.game_finished(controller)
    assert client.results == seen == []
    assert client.unfinished == [controller.result]
    assert "g" not in client.games

    controller.result = _result(winner_pn=0)
    client.game_finished(controller)
    assert client.results == seen == [controller.result]


def test_bridge_leaves_games_abandoned_from_another_thread():
    client = BridgeClient("127.0.0.1", 0, "cookie", None)
    client.conn = FakeConnection()
    client.games["g"] = object()
    stops = iter([False, True])

    thread = threading.Thread(target=client.abandon, args=("g",))
    thread.start()
    thread.join()
    client.run(should_stop=lambda: next(stops))

    assert client.conn.sent == [f"{p.LEAVEGAME}|catanatron,-,g"]
    assert "g" not in client.games
