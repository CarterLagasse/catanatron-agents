"""JSettlers2 wire protocol: framing, message type ids, and encoders.

Each message is sent with Java's DataOutputStream.writeUTF: a 2-byte
big-endian length, then (modified) UTF-8 text of the form "TYPE|data".
Most message types separate their data fields with ","; the multi-param
types (soc.message.SOCMessageTemplateMi subclasses) use "|" instead.
Formats follow JSettlers 2.6.10's soc.message classes.
"""

import select
import socket
import struct
from dataclasses import dataclass

# Message type ids (soc.message.SOCMessage)
PUTPIECE = 1009
LEAVEGAME = 1011
SITDOWN = 1012
JOINGAME = 1013
BOARDLAYOUT = 1014
DELETEGAME = 1015
GAMEMEMBERS = 1017
STARTGAME = 1018
JOINGAMEAUTH = 1021
IMAROBOT = 1022
BOTJOINGAMEREQUEST = 1023
PLAYERELEMENT = 1024
GAMESTATE = 1025
TURN = 1026
DICERESULT = 1028
DISCARDREQUEST = 1029
ROLLDICE = 1031
ENDTURN = 1032
DISCARD = 1033
MOVEROBBER = 1034
CHOOSEPLAYER = 1035
CHOOSEPLAYERREQUEST = 1036
REJECTOFFER = 1037
ACCEPTOFFER = 1039
BANKTRADE = 1040
MAKEOFFER = 1041
BUYDEVCARDREQUEST = 1045
DEVCARDACTION = 1046
PLAYDEVCARDREQUEST = 1049
PICKRESOURCES = 1052
PICKRESOURCETYPE = 1053
FIRSTPLAYER = 1054
SETTURN = 1055
ROBOTDISMISS = 1056
REJECTCONNECTION = 1059
GAMESTATS = 1061
RESOURCECOUNT = 1063
LONGESTROAD = 1066
LARGESTARMY = 1067
STATUSMESSAGE = 1069
UPDATEROBOTPARAMS = 1071
CANCELBUILDREQUEST = 1044
BOARDLAYOUT2 = 1084
PLAYERSTATS = 1085
PLAYERELEMENTS = 1086
SIMPLEACTION = 1090
DICERESULTRESOURCES = 1092
GAMEELEMENTS = 1096
ROBBERYRESULT = 1102
BOTGAMEDATACHECK = 1103
DECLINEPLAYERREQUEST = 1104
VERSION = 9998
SERVERPING = 9999

# Types whose fields are all separated by "|" (SOCMessageTemplateMi and friends)
MULTI_TYPES = {
    PLAYERSTATS,
    PLAYERELEMENTS,
    DICERESULTRESOURCES,
    GAMEELEMENTS,
    BOTGAMEDATACHECK,
}

# Placeholder for an empty string field (SOCMessage.EMPTYSTR)
EMPTYSTR = "\t"

# The version we claim to speak; dev card and message formats depend on it
VERSION_NUMBER = 2610
VERSION_STRING = "2.6.10"
VERSION_BUILD = "JM20220705"


@dataclass
class Message:
    type: int
    fields: list[str]

    @property
    def game(self) -> str:
        return self.fields[0] if self.fields else ""

    def ints(self, start: int = 1) -> list[int]:
        return [int(f) for f in self.fields[start:]]


def parse(text: str) -> Message:
    type_str, _, data = text.partition("|")
    msg_type = int(type_str)
    if msg_type in MULTI_TYPES:
        fields = data.split("|")
    else:
        # The game name is followed by "," fields, except in a few types
        # (e.g. STATUSMESSAGE) whose free text we don't need to parse.
        fields = data.split(",")
    return Message(msg_type, fields)


def encode(msg_type: int, *fields) -> str:
    return f"{msg_type}|" + ",".join(str(f) for f in fields)


class Connection:
    """A TCP connection speaking JSettlers' writeUTF/readUTF framing."""

    def __init__(self, host: str, port: int, timeout: float | None = None):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(None)
        self._buffer = bytearray()

    def send(self, text: str):
        data = text.encode("utf-8")
        self.sock.sendall(struct.pack(">H", len(data)) + data)

    def recv(self, timeout: float | None = None) -> str | None:
        """Blocks until a message arrives, or returns None if none arrives in
        timeout seconds. Raises ConnectionError on close."""
        while True:
            if len(self._buffer) >= 2:
                (length,) = struct.unpack(">H", self._buffer[:2])
                if len(self._buffer) >= 2 + length:
                    data = bytes(self._buffer[2 : 2 + length])
                    del self._buffer[: 2 + length]
                    return data.decode("utf-8", errors="replace")
            if timeout is not None:
                readable, _, _ = select.select([self.sock], [], [], timeout)
                if not readable:
                    return None
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("JSettlers server closed the connection")
            self._buffer += chunk

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
