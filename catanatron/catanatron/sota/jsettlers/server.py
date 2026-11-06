"""Downloads and runs a JSettlers2 server for robot-only games.

Games are started by starter.py with the *STARTBOTGAME* debug command. The
server fills their seats with its own robots, except one seat which goes to
our bridge, connected as a "third-party" robot (jsettlers.bots.percent3p).
See JSettlers2's doc/Readme.developer.md, "Robots (AI)".
"""

import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path

from catanatron.sota._download import cache_dir, fetch

VERSION = "2.6.10"
JAR_URL = (
    "https://github.com/jdmonin/JSettlers2/releases/download/"
    f"release-{VERSION}/jsettlers-{VERSION}-server.jar"
)
JAR_SHA256 = "9b26cb20ec943213600795d660fad2da32fc72629c348e180bd171cc26d6b4e2"


def server_jar() -> Path:
    dest = cache_dir() / "jsettlers" / f"jsettlers-{VERSION}-server.jar"
    return fetch(JAR_URL, dest, JAR_SHA256)


class JSettlersServer:
    """Context manager running a local JSettlers server for bot-only games.

    Args:
        robots: built-in robots to start. JSettlers makes 30% of them (rounded
            down) "fast" bots and the rest "smart" bots, so 3 gives all smart.
        fast_pause_percent: robots' pause length in bot-only games, as a percent
            of their normal pauses (which imitate human play).
        turn_timeout: seconds the bridge may take per turn before the server
            ends its turn for it.

    While running, the server listens on all network interfaces (JSettlers has no
    bind-address setting) and enables its passwordless "debug" user, which
    starter.py needs to start bot games. Anyone who can reach the port can log in
    as that user, so keep it firewalled from other machines.
    """

    def __init__(
        self,
        port: int = 8880,
        robots: int = 3,
        fast_pause_percent: int = 1,
        turn_timeout: int = 300,
        check_tracking: bool = False,
        java: str = "java",
        log_path: Path | None = None,
    ):
        self.port = port
        self.cookie = secrets.token_hex(8)
        self.java = shutil.which(java) or java
        self.log_path = log_path
        self.props = {
            "jsettlers.startrobots": robots,
            "jsettlers.bots.cookie": self.cookie,
            "jsettlers.bots.percent3p": 25,  # one seat in 2- to 4-player games (rounded)
            # != 0 enables *STARTBOTGAME*; < 0 doesn't start any games by itself
            "jsettlers.bots.botgames.total": -1,
            "jsettlers.bots.fast_pause_percent": fast_pause_percent,
            "jsettlers.bots.timeout.turn": turn_timeout,
            "jsettlers.allow.debug": "Y",
            # Lets games use the "fully observable" option (see starter.py)
            "jsettlers.gameopts.activate": "PLAY_FO",
        }
        if check_tracking:
            # Send bots everyone's true resources at each turn end (see GameController)
            self.props["jsettlers.debug.bots.datacheck.rsrc"] = "Y"
        self.process = None
        self._log_file = None

    def command(self):
        args = [self.java, "-jar", str(server_jar())]
        args += [f"-D{key}={value}" for key, value in self.props.items()]
        return args + [str(self.port), "50"]

    def __enter__(self):
        self.check_port_free()
        self._log_file = (
            open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        )
        self.process = subprocess.Popen(
            self.command(), stdout=self._log_file, stderr=subprocess.STDOUT
        )
        self.wait_until_listening()
        return self

    def check_port_free(self):
        # Another program on 127.0.0.1:port would take our connections even if
        # JSettlers manages to listen on the other interfaces (Windows allows it).
        for host in ("127.0.0.1", "0.0.0.0"):
            with socket.socket() as s:
                try:
                    s.bind((host, self.port))
                except OSError as e:
                    raise RuntimeError(
                        f"Port {self.port} is already in use on {host}; choose another with --port"
                    ) from e

    def wait_until_listening(self, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"JSettlers server exited with code {self.process.returncode}"
                    + (f"; see {self.log_path}" if self.log_path else "")
                )
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    return
            except OSError:
                time.sleep(0.2)
        raise TimeoutError(f"JSettlers server didn't open port {self.port}")

    def __exit__(self, *exc):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self._log_file not in (None, subprocess.DEVNULL):
            self._log_file.close()
