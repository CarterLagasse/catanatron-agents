import socket

import pytest

from catanatron.sota.jsettlers.server import JSettlersServer


def test_refuses_a_port_another_program_listens_on():
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        with pytest.raises(RuntimeError, match="already in use"):
            JSettlersServer(port=port).check_port_free()
