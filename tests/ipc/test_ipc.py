import os, time, sys, subprocess, socket
import unittest, pytest
import numpy

from jaqalpaq.ipc.ipc import IPCBackend
from jaqalpaq.parser import parse_jaqal_file
from jaqalpaq.run import run_jaqal_circuit

IPC_SOCKET = "/tmp/ipc_test"


@unittest.skipIf(sys.platform.startswith("win"), "IPC tests use Unix sockets")
class IPCTester(unittest.TestCase):
    """Test interacting through the IPC interface."""

    def run_jaqal(self, circuit, **kwargs):
        host_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            P = subprocess.Popen([sys.executable, "-mtests.ipc._mock_server"])
            time.sleep(1)
            host_socket.connect(IPC_SOCKET)

            backend = IPCBackend(host_socket=host_socket)
            exe = run_jaqal_circuit(circuit, backend=backend, **kwargs)
        finally:
            host_socket.close()
            P.send_signal(subprocess.signal.SIGTERM)
            P.communicate()

        return exe

    def test_bell_prep(self):
        """Run bell_prep.jaqal through the IPC mock server, and parse the results"""

        circ = parse_jaqal_file("examples/jaqal/bell_prep.jaqal")
        exe = self.run_jaqal(circ)

        sc0, sc1 = exe.subcircuits
        assert numpy.allclose(sc0.relative_frequency_by_int, [0, 1 / 2, 1 / 2, 0])
        assert numpy.allclose(sc1.relative_frequency_by_int, [1, 0, 0, 0])
