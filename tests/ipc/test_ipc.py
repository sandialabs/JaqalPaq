import os, time, sys, subprocess, socket
import unittest, pytest
import numpy

from jaqalpaq.ipc.client import IPCBackend
from jaqalpaq.parser import parse_jaqal_file
from jaqalpaq.run import run_jaqal_circuit

IPC_SOCKET = "/tmp/ipc_test"


@unittest.skipIf(sys.platform.startswith("win"), "IPC tests use Unix sockets")
class IPCTester(unittest.TestCase):
    """Test interacting through the IPC interface."""

    def run_jaqal(self, circuit, **kwargs):
        host_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            P = subprocess.Popen(
                [sys.executable, "-mjaqalpaq.ipc.server", "-U", IPC_SOCKET, "-r0"]
            )

            time.sleep(1)
            host_socket.connect(IPC_SOCKET)

            backend = IPCBackend(host_socket=host_socket)
            exe = run_jaqal_circuit(circuit, backend=backend, **kwargs)
        finally:
            host_socket.close()
            time.sleep(0.1)
            P.send_signal(subprocess.signal.SIGTERM)
            P.communicate()

        return exe

    def test_prep_meas_equiv(self):
        """Run bell_prep.jaqal through the IPC mock server, and parse the results"""

        circ = parse_jaqal_file("examples/jaqal/prepare-measure-equiv.jaqal")
        exe = self.run_jaqal(circ)

        sc0, sc1, sc2 = exe.by_subbatch[0].by_subcircuit
        assert numpy.allclose(
            sc0.normalized_counts.by_int_dense,
            [numpy.array((v,)) for v in [0.0, 1.0]],
        )
        assert numpy.allclose(
            sc1.normalized_counts.by_int_dense,
            [
                numpy.array(v, dtype=float)
                for v in [(0, 0, 0, 1, 0, 1, 0), (1, 1, 1, 0, 1, 0, 1)]
            ],
        )
        assert numpy.allclose(
            sc2.normalized_counts.by_int_dense,
            [numpy.array((v,)) for v in [0.0, 1.0]],
        )

    def test_bell_prep(self):
        """Run bell_prep.jaqal through the IPC mock server, and parse the results"""

        circ = parse_jaqal_file("examples/jaqal/bell_prep.jaqal")
        exe = self.run_jaqal(circ)

        sc0, sc1 = exe.subcircuits
        assert numpy.allclose(sc0.relative_frequency_by_int, [0, 0, 1, 0])
        assert numpy.allclose(sc1.relative_frequency_by_int, [1, 0, 0, 0])

        rf0, rf1 = (i.normalized_counts for i in exe.by_time)
        assert numpy.allclose(rf0.by_int_dense, [0, 0, 1, 0])
        assert numpy.allclose(rf1.by_int_dense, [1, 0, 0, 0])

    def test_bell_prep_subcircuit(self):
        """Run bell_prep.jaqal through the IPC mock server, and parse the results"""

        circ = parse_jaqal_file("examples/jaqal/bell_prep_subcircuit.jaqal")
        exe = self.run_jaqal(circ)

        rf0, rf1 = (i.normalized_counts for i in exe.by_time)
        assert numpy.allclose(rf0.by_int_dense, [0, 10 / 30, 20 / 30, 0])
        assert numpy.allclose(rf1.by_int_dense, [1, 0, 0, 0])

    def test_subcircuit_batching(self):
        """Run bell_prep.jaqal through the IPC mock server, and parse the results"""

        circ = parse_jaqal_file("examples/jaqal/subcircuit.jaqal")
        exe = self.run_jaqal(circ, overrides=dict(A=[0, 1, 2], B=[numpy.pi / 2, 2, 0]))

        rf0, rf1 = (i.normalized_counts for i in exe.by_subbatch[0].by_time)
        assert numpy.allclose(rf0.by_int_dense, [20 / 20, 0, 0, 0])
        assert numpy.allclose(rf1.by_int_dense, [14 / 30, 0, 16 / 30, 0])

        rf0, rf1 = (i.normalized_counts for i in exe.by_subbatch[1].by_time)
        assert numpy.allclose(rf0.by_int_dense, [17 / 20, 3 / 20, 0, 0])
        assert numpy.allclose(rf1.by_int_dense, [12 / 30, 0, 18 / 30, 0])

        rf0, rf1 = (i.normalized_counts for i in exe.by_subbatch[2].by_time)
        assert numpy.allclose(rf0.by_int_dense, [3 / 20, 17 / 20, 0, 0])
        assert numpy.allclose(rf1.by_int_dense, [30 / 30, 0, 0, 0])

    def test_subcircuit(self):
        """Run bell_prep.jaqal through the IPC mock server, and parse the results"""

        circ = parse_jaqal_file("examples/jaqal/subcircuit.jaqal")
        exe = self.run_jaqal(circ)

        sc0, sc1 = exe.by_subbatch[0].by_subcircuit
        assert numpy.allclose(
            sc0.normalized_counts.by_int_dense,
            [numpy.array((v,)) for v in [6 / 20, 14 / 20, 0, 0]],
        )
        assert numpy.allclose(
            sc1.normalized_counts.by_int_dense,
            [numpy.array((v,)) for v in [14 / 30, 0, 16 / 30, 0]],
        )
