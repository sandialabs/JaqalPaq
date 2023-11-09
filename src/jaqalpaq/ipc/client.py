# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
"""
(internal) All members of this module should be considered internal API.
"""

import math
import os
import socket, select
import json
import time

from jaqalpaq.error import JaqalError
from jaqalpaq.generator import generate_jaqal_program

from jaqalpaq.run import result
from jaqalpaq.run.backend import IndependentSubcircuitsBackend
from jaqalpaq.ipc.header import IPCHeader


def emit_jaqal_for_hardware(circuit, overrides):
    """(internal) Generate Jaqal appropriate for running on the hardware."""
    jaqal = generate_jaqal_program(circuit)
    return "".join((jaqal, "\n// OVERRIDES: ", json.dumps(overrides or {}), "\n"))


class IPCBackend(IndependentSubcircuitsBackend):
    # How long to wait for a response from the HW or other component
    # executing the circuit. In seconds.
    long_timeout = 20 * 60

    # Size, in bytes, that we read in at once when getting our
    # results. This size is recommended by Python docs, and it is
    # unlikely you will benefit from changing it.
    block_size = 4096

    def __init__(self, *, address=None, host_socket=None):
        super().__init__()
        if host_socket is not None:
            self._host_socket = host_socket
            if address is not None:
                raise TypeError("host_socket and address cannot both be specified")
        else:
            if address is None:
                try:
                    ipc_port = int(os.environ["JAQALPAQ_RUN_PORT"])
                except (KeyError, ValueError) as e:
                    raise TypeError("host_socket or address must be specified")
                address = ("localhost", ipc_port)

            self._address = address

    def get_host_socket(self):
        try:
            return self._host_socket
        except AttributeError:
            pass

        try:
            self._host_socket = socket.create_connection(self._address)
        except Exception as exc:
            raise JaqalError(
                f"Could not connect to host {':'.join(self._address)}: {exc}"
            )

        return self._host_socket

    def _communicate(self, socket, data):
        hdr = IPCHeader.from_body(data)
        hdr.send(socket)
        socket.send(data)

        # The response is serialized JSON. Each entry in the array is a measurement
        # in the Jaqal file, and each entry in those entries represents a state
        resp_list = []
        hdr = IPCHeader.recv(socket)
        if hdr is None:
            raise JaqalError("Connection closed without sending results")
        length = hdr.size

        while length > 0:
            packet = self._recv_responsive(socket, min(length, self.block_size))
            if not packet:
                raise JaqalError(f"Did not receive full response")
            length -= len(packet)
            resp_list.append(packet.decode())

        assert length == 0, "Wrong amount read"

        resp_text = "".join(resp_list)

        # Deserialize the JSON into a list of lists of floats
        try:
            results = json.loads(resp_text)
        except Exception as exc:
            print(resp_text)
            raise JaqalError(f"Bad response: {exc}") from exc

        return results

    def _recv_responsive(self, socket, length):
        """Receive up to length bytes while remaining responsive to
        ctrl-c and other signals. Windows in particular will not wake
        a process waiting on a network action with a signal and can
        cause hard-to-kill processes in the case of an error without
        this extra machinery."""

        polling_timeout = 1
        start_time = time.time()

        while True:
            events = select.select([socket], [], [socket], polling_timeout)
            if any(events):
                break
            if time.time() - start_time < self.long_timeout:
                raise JaqalError(f"No response before timeout")

        try:
            return socket.recv(length)
        except Exception as exc:
            raise JaqalError(f"Error while receiving response: {exc}") from exc

    def _ipc_protocol(self, circuit, overrides=None):
        jaqal = emit_jaqal_for_hardware(circuit, overrides)
        sock = self.get_host_socket()
        results = self._communicate(sock, jaqal.encode())

        qubit_count = math.log2(len(results[0]))
        if 2**qubit_count != len(results[0]):
            import warnings

            warnings.warn("Invalid normalized counts")

        return results

    def _execute_job(self, job):
        circ = job.expanded_circuit
        exe_res = result.ExecutionResult(circ, job.overrides)

        freqs = self._ipc_protocol(circ, job.overrides)
        parser = exe_res.accept_normalized_counts()

        parser.pass_data(freqs)

        if not parser.done:
            raise JaqalError("Unable to parse output: not enough values")

        return exe_res
