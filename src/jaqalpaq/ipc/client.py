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

from jaqalpaq.generator import generate_jaqal_program
from jaqalpaq.core.result import ExecutionResult, RelativeFrequencySubcircuit
from jaqalpaq.error import JaqalError
from jaqalpaq.run.backend import IndependentSubcircuitsBackend


class IPCBackend(IndependentSubcircuitsBackend):
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
            return socket.create_connection(self._address)
        except Exception as exc:
            raise JaqalError(f"Could not connect to host {':'.join(address)}: {exc}")

    def _communicate(self, socket, data):
        socket.send(data)

        # The response is serialized JSON. Each entry in the array is a measurement
        # in the Jaqal file, and each entry in those entries represents
        resp_list = []
        polling_timeout = 0.1
        started = False
        while True:
            block_size = 4096  # size recommended by Python docs
            events = select.select([socket], [], [socket], polling_timeout)
            if any(events):
                packet = socket.recv(block_size)
                if packet:
                    resp_list.append(packet.decode())
                    started = True
                    continue

            if started:
                break
        resp_text = "".join(resp_list)

        # Deserialize the JSON into a list of lists of floats
        try:
            results = json.loads(resp_text)
        except Exception as exc:
            print(resp_text)
            raise JaqalError(f"Bad response: {exc}") from exc

        return results

    def _ipc_protocol(self, circuit):
        jaqal = generate_jaqal_program(circuit)
        sock = self.get_host_socket()
        try:
            results = self._communicate(sock, jaqal.encode())
        finally:
            # Can this be reused?
            sock.close()

        qubit_count = math.log2(len(results[0]))
        if 2**qubit_count != len(results[0]):
            import warnings

            warnings.warn("Invalid normalized counts")

        return results

    def _execute_job(self, job):
        freqs = self._ipc_protocol(job.circuit)
        n_qubits = self.get_n_qubits(job.circuit)
        subcircs = []
        for n, tr in enumerate(job.traces):
            subcircs.append(
                RelativeFrequencySubcircuit(
                    tr, n, normalized_counts=[float(rf) for rf in freqs[n]]
                )
            )

        if n + 1 < len(freqs):
            raise JaqalError("Unable to parse output: too many values")

        return ExecutionResult(subcircs, None)
