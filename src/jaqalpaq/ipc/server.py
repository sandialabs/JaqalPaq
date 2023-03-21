# This is a reference implementation of the IPC server used in
# the experiment.  It simply forwards all requests to another
# execution backend.

import socket, select, os, sys, time
from contextlib import contextmanager
from pathlib import Path
import json
import numpy
from jaqalpaq.parser import parse_jaqal_string

BLOCK_SIZE = 4096  # size recommended by Python docs
POLLING_TIMEOUT = 0.1


def ipc_protocol_read(conn):
    resp_list = []
    started = False
    while True:
        events = select.select([conn], [], [conn], POLLING_TIMEOUT)
        if any(events):
            packet = conn.recv(BLOCK_SIZE)
            if packet:
                resp_list.append(packet.decode())
                started = True
                continue

        if started:
            break

    return "".join(resp_list)


def ipc_protocol_parse(resp_text):
    overrides = None
    overrides_prefix = "// OVERRIDES:"
    for line in resp_text.split("\n"):
        if line.startswith(overrides_prefix):
            if overrides is not None:
                raise JaqalError("OVERRIDES may only be provided once per Jaqal file")
            overrides = json.loads(line[len(overrides_prefix) :])

    # Unvalidated and unauthenticated network-received data is being passed to
    # the Jaqal emulator here.
    circ = parse_jaqal_string(resp_text)
    return overrides, circ


def ipc_protocol_write(conn, exe_res):
    results = []

    for exp in exe_res.by_time:
        results.append(list(exp.normalized_counts.by_int_dense))

    return conn.send(json.dumps(results).encode())


@contextmanager
def choose_socket(ns):
    if ns.unix_domain:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(ns.socket_address)
    elif ns.tcp:
        raise NotImplementedError()

    try:
        yield server
    finally:
        server.close()
        if ns.unix_domain:
            os.unlink(ns.socket_address)


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(
        prog="jaqal-ipcserver",
        description="Reference IPC server forwarding to an execution backend",
    )
    parser.add_argument(
        "socket_address",
        help="Address to listen on",
    )
    parser.add_argument(
        "--backend",
        "-b",
        dest="backend",
        default="jaqalpaq.emulator.unitary.UnitarySerializedEmulator()",
        help="Backend to forward all execution requests to",
    )
    parser.add_argument(
        "-U",
        dest="unix_domain",
        action="store_true",
        help="Listen on a unix domain socket",
    )
    parser.add_argument(
        "--random-seed",
        "-r",
        default=(int(10 * time.time()) % (2**32),),
        dest="seed",
        nargs=1,
        help="Choose the random seed that numpy uses. Defaults to a function of the current time.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Start a Jupyter server after reading from the socket.",
    )

    ns = parser.parse_args(argv)

    try:
        seed = int(ns.seed[0])
    except ValueError:
        print("Invalid random seed provided.  Must be an integer.")
        return 2

    numpy.random.seed(seed)

    modname = ns.backend.split("(", maxsplit=1)[0].rsplit(".", maxsplit=1)[0]
    exec(f"import {modname}")
    backend = eval(ns.backend)

    with choose_socket(ns) as server:
        server.listen(1)
        conn, addr = server.accept()
        resp_text = ipc_protocol_read(conn)
        if ns.debug:
            import IPython

            IPython.embed_kernel()
        overrides, circ = ipc_protocol_parse(resp_text)
        exe_res = backend(circ, overrides=overrides).execute()
        ipc_protocol_write(conn, exe_res)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
