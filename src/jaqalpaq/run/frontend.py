# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
"""
This module contains several mechanisms to execute Jaqal code, either with
perfect "emulation," noisy simulation, or on actual quantum hardware (via an
inter-process communication protocol).

Environment variables control the behavior of these functions:

 - JAQALPAQ_RUN_EMULATOR -- If this environment variable is set and has
    a value starting with '1', 't', or 'T', then unconditionally do NOT
    use the IPC mechanism.

 - JAQALPAQ_RUN_PORT -- If this variable is set, and JAQALPAQ_RUN_EMULATOR
    does not indicate to use the emulator, communicate with another process
    over a local tcp socket on the given port.
"""
from jaqalpaq.error import JaqalError
from jaqalpaq.parser import parse_jaqal_file, parse_jaqal_string


DEFAULT_BACKEND = None
FORCE_BACKEND = None


def _get_backend(backend):
    global DEFAULT_BACKEND

    if FORCE_BACKEND is not None:
        return FORCE_BACKEND

    if backend is not None:
        return backend

    if DEFAULT_BACKEND is None:
        import os

        if os.environ.get("JAQALPAQ_RUN_PORT", False) and not os.environ.get(
            "JAQALPAQ_RUN_EMULATOR", ""
        ).startswith(("1", "t", "T")):
            from jaqalpaq.ipc.client import IPCBackend

            DEFAULT_BACKEND = IPCBackend()
        else:
            from jaqalpaq.emulator.unitary import UnitarySerializedEmulator

            DEFAULT_BACKEND = UnitarySerializedEmulator()

    return DEFAULT_BACKEND


def _should_autoload_pulses(backend=None, **_kwargs):
    import os

    if not os.environ.get("JAQALPAQ_RUN_PORT", False):
        return True

    backend = _get_backend(backend)

    from jaqalpaq.ipc.client import IPCBackend

    return not isinstance(backend, IPCBackend)


def run_jaqal_circuit(
    circuit, backend=None, force_sim=False, emulator_backend=None, **kwargs
):
    """Execute a Jaqal :class:`~jaqalpaq.core.Circuit` using either an
    emulator or by communicating over IPC with another process.

    :param Circuit circuit: The Jaqalpaq circuit to be run.
    :param backend: The backend to perform the circuit simulation/emulation.
        Defaults to UnitarySerializedEmulator.

    :rtype: ExecutionResult

    .. note::
        Random seed is controlled by numpy.random.seed.  Consider calling ::

            numpy.random.seed(int(time.time()))

        for random behavior.

    """
    if (force_sim is not False) or (emulator_backend is not None):
        raise JaqalError("Specify backend, DEFAULT_BACKEND, or FORCE_BACKEND")

    return _get_backend(backend)(circuit, **kwargs).execute()


def run_jaqal_string(jaqal, import_path=None, **kwargs):
    """Execute a Jaqal string using either an emulator or by communicating
    over IPC with another process.

    :param str jaqal: The literal Jaqal program text.
    :param str import_path: The path to perform relative Jaqal imports from.
        Defaults to the current directory.

    :rtype: ExecutionResult

    .. note::
        See :meth:`run_jaqal_circuit` for additional arguments
    """
    return run_jaqal_circuit(
        parse_jaqal_string(
            jaqal,
            autoload_pulses=_should_autoload_pulses(**kwargs),
            import_path=import_path,
        ),
        **kwargs
    )


def run_jaqal_file(fname, import_path=None, **kwargs):
    """Execute a Jaqal program in a file using either an emulator or by communicating
    over IPC with another process.

    :param str fname: The path to a Jaqal file to execute.
    :param str import_path: The path to perform relative Jaqal imports from.
        Defaults to parent directory of the file.

    :rtype: ExecutionResult

    .. note::
        See :meth:`run_jaqal_circuit` for additional arguments

    """
    return run_jaqal_circuit(
        parse_jaqal_file(
            fname,
            autoload_pulses=_should_autoload_pulses(**kwargs),
            import_path=import_path,
        ),
        **kwargs
    )


__all__ = [
    "run_jaqal_string",
    "run_jaqal_file",
    "run_jaqal_circuit",
]
