# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
from .frontend import run_jaqal_circuit, run_jaqal_file, run_jaqal_string


class BCResult:
    """(deprecated) Thin layer providing access to old-style batched Jaqal responses.

    *Only subcircuits* are available through this interface.
    """

    def __init__(self, circuitindex):
        self.subcircuits = [circuitindex]


def run_jaqal_batch(code, overrides):
    """(deprecated) Thin later providing access to old-style batched Jaqal.

    BCResult objects are provided, which only allow for access to subcircuit information.
    """
    return [
        BCResult(circuitindex)
        for circuitindex in run_jaqal_string(code, overrides=overrides).by_time
    ]


__all__ = [
    "run_jaqal_circuit",
    "run_jaqal_file",
    "run_jaqal_string",
]
