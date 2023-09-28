# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import warnings
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
    Do not use this.  Instead, use run_jaqal_string with an overrides dictionary.

    If using the emulator, you probably also want to set repeats to 1000 (or as desired),
    either in the overrides dictionary as __repeats__, or by setting
    jaqalpaq.run.DEFAULT_NUM_REPEATS

    """
    try:
        import jaqalapp
    except ImportError:
        warnings.warn(run_jaqal_batch.__doc__)

    return [
        BCResult(circuitindex)
        for circuitindex in run_jaqal_string(code, overrides=overrides).by_time
    ]


__all__ = [
    "run_jaqal_circuit",
    "run_jaqal_file",
    "run_jaqal_string",
]
