# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import abc
import itertools

from jaqalpaq.core.algorithm import expand_macros
from jaqalpaq.error import JaqalError


class JaqalJob:
    """Jaqal Compute Job"""

    def __init__(self, backend, circuit, overrides=None):
        self.backend = backend
        self.circuit = circuit
        self.expanded_circuit = expand_macros(circuit)
        self.overrides = overrides

    def __repr__(self):
        return f"<{type(self)} of {self.backend}>"

    def execute(self):
        """Executes the job on the backend"""
        return self.backend._execute_job(self)


class AbstractBackend:
    """Abstract Jaqal Execution Backend"""

    @abc.abstractmethod
    def _execute_job(self, job):
        """(internal) Performs the backend-specific job execution."""

    @abc.abstractmethod
    def __call__(self, circ):
        """Creates a job object for circ

        :param Circuit circ: circuit to run
        """

    def get_n_qubits(self, circ):
        """Returns the number of qubits the backend will execute

        Specifically, it will be the number of qubits in the considered circuit.

        :param circ: The circuit object being executed
        """

        registers = circ.fundamental_registers()

        try:
            (register,) = registers
        except ValueError:
            raise JaqalError("Multiple fundamental registers unsupported.")

        return register.size


class IndependentSubcircuitsBackend(AbstractBackend):
    """Abstract backend for subcircuits that are independent"""

    def __call__(self, circuit, *, overrides=None):
        """Attaches the backend to a particular circuit, creating a Job object.

        Calculates the probabilities of outcomes for every subcircuit.

        :param Circuit circuit: parent circuit

        :returns IndependentSubcircuitsJob:
        """
        job = JaqalJob(self, circuit, overrides=overrides)

        return job
