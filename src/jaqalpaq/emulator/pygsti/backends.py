# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import itertools

from numpy import zeros

from pygsti.protocols import ModelFreeformSimulator

from jaqalpaq.core.algorithm.walkers import TraceSerializer, Trace
from jaqalpaq.run.cursor import SubcircuitCursor, State
from jaqalpaq.run.result import Subcircuit, ReadoutTreeNode, validate_probabilities
from jaqalpaq.emulator import backend

from .circuit import pygsti_circuit_from_circuit
from .model import build_noisy_native_model

# Set this to True if you would like Subcircuit objects to retain the pyGSTi objects
# used to generate probabilities during their emulation.
KEEP_PYGSTI_OBJECTS = False


class CircuitEmulator(backend.EmulatedIndependentSubcircuitsBackend):
    """Emulator using pyGSTi circuit objects

    This object should be treated as an opaque symbol to be passed to run_jaqal_circuit.
    """

    def __init__(self, *args, model=None, gate_durations=None, **kwargs):
        self.model = model
        self.gate_durations = gate_durations if gate_durations is not None else {}
        super().__init__(*args, **kwargs)

    def _make_subcircuit(self, circ, index, start, end):
        """Generate the probabilities of outcomes of a subcircuit

        :param Trace trace: the subcircut of circ to generate probabilities for
        :return: A pyGSTi outcome dictionary.
        """
        cursor = SubcircuitCursor.terminal_cursor(end)
        trace = Trace(list(start.address), list(end.address))

        pc = pygsti_circuit_from_circuit(
            circ, trace=trace, durations=self.gate_durations
        )

        model = self.model
        mfs = ModelFreeformSimulator(None)
        rho, prob_dict = mfs.compute_final_state(model, pc, include_probabilities=True)

        probs = zeros(len(prob_dict), dtype=float)
        for k, v in prob_dict.items():
            probs[int(k[::-1], 2)] = v

        p = validate_probabilities(probs)

        tree = ReadoutTreeNode(cursor)
        tree.simulated_density_matrix = rho

        for k, v in enumerate(p):
            nxt_cursor = cursor.copy()
            nxt_cursor.next_measure()
            node = tree.subsequent[k] = ReadoutTreeNode(nxt_cursor)
            node.simulated_probability = v

        ret = Subcircuit(index, start, end, tree=tree)

        if KEEP_PYGSTI_OBJECTS:
            ret._pygsti_circuit = pc
            ret._pygsti_model = model

        return ret


class AbstractNoisyNativeEmulator(backend.ExtensibleBackend, CircuitEmulator):
    """(abstract) Noisy emulator using pyGSTi circuit objects

    Provides helper functions to make the generation of a noisy native model simpler.

    Every gate to be emulated should have a corresponding gate_{name} and
      gateduration_{name} method defined.  These will be automatically converted into
      pyGSTi-appropriate objects for model construction.  See build_model for more
      details.
    """

    def __init__(
        self, n_qubits, model=None, gate_durations=None, stretched_gates=None, **kwargs
    ):
        self.n_qubits = n_qubits
        self.stretched_gates = stretched_gates
        model, gate_durations = self.build_model()
        super().__init__(model=model, gate_durations=gate_durations, **kwargs)

    def get_n_qubits(self, circ):
        """Returns the number of qubits the backend will be simulating.

        :param circ: The circuit object being emulated/simulated.
        """
        circuit_qubits = super().get_n_qubits(circ)
        if circuit_qubits > self.n_qubits:
            raise ValueError(f"{self} emulates fewer qubits than {circ} uses")
        return self.n_qubits

    def build_model(self):
        return build_noisy_native_model(
            self.jaqal_gates,
            self.collect_gate_models(),
            self.idle,
            self.n_qubits,
            stretched_gates=self.stretched_gates,
        )
