# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import itertools

from numpy import zeros

from pygsti.protocols import ModelFreeformSimulator

from jaqalpaq.run.cursor import SubcircuitCursor, State
from jaqalpaq.run import result
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

    def _simulate_subcircuit(self, job, subcirc):
        start = subcirc.start
        end = subcirc.end
        circ = subcirc.filled_circuit

        pc = pygsti_circuit_from_circuit(
            circ, trace=(start, end), durations=self.gate_durations
        )

        model = self.model
        mfs = ModelFreeformSimulator(None)
        rho, prob_dict = mfs.compute_final_state(model, pc, include_probabilities=True)

        probs = zeros(len(prob_dict), dtype=float)
        for k, v in prob_dict.items():
            probs[int(k[::-1], 2)] = v

        p = result.validate_probabilities(probs)

        tree = subcirc.tree
        tree.simulated_density_matrix = rho

        for k, v in enumerate(p):
            node = tree.force_get(k)
            node.simulated_probability = v

        if KEEP_PYGSTI_OBJECTS:
            subcirc._pygsti_circuit = pc
            subcirc._pygsti_model = model


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
