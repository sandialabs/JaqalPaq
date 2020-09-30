# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
from numpy import array

import pygsti

from jaqalpaq.core.algorithm.walkers import TraceSerializer
from jaqalpaq.emulator.backend import IndependentSubcircuitsBackend

from .circuit import pygsti_circuit_from_gatelist, pygsti_circuit_from_circuit
from .model import (
    build_noiseless_native_model,
    pygsti_independent_noisy_gate,
    JaqalOpFactory,
)


class pyGSTiEmulator(IndependentSubcircuitsBackend):
    """(abstract) pyGSTi emulator
    Collects common helper functions required by pyGSTi backends.
    """

    ZERO_CUTOFF = 1e-13

    def _probs_from_model(self, model, pc):
        res = []
        for k, v in model.probs(pc).items():
            if (v < 0) and (v > -self.ZERO_CUTOFF):
                v = 0
            res.append((int(k[0][::-1], 2), v))

        probs = array(res)
        return probs[probs[:, 0].argsort()][:, 1].copy()


class UnitarySerializedEmulator(pyGSTiEmulator):
    """Serialized emulator using pyGSTi circuit objects

    This object should be treated as an opaque symbol to be passed to run_jaqal_circuit.
    """

    def _probability(self, job, trace):
        """Generate the probabilities of outcomes of a subcircuit

        :param Trace trace: the subcircut of circ to generate probabilities for
        :return: A pyGSTi outcome dictionary.
        """

        circ = job.circuit
        try:
            (register,) = circ.fundamental_registers()
        except ValueError:
            raise NotImplementedError("Multiple fundamental registers unsupported.")

        n_qubits = register.size

        s = TraceSerializer(trace)
        pc = pygsti_circuit_from_gatelist(list(s.visit(circ)), n_qubits)
        model = build_noiseless_native_model(n_qubits, circ.native_gates)
        return self._probs_from_model(model, pc)


class CircuitEmulator(pyGSTiEmulator):
    """Emulator using pyGSTi circuit objects

    This object should be treated as an opaque symbol to be passed to run_jaqal_circuit.
    """

    def __init__(self, *args, model=None, gate_durations=None, **kwargs):
        self.model = model
        self.gate_durations = gate_durations if gate_durations is not None else {}
        super().__init__(*args, **kwargs)

    def _probability(self, job, trace):
        """Generate the probabilities of outcomes of a subcircuit

        :param Trace trace: the subcircut of circ to generate probabilities for
        :return: A pyGSTi outcome dictionary.
        """

        pc = pygsti_circuit_from_circuit(
            job.circuit, trace=trace, durations=self.gate_durations
        )
        return self._probs_from_model(self.model, pc)


class AbstractNoisyNativeEmulator(CircuitEmulator):
    """(abstract) Noisy emulator using pyGSTi circuit objects

    Provides helper functions to make the generation of a noisy native model simpler.

    Every gate to be emulated should have a corresponding gate_{name} and
      gateduration_{name} method defined.  These will be automatically converted into
      pyGSTi-appropriate objects for model construction.  See build_model for more
      details.
    """

    def __init__(self, n_qubits, **kwargs):
        """(abstract) Perform part of the construction of a noisy model.

        :param n_qubits: The number of qubits to simulate
        """
        self.n_qubits = n_qubits
        model, durations = self.build_model()
        super().__init__(model=model, gate_durations=durations, **kwargs)

    def build_model(self):
        """
        Parse the dictionary of the **current** class for entries named gate_{name},
          convert them to pyGSTi-appropriate objects using pygsti_independent_noisy_gate,
          and then submit them to build_localnoise_model to produce a pyGSTi local noise
          model.  Furthermore, collect corresponding gateduration_{name} objects into a
          dictionary.

        Both the gate and gateduration functions must take a signature identical to
          each other and their corresponding Jaqal gate.  Currently, all qubit parameters
          will be passed None, but future versions may pass a description of the qubit
          to simulate (allowing qubit-dependend noise models).

        :return tuple: of pyGSTi local noise model and dictionary (of duration functions)
        """
        gates = {}
        durations = {}
        availability = {}

        for gate_name in type(self).__dict__:
            if not gate_name.startswith("gate_"):
                continue

            name = gate_name[5:]
            durations[name] = getattr(self, f"gateduration_{name}")

            pygsti_name = f"GJ{name}"
            jaqal_gate = self.jaqal_gates[name]
            func = getattr(self, gate_name)
            gates[pygsti_name] = pygsti_independent_noisy_gate(jaqal_gate, func)
            if len(jaqal_gate.quantum_parameters) > 1:
                availability[pygsti_name] = "all-permutations"

        gates["Gidle"] = JaqalOpFactory(self.idle)

        target_model = pygsti.construction.build_localnoise_model(
            nQubits=self.n_qubits,
            availability=availability,
            gate_names=list(gates.keys()),
            custom_gates=gates,
            parameterization="full",
            evotype="densitymx",
        )

        return target_model, durations

    def set_defaults(self, kwargs, **values):
        """Helper function to set default values.
        For every value passed as a keyword argument or in kwargs, set it in the object's
          namespace, with values in kwargs taking precedence.

        :param kwargs: a dictionary of your function's keyword arguments
        """
        for k, v in values.items():
            setattr(self, k, kwargs.pop(k, v))

    @staticmethod
    def _curry(params, *ops):
        """Helper function to make defining related gates easier.
        Curry every function in ops, using the signature description in params.  For
          every non-None entry of params, pass that value to the function.

        :param params: List of parameters to pass to each op in ops, with None allpwing
          passthrough of values in the new function
        :param ops: List of functions to curry
        :return List[functions]: A list of curried functions
        """

        def _inner(op):
            def newop(self, *args, **kwargs):
                args = iter(args)
                argv = [next(args) if param is None else param for param in params]
                try:
                    while True:
                        argv.append(next(args))
                except StopIteration:
                    pass
                return op(self, *argv, **kwargs)

            return newop

        newops = []
        for op in ops:
            newops.append(_inner(op))

        return newops
