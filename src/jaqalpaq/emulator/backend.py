# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import abc

from numpy.random import choice

from jaqalpaq.core.locus import Locus
from jaqalpaq.core.block import BlockStatement
from jaqalpaq.core.algorithm.walkers import walk_circuit, discover_subcircuits

from jaqalpaq.run import cursor
from jaqalpaq.run.result import ExecutionResult, Readout
from jaqalpaq.run.backend import IndependentSubcircuitsBackend, AbstractBackend


class ExtensibleBackend(AbstractBackend):
    """Abstract mixin providing an interface for extending a backend.

    Every gate to be emulated should have a corresponding gate_{name} and
      gateduration_{name} method defined.
    """

    def __init__(self, *args, stretched_gates=None, **kwargs):
        """(abstract) Perform part of the construction of a noisy model.

        :param stretched_gates: (default False)  Add stretched gates to the model:
          - If None, do not modify the gates.
          - If 'add', add gates with '_stretched' appended that take an extra parameter,
            a stretch factor.
          - Otherwise, stretched_gates must be the numerical stretch factor that is
            applied to all gates (no extra stretched gates are added
        """
        self.stretched_gates = stretched_gates
        super().__init__(*args, **kwargs)

    def set_defaults(self, kwargs, **defaults):
        """Set parameters from a list of defaults and function kwargs.

        For every value passed as a keyword argument (into **defaults), set it in the
          object's namespace.  Values in kwargs overrided the default.  Values used from
          kwargs are removed from kwargs.

        :param kwargs: a dictionary of your function's keyword arguments, mutated to
          only contain unused values.
        """
        for k, v in defaults.items():
            setattr(self, k, kwargs.pop(k, v))

    @staticmethod
    def _curry(params, *ops):
        """Helper function to make defining related gates easier.
        Curry every function in ops, using the signature description in params.  For
          every non-None entry of params, pass that value to the function.

        :param params: List of parameters to pass to each op in ops, with None allowing
          passthrough of values in the new function
        :param ops: List of functions to curry
        :return List[functions]: A list of curried functions
        """

        def _inner(op):
            def newop(self, *args, **kwargs):
                args = iter(args)
                argv = [next(args) if param is None else param for param in params]
                argv.extend(args)
                return op(self, *argv, **kwargs)

            return newop

        newops = []
        for op in ops:
            newops.append(_inner(op))

        return newops

    def collect_gate_models(self):
        """Return a dictionary of tuples of gate models and gate durations.

        This combs through the class's definition for all parameters named gate_*, and
          adds a corresponding entry in the returned dictionary, keyed by the associated
          gate name, of the gate model (i.e., the noisy process model) and the duration
          that the gate operates.
        : return dict: A dictionary of the models of the gates
        """
        gate_models = {}

        for gate_name in dir(type(self)):
            if not gate_name.startswith("gate_"):
                continue

            name = gate_name[5:]
            gate_models[name] = (
                getattr(self, gate_name),
                getattr(self, f"gateduration_{name}"),
            )

        return gate_models


class EmulatedIndependentSubcircuitsBackend(IndependentSubcircuitsBackend):
    """Abstract emulator backend for subcircuits that are independent"""

    @abc.abstractmethod
    def _make_subcircuit(self, job, index, start, end):
        """(internal) Produce a subcircuit given a trace"""

    def _make_readouts(self, subcircuit, results):
        def _make_all_readouts():
            node = subcircuit._tree
            while not node.classical_state.state == cursor.State.shutdown:
                keys = list(node.subsequent.keys())
                p = [node.subsequent[k].simulated_probability for k in keys]
                nxt = choice(keys, p=p)
                mr = Readout(nxt, len(results), node)
                results.append(mr)
                yield mr
                node = node[nxt]

        subcircuit.accept_readouts(_make_all_readouts())

    def _execute_job(self, job):
        """(internal) Execute the job on the backend"""
        circ = job.expanded_circuit
        subcircs = []
        for n, (start, end) in enumerate(discover_subcircuits(circ)):
            subcirc = self._make_subcircuit(job, n, start, end)
            subcirc._simulated = True
            subcirc.reset_readouts()
            subcircs.append(subcirc)
        results = []

        for index in walk_circuit(circ, [t._start for t in subcircs]):
            # The subcircuit directive is not handled separately from the block
            # that it contains, so we handle it manually here.
            subcirc = subcircs[index]
            start_obj = subcirc._start.object
            if isinstance(start_obj, BlockStatement):
                assert start_obj.subcircuit
                iterations = start_obj.iterations
            else:
                iterations = 1
            for _ in range(iterations):
                self._make_readouts(subcirc, results)

        for subcirc in subcircs:
            subcirc.normalize_counts()

        return ExecutionResult(subcircs, results)
