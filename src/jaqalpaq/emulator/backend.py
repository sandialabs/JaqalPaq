# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import abc

from numpy.random import choice

from jaqalpaq.error import JaqalError
from jaqalpaq.core.locus import Locus
from jaqalpaq.core.block import BlockStatement
from jaqalpaq.core.algorithm.walkers import walk_circuit, discover_subcircuits
from jaqalpaq.core.algorithm import fill_in_let

from jaqalpaq.run import result, cursor
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
        super().__init__(*args, **kwargs)
        self.stretched_gates = stretched_gates

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

    def simulate_subcircuit(self, job, subcircuit):
        self._simulate_subcircuit(job, subcircuit)
        subcircuit.simulated = True
        subcircuit.tree.simulated_probability = 1

    @abc.abstractmethod
    def _simulate_subcircuit(self, job, subcircuit):
        """(internal) Populate a subcircuit object with simulated data
        :param JaqalJob job: The job, describing the circuit and overrides
        :param SubcircuitResult subcirc: Data-carrying object of ExecutionResulsts
        """
        raise NotImplementedError()

    def _simulate_ci(self, ci, job):
        subcircuit = ci._subcircuit
        for _ in range(ci.num_repeats):
            node = subcircuit.tree
            while not node.classical_state.state == cursor.State.shutdown:
                keys = list(node.subsequent.keys())
                p = [node.subsequent[k].simulated_probability for k in keys]
                nxt = choice(keys, p=p)
                mr = result.Readout(nxt, job.meas_count, node)
                yield mr
                job.meas_count += 1
                node = node[nxt]

    def _execute_job(self, job):
        """(internal) Execute the job on the backend"""

        exe_res = result.ExecutionResult(job.expanded_circuit, job.overrides)
        for sb in exe_res.by_subbatch:
            for sc in sb.by_subcircuit:
                self.simulate_subcircuit(job, sc._subcircuit)

        parser = exe_res.accept_readouts()

        job.meas_count = 0
        for ci in exe_res.by_time:
            parser.pass_data(self._simulate_ci(ci, job))

        if not parser.done:
            raise JaqalError("Not enough readouts passed to ExecutionResults.")

        return exe_res
