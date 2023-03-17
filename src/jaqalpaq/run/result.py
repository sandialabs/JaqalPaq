# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import warnings
from collections import OrderedDict
import time

from jaqalpaq.core.branch import BranchStatement
from jaqalpaq.core.block import BlockStatement
from jaqalpaq.core.algorithm import fill_in_let, expand_macros
from jaqalpaq.core.algorithm.visitor import Visitor
from jaqalpaq.core.algorithm.walkers import walk_circuit, discover_subcircuits
from jaqalpaq.error import JaqalError
from jaqalpaq.run.cursor import SubcircuitCursor, State
from .cursor import SubcircuitCursor


# Warn if probabilities/normalized_counts are greater than CUTOFF_WARN, and
# throw an exception if it's above CUTOFF_FAIL.
CUTOFF_FAIL = 2e-6
CUTOFF_WARN = 1e-13
CUTOFF_ZERO = 1e-13
assert CUTOFF_WARN <= CUTOFF_FAIL
assert CUTOFF_ZERO <= CUTOFF_WARN


def parse_jaqal_output_list(circuit, output, overrides=None):
    """Parse experimental output into an :class:`ExecutionResult` providing collated and
    uncollated access to the output.

    :param Circuit circuit: The circuit under consideration.
    :param output: The measured qubit state, encoded as a string of 1s and 0s, or as an
        int with state of qubit 0 encoded as the least significant bit, and so on.
        For example, Measuring ``100`` is encoded as 1, and ``001`` as 4.
    :type output: list[int or str]
    :returns: The parsed output.
    :rtype: ExecutionResult
    """
    circuit = expand_macros(fill_in_let(circuit))

    subcircuits = []
    for n, (sc, sc_end) in enumerate(discover_subcircuits(circuit)):
        subcircuit = SubcircuitResult(n, sc, sc_end, circuit)
        subcircuit.reset_readouts()
        subcircuits.append(subcircuit)

    res = []
    breaks = [t._start for t in subcircuits]

    def _emit_readouts():
        data = iter(output)
        readout_index = 0
        while True:
            try:
                nxt = next(data)
            except StopIteration:
                return
            if isinstance(nxt, str):
                nxt = int(nxt[::-1], 2)
            mr = Readout(nxt, readout_index, None)
            readout_index += 1
            res.append(mr)
            yield mr

    emitter = _emit_readouts()

    for readout_index, index in enumerate(walk_circuit(circuit, breaks)):
        subcircuit = subcircuits[index]
        try:
            subcircuit.accept_readouts(emitter)
        except StopIteration:
            raise JaqalError("Unable to parse output: too few values")

    try:
        next(emitter)
    except StopIteration:
        pass
    else:
        raise JaqalError("Unable to parse output: too many values")

    return ExecutionResult(circuit, subcircuits, overrides, res)


class ExecutionResult:
    "Captures the results of a Jaqal program's execution, on hardware or an emulator."

    def __init__(self, circuit, subcircuits, overrides, readouts=None, *, timestamp=None):
        """(internal) Initializes an ExecutionResult object.

        :param Circuit circuit:  The circuit for which these results will represent.
        :param list[Subcircuit] output:  The subcircuits bounded at the beginning by a
            prepare_all statement, and at the end by a measure_all statement.
        :param dict overrides: The overrides, if any, made to the Jaqal circuit.
        :param list[Readout] output:  The measurements made during the running of the
            Jaqal problem.
        """
        self._circuit = circuit
        self.time = time.time() if timestamp is None else timestamp
        self._subcircuits = subcircuits
        self._readouts = readouts

    def _repr_pretty_(self, printer, cycle=False):
        printer.text(f"<ExecutionResult@{self._repr_time()} of ")
        printer.pretty(self._circuit)
        printer.text(">")

    def _repr_time(self):
        return f"{time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime(self.time))}.{int((self.time%1)*1000000)}Z"

    def __repr__(self):
        return f"<ExecutionResult@{self._repr_time()} of {self._circuit}>"


class Readout:
    """Encapsulate the result of measurement of some number of qubits."""

    def __init__(self, result, index, node):
        """(internal) Instantiate a Readout object

        Contains the actual results of a measurement.
        """
        self._result = result
        self._index = index
        self._node = node

    @property
    def index(self):
        """The temporal index of this measurement in the parent circuit."""
        return self._index

    @property
    def subcircuit(self):
        """Return the associated prepare_all/measure_all block in the parent circuit."""
        return self._subcircuit

    @property
    def as_int(self):
        """The measured result encoded as an integer, with qubit 0 represented by the
        least significant bit."""
        return self._result

    @property
    def as_str(self):
        """The measured result encoded as a string of qubit values."""
        return f"{self._result:b}".zfill(self._node.bits)[::-1]

    def _repr_pretty_(self, printer, cycle=False):
        printer.text(f"<Readout {self.as_str} index {self._index} from ")
        printer.pretty(self._subcircuit)
        printer.text(">")

    def __repr__(self):
        return f"<Readout {self.as_str} index {self._index} from {self._subcircuit}>"


class ReadoutTreeNode:
    def __init__(self, classical_state=None, subsequent=None):
        self.classical_state = classical_state
        if subsequent is None:
            self.subsequent = {}
        else:
            self.subsequent = subsequent

    def __getitem__(self, index):
        if isinstance(index, str):
            index = int(index[::-1], 2)

        return self.subsequent[index]

    def force_get(self, index, cursor=None):
        try:
            return self[index]
        except KeyError:
            return self._spawn(index, cursor=cursor)

    def _spawn(self, index, cursor=None):
        if cursor is None:
            cursor = self.classical_state.copy()
            cursor.next_measure()
            if not cursor.state == State.shutdown:
                cursor.report_measurement(index)
                while cursor.state == State.gate:
                    _ = cursor.next_gate()
                    cursor.report_gate_executed()

        new = self.subsequent[index] = ReadoutTreeNode(cursor)
        owner = new._owner = self._owner
        owner._prepare_tree_node(new)
        return new

    @property
    def bits(self):
        all_qubits = len(self._owner.measured_qubits)

        if hasattr(self.classical_state.locus.object, "iterations"):
            return all_qubits

        gd = self.classical_state.locus.object.gate_def
        if gd.name == "measure_all":
            return all_qubits
        else:
            raise NotImplementedError(f"Unsupported measurement {gd.name}")

    @staticmethod
    def deref(tree, index):
        yield None, tree

        for n, i in enumerate(index):
            if isinstance(i, str):
                i = int(i[::-1], 2)
            elif not isinstance(i, int):
                raise TypeError(
                    f"JaqalPaq: lookup address must by integers or strings, not {type(i)}"
                )

            try:
                tree = tree.subsequent[i]
            except KeyError:
                if n == len(index) - 1:
                    raise TreeAccessDefault()
                else:
                    raise TreeAccessOutOfBounds()
            else:
                yield i, tree


class TreeAccessOutOfBounds(JaqalError):
    pass


class TreeAccessDefault(JaqalError):
    pass


def update_tree(update_node, tree):
    def _inner(node):
        if node.subsequent:
            for child in node.subsequent.values():
                _inner(child)
        update_node(node)

    _inner(tree)


class SubcircuitResult:
    """(internal) Encapsulate results from the part of a circuit between a prepare_all and measure_all gate."""

    def __init__(self, index, start, end, circuit, *, tree=None):
        """(internal) Instantiate a Subcircuit"""
        self._start = start
        if isinstance(start.object, BlockStatement) and start.object.subcircuit:
            assert end == start
        self._end = end
        self._index = int(index)
        self._circuit = circuit
        if tree is not None:
            self._tree = tree
            update_tree(lambda node: setattr(node, "_owner", self), self._tree)
        else:
            tree = self._tree = ReadoutTreeNode(SubcircuitCursor.terminal_cursor(end))
            tree._owner = self
        tree.simulated_probability = tree.normalized_count = 1
        self._prepare_actions = []
        self._simulated = False

    @property
    def index(self):
        return self._index

    @property
    def circuit(self):
        return self._circuit

    @property
    def measured_qubits(self):
        """A list of the qubits that are measured, in their display order."""
        try:
            (reg,) = next(iter(self._start.lineage)).object.registers.values()
        except ValueError as exc:
            raise JaqalError("Circuits must have exactly one register") from exc
        return list(reg)

    def _repr_pretty_(self, printer, cycle=False):
        printer.text(f"<SubcircuitResult {self._index}@{self._start} of ")
        printer.pretty(self.circuit)
        printer.text(">")

    def __repr__(self):
        return f"<SubcircuitResult {self._index}@{self._start} of {self.circuit}>"

    def copy(self):
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def normalize_counts(self):
        assert self._prepare_actions
        self._prepare_actions = []

        def _update(node):
            for child in node.subsequent.values():
                child.normalized_count /= node.normalized_count

        update_tree(_update, self._tree)
        self._tree.num_repeats = self._tree.normalized_count
        self._tree.normalized_count = 1

    def _prepare_tree_node(self, new):
        for action in self._prepare_actions:
            action(new)


def validate_probabilities(probabilities):
    # Don't require numpy in the experiment
    import numpy

    p = numpy.asarray(probabilities)

    # We normalize the probabilities if they are outside the range [0,1]
    p_clipped = numpy.clip(p, 0, 1)
    clip_err = numpy.abs(p_clipped - p).max()

    p = p_clipped

    # We also normalize their sum.
    total = p.sum()
    total_err = numpy.abs(total - 1)
    if total_err > 0:
        p /= total

    #
    # This is done to account for minor numerical error. If the change
    # is significant, you should be suspicious of the results.
    #

    err = max(total_err, clip_err)
    if err > CUTOFF_WARN:
        msg = f"Error in probabilities {err}"
        if err > CUTOFF_FAIL:
            raise JaqalError(msg)
        warnings.warn(msg, category=RuntimeWarning)

    return p


__all__ = [
    "ExecutionResult",
    "parse_jaqal_output_list",
    "Subcircuit",
    "Readout",
]
