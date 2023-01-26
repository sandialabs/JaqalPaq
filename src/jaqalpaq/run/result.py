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

    @property
    def readouts(self):
        """An indexable, iterable view of :class:`Readout` objects, containing the
        time-ordered measurements and auxiliary data."""
        return self._readouts

    @property
    def subcircuits(self):
        """An indexable, iterable view of the :class:`Subcircuit` objects, in
        :term:`flat order`, containing the readouts due to that subcircuit, as well as
        additional auxiliary data."""
        return self._subcircuits


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


class tree_accessor:
    def __init__(self, func):
        self._func = func

    def __get__(self, instance, owner=None):
        return TreeProxy(instance, self._func)


class TreeProxy:
    def __init__(self, subcircuit, func, prefix=()):
        self._subcirc = subcircuit
        self._func = func
        self._prefix = prefix
        self._target = None

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        sliced = False
        for i in index:
            if sliced:
                raise TypeError("Slice must be final")
            if isinstance(i, slice):
                sliced = True

        if sliced:
            return TreeProxy(
                self._subcirc, self._func, prefix=self._prefix + index[:-1]
            )

        return self._func(
            ReadoutTreeNode.deref(self._subcirc._tree, self._prefix + index)
        )

    def __iter__(self):
        if self._target is None:
            val = self._subcirc._tree
            for step in ReadoutTreeNode.deref(self._subcirc._tree, self._prefix):
                _, val = step
            self._target = val
        else:
            val = self._target
        return iter(val.subsequent.keys())

    def __repr__(self):
        ret = ["Tree{"]
        first = True
        for k in self:
            if first:
                first = False
            else:
                ret.append(", ")
            ret.append(f"{k:b}".zfill(self._target.bits)[::-1])
            ret.append(f": {self[k]}")
        ret.append("}")
        return "".join(ret)


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

    @property
    def readouts(self):
        """An indexable, iterable view of :class:`Readout` objects, containing the
        time-ordered measurements and auxiliary data, restricted to this Subcircuit."""
        return self._readouts

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

    def _prepare_readouts(self, new):
        new.normalized_count = 0

    def reset_readouts(self):
        self._readouts = []
        update_tree(self._prepare_readouts, self._tree)
        self._prepare_actions = [self._prepare_readouts]

    def accept_readout(self, readout):
        total = self.accept_readouts(iter((readout,)))
        if total != 1:
            raise JaqalError("Unable to accept readout")

    def accept_readouts(self, readouts):
        assert self._prepare_actions
        node = self._tree
        total = 0
        node.normalized_count += 1
        while not node.classical_state.state == State.shutdown:
            readout = next(readouts)
            readout._subcircuit = self
            readout._node = node
            self._readouts.append(readout)
            total += 1
            node = node.force_get(readout.as_int)
            node.normalized_count += 1
        return total

    @property
    def relative_frequency_by_int(self):
        """(deprecated) Return the relative frequency of each measurement result as a list,
        ordered by the integer representation of the result, with least significant bit
        representing qubit 0.  I.e., "000" for 0b000, "100" for 0b001, "010" for 0b010,
        etc.
        """
        import numpy

        qubits = len(self.measured_qubits)
        ret = numpy.zeros(2**qubits)
        for n, v in self._tree.subsequent.items():
            ret[n] = v.normalized_count
        return ret

    @property
    def relative_frequency_by_str(self):
        """(deprecated) Return the relative frequency associated with each measurement
        result as a dictionary mapping result strings to their respective probabilities.
        """
        qubits = len(self.measured_qubits)
        ret = OrderedDict()
        for n in range(2**qubits):
            k = f"{n:b}".zfill(qubits)[::-1]
            try:
                ret[k] = self._tree.subsequent[n].normalized_count
            except KeyError:
                ret[k] = 0
        return ret

    @property
    def probability_by_int(self):
        """(deprecated) synonym of relative_frequency_by_int (if experimental)
        or simulated_probability_by_int (if simulated)"""
        if self._simulated:
            return self.simulated_probability_by_int
        else:
            return self.relative_frequency_by_int

    @property
    def probability_by_str(self):
        """(deprecated) synonym of relative_frequency_by_str (if experimental)
        or simulated_probability_by_str (if simulated)"""
        if self._simulated:
            return self.simulated_probability_by_str
        else:
            return self.relative_frequency_by_str

    @property
    def simulated_probability_by_int(self):
        """Return the probability associated with each measurement result as a list,
        ordered by the integer representation of the result, with least significant bit
        representing qubit 0.  I.e., "000" for 0b000, "100" for 0b001, "010" for 0b010,
        etc.
        """
        import numpy

        qubits = len(self.measured_qubits)
        ret = numpy.zeros(2**qubits)
        for n, v in self._tree.subsequent.items():
            ret[n] = v.simulated_probability
        return ret

    @property
    def simulated_probability_by_str(self):
        """Return the probability associated with each measurement result formatted as a
        dictionary mapping result strings to their respective probabilities."""
        qubits = len(self.measured_qubits)
        ret = OrderedDict()
        for n in range(2**qubits):
            k = f"{n:b}".zfill(qubits)[::-1]
            try:
                ret[k] = self._tree.subsequent[n].simulated_probability
            except KeyError:
                ret[k] = 0
        return ret

    @tree_accessor
    def conditional_simulated_probabilities(path):
        path_iter = iter(path)
        while True:
            try:
                _, final = next(path_iter)
            except StopIteration:
                return final.simulated_probability
            except TreeAccessDefault:
                return 0.0

    @tree_accessor
    def simulated_probabilities(path):
        path_iter = iter(path)
        ret = 1
        while True:
            try:
                _, node = next(path_iter)
            except StopIteration:
                return ret
            except TreeAccessDefault:
                return 0.0
            else:
                ret *= node.simulated_probability

    @tree_accessor
    def num_repeats(path):
        path_iter = iter(path)
        _, ret = next(path_iter).num_repeats
        while True:
            try:
                _, node = next(path_iter)
            except StopIteration:
                assert abs(int(round(ret, 0)) - ret) < CUTOFF_ZERO
                return int(ret)
            except TreeAccessDefault:
                return 0
            else:
                ret *= node.normalized_count

    @tree_accessor
    def conditional_normalized_counts(path):
        path_iter = iter(path)
        while True:
            try:
                _, node = next(path_iter)
            except StopIteration:
                return node.normalized_count
            except TreeAccessDefault:
                return 0.0

    @tree_accessor
    def normalized_counts(path):
        path_iter = iter(path)
        ret = 1
        while True:
            try:
                _, node = next(path_iter)
            except StopIteration:
                return ret
            except TreeAccessDefault:
                return 0.0
            else:
                ret *= node.normalized_count

    @tree_accessor
    def state_vector(path):
        path_iter = iter(path)
        while True:
            try:
                _, node = next(path_iter)
            except StopIteration:
                return node.state_vector
            except TreeAccessDefault:
                raise AttributeError()

    @tree_accessor
    def state_vector_postmeasure(path):
        path_iter = iter(path)
        nxt = None
        while True:
            try:
                meas_val, node = next(path_iter)
            except StopIteration:
                pass
            except TreeAccessDefault as e:
                meas_val = e.measure
            else:
                continue
            return whatever(meas_val)


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
