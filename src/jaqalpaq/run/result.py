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
from jaqalpaq.core.locus import Locus
from jaqalpaq.error import JaqalError
from ._view import Accessor, ArrayAccessor, cachedproperty
from .cursor import SubcircuitCursor, State
from .classical_cursor import ClassicalCursor


# Warn if probabilities/normalized_counts are greater than CUTOFF_WARN, and
# throw an exception if it's above CUTOFF_FAIL.
CUTOFF_FAIL = 2e-6
CUTOFF_WARN = 1e-13
CUTOFF_ZERO = 1e-13
assert CUTOFF_WARN <= CUTOFF_FAIL
assert CUTOFF_ZERO <= CUTOFF_WARN


class Acceptor:
    __slots__ = ("done", "_coroutine")

    def __init__(self, *args, **kwargs):
        # Hide the actual function inside a tuple to avoid munging it
        self._coroutine = self._func[0](self._instance, *args, **kwargs)
        try:
            self._coroutine.send(None)
        except StopIteration:
            # This happens if the circuit is empty.
            self.done = True
        else:
            self.done = False

    def pass_data(self, data):
        for datum in data:
            try:
                self._coroutine.send(datum)
            except StopIteration:
                if self.done:
                    raise JaqalError("Coroutine ended early")
                self.done = True

    def pass_datum(self, datum):
        self.pass_data((datum,))


class acceptor:
    __slots__ = ("func",)

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner=None):
        func = self.func
        return type(
            func.__name__,
            (Acceptor,),
            dict(__slots__=(), _instance=instance, _func=(func,)),
        )


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
    exe_res = ExecutionResult(expand_macros(circuit), overrides)

    parser = exe_res.accept_readouts()

    readout_i = 0
    for datum in output:
        if isinstance(datum, str):
            datum = int(datum[::-1], 2)
        readout = Readout(datum, readout_i, None)
        readout_i += 1

        parser.pass_datum(readout)

    if not parser.done:
        raise JaqalError("Too many readouts passed to ExecutionResults.")

    return exe_res


class ExecutionResult:
    "Captures the results of a Jaqal program's execution, on hardware or an emulator."

    def __init__(self, circuit, overrides, *, timestamp=None, **kwargs):
        """(internal) Initializes an ExecutionResult object.

        :param Circuit circuit:  The circuit for which these results will represent.
        :param dict overrides: The overrides, if any, made to the Jaqal circuit.
        :param float timestamp: The time at which these results were generated.
        :param list[Readout] output:  The measurements made during the running of the
            Jaqal problem.
        """
        self.time = time.time() if timestamp is None else timestamp
        self._circuit = circuit
        subcirc_loci = self._subcircuits_loci = list(discover_subcircuits(circuit))
        sno = list(walk_circuit(circuit, [start for (start, end) in subcirc_loci]))

        self._classical_cursor = cc = ClassicalCursor(overrides, subcirc_list=sno)
        if kwargs:
            raise JaqalError(
                "Setting _attributes through the constructor is not supported"
            )
        self._attributes = kwargs

        self._subcircuits = subcircs = []
        for sb in cc.by_subbatch:
            filled_circ = fill_in_let(circuit, override_dict=sb.overrides)
            scs = {}
            subcircs.append(scs)
            for sc in sb.by_subcircuit:
                sc_i = sc.subcircuit_i
                sc_start, sc_end = subcirc_loci[sc_i]
                # Need this to reference filled_circ
                sc_start = Locus.from_address(filled_circ, sc_start.address)
                if sc_start is not sc_end:
                    sc_end = Locus.from_address(filled_circ, sc_end.address)
                else:
                    sc_end = sc_start
                scs[sc_i] = SubcircuitResult(
                    sc_i, sc_start, sc_end, sb.index, len(sc.by_time), circuit
                )

    def _repr_pretty_(self, printer, cycle=False):
        printer.text(f"<ExecutionResult@{self._repr_time()} of ")
        printer.pretty(self._circuit)
        printer.text(">")

    def _repr_time(self):
        return f"{time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime(self.time))}.{int((self.time%1)*1000000)}Z"

    def __repr__(self):
        return f"<ExecutionResult@{self._repr_time()} of {self._circuit}>"

    @property
    def subcircuits_loci(self):
        return self._subcircuits_loci

    @property
    def readouts(self):
        """An indexable, iterable view of :class:`Readout` objects, containing the
        time-ordered measurements and auxiliary data."""
        lst = []
        for sb in self._classical_cursor.by_subbatch:
            per_sb = {
                k: iter(v) for k, v in self._attributes["readouts"][sb.index].items()
            }
            for ci in sb.by_time:
                lst.append(next(per_sb[ci.subcircuit_i]))
            for k, v in per_sb.items():
                try:
                    next(v)
                except StopIteration:
                    pass
                else:
                    raise JaqalError("Not all subcircuit processed!")

        return lst

    @cachedproperty
    def subcircuits(self):
        """(deprecated) An indexable, iterable view of the :class:`DeprecatedSubcircuitView`
        objects in :term:`flat order`, containing the readouts due to that subcircuit, as well as
        additional auxiliary data.

        This interface is not compatible with batched results.
        """
        (sb,) = self._subcircuits
        return [DeprecatedSubcircuitView(self, sc) for sc in sb.values()]

    class by_subbatch(ArrayAccessor):
        @ArrayAccessor.getitem
        def __getitem__(self, subbatch_i):
            return SubbatchView(self, subbatch_i)

        def __len__(self):
            return len(self._subcircuits)

    class by_time(ArrayAccessor):
        @ArrayAccessor.getitem
        def __getitem__(self, time_i):
            cc = self._classical_cursor
            sb_i, per_sb_time_i = cc.get_sb_i_from_time_i(time_i)
            return CircuitIndexView(
                self, cc.by_subbatch[sb_i].by_time[per_sb_time_i], time_i=time_i
            )

        def __len__(self):
            return len(self._classical_cursor.by_time)

    @acceptor
    def accept_readouts(self):
        for sb in self._subcircuits:
            for sc in sb.values():
                sc._reset_normalized_counts()

        readout_i = 0
        assert self._attributes.get("readouts", None) is None
        readouts = self._attributes["readouts"] = []
        for sb in self.by_subbatch:
            readouts_for_sb = {}
            readouts.append(readouts_for_sb)
            for ci in sb.by_time:
                sc = ci._subcircuit
                readouts_for_subcirc = readouts_for_sb.get(sc.index, None)
                if readouts_for_subcirc is None:
                    readouts_for_subcirc = readouts_for_sb[sc.index] = []
                for _ in range(ci.num_repeats):
                    subcirc_accept = sc.accept_readouts(ci._ci.per_sc_time_i)
                    while not subcirc_accept.done:
                        readout = yield
                        subcirc_accept.pass_datum(readout)
                        readouts_for_subcirc.append(readout)

            for subcirc in sb.by_subcircuit:
                subcirc._subcircuit.normalize_counts()

    @acceptor
    def accept_normalized_counts(self):
        for sb in self._subcircuits:
            for sc in sb.values():
                sc._reset_normalized_counts()
                sc._tree.normalized_count[:] = 1

        assert self._attributes.get("normalized_count", None) is None
        freqs = self._attributes["normalized_count"] = []
        for sb in self.by_subbatch:
            freqs_for_sb = {}
            freqs.append(freqs_for_sb)
            for ci in sb.by_time:
                sc_i = ci.subcircuit_i
                freqs_for_subcirc = freqs_for_sb.get(sc_i, None)
                if freqs_for_subcirc is None:
                    freqs_for_subcirc = freqs_for_sb[sc_i] = []
                subcirc_accept = ci._subcircuit.accept_normalized_counts(
                    ci._ci.per_sc_time_i
                )
                while not subcirc_accept.done:
                    freq = yield
                    subcirc_accept.pass_datum(freq)
                    freqs_for_subcirc.append(freq)

            for sc in sb.by_subcircuit:
                sc._subcircuit._prepare_actions = []


class SubbatchView:
    def __init__(self, result, subbatch_i):
        self.result = result
        self.index = subbatch_i

    class by_subcircuit(Accessor):
        def __getitem__(self, subcircuit_i):
            return SubcircuitView(
                self.result, self.result._subcircuits[self.index][subcircuit_i]
            )

        def __len__(self):
            return len(self.result._subcircuits[self.index])

        def keys(self):
            return self.result._subcircuits[self.index].values()

        @Accessor.direct
        def __iter__(self):
            for sc_i in self.keys():
                yield SubcircuitView(self.instance.result, sc_i)

    class by_time(ArrayAccessor):
        @ArrayAccessor.getitem
        def __getitem__(self, per_sb_time_i):
            cc = self.result._classical_cursor
            return CircuitIndexView(
                self.result, cc.by_subbatch[self.index].by_time[per_sb_time_i]
            )

        def __len__(self):
            return len(self.result._classical_cursor.by_subbatch[self.index].by_time)

    def _repr_pretty_(self, printer, cycle=False):
        printer.text(f"<Subbatch {self.index}/{len(self.result._subcircuits)} of ")
        printer.pretty(self.result)
        printer.text(">")

    def __repr__(self):
        return (
            f"<Subbatch {self.index}/{len(self.result._subcircuits)} of {self.result}>"
        )


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
        return DeprecatedSubcircuitView(self._result, self._subcircuit)

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


class AbstractTreeAccessor(Accessor):
    __slots__ = ("_prefix", "_target")

    def __init__(self, instance, owner=None, *, target=None, prefix=()):
        self._prefix = prefix
        Accessor.__init__(self, instance, owner=owner)

    def _func(self, index):
        raise NotImplementedError()

    @Accessor.direct
    def _getattr(self, node):
        return getattr(node, self.attrname)

    @Accessor.direct
    def walk(self, index):
        return ReadoutTreeNode.deref(
            self.instance._subcircuit._tree, self._prefix + index
        )

    @Accessor.direct
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        sliced = False
        for i in index:
            if sliced:
                raise TypeError(
                    "JaqalPaq: Slices are only supported on the final index"
                )
            if isinstance(i, slice):
                sliced = True

        if not sliced:
            return self._func(index)

        if len(index) == 1:
            return self

        return self.add_to_prefix(index[:-1])

    @Accessor.direct
    def add_to_prefix(self, extend):
        return type(self)(self.instance, prefix=self._prefix + extend)

    @Accessor.direct
    @property
    def by_int(self):
        return self

    @Accessor.direct
    @property
    def by_str(self):
        return {f"{k:b}".zfill(self.target.bits)[::-1]: v for k, v in self.items()}

    @Accessor.direct
    @property
    def by_int_dense(self):
        # Don't require numpy in the experiment
        import numpy

        bits = self.target.bits
        return numpy.array([self[k] for k in range(1 << bits)])

    @Accessor.direct
    @property
    def by_str_dense(self):
        bits = self.target.bits
        ret = {}
        for k in range(1 << bits):
            ret[f"{k:b}".zfill(self.target.bits)[::-1]] = self[k]
        return ret

    @Accessor.direct
    @property
    def target(self):
        try:
            return self._target
        except AttributeError:
            pass

        for _, value in ReadoutTreeNode.deref(
            self.instance._subcircuit._tree, self._prefix
        ):
            pass

        self._target = value
        return value

    @Accessor.direct
    def __iter__(self):
        return iter(self.target.subsequent.keys())

    @Accessor.direct
    def items(self):
        target = self.target
        for k in target.subsequent.keys():
            yield k, self[k]

    @Accessor.direct
    def __repr__(self):
        ret = ["Tree{"]
        first = True
        for k, v in self.items():
            if first:
                first = False
            else:
                ret.append(", ")
            ret.append(f"{k:b}".zfill(self.target.bits)[::-1])
            ret.append(f": {v}")
        ret.append("}")
        return "".join(ret)


class CumulativeTreeAccessor(AbstractTreeAccessor):
    @Accessor.direct
    def _func(self, index):
        ret = self.start
        try:
            for _, node in self.walk(index):
                ret = self.reduce(ret, self._getattr(node))
        except TreeAccessDefault:
            return self.default
        return ret


class TreeAccessor(AbstractTreeAccessor):
    @Accessor.direct
    def _func(self, index):
        try:
            for _, final in self.walk(index):
                continue
        except TreeAccessDefault:
            return self.default
        return self._getattr(final)


def update_tree(update_node, tree):
    def _inner(node):
        if node.subsequent:
            for child in node.subsequent.values():
                _inner(child)
        update_node(node)

    _inner(tree)


class SubcircuitResult:
    """(internal) Encapsulate results from the part of a circuit between a prepare_all and measure_all gate."""

    def __init__(self, index, start, end, subbatch_i, circuitindex_c, circuit):
        """(internal) Instantiate a Subcircuit"""
        self._start = start
        if isinstance(start.object, BlockStatement) and start.object.subcircuit:
            assert end == start
        self._end = end
        self._index = int(index)
        self.circuitindex_c = circuitindex_c
        self._circuit = circuit
        try:
            _head = next(iter(start.lineage))
        except StopIteration as exc:
            raise JaqalError("Start of subcircuit has empty lineage") from exc
        self._filled_circuit = _head.object
        tree = self._tree = ReadoutTreeNode(SubcircuitCursor.terminal_cursor(end))
        tree._owner = self
        self._prepare_actions = []
        self.simulated = False
        self.subbatch_i = subbatch_i

    @property
    def index(self):
        return self._index

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def circuit(self):
        return self._circuit

    @property
    def filled_circuit(self):
        """
        The circuit object, with any overriden let values applied, for which
        this SubcircuitResult object contains data.
        """
        return self._filled_circuit

    @property
    def tree(self):
        return self._tree

    @property
    def measured_qubits(self):
        """A list of the qubits that are measured, in their display order."""
        qubits = []
        for reg in self.filled_circuit.registers.values():
            qubits.extend(reg)
        return reg

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
        self._tree.normalized_count[:] = 1

    def _prepare_tree_node(self, new):
        for action in self._prepare_actions:
            action(new)

    def _result_array(self):
        import numpy

        return numpy.zeros(self.circuitindex_c)

    def _prepare_normalized_count(self, new):
        new.normalized_count = self._result_array()

    def _reset_normalized_counts(self):
        update_tree(self._prepare_normalized_count, self._tree)
        self._prepare_actions = [self._prepare_normalized_count]

    @acceptor
    def accept_readouts(self, time_i):
        assert self._prepare_actions
        node = self._tree
        node.normalized_count[time_i] += 1
        while not node.classical_state.state == State.shutdown:
            readout = yield
            readout._subcircuit = self
            readout._node = node
            node = node.force_get(readout.as_int)
            node.normalized_count[time_i] += 1

    @acceptor
    def accept_normalized_counts(self, time_i):
        assert self._prepare_actions
        data = yield
        for meas_as_int, nc in enumerate(data):
            node = self._tree.force_get(meas_as_int)
            assert node.classical_state.state == State.shutdown
            node.normalized_count[time_i] += nc


class BackwardsCompatibleView:
    @property
    def _probabilities(self):
        if self._subcircuit.simulated:
            return self._simulated_probabilities
        else:
            return self._relative_frequencies

    @property
    def probability_by_int(self):
        return self._probabilities.by_int_dense

    @property
    def probability_by_str(self):
        return self._probabilities.by_str_dense

    @property
    def simulated_probability_by_int(self):
        return self._simulated_probabilities.by_int_dense

    @property
    def simulated_probability_by_str(self):
        return self._simulated_probabilities.by_str_dense

    @property
    def relative_frequency_by_int(self):
        return self._relative_frequencies.by_int_dense

    @property
    def relative_frequency_by_str(self):
        return self._relative_frequencies.by_str_dense

    @property
    def _pygsti_circuit(self):
        return self._subcircuit._pygsti_circuit

    @property
    def _pygsti_model(self):
        return self._subcircuit._pygsti_model


class SubcircuitView:
    def __init__(self, result, subcircuit):
        self._subcircuit = subcircuit
        self.result = result

    class by_time(ArrayAccessor):
        @ArrayAccessor.getitem
        def __getitem__(self, per_sc_time_i):
            cc = self.result._classical_cursor
            return CircuitIndexView(
                self.result,
                cc.by_subbatch[self.subbatch.index]
                .by_subcircuit[self.index]
                .by_time[per_sc_time_i],
            )

        def __len__(self):
            cc = self.result._classical_cursor
            return len(cc.by_subbatch[self.subbatch_i].by_subcircuit[self.index])

    def _repr_pretty_(self, printer, cycle=False):
        sc = self._subcircuit
        printer.text(f"<SubcircuitView {sc._index}@{sc._start} of ")
        printer.pretty(sc.circuit)
        printer.text(">")

    def __repr__(self):
        sc = self._subcircuit
        return f"SubcircuitView {sc._index}@{sc._start} of {sc.circuit}>"

    @property
    def index(self):
        return self._subcircuit.index

    @property
    def subbatch_i(self):
        return self._subcircuit.subbatch_i

    @property
    def subbatch(self):
        return self.result.by_subbatch[self.subbatch_i]

    @property
    def num_repeats(self):
        return [ci.num_repeats for ci in self.by_time]

    class simulated_probabilities(CumulativeTreeAccessor):
        attrname = "simulated_probability"

        @property
        def default(self):
            return self._subcircuit._result_array()

        @property
        def start(self):
            ret = self._subcircuit._result_array()
            ret[:] = 1
            return ret

        def reduce(self, cur, nxt):
            return cur * nxt

    class conditional_simulated_probabilities(TreeAccessor):
        attrname = "simulated_probability"

        @property
        def default(self):
            return self._subcircuit._result_array()

        @Accessor.direct
        def _func(self, index):
            wide = self.default
            wide[:] = 1
            return super()._func(index) * wide

    class normalized_counts(CumulativeTreeAccessor):
        attrname = "normalized_count"

        @property
        def default(self):
            return self._subcircuit._result_array()

        @property
        def start(self):
            ret = self._subcircuit._result_array()
            ret[:] = 1
            return ret

        def reduce(self, cur, nxt):
            return cur * nxt

        class by_time(ArrayAccessor):
            def __len__(self):
                return len(self.instance.by_time)

            @ArrayAccessor.getitem
            def __getitem__(self, i):
                return self.instance.by_time[i].normalized_counts

    class conditional_normalized_counts(TreeAccessor):
        attrname = "normalized_count"
        default = 0.0

        class by_time(ArrayAccessor):
            def __len__(self):
                return len(self.instance.by_time)

            @ArrayAccessor.getitem
            def __getitem__(self, i):
                return self.instance.by_time[i].conditional_normalized_counts


class DeprecatedSubcircuitView(SubcircuitView, BackwardsCompatibleView):
    class _relative_frequencies(CumulativeTreeAccessor):
        # The existing API collected all prior runs of the subcircuit and gave the
        # relative frequency of all of those runs.
        attrname = "normalized_count"
        default = 0.0
        start = 1.0

        def reduce(self, cur, nxt):
            (l,) = nxt.shape
            return cur * nxt.sum() / l

    class _simulated_probabilities(CumulativeTreeAccessor):
        # The existing API did not distinguish different circuit indexes
        attrname = "simulated_probability"
        default = 0.0
        start = 1.0

        def reduce(self, cur, nxt):
            return cur * nxt

    @property
    def readouts(self):
        """An indexable, iterable view of :class:`Readout` objects, containing the
        time-ordered measurements and auxiliary data."""
        # Readouts came as an actual list
        return self.result._attributes["readouts"][self.subbatch_i][self.index]

    @property
    def measured_qubits(self):
        return self._subcircuit.measured_qubits


class AttributeView:
    __slots__ = ("attribute", "attribute_name", "_subcircuit")

    def __init__(self, subcircuit, attribute_name, attribute):
        self._subcircuit = subcircuit
        self.attribute_name = attribute_name
        self.attribute = attribute

    class by_time(Accessor):
        def __getitem__(self, time_i):
            sc = self._subcircuit
            return self.attribute[sc.subbatch_i][sc._index][time_i]

        def __len__(self):
            sc = self._subcircuit
            return len(self.attribute[sc.subbatch_i][sc._index])

        def __iter__(self):
            sc = self._subcircuit
            yield from self.attribute[sc.subbatch_i][sc._index]


class CITreeAccessor(AbstractTreeAccessor):
    @Accessor.direct
    def _getattr(self, node):
        return getattr(node, self.attrname)[self.instance.per_subcircuit_time_index]


class CircuitIndexView:
    def __init__(self, result, circuitindex, *, time_i=None):
        self.result = result
        self._ci = circuitindex
        sb_i = circuitindex.subbatch.index
        sc_i = circuitindex.subcircuit_i
        self._subcircuit = result._subcircuits[sb_i][sc_i]
        if time_i is not None:
            self.__dict__["time_i"] = time_i

    def _repr_pretty_(self, printer, cycle=False):
        ci = self._ci
        printer.text(
            f"<CircuitIndexView {ci.subbatch.index}/{ci.subcircuit_i}:{self.per_subcircuit_time_index} ({self.time_i}) of "
        )
        printer.pretty(self.result)
        printer.text(">")

    def __repr__(self):
        ci = self._ci
        return f"<CircuitIndexView {ci.subbatch.index}/{ci.subcircuit_i}:{self.per_subcircuit_time_index} ({self.time_i}) of {self.result}>"

    @property
    def num_repeats(self):
        shots = self._ci.subbatch.get_override("__repeats__")
        if shots is not None:
            return shots

        start_obj = self._subcircuit._start.object
        if isinstance(start_obj, BlockStatement):
            assert start_obj.subcircuit
            shots = start_obj.iterations
        else:
            # This global setting does not reflect the typical behavior of
            # QSCOUT hardware, which is by default much closer to 100
            # repeats.  Changing it here, however, breaks API, so we will
            # wait until 2.0 to set this to a more reasonable default.
            # You can control this by setting the __repeats__ override.
            shots = 1
        return shots

    @property
    def subcircuit(self):
        return SubcircuitView(self.result, self._subcircuit)

    @property
    def subcircuit_i(self):
        return self._ci.subcircuit_i

    class simulated_probabilities(CumulativeTreeAccessor):
        attrname = "simulated_probability"
        default = 0.0
        start = 1.0

        def reduce(self, cur, nxt):
            return cur * nxt

    class conditional_simulated_probabilities(TreeAccessor):
        attrname = "simulated_probability"
        default = 0.0

    class normalized_counts(CumulativeTreeAccessor, CITreeAccessor):
        attrname = "normalized_count"
        default = 0.0
        start = 1.0

        def reduce(self, cur, nxt):
            return cur * nxt

    class conditional_normalized_counts(TreeAccessor, CITreeAccessor):
        attrname = "normalized_count"
        default = 0.0

    @cachedproperty
    def per_subcircuit_time_index(self):
        ci = self._ci
        return self.result._classical_cursor.get_per_sc_time_i(
            ci.subbatch.index, ci.subcircuit_i, ci.per_sb_time_i
        )

    @cachedproperty
    def time_i(self):
        return self._ci.time_i


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
