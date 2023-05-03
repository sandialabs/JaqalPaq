# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import warnings
from collections import OrderedDict

from .algorithm import fill_in_let, expand_macros
from .algorithm.walkers import *
from jaqalpaq.error import JaqalError


# Warn if a probability is greater than CUTOFF_WARN, and
# throw an exception if it's above CUTOFF_FAIL.
CUTOFF_FAIL = 2e-6
CUTOFF_WARN = 1e-13
assert CUTOFF_WARN <= CUTOFF_FAIL


def parse_jaqal_output_list(circuit, output):
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
    visitor = DiscoverSubcircuits()
    traces = visitor.visit(circuit)

    subcircuits = [ReadoutSubcircuit(sc, n) for n, sc in enumerate(traces)]
    res = []
    readout_index = 0
    data = iter(output)
    breaks = [t.start for t in traces]

    for readout_index, index in enumerate(walk_circuit(circuit, breaks)):
        subcircuit = subcircuits[index]
        try:
            nxt = next(data)
        except StopIteration:
            raise JaqalError("Unable to parse output: too few values")
        if isinstance(nxt, str):
            nxt = int(nxt[::-1], 2)
        mr = Readout(nxt, readout_index)
        subcircuit.accept_readout(mr)
        res.append(mr)

    try:
        next(data)
    except StopIteration:
        pass
    else:
        raise JaqalError("Unable to parse output: too many values")

    return ExecutionResult(subcircuits, res)


class ExecutionResult:
    "Captures the results of a Jaqal program's execution, on hardware or an emulator."

    def __init__(self, subcircuits, readouts=None):
        """(internal) Initializes an ExecutionResult object.

        :param list[Subcircuit] output:  The subcircuits bounded at the beginning by a
            prepare_all statement, and at the end by a measure_all statement.
        :param list[Readout] output:  The measurements made during the running of the
            Jaqal problem.

        """
        self._subcircuits = subcircuits
        self._readouts = readouts

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

    def __init__(self, result, index):
        """(internal) Instantiate a Readout object

        Contains the actual results of a measurement.
        """
        self._result = result
        self._index = index

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
        return f"{self._result:b}".zfill(len(self.subcircuit.measured_qubits))[::-1]

    def __repr__(self):
        return f"<{type(self).__name__} {self.as_str} index {self._index} from {self._subcircuit.index}>"


class Subcircuit:
    """Encapsulate one part of the circuit between a prepare_all and measure_all gate."""

    def __init__(self, trace, index):
        """(internal) Instantiate a Subcircuit"""
        self._trace = trace
        self._index = int(index)

    @property
    def index(self):
        """The :term:`flat order` index of this object in the (unrolled) parent circuit."""
        return self._index

    @property
    def measured_qubits(self):
        """A list of the qubits that are measured, in their display order."""
        return self._trace.used_qubits

    def __repr__(self):
        return f"<{type(self).__name__} {self._index}@{self._trace.end}>"

    def copy(self):
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new


class RelativeFrequencySubcircuit(Subcircuit):
    """Encapsulate one part of the circuit between a prepare_all and measure_all gate.

    Additionally, track the relative frequencies of each measurement result.
    """

    def __init__(self, *args, normalized_counts=None, **kwargs):
        super().__init__(*args, **kwargs)
        if normalized_counts is not None:
            self._normalized_counts = validate_probabilities(normalized_counts)

    @property
    def relative_frequency_by_int(self):
        """Return the relative frequency of each measurement result as a list,
        ordered by the integer representation of the result, with least significant bit
        representing qubit 0.  I.e., "000" for 0b000, "100" for 0b001, "010" for 0b010,
        etc.
        """
        return self._normalized_counts

    @property
    def relative_frequency_by_str(self):
        """Return the relative frequency associated with each measurement result formatted as a
        dictionary mapping result strings to their respective probabilities."""
        qubits = len(self._trace.used_qubits)
        rf = self._normalized_counts
        return OrderedDict(
            [(f"{n:b}".zfill(qubits)[::-1], v) for n, v in enumerate(rf)]
        )

    @property
    def probability_by_int(self):
        """(deprecated) Return the relative frequency of each measurement result as a list,
        ordered by the integer representation of the result, with least significant bit
        representing qubit 0.  I.e., "000" for 0b000, "100" for 0b001, "010" for 0b010,
        etc.

        Use relative_frequency_by_int.
        """
        return self.relative_frequency_by_int

    @property
    def probability_by_str(self):
        """(deprecated) Return the relative frequency associated with each measurement result formatted as a
        dictionary mapping result strings to their respective probabilities.

        Use relative_frequency_by_str.
        """
        return self.relative_frequency_by_str


class ReadoutSubcircuit(RelativeFrequencySubcircuit):
    """Encapsulate one part of the circuit between a prepare_all and measure_all gate.

    Additionally, track the specific measurement results, and their relative frequencies.
    """

    def __init__(self, trace, index, **kwargs):
        # Don't require numpy in the experiment
        import numpy

        assert "normalized_counts" not in kwargs
        assert "readouts" not in kwargs

        self._readouts = []
        super().__init__(trace, index, normalized_counts=None, **kwargs)
        self._counts = numpy.zeros(2 ** len(self.measured_qubits))

    def _recalculate_counts(self):
        try:
            del self._normalized_counts
        except AttributeError:
            pass

        counts = self._counts
        counts[:] = 0
        for ro in self.readouts:
            counts[ro.as_int] += 1

    def accept_readout(self, readout):
        try:
            del self._normalized_counts
        except AttributeError:
            pass

        readout._subcircuit = self
        self._readouts.append(readout)
        self._counts[readout.as_int] += 1

    @property
    def relative_frequency_by_int(self):
        try:
            return self._normalized_counts
        except AttributeError:
            counts = self._counts
            rf = self._normalized_counts = counts / counts.sum()
            return rf

    @property
    def readouts(self):
        """An indexable, iterable view of :class:`Readout` objects, containing the
        time-ordered measurements and auxiliary data, restricted to this Subcircuit."""
        return self._readouts


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


class ProbabilisticSubcircuit(Subcircuit):
    """Encapsulate one part of the circuit between a prepare_all and measure_all gate.

    Also track the theoretical probability distribution of measurement results.
    """

    def __init__(self, *args, probabilities, **kwargs):
        """(internal) Instantiate a Subcircuit"""
        super().__init__(*args, **kwargs)
        self._probabilities = validate_probabilities(probabilities)

    @property
    def simulated_probability_by_int(self):
        """Return the probability associated with each measurement result as a list,
        ordered by the integer representation of the result, with least significant bit
        representing qubit 0.  I.e., "000" for 0b000, "100" for 0b001, "010" for 0b010,
        etc.
        """
        return self._probabilities

    @property
    def simulated_probability_by_str(self):
        """Return the probability associated with each measurement result formatted as a
        dictionary mapping result strings to their respective probabilities."""
        qubits = len(self._trace.used_qubits)
        p = self._probabilities
        return OrderedDict([(f"{n:b}".zfill(qubits)[::-1], v) for n, v in enumerate(p)])

    @property
    def probability_by_int(self):
        """(deprecated) Return the probability associated with each measurement result as a list,
        ordered by the integer representation of the result, with least significant bit
        representing qubit 0.  I.e., "000" for 0b000, "100" for 0b001, "010" for 0b010,
        etc.

        Use simulated_probability_by_int or relative_frequency_by_int.
        """
        return self.simulated_probability_by_int

    @property
    def probability_by_str(self):
        """(deprecated) Return the probability associated with each measurement result formatted as a
        dictionary mapping result strings to their respective probabilities.

        Use simulated_probability_by_str or relative_frequency_by_str.
        """
        return self.simulated_probability_by_str


__all__ = [
    "ExecutionResult",
    "parse_jaqal_output_list",
    "Subcircuit",
    "Readout",
]
