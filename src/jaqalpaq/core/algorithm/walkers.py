# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
from collections import defaultdict
from itertools import chain

from .used_qubit_visitor import UsedQubitIndicesVisitor
from .visitor import Visitor
from jaqalpaq.core.locus import Locus
from jaqalpaq.error import JaqalError
from jaqalpaq.core.block import BlockStatement, LoopStatement
from jaqalpaq.core.branch import BranchStatement


class Trace:
    """(deprecated) Describes a portion of a Circuit traced out by start and stop locations."""

    def __init__(self, start=None, end=None, used_qubits=None):
        if start is None:
            self.start = []
        else:
            self.start = start

        self.end = end
        self.used_qubits = used_qubits

    def __repr__(self):
        if self.end is None:
            return f"Trace({self.start})"
        else:
            return f"Trace({self.start}, {self.end})"


class TraceSerializer(Visitor):
    """Returns a serialized representation of all gates called during a circuit trace."""

    def __init__(self, trace=None, **kwargs):
        super().__init__(trace=trace, **kwargs)

    def visit_Circuit(self, circuit):
        yield from self.visit(circuit.body)

    def visit_BlockStatement(self, block):
        for n, sub_obj in self.trace_statements(block.statements):
            yield from self.visit(sub_obj)

    def visit_BranchStatement(self, branch):
        raise JaqalError("Branch statements not supported in trace serializing")

    def visit_CaseStatement(self, block):
        raise JaqalError("Branch cases not supported in trace serializing")

    def visit_LoopStatement(self, loop):
        if self.started:
            for n in range(loop.iterations):
                yield from self.visit(loop.statements)
        else:
            yield from self.visit(loop.statements)

    def visit_GateStatement(self, gate):
        yield gate


class DiscoverSubcircuits(UsedQubitIndicesVisitor):
    """Walks a Circuit, identifying subcircuits bounded by prepare_all and measure_all"""

    # While this *is* the behavior of DiscoverSubcircuits, this flag does nothing.
    validate_parallel = True

    def __init__(self, *args, p_gate="prepare_all", m_gate="measure_all", **kwargs):
        super().__init__(*args, **kwargs)
        self.current = None
        self.qubits = None
        self.subcircuits = []
        self.p_gate = p_gate
        self.m_gate = m_gate

    def visit_Circuit(self, circuit, context=None):
        # All qubits will be used in every subcircuit, because it is bounded by
        # prepare_all and measure_all.  In the future, we presumably will employ the used
        # qubit functionality of the superclass, and separately report the measured
        # qubits.  But we do not support mid-circuit measurements yet.
        if self.qubits is not None:
            raise RuntimeError("Cannot reuse DiscoverSubcircuit object")
        self.qubits = list(chain.from_iterable(circuit.fundamental_registers()))
        super().visit_Circuit(circuit, context=context)

        if self.current is not None:
            raise JaqalError("Subcircuit did not end")

        subcircuits = self.subcircuits
        if len(subcircuits) == 0:
            return ()

        if subcircuits[-1].end is None:
            return subcircuits[:-1]
        else:
            return subcircuits[:]

    def visit_LoopStatement(self, obj, context=None):
        return self.visit(obj.statements, context=context, reps=obj.iterations)

    def visit_CaseStatement(self, obj, context=None):
        return self.visit(obj.statements, context=context, state=obj.state)

    def visit_BranchStatement(self, obj, context=None):
        return self.visit(obj.cases, context=context)

    def visit_BlockStatement(self, block, context=None, reps=1):
        # Calling UsedQubitIndicesVisitor as super() is
        # far too inflexible for the purposes here.
        indices = defaultdict(set)

        count = len(self.subcircuits)
        had_started = self.current is not None

        if block.subcircuit:
            self.start_trace(context=context)

        # XXX: using a trace restriction here is untested
        for n, stmt in self.trace_statements(block.statements):
            self.merge_into(
                indices, self.visit(stmt, context=context), disjoint=block.parallel
            )

        if block.subcircuit:
            self.end_trace(context=context)

        if had_started and (len(self.subcircuits) != count) and (reps > 1):
            raise JaqalError("measure_all -> prepare_all not supported in loops")

        return indices

    def visit_GateStatement(self, gate, context=None):
        if gate.name == self.p_gate:
            self.start_trace(context=context)
        elif gate.name == self.m_gate:
            self.end_trace(context=context)
        else:
            if self.current is None:
                raise JaqalError(f"gates must follow a {self.p_gate}")

        return super().visit_GateStatement(gate, context=context)

    def start_trace(self, context=None):
        # We do not allow for multiple prepare_all's in a row.  Notice also, if we
        # were doing mid-circuit measurements, that we would not know what the measured
        # or used qubits are until until the measurement.
        if self.current is not None:
            raise JaqalError("Nested subcircuit are not allowed")
        self.current = Trace(self.address[:])

    def end_trace(self, context=None):
        if self.current is None:
            raise JaqalError(f"{self.m_gate} must follow a {self.p_gate}")
        current = self.current
        current.end = self.address[:]
        current.used_qubits = self.qubits
        self.subcircuits.append(current)
        self.current = None


def discover_subcircuits(circuit):
    ds = DiscoverSubcircuits()
    for tr in ds.visit(circuit):
        locus = Locus.from_address(circuit, tr.start)
        if tr.start != tr.end:
            end_locus = Locus.from_address(circuit, tr.end)
        else:
            end_locus = locus
        last = False
        for parent in locus.lineage:
            if last:
                raise JaqalError("Cannot nest subcircuit in any other block")
            obj = parent.object
            if isinstance(obj, BranchStatement):
                raise JaqalError("Cannot start subcircuit inside a branch")
            elif isinstance(obj, BlockStatement) and obj.subcircuit:
                last = True
        yield locus, end_locus


class CircuitWalker(Visitor):
    """(internal) Walk a circuit in execution order, yielding at each breakpoint.

    This requires the breakpoints to be given in the order they are
    found by DiscoverSubcircuits.
    """

    def __init__(self, breakpoints):
        self.breakpoints = breakpoints
        self.address = []
        self.index = 0

    def visit_Circuit(self, circuit):
        if len(self.breakpoints) == 0:
            return
        self.objective = list(self.breakpoints[self.index].address)

        yield from self.visit(circuit.body)

    def visit_BlockStatement(self, block):
        if block.subcircuit:
            yield from self.do_loop(self.iterations, block)
        else:
            yield from self.do_block(block)

    def do_block(self, block):
        first = True
        address = self.address
        while self.objective:
            if address != self.objective[: len(address)]:
                assert not first
                assert address < self.objective[: len(address)]
                return

            first = False

            n = self.objective[len(address)]
            nxt = block.statements[n]
            if (len(address) + 1) == len(self.objective):
                yield self.index
                self.index += 1
                if self.index == len(self.breakpoints):
                    # We've found all the traces.  We're done!
                    self.objective = None
                    return
                else:
                    self.objective = list(self.breakpoints[self.index].address)
            else:
                address.append(n)
                yield from self.visit(nxt)
                address.pop()

    def visit_LoopStatement(self, loop):
        yield from self.do_loop(loop.iterations, loop.statements)

    def do_loop(self, iterations, block):
        # store the walk status
        index = self.index
        address = self.address[:]
        objective = self.objective

        # loop over the classical parts
        for n in range(iterations):
            # Restore the walk status at the start of every loop
            self.objective = objective
            self.address[:] = address[:]
            self.index = index
            yield from self.visit(block)

    def visit_CaseStatement(self, case):
        raise NotImplementedError("case statements are not supporetd")

    def visit_BranchStatement(self, branch):
        raise JaqalError("Tracing a circuit with a branch not supported")


def walk_circuit(circuit, breakpoints):
    """Walk a circuit in execution order, yielding at each breakpoint.

    This requires the breakpoints to be given in the order they are
    found by DiscoverSubcircuits.
    """
    yield from CircuitWalker(breakpoints).visit(circuit)
