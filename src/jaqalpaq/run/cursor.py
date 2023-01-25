import enum

from jaqalpaq.core.circuit import Circuit
from jaqalpaq.core.locus import Locus
from jaqalpaq.core.block import BlockStatement, LoopStatement
from jaqalpaq.core.gate import GateStatement
from jaqalpaq.error import JaqalError


class State(enum.Enum):
    shutdown = 0
    gate = 1
    gate_stalled = -1
    final_measurement = 2
    midcircuit_measurement = 3
    midcircuit_measurement_report = -3


class SubcircuitCursor:
    def __init__(self, locus, end):
        self._locus = locus
        self._end = end
        self._loop_stack = []
        self._do_initialize()

    @classmethod
    def terminal_cursor(klass, end):
        ret = klass.__new__(klass)
        ret._locus = end
        ret._end = end
        ret._loop_stack = []
        ret._state = State.final_measurement
        return ret

    @property
    def locus(self):
        return self._locus

    @property
    def state(self):
        return self._state

    def walk(self, descend_first=False):
        loc = self._locus
        end = self._end.address

        go_forward = not descend_first
        classify = False

        while not classify:
            while go_forward:  # forward
                obj = loc.parent.object
                next_sibling = loc.index + 1
                if len(loc.parent.children) == next_sibling:
                    loc = loc.parent
                    # TODO: calling loc.address repeatedly is quadratic
                    if isinstance(obj, BlockStatement) and obj.subcircuit:
                        # subcircuit case
                        assert loc.address == end
                        self._locus = loc
                        classify = True
                        break
                    else:
                        assert loc.address != end
                    if isinstance(obj, BlockStatement):
                        continue
                    elif isinstance(obj, LoopStatement):
                        try:
                            loop_term = self._loop_stack[-1]
                        except IndexError as e:
                            raise JaqalError("Invalid loop") from e

                        if loop_term == obj.iterations:
                            self._loop_stack.pop()
                            continue
                        else:
                            self._loop_stack[-1] += 1
                            loc = loc[0]
                            assert self._loop_stack[-1] <= obj.iterations
                            break
                    elif isinstance(obj, Circuit):
                        raise JaqalError("Subcircuit did not end")
                    else:
                        raise RuntimeError("Unrecognized gate")
                else:
                    loc = loc.parent[next_sibling]
                    break

            go_forward = True
            while not classify:  # descend
                obj = loc.object
                if isinstance(obj, BlockStatement):
                    if len(loc.children) == 0:
                        break
                    assert not obj.subcircuit
                    loc = loc[0]
                elif isinstance(obj, LoopStatement):
                    if (obj.iterations == 0) or (len(loc.children) == 0):
                        break
                    self._loop_stack.append(1)
                    loc = loc[0]
                elif isinstance(obj, GateStatement):
                    self._locus = loc
                    classify = True
                    break
                else:
                    raise JaqalError("Unexpected gate")

        obj = self._locus.object

        is_end = (isinstance(obj, GateStatement) and obj.name == "measure_all") or (
            isinstance(obj, BlockStatement) and obj.subcircuit
        )
        if not is_end:
            assert self._locus != self._end

        # Classify the state
        if is_end:
            self._state = State.final_measurement
            assert self._locus == self._end
        elif isinstance(obj, GateStatement):
            assert obj.gate_def.unitary
            self._state = State.gate
        else:
            raise JaqalError("Unexpected JaqalStatement")

    def _do_initialize(self):
        obj = self._locus.object

        if isinstance(obj, GateStatement) and (obj.name == "prepare_all"):
            self.walk()
            return
        elif isinstance(obj, BlockStatement) and obj.subcircuit:
            assert self._end == self._locus
            if len(self._locus.children) == 0:
                # empty subcircuit; nothing to advance, just
                self._state = State.final_measurement
                return
            self._locus = self._locus[0]
            self.walk(descend_first=True)
            return
        else:
            raise JaqalError()

    def next_gate(self):
        assert self._state == State.gate

        obj = self._locus.object
        self._state = State.gate_stalled
        return obj

    def report_gate_executed(self):
        self.walk()

    def next_measure(self):
        obj = self._locus.object

        if self._state == State.final_measurement:
            self._state = State.shutdown
        elif self._state == State.midcircuit_measurement:
            self._state = State.midcircuit_measurement_report
        else:
            raise JaqalError("Cursor not ready to provide measurement")

        return obj

    def copy(self):
        klass = type(self)
        ret = klass.__new__(klass)
        ret._locus = self._locus
        ret._end = self._end
        ret._state = self._state
        ret._loop_stack = self._loop_stack.copy()
        return ret

    def __eq__(self, other):
        if not isinstance(other, SubcircuitCursor):
            return False
        if self._locus != other._locus:
            return False
        if self._loop_stack != other._loop_stack:
            return False
        assert self._state == other._state
        return True
