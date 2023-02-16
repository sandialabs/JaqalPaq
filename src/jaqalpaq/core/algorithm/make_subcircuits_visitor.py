# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.

"""Make subcircuits in a Circuit from prepare and measure gates."""

from jaqalpaq.core.algorithm.visitor import Visitor
from jaqalpaq.core import (
    BlockStatement,
    Circuit,
    GateStatement,
    GateDefinition,
    LoopStatement,
    Macro,
)
from jaqalpaq.core.branch import BranchStatement, CaseStatement
from jaqalpaq.error import JaqalError


def make_subcircuits(circuit, prepare_def=None, measure_def=None):
    """Expand subcircuit blocks by adding a prepare and measure gate as
    the first and last gates in the sequential block.

    :param Circuit circuit: The circuit in which to expand subcircuits.

    :param str or GateDefinition prepare_def: The definition of the
        gate to locate in the beginning of each subcircuit. If a string is
        provided, look up in the circuit's native gates. If not given,
        create a new definition using this string or 'prepare_all'.

    :param str or GateDefinition measure_def: The definition of the
        gate to locate at the end of each subcircuit. If a string is
        provided, look up in the circuit's native gates. If not given,
        create a new definition using this string or 'measure_all'.

    """

    prepare_def = _choose_bounding_gate(prepare_def, "prepare_all", circuit)
    measure_def = _choose_bounding_gate(measure_def, "measure_all", circuit)
    visitor = SubcircuitMaker(prepare_def, measure_def)
    return visitor.visit(circuit)


def _choose_bounding_gate(user_def, default_name, circuit):
    """Choose a gate definition from either the one provided by the user,
    one available in circuit, or a default one created on the spot."""

    if not isinstance(user_def, str) and user_def is not None:
        return user_def

    if isinstance(user_def, str):
        name = user_def
    else:
        name = default_name

    try:
        return circuit.native_gates[name]
    except KeyError:
        pass

    return GateDefinition(name)


class SubcircuitMaker(Visitor):
    """Go through a circuit tree to create subcircuits from
    prepare/measure gates and the gates between them. There are some
    rules to subcircuits which are also verified. A subcircuit cannot
    cross another block boundary. If the prepare statement is at one
    level of indentation, the measure must be at the same. Parallel
    gates may not directly contain prepare or measure
    gates. Subcircuits cannot be nested. All gates must be contained
    within a subcircuit. However, subcircuits may be inside blocks, so
    long as the above rules are not violated.

    Some special handling is done for macros. A macro may contain a
    subcircuit or not, but a subcircuit may not start in a macro
    unless it ends in the same macro, nor may it end in one it did not
    start in. The use of macros may not be used to evade any of the
    above rules. An empty macro may be used anywhere.

    All visit calls except for visit_Circuit return the reconstructed
    portion of the tree and a boolean representing whether a
    subcircuit is encountered (which may be None if it is empty of
    gates). If any level of the tree has a subcircuit, all gates or
    sub-blocks must be in a subcircuit as well at that level.

    """

    def __init__(self, prepare_def, measure_def):
        self.prepare_def = prepare_def
        self.measure_def = measure_def
        # A dictionary mapping macro names to whether they contain a
        # subcircuit.
        self._macro_has_subcircuit = {}
        # Store the original macros for a circuit to detect nested
        # macro calls
        self._macros = None

    def visit_default(self, obj):
        """By default we leave all objects alone. We do not copy the object,
        so the input and output circuits will share memory."""
        return obj, None

    def visit_Circuit(self, circuit):
        try:
            new_circuit = Circuit(native_gates=circuit.native_gates)
            self._macros = circuit.macros
            new_circuit.macros.update(self._process_macros())
            new_circuit.constants.update(circuit.constants)
            new_circuit.registers.update(circuit.registers)
            body, _ = self.visit(circuit.body)
            new_circuit.body.statements.extend(body.statements)
            return new_circuit
        finally:
            self._macros = None

    def _process_macros(self):
        # Macros are marked first pending; then either True, False, or
        # None if they contain a subcircuit, do not contain a
        # subcircuit, or are empty. This system is used to detect
        # circular references and forward references, both of which
        # are difficult to introduce through Jaqalpaq
        visited_macros = {}
        self._macro_has_subcircuit = {name: "pending" for name in self._macros}
        for name, macro in self._macros.items():
            visited_macros[name] = self._process_macro(name)
        return visited_macros

    def _process_macro(self, name):
        visited, has_subcircuit = self.visit(self._macros[name])
        assert not isinstance(has_subcircuit, str), "Macro not properly resolved"
        self._macro_has_subcircuit[name] = has_subcircuit
        return visited

    def visit_BlockStatement(self, block):
        # Iterate through the gates to create subcircuit blocks. Make
        # sure everything not in a subcircuit block has subcircuits
        # when visited.
        has_subcircuit = None
        subcircuit = None
        statements = []

        stmt_iter = iter(block.statements)
        for stmt in stmt_iter:
            if self._is_prepare(stmt):
                if block.parallel:
                    raise JaqalError("Cannot have prepare in parallel statement")
                visited = self._make_subcircuit(stmt_iter)
                has_subcircuit = self._update_has_subcircuit(has_subcircuit, True)
            elif self._is_measure(stmt):
                raise JaqalError("No prepare statement found to match with a measure")
            else:
                visited, visited_has_subcircuit = self.visit(stmt)
                has_subcircuit = self._update_has_subcircuit(
                    has_subcircuit, visited_has_subcircuit
                )

            statements.append(visited)

        if block.subcircuit and has_subcircuit:
            raise JaqalError("Cannot nest subcircuits")

        # The order here matters! We distinguish None from False, so
        # has_subcircuit must be last
        has_subcircuit = block.subcircuit or has_subcircuit
        return (
            BlockStatement(
                parallel=block.parallel,
                subcircuit=block.subcircuit,
                iterations=block.iterations,
                statements=statements,
            ),
            has_subcircuit,
        )

    def _make_subcircuit(self, stmt_iter):
        """Knowing that we just processed a prepare statement, create a new
        subcircuit block with all the statements between it and the measure
        statement."""

        statements = []

        for stmt in stmt_iter:
            if self._is_measure(stmt):
                return BlockStatement(subcircuit=True, statements=statements)
            visited, has_subcircuit = self.visit(stmt)
            if has_subcircuit:
                raise JaqalError("Cannot nest subcircuits")
            statements.append(visited)

        raise JaqalError("No measure statement found to correspond to a prepare")

    def _is_prepare(self, statement):
        return (
            isinstance(statement, GateStatement)
            and statement.gate_def == self.prepare_def
        )

    def _is_measure(self, statement):
        return (
            isinstance(statement, GateStatement)
            and statement.gate_def == self.measure_def
        )

    def _update_has_subcircuit(self, block_has_subcircuit, statement_has_subcircuit):
        """Return whether the block has a subcircuit. If the block and
        statement subcircuits are incompatible, raise an exception."""
        if block_has_subcircuit is None:
            return statement_has_subcircuit

        if statement_has_subcircuit is None:
            # This can happen if the statement is an empty block for
            # instance.
            return block_has_subcircuit

        if block_has_subcircuit != statement_has_subcircuit:
            raise JaqalError("All gates at a layer must be in or out of a subcircuit")

        return statement_has_subcircuit

    def visit_LoopStatement(self, loop):
        visited, has_subcircuit = self.visit(loop.statements)
        return (
            LoopStatement(self.visit(loop.iterations)[0], statements=visited),
            has_subcircuit,
        )

    def visit_BranchStatement(self, branch):
        cases = []
        has_subcircuit = None
        for case in branch.cases:
            visited, visited_has_subcircuit = self.visit(case)
            has_subcircuit = self._update_has_subcircuit(
                has_subcircuit, visited_has_subcircuit
            )
            cases.append(visited)
        ret = BranchStatement(cases=cases)
        return ret, has_subcircuit

    def visit_CaseStatement(self, case):
        visited, has_subcircuit = self.visit(case.statements)
        # has_subcircuit could be None if there are no gates, but for
        # cases we only allow a subcircuit block in one if all of them
        # have subcircuit blocks.
        return CaseStatement(case.state, visited), bool(has_subcircuit)

    def visit_GateStatement(self, gate):
        has_subcircuit = self._macro_has_subcircuit.get(gate.name, False)
        if has_subcircuit == "pending":
            raise JaqalError(f"Macro {gate.name} unresolved or in a cycle")
        assert not isinstance(has_subcircuit, str), "Macro not properly resolved"
        return gate, has_subcircuit

    def visit_Macro(self, macro):
        body, has_subcircuit = self.visit(macro.body)
        assert (has_subcircuit is None) == (len(macro.body.statements) == 0)
        return Macro(macro.name, macro.parameters, body), has_subcircuit
