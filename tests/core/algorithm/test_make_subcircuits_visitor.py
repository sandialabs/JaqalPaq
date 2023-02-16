# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.

import unittest

from jaqalpaq.core.algorithm.make_subcircuits_visitor import make_subcircuits
from jaqalpaq.parser import parse_jaqal_string
from jaqalpaq.error import JaqalError
import jaqalpaq.core as core
from jaqalpaq.core.circuitbuilder import build
from contextlib import contextmanager


@contextmanager
def enable_branch():
    import jaqalpaq.core.branch

    try:
        old = jaqalpaq.core.branch.USE_EXPERIMENTAL_BRANCH
        jaqalpaq.core.branch.USE_EXPERIMENTAL_BRANCH = True
        yield
    finally:
        jaqalpaq.core.branch.USE_EXPERIMENTAL_BRANCH = old


class MakeSubcircuitsTester(unittest.TestCase):
    def test_noop(self):
        """Test expanding subcircuits in a circuit without any."""
        text = "{ foo; < bar | baz > }"
        exp = "{ foo; < bar | baz > }"
        self.run_test(text, exp)

    def test_noop_with_usepulses(self):
        """Regression test since an old version didn't copy usepulses."""
        text = "from qscout.v1.std usepulses *"
        exp = "from qscout.v1.std usepulses *"
        self.run_test(text, exp)

    def test_top_level(self):
        """Test at top level with no other gates."""
        text = "prepare_all; measure_all"
        exp = "subcircuit {}"
        self.run_test(text, exp)

    def test_nonempty_top_level(self):
        """Test at top level with other gates."""
        text = "prepare_all; foo; bar; measure_all"
        exp = "subcircuit {foo; bar}"
        self.run_test(text, exp)

    def test_multiple_top_level(self):
        """Test multiple subcircuits at top level."""
        text = (
            "prepare_all; foo; bar; measure_all; " + "prepare_all; a; b; c; measure_all"
        )
        exp = "subcircuit {foo; bar}; subcircuit {a; b; c}"
        self.run_test(text, exp)

    def test_subcircuit_top_level(self):
        """Test a pre-existing subcircuit at top level."""
        text = "subcircuit {foo; bar}"
        exp = "subcircuit {foo; bar}"
        self.run_test(text, exp)

    def test_mixed_top_level(self):
        """Test pre-existing subcircuits with prepare measure."""
        text = "subcircuit {foo; bar}; prepare_all; a; b; c; measure_all"
        exp = "subcircuit {foo; bar}; subcircuit {a; b; c}"
        self.run_test(text, exp)

    def test_sequential_block(self):
        """Test making a subcircuit in a sequential block."""
        text = "{prepare_all; foo; bar; measure_all}"
        exp = "{subcircuit{ foo; bar }}"
        self.run_test(text, exp)

    def test_parallel_block(self):
        """Test rejecting a subcircuit directly in a parallel block."""
        text = "<prepare_all | {foo ; prepare_all}>; measure_all"
        self.run_failure_test(text)

    def test_loop(self):
        """Test making a subcircuit in a loop."""
        text = "loop 10 { prepare_all; foo; bar; measure_all }"
        exp = "loop 10 { subcircuit { foo; bar } }"
        self.run_test(text, exp)

    def test_unmatched_prepare(self):
        """Test a block with a prepare_all statement without a measure_all."""
        text = "{prepare_all; foo}"
        self.run_failure_test(text)
        text = "{prepare_all}"
        self.run_failure_test(text)

    def test_unmatched_measure(self):
        """Test a block with a measure_all statement without a prepare_all."""
        text = "{foo; measure_all; bar}"
        self.run_failure_test(text)
        text = "{measure_all; prepare_all}"
        self.run_failure_test(text)
        text = "{measure_all}"
        self.run_failure_test(text)

    def test_making_nested_subcircuits(self):
        """Test making subcircuits nested in each other."""
        text = "prepare_all; prepare_all; measure_all; measure_all"
        self.run_failure_test(text)
        text = "prepare_all; foo; prepare_all; bar; measure_all; measure_all"
        self.run_failure_test(text)

    def test_prepare_measure_in_subcircuit(self):
        """Test rejecting a prepare measure block in a subcircuit."""
        text = "subcircuit { prepare_all; measure_all }"
        self.run_failure_test(text)

    def test_subcircuit_in_prepare_measure(self):
        """Test rejecting a subcircuit in a prepare measure block."""
        text = "prepare_all; subcircuit {foo; bar}; measure_all"
        self.run_failure_test(text)

    def test_gate_outside_subcircuit(self):
        """Test a gate not in a subcircuit in a block with gates that are."""
        text = "{foo; prepare_all; measure_all}"
        self.run_failure_test(text)
        text = "{prepare_all; measure_all; foo}"
        self.run_failure_test(text)
        text = "{foo; subcircuit {}}"
        self.run_failure_test(text)

    def test_empty_block_alongside_subcircuits(self):
        """Test an empty block in another block next to a subcircuit."""
        text = "{ <>; prepare_all; measure_all }"
        exp = "{ <>; subcircuit {} }"
        self.run_test(text, exp)

        text = "{ prepare_all; measure_all; <> }"
        exp = "{ subcircuit {}; <> }"
        self.run_test(text, exp)

        text = "{ foo; <> }"
        exp = "{ foo; <> }"
        self.run_test(text, exp)

    def test_subcircuit_in_branch(self):
        """Test a branch with subcircuits."""
        text = (
            "branch { '0' : { prepare_all; measure_all }\n"
            + "'1' : { prepare_all; foo; measure_all }}"
        )
        exp = "branch { '0' : { subcircuit {} }\n" + "'1' : { subcircuit { foo } }}"
        with enable_branch():
            self.run_test(text, exp)

    def test_branch_in_subcircuit(self):
        """Test a branch inside a subcircuit."""
        text = "prepare_all; " + "branch { '0' : {foo}; '1' : {bar}}; " + "measure_all"
        exp = "subcircuit { " + "branch { '0' : {foo}; '1' : {bar}}; " + "}"
        with enable_branch():
            self.run_test(text, exp)

    def test_inconsistent_subcircuits_in_branch(self):
        """Test failure on a branch with some subcircuits and some not."""
        text = "branch { '0' : { prepare_all; measure_all }\n" + "'1' : { foo }}"
        with enable_branch():
            self.run_failure_test(text)

    def test_macro_without_subcircuit(self):
        """Test using a macro without a subcircuit inside one."""
        text = "macro Foo a { x a; b; c; }; {Foo 5}"
        exp = "macro Foo a { x a; b; c; }; {Foo 5}"
        self.run_test(text, exp)

    def test_macro_with_subcircuit(self):
        """Test a macro with a subcircuit."""
        text = "macro Foo a { prepare_all; x a; measure_all; }; {Foo 5}"
        exp = "macro Foo a { subcircuit {x a} }; {Foo 5}"
        self.run_test(text, exp)

    def test_macro_in_subcircuit(self):
        """Test a macro with a subcircuit inside a subcircuit."""
        text = (
            "macro Foo a { prepare_all; x a; measure_all; }; "
            + "prepare_all; Foo 5; measure_all"
        )
        self.run_failure_test(text)

    def test_macro_with_indirect_subcircuit(self):
        """Test a macro that calls a macro with a subcircuit."""
        # I want to make sure this works regardless of the stored
        # order of Foo and Bar.
        text = (
            "macro Bar a { prepare_all; x a; measure_all; }; "
            + "macro Foo a { Bar a };"
            + "prepare_all; Foo 5; measure_all"
        )
        self.run_failure_test(text)

        text = (
            "macro Foo a { prepare_all; x a; measure_all; }; "
            + "macro Bar a { Foo a };"
            + "prepare_all; Bar 5; measure_all"
        )
        self.run_failure_test(text)

    def test_macro_with_circular_reference(self):
        """Test correctly rejecting an illegal macro with a circular
        reference."""
        # The parser and builder both prevent circular references, so
        # we have to go to some lengths to introduce one.
        foo = core.Macro("foo", core.BlockStatement())

        bar = core.Macro("bar", core.BlockStatement())

        foo.body.statements.append(core.GateStatement(bar))
        bar.body.statements.append(core.GateStatement(foo))

        circ = core.Circuit()
        circ._macros = {"foo": foo, "bar": bar}

        with self.assertRaises(JaqalError):
            make_subcircuits(circ)

    def test_empty_macro_beside_subcircuit(self):
        """Test an empty macro is okay beside a subcircuit."""
        text = "macro Foo {}; {Foo; prepare_all; measure_all }"
        exp = "macro Foo {}; {Foo; subcircuit{} }"
        self.run_test(text, exp)

    def test_empty_macro_in_subcircuit(self):
        """Test an empty macro is okay in a subcircuit."""
        text = "macro Foo {}; {prepare_all; Foo; measure_all }"
        exp = "macro Foo {}; {subcircuit{Foo} }"
        self.run_test(text, exp)

    def test_alternative_definitions(self):
        """Test providing your own measure and prepare gate defintions."""
        text = "prep; meas"
        exp = "subcircuit {}"
        prepare_def = core.GateDefinition("prep")
        measure_def = core.GateDefinition("meas")
        self.run_test(text, exp, prepare_def=prepare_def, measure_def=measure_def)

    def test_alternative_names(self):
        """Test providing different names for the prep and measure gates and
        having them created."""
        text = "prep; meas"
        exp = "subcircuit {}"
        self.run_test(text, exp, prepare_def="prep", measure_def="meas")

    def run_test(self, text, exp, prepare_def=None, measure_def=None):
        act_parsed = parse_jaqal_string(text, autoload_pulses=False)
        act_circuit = make_subcircuits(
            act_parsed, prepare_def=prepare_def, measure_def=measure_def
        )
        if isinstance(exp, str):
            exp_circuit = parse_jaqal_string(exp, autoload_pulses=False)
        else:
            exp_circuit = exp
        if exp_circuit != act_circuit:
            print(f"Expected:\n{exp_circuit}")
            print(f"Actual:\n{act_circuit}")
        self.assertEqual(exp_circuit, act_circuit)

    def run_failure_test(self, text, prepare_def=None, measure_def=None):
        act_parsed = parse_jaqal_string(text, autoload_pulses=False)
        with self.assertRaises(JaqalError):
            act_circuit = make_subcircuits(
                act_parsed, prepare_def=prepare_def, measure_def=measure_def
            )
