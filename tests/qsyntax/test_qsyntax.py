import unittest

from jaqalpaq.qsyntax import circuit
from jaqalpaq.parser import parse_jaqal_string
from jaqalpaq.core import GateDefinition, Parameter, ParamType
from jaqalpaq.generator import generate_jaqal_program
from jaqalpaq.error import JaqalError
import jaqalpaq.core.branch


class QsyntaxTester(unittest.TestCase):
    """Test generating code using Qsyntax and ensuring it creates the
    expected IR."""

    def setUp(self):
        jaqalpaq.core.branch.USE_EXPERIMENTAL_BRANCH = True
        circuit.clear()

    def tearDown(self):
        jaqalpaq.core.branch.USE_EXPERIMENTAL_BRANCH = False

    def test_empty_circuit(self):
        """Test an empty circuit, which really contains an implicit
        prepare_all and measure_all statement."""

        @circuit
        def func(Q):
            pass

        text = "prepare_all; measure_all"
        self.run_test(func, text)

    def test_let_constant(self):
        @circuit
        def func(Q):
            Q.let(5)
            Q.let(10, "mylet")

        text = "let __c0 5; let mylet 10; prepare_all; measure_all"
        self.run_test(func, text)

    def test_deconflict_let_names(self):
        @circuit
        def func(Q):
            Q.let(0, "__c0")
            Q.let(1)

        text = "let __c0 0; let __c1 1; prepare_all; measure_all"
        self.run_test(func, text)

    def test_unnamed_register(self):
        @circuit
        def func(Q):
            Q.register(2)

        text = "register __r0[2]; prepare_all; measure_all"
        self.run_test(func, text)

    def test_named_register(self):
        @circuit
        def func(Q):
            Q.register(2, "r")

        text = "register r[2]; prepare_all; measure_all"
        self.run_test(func, text)

    def test_register_constant_size(self):
        @circuit
        def func(Q):
            n = Q.let(2, "n")
            Q.register(n, "r")

        text = "let n 2; register r[n]; prepare_all; measure_all"
        self.run_test(func, text)

    def test_gate(self):
        @circuit
        def func(Q):
            Q.Foo(1, 3.14)

        text = "prepare_all; Foo 1 3.14; measure_all"
        self.run_test(func, text)

    def test_index_qubit(self):
        @circuit
        def func(Q):
            r = Q.register(2, "r")
            c = Q.let(1, "c")
            Q.Foo(r[0], r[c])

        text = "register r[2]; let c 1; prepare_all; Foo r[0] r[c]; measure_all"
        self.run_test(func, text)

    def test_usepulses(self):
        @circuit
        def func(Q):
            Q.usepulses("sample.v1")

        text = "from sample.v1 usepulses *; prepare_all; measure_all"
        self.run_test(func, text)

    def test_sequential_block(self):
        @circuit
        def func(Q):
            with Q.sequential():
                Q.Foo(1, 2, 3)

        text = "prepare_all; {Foo 1 2 3}; measure_all"
        self.run_test(func, text)

    def test_sequential_block_noparen(self):
        @circuit
        def func(Q):
            with Q.sequential:
                Q.Foo(1, 2, 3)

        text = "prepare_all; {Foo 1 2 3}; measure_all"
        self.run_test(func, text)

    def test_parallel_block(self):
        @circuit
        def func(Q):
            with Q.parallel():
                Q.Foo(1, 2, 3)
                Q.Bar(1, 2, 3)

        text = "prepare_all; <Foo 1 2 3|Bar 1 2 3>; measure_all"
        self.run_test(func, text)

    def test_parallel_block_noparen(self):
        @circuit
        def func(Q):
            with Q.parallel:
                Q.Foo(1, 2, 3)
                Q.Bar(1, 2, 3)

        text = "prepare_all; <Foo 1 2 3|Bar 1 2 3>; measure_all"
        self.run_test(func, text)

    def test_subcircuit_block(self):
        @circuit
        def func(Q):
            with Q.subcircuit(100):
                Q.Foo(1, 2, 3)
                Q.Bar(1, 2, 3)

        text = "subcircuit 100 {Foo 1 2 3 ; Bar 1 2 3}"
        self.run_test(func, text)

    def test_subcircuit_block_no_arg(self):
        @circuit
        def func(Q):
            with Q.subcircuit():
                Q.Foo(1, 2, 3)
                Q.Bar(1, 2, 3)

        text = "subcircuit {Foo 1 2 3 ; Bar 1 2 3}"
        self.run_test(func, text)

    def test_loop(self):
        @circuit
        def func(Q):
            with Q.loop(150):
                Q.Foo(1, 2, 3)

        text = "prepare_all; loop 150 { Foo 1 2 3 }; measure_all"
        self.run_test(func, text)

    def test_loop_constant_arg(self):
        @circuit
        def func(Q):
            n = Q.let(150, "n")
            with Q.loop(n):
                Q.Foo(1, 2, 3)

        text = "let n 150; prepare_all; loop n { Foo 1 2 3 }; measure_all"
        self.run_test(func, text)

    def test_branch_case(self):
        @circuit
        def func(Q):
            with Q.branch():
                with Q.case(0b0):
                    Q.Foo(1)
                with Q.case(0b1):
                    Q.Foo(2)

        text = "prepare_all; branch { '0' : { Foo 1}; '1' : {Foo 2}}; measure_all"
        self.run_test(func, text)

    def test_function_call(self):
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; prepare_all; Foo r[0]; measure_all"
        self.run_test(func, text)

    def test_arguments_to_function(self):
        n = 2

        @circuit
        def func(Q, n):
            r = Q.register(n, "r")

        text = f"register r[{n}]; prepare_all; measure_all"
        self.run_test(func, text, n)

    def test_variadic_gate(self):
        gatedefs = {
            "prepare_all": GateDefinition("prepare_all", []),
            "measure_all": GateDefinition("measure_all", []),
            "Foo": GateDefinition(
                "Foo", [Parameter("a", ParamType.INT, variadic=True)]
            ),
        }

        @circuit(inject_pulses=gatedefs)
        def func(Q):
            Q.Foo(1, 2, 3, 4)

        text = "prepare_all; Foo 1 2 3 4; measure_all"
        self.run_test(func, text, inject_pulses=gatedefs)

    def test_sequential_function(self):
        """Test using a function with a Q.sequential decorator."""

        @circuit
        def func(Q):
            @Q.sequential
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_sequential_function_redefine_gate(self):
        """Test a sequential function attempting to redefine a gate."""

        native_gates = {
            "prepare_all": GateDefinition("prepare_all"),
            "measure_all": GateDefinition("measure_all"),
            "Foo": GateDefinition(
                "Foo", parameters=[Parameter("r", ParamType.REGISTER)]
            ),
            "Sx": GateDefinition("Sx", parameters=[Parameter("q", ParamType.QUBIT)]),
        }

        @circuit(inject_pulses=native_gates)
        def func(Q):
            @Q.sequential
            def Foo(Q, r):
                Q.Sx(r[0])

            r = Q.register(2, "r")
            Q.Foo(r)

        with self.assertRaises(JaqalError):
            func()

    def test_sequential_function_standalone(self):
        """Test using a function with a Q.sequential decorator."""

        @circuit
        def func(Q):
            @Q.sequential
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; prepare_all; {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_sequential_function_inst(self):
        """Test using a function with a Q.sequential decorator."""

        class Body:
            def __init__(self, name, idx):
                self.__name__ = name
                self.idx = idx

            def __call__(self, Q, r):
                Q.Foo(r[self.idx])

        @circuit
        def func(Q):
            Q.sequential(Body("make_body", 0))

            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_global_sequential_function_inst(self):
        """Test using a function with a Q.sequential decorator."""

        class Body:
            def __init__(self, name, idx):
                self.__name__ = name
                self.idx = idx

            def __call__(self, Q, r):
                Q.Foo(r[self.idx])

        circuit.sequential(Body("make_body", 0))

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_sequential_function_inst_noname(self):
        """Test using a function with a Q.sequential decorator."""

        class Body:
            def __init__(self, idx):
                self.idx = idx

            def __call__(self, Q, r):
                Q.Foo(r[self.idx])

        @circuit
        def func(Q):
            Q.sequential(Body(0))

            r = Q.register(2, "r")
            Q.make_body(r)

        with self.assertRaises(JaqalError):
            func()

    def test_sequential_function_standalone_no_q(self):
        """Test using a function with a Q.sequential decorator."""

        @circuit
        def func(Q):
            @Q.sequential
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            make_body(r)

        with self.assertRaises(JaqalError):
            func()

    def test_global_sequential_function(self):
        """Test a function outside a circuit with a circuit.sequential decorator."""

        @circuit.sequential()
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_global_sequential_function_standalone(self):
        """Test a function outside a circuit with a circuit.sequential decorator."""

        @circuit.sequential()
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; prepare_all; {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_global_sequential_function_noparen(self):
        """Test a function outside a circuit with a circuit.sequential decorator."""

        @circuit.sequential
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_parallel_function(self):
        """Test using a function with a Q.parallel decorator."""

        @circuit
        def func(Q):
            @Q.parallel
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; <Foo r[0]>; measure_all"
        self.run_test(func, text)

    def test_parallel_function_standalone(self):
        """Test using a function with a Q.parallel decorator."""

        @circuit
        def func(Q):
            @Q.parallel
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; prepare_all; <Foo r[0]>; measure_all"
        self.run_test(func, text)

    def test_global_parallel_function(self):
        """Test a function outside a circuit with a circuit.parallel decorator."""

        @circuit.parallel()
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; <Foo r[0]>; measure_all"
        self.run_test(func, text)

    def test_global_parallel_function_standalone(self):
        """Test a function outside a circuit with a circuit.parallel decorator."""

        @circuit.parallel()
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; prepare_all; <Foo r[0]>; measure_all"
        self.run_test(func, text)

    def test_global_parallel_function_noparen(self):
        """Test a function outside a circuit with a circuit.parallel decorator."""

        @circuit.parallel
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; <Foo r[0]>; measure_all"
        self.run_test(func, text)

    def test_subcircuit_function(self):
        """Test using a function with a Q.subcircuit decorator."""

        @circuit
        def func(Q):
            @Q.subcircuit
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; subcircuit {Foo r[0]}"
        self.run_test(func, text)

    def test_subcircuit_function_standalone(self):
        """Test using a function with a Q.subcircuit decorator."""

        @circuit
        def func(Q):
            @Q.subcircuit
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; subcircuit {Foo r[0]}"
        self.run_test(func, text)

    def test_subcircuit_function_with_arg(self):
        """Test using a function with a Q.subcircuit decorator."""

        @circuit
        def func(Q):
            @Q.subcircuit(42)
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; subcircuit 42 {Foo r[0]}"
        self.run_test(func, text)

    def test_subcircuit_function_with_arg_standalone(self):
        """Test using a function with a Q.subcircuit decorator."""

        @circuit
        def func(Q):
            @Q.subcircuit(42)
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; subcircuit 42 {Foo r[0]}"
        self.run_test(func, text)

    def test_global_subcircuit_function(self):
        """Test a function outside a circuit with a circuit.subcircuit decorator."""

        @circuit.subcircuit()
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; subcircuit {Foo r[0]}"
        self.run_test(func, text)

    def test_global_subcircuit_function_noparen(self):
        """Test a function outside a circuit with a circuit.subcircuit decorator."""

        @circuit.subcircuit
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; subcircuit {Foo r[0]}"
        self.run_test(func, text)

    def test_global_subcircuit_function_with_arg(self):
        """Test a function outside a circuit with a circuit.subcircuit decorator."""

        @circuit.subcircuit(42)
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; subcircuit 42 {Foo r[0]}"
        self.run_test(func, text)

    def test_global_subcircuit_function_with_arg_standalone(self):
        """Test a function outside a circuit with a circuit.subcircuit decorator."""

        @circuit.subcircuit(42)
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; subcircuit 42 {Foo r[0]}"
        self.run_test(func, text)

    def test_loop_function(self):
        """Test using a function with a Q.loop decorator."""

        @circuit
        def func(Q):
            @Q.loop(5)
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; loop 5 {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_loop_function_standalone(self):
        """Test using a function with a Q.loop decorator."""

        @circuit
        def func(Q):
            @Q.loop(5)
            def make_body(Q, r):
                Q.Foo(r[0])

            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; prepare_all; loop 5 {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_global_loop_function(self):
        """Test a function outside a circuit with a circuit.loop decorator."""

        @circuit.loop(10)
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            Q.make_body(r)

        text = "register r[2]; prepare_all; loop 10 {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def test_global_loop_function_standalone(self):
        """Test a function outside a circuit with a circuit.loop decorator."""

        @circuit.loop(10)
        def make_body(Q, r):
            Q.Foo(r[0])

        @circuit
        def func(Q):
            r = Q.register(2, "r")
            make_body(Q, r)

        text = "register r[2]; prepare_all; loop 10 {Foo r[0]}; measure_all"
        self.run_test(func, text)

    def run_test(self, func, text, *args, inject_pulses=None):
        func_circ = func(*args)
        text_circ = parse_jaqal_string(
            text, autoload_pulses=False, inject_pulses=inject_pulses
        )
        self.assertEqual(func_circ, text_circ)
