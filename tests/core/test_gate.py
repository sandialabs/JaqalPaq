import unittest

from jaqalpaq.core import GateStatement

from .randomize import random_identifier, random_whole
from . import common


class GateTester(unittest.TestCase):
    def test_create_gate_no_parameters(self):
        """Test creating a gate without parameters."""
        gate, definition, _ = common.make_random_gate_statement(
            count=0, return_params=True
        )
        self.assertEqual(definition.name, gate.name)
        self.assertEqual({}, gate.parameters_by_name)
        self.assertEqual(definition, gate.gate_def)
        # This tests the constructor for GateStatement when not given
        # parameters
        self.assertEqual(gate, GateStatement(definition))

    def test_create_gate_with_parameters(self):
        """Test creating a gate with parameters."""
        count = random_whole()
        gate, definition, arguments = common.make_random_gate_statement(
            count=count, return_params=True
        )
        self.assertEqual(definition.name, gate.name)
        self.assertEqual(arguments, gate.parameters_by_name)
        self.assertEqual(definition, gate.gate_def)

    def test_parameters_by_name(self):
        count = random_whole()
        gate, definition, arguments = common.make_random_gate_statement(
            count=count, return_params=True
        )
        self.assertEqual(arguments, gate.parameters_by_name)

    def test_parameters_linear(self):
        count = random_whole()
        gate, definition, arguments = common.make_random_gate_statement(
            count=count, return_params=True
        )
        self.assertEqual(list(arguments.values()), gate.parameters_linear)

    def test_parameters_with_types(self):
        count = random_whole()
        gate, definition, arguments = common.make_random_gate_statement(
            count=count, return_params=True
        )
        typed_args = list(zip(arguments.values(), definition.parameters))
        self.assertEqual(typed_args, gate.parameters_with_types)

    def test_parameters_by_name_variadic(self):
        count = random_whole()
        gate, definition, arguments = common.make_random_gate_statement(
            count=count, return_params=True, variadic=True
        )
        for param, arg in zip(definition.parameters, arguments.values()):
            self.assertEqual(param.variadic, isinstance(arg, list))
        self.assertEqual(arguments, gate.parameters_by_name)

    def test_parameters_linear_variadic(self):
        count = random_whole()
        gate, definition, arguments = common.make_random_gate_statement(
            count=count, return_params=True, variadic=True
        )
        linear_args = []
        for param, arg in zip(definition.parameters, arguments.values()):
            if param.variadic:
                linear_args.extend(arg)
            else:
                linear_args.append(arg)
        self.assertEqual(linear_args, gate.parameters_linear)

    def test_parameters_with_types_variadic(self):
        count = random_whole()
        count = 2
        gate, definition, arguments = common.make_random_gate_statement(
            count=count, return_params=True, variadic=True
        )
        typed_args = []
        for arg, param in zip(arguments.values(), definition.parameters):
            if not param.variadic:
                typed_args.append((arg, param))
            else:
                typed_args.extend((a, param) for a in arg)
        self.assertEqual(typed_args, gate.parameters_with_types)
