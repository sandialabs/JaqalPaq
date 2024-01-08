import unittest

from .abstractgate import AbstractGateTesterBase
from . import common
from jaqalpaq.core import GateDefinition
from jaqalpaq.error import JaqalError


class GateDefinitionTester(AbstractGateTesterBase, unittest.TestCase):
    def create_random_instance(self, **kwargs):
        """Create an instance of a gate, as required by the base class
        for this testing suite."""
        return common.make_random_gate_definition(**kwargs)

    @property
    def tested_type(self):
        """The type of object under test, as required by the base
        class for this testing suite."""
        return GateDefinition

    def test_reject_multiple_variadic(self):
        """Test that we properly reject multiple variadic parameters."""
        params0 = common.make_random_parameter_list(count=1, variadic=True)
        params1 = common.make_random_parameter_list(
            count=common.random_integer(lower=1, upper=16),
            variadic=bool(common.random_integer(lower=0, upper=1)),
        )
        params2 = common.make_random_parameter_list(count=1, variadic=True)
        params = params0 + params1 + params2
        with self.assertRaises(JaqalError):
            GateDefinition(common.random_identifier(), parameters=params)

    def test_variadic(self):
        """Test that we properly instantiate a single variadic
        parameter in any position."""
        params0 = common.make_random_parameter_list(
            count=common.random_integer(lower=0, upper=4),
        )
        params1 = common.make_random_parameter_list(count=1, variadic=True)
        params2 = common.make_random_parameter_list(
            count=common.random_integer(lower=0, upper=4),
        )
        params = params0 + params1 + params2
        gatedef = GateDefinition(common.random_identifier(), parameters=params)

        arg_types = [(param.kind, param.variadic) for param in params]
        arguments = common.make_random_argument_list(arg_types)

        arg_dict = {}
        for param, arg in zip(params0, arguments):
            arg_dict[param.name] = arg
        variadic = arguments[len(params0) : len(arguments) - len(params2)]
        arg_dict[params1[0].name] = variadic
        for param, arg in zip(params2, arguments[len(params0) + len(variadic) :]):
            arg_dict[param.name] = arg

        gate = gatedef(*arguments)
        self.assertEqual(arg_dict, gate.parameters_by_name)
        self.assertEqual(gatedef.name, gate.name)
        gate = gatedef(**arg_dict)
        self.assertEqual(arg_dict, gate.parameters_by_name)
        self.assertEqual(gatedef.name, gate.name)
