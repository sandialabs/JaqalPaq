import unittest

from .abstractgate import AbstractGateTesterBase
from . import common
from jaqalpaq.core import GateDefinition
from jaqalpaq.error import JaqalError


class GateDefinitionTester(AbstractGateTesterBase, unittest.TestCase):
    def create_random_instance(self, **kwargs):
        return common.make_random_gate_definition(**kwargs)

    @property
    def tested_type(self):
        return GateDefinition

    def test_trailing_variadic_arg(self):
        """Only gates can be variadic for now (not macros), but if we allow
        variadic macros in the future this test should be moved to the base
        class."""
        for _ in range(20):
            gatedef = self.create_random_instance(variadic=True)
            arg_types = [(param.kind, param.variadic) for param in gatedef.parameters]
            arguments = common.make_random_argument_list(arg_types)

            gate = gatedef(*arguments)

            # If this triggers, then variadic arguments were expanded to
            # be able to be in the middle of a gate and we need to update
            # this test.
            assert all(
                not param.variadic for param in gatedef.parameters[:-1]
            ), "Variadic arguments only tested in final position"

            arg_dict = {
                param.name: arg
                for param, arg in zip(gatedef.parameters, arguments)
                if not param.variadic
            }

            if gatedef.parameters and gatedef.parameters[-1].variadic:
                last = gatedef.parameters[-1]
                arg_dict[last.name] = list(arguments[len(gatedef.parameters) - 1 :])

            self.assertEqual(arg_dict, gate.parameters_by_name)
            self.assertEqual(gatedef.name, gate.name)
            gate = gatedef(*arguments)
            self.assertEqual(arg_dict, gate.parameters_by_name)
            self.assertEqual(gatedef.name, gate.name)

    def test_variadic_keyword_arg(self):
        for _ in range(20):
            gatedef = self.create_random_instance(variadic=True)
            arg_types = [(param.kind, param.variadic) for param in gatedef.parameters]
            arguments = common.make_random_argument_list(arg_types)

            arg_dict = {
                param.name: arg
                for param, arg in zip(gatedef.parameters, arguments)
                if not param.variadic
            }
            if gatedef.parameters and gatedef.parameters[-1].variadic:
                last = gatedef.parameters[-1]
                arg_dict[last.name] = list(arguments[len(gatedef.parameters) - 1 :])

            gate = gatedef(**arg_dict)

            # If this triggers, then variadic arguments were expanded to
            # be able to be in the middle of a gate and we need to update
            # this test.
            assert all(
                not param.variadic for param in gatedef.parameters[:-1]
            ), "Variadic arguments only tested in final position"

            self.assertEqual(arg_dict, gate.parameters_by_name)
            self.assertEqual(gatedef.name, gate.name)
            gate = gatedef(*arguments)
            self.assertEqual(arg_dict, gate.parameters_by_name)
            self.assertEqual(gatedef.name, gate.name)

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

    def test_nonfinal_variadic(self):
        """Test that we properly handle a nonfinal variadic parameter."""
        params0 = common.make_random_parameter_list(
            count=common.random_integer(lower=1, upper=4),
        )
        params1 = common.make_random_parameter_list(count=1, variadic=True)
        params2 = common.make_random_parameter_list(
            count=common.random_integer(lower=1, upper=4),
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
