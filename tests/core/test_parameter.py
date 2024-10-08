import unittest

from jaqalpaq.error import JaqalError
from jaqalpaq.core.parameter import ParamType, Parameter
from jaqalpaq.core.constant import Constant

from jaqalpaq.core.register import Register, NamedQubit
from . import common
from .randomize import random_float, random_integer


class ParameterTester(unittest.TestCase):
    def setUp(self):
        reg = common.make_random_register()
        self.example_values = [
            (reg, [ParamType.REGISTER, ParamType.NONE]),
            (
                common.choose_random_qubit_getitem(reg),
                [ParamType.QUBIT, ParamType.NONE],
            ),
            (random_float(), [ParamType.FLOAT, ParamType.NONE]),
            (float(random_integer()), [ParamType.FLOAT, ParamType.INT, ParamType.NONE]),
            (random_integer(), [ParamType.FLOAT, ParamType.INT, ParamType.NONE]),
        ]
        self.example_params = {
            kind: common.make_random_parameter(allowed_types=[kind])
            for kind in ParamType
        }

    def test_create(self):
        """Test creating a parameter with a name and kind."""
        for kind in ParamType:
            param, name, _ = common.make_random_parameter(
                allowed_types=[kind], return_params=True
            )
            self.assertEqual(name, param.name)
            self.assertEqual(kind, param.kind)

    def test_variadic_representation(self):
        """Test that the variadic repr of a Parameter differs from the non-variadic."""
        # Create a random parameter and store its repr string
        param = common.make_random_parameter()
        rep = repr(param)

        # Make a copy to make sure we're doing the representation
        # correctly as a control
        cparam = Parameter(param.name, param.kind, variadic=param.variadic)
        crep = repr(param)

        # Do a copy but switch up whether it's variadic
        vparam = Parameter(param.name, param.kind, variadic=not param.variadic)
        vrep = repr(vparam)

        self.assertNotEqual(vrep, rep)
        self.assertEqual(crep, rep)

    def test_validate(self):
        """Test if the validate method returns True for valid values and False for
        invalid ones."""
        for ex_value, ex_types in self.example_values:
            for param_kind, param in self.example_params.items():
                if param.kind in ex_types:
                    param.validate(ex_value)
                else:
                    with self.assertRaises(Exception):
                        param.validate(ex_value)

    def test_validate_constant(self):
        """Test validating against let constants with certain types."""
        # Let-Constants cannot take on these values and are thus
        # skipped for this test.
        bad_types = {ParamType.REGISTER, ParamType.QUBIT}
        for ex_value, ex_types in self.example_values:
            if bad_types & set(ex_types):
                continue
            const_value = Constant("c", ex_value)
            for param_kind, param in self.example_params.items():
                if param.kind in ex_types:
                    param.validate(const_value)
                else:
                    with self.assertRaises(Exception):
                        param.validate(const_value)

    def test_validate_none_parameter(self):
        """Test validating against another parameter with no value like in a macro."""
        for ex_value, ex_types in self.example_values:
            macro_param = Parameter(ex_value, ParamType.NONE)
            for param_kind, param in self.example_params.items():
                param.validate(macro_param)

    def test_resolve_value(self):
        """Test resolving a value within some context."""
        for param_kind, param in self.example_params.items():
            # Note: The Parameter class could, but does not, do type checking here.
            values = [
                value
                for value, value_types in self.example_values
                if param_kind in value_types
            ]
            for value in values:
                context = {param.name: value}
                resolved_value = param.resolve_value(context)
                common.assert_values_same(self, value, resolved_value)
                with self.assertRaises(Exception):
                    param.resolve_value()

    def test_classical(self):
        """Test that the right parameter types are classical."""
        classical_kinds = [ParamType.FLOAT, ParamType.INT]
        for param_kind, param in self.example_params.items():
            if param_kind == ParamType.NONE:
                with self.assertRaises(Exception):
                    # Note: This behavior implies that classical checking is only reasonable when a
                    # parameter has a known type.
                    param.classical
            else:
                self.assertEqual(param_kind in classical_kinds, param.classical)

    def test_getitem(self):
        """Treat a parameter like a register or register alias."""
        allowed_kinds = [ParamType.REGISTER, ParamType.NONE]
        for param_kind, param in self.example_params.items():
            if param_kind in allowed_kinds:
                # Note: these don't seem to have an application in Jaqal directly.
                self.assertTrue(isinstance(param[0], NamedQubit))
                self.assertTrue(isinstance(param[0:2], Register))
            else:
                with self.assertRaises(Exception):
                    param[0]
                with self.assertRaises(Exception):
                    param[0:1]


class ParamTypeTester(unittest.TestCase):
    def test_types(self):
        """Test the types property that excludes the NONE item"""
        for typ in ParamType:
            if typ != ParamType.NONE:
                self.assertIn(typ, ParamType.types)
            else:
                self.assertNotIn(typ, ParamType.types)
        for typ in ParamType.types:
            self.assertIn(typ, ParamType)

    def test_make(self):
        """Test that the make classmethod raises a JaqalError or does the same
        as the __call__ classmethod."""

        with self.assertRaises(JaqalError):
            ParamType.make("nonsense")
