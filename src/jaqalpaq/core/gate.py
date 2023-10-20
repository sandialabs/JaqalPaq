# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
from itertools import zip_longest
import math
import warnings

from jaqalpaq.error import JaqalError


class GateStatement:
    """
    Represents a Jaqal gate statement.

    :param GateDefinition gate_def: The gate to call.
    :param dict parameters: A map from gate parameter names to the values to pass for those parameters. Can be omitted for gates that have no parameters.
    """

    def __init__(self, gate_def, parameters=None):
        self._gate_def = gate_def
        if parameters is None:
            self._parameters = {}
        else:
            self._parameters = parameters

    def __repr__(self):
        params = ", ".join(
            [repr(self.name)] + [repr(param) for param in self._parameters.values()]
        )
        return f"GateStatement({params})"

    def __eq__(self, other):
        def are_equal(p0, p1):
            if isinstance(p0, float) and math.isnan(p0):
                return isinstance(p1, float) and math.isnan(p1)
            return p0 == p1

        try:
            return self.name == other.name and all(
                are_equal(sparam, oparam)
                for sparam, oparam in zip_longest(
                    self.parameters_linear, other.parameters_linear
                )
            )
        except AttributeError:
            return False

    @property
    def name(self):
        """
        The name of the gate being called.
        """
        return self._gate_def.name

    @property
    def gate_def(self):
        """
        The :class:`GateDefinition` of the gate being called.
        """
        return self._gate_def

    @property
    def used_qubits(self):
        """
        An iterator over the qubits acted on by this gate statement. This may
        include the special symbol `all` indicating the gate operates on all qubits.
        """
        for param in self.gate_def.used_qubits:
            if param is all:
                yield all
                return
            yield self._parameters[param.name]

    @property
    def parameters(self):
        """(deprecated) Read-only access to the dictionary mapping
        gate parameter names to the associated values.
        """
        warnings.warn(
            "Please use one of the Gate.parameters_* properties", DeprecationWarning
        )
        return self._parameters

    @property
    def parameters_by_name(self):
        """Return a dictionary mapping parameter names to the
        associated values. If a parameter is variadic, it will be
        mapped to a list of its values.
        """
        return self._parameters

    @property
    def parameters_linear(self):
        """Return an ordered list of all parameter values. Variadic
        parameters are listed indistinguishable from others."""
        res = []
        for val in self._parameters.values():
            if not isinstance(val, list):
                res.append(val)
            else:
                res.extend(val)
        return res

    @property
    def parameters_with_types(self):
        """Return a list of 2-tuples of (value, type) for each
        parameter. Variadic parameters are included like all other
        parameters."""
        res = []
        for val, typ in zip(self._parameters.values(), self._gate_def.parameters):
            if not typ.variadic:
                res.append((val, typ))
            else:
                res.extend((v, typ) for v in val)
        return res
