from itertools import zip_longest
import math

from jaqal import JaqalError


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
            [repr(self.name)] + [repr(param) for param in self.parameters.values()]
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
                    self.parameters.values(), other.parameters.values()
                )
            )
        except AttributeError:
            return False

    @property
    def name(self):
        return self._gate_def.name

    @property
    def gate_def(self):
        return self._gate_def

    @property
    def parameters(self):
        """
        Read-only access to the dictionary mapping gate parameter names to the associated values.
        """
        return self._parameters

    def moment_iter(self, parameters=None):
        yield [self]