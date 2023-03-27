# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
from collections import OrderedDict

from jaqalpaq.error import JaqalError
from .gate import GateStatement

import warnings


class AbstractGate:
    """
    The abstract base class for gate definitions. Everything here can be used whether the
    gate is defined by a macro in Jaqal, or is a gate defined by a pulse sequence in a
    gate definition file.

    :param str name: The name of the gate.
    :param parameters: What arguments (numbers, qubits, etc) the gate should be called
        with. If None, the gate takes no parameters.
    :type parameters: list(Parameter) or None
    :param function ideal_unitary: (deprecated) A function mapping a list of all classical
        arguments to a numpy 2D array representation of the gate's ideal action in the
        computational basis.
    """

    def __init__(self, name, parameters=None, ideal_unitary=None):
        self._name = name
        if parameters is None:
            self._parameters = []
        else:
            self._parameters = parameters

        if ideal_unitary is not None:
            warnings.warn("Define unitary in <path>.jaqal_action", DeprecationWarning)
            self._ideal_unitary = ideal_unitary

    def __repr__(self):
        return f"{type(self).__name__}({self.name}, {self.parameters})"

    def __eq__(self, other):
        try:
            return self.name == other.name and self.parameters == other.parameters
        except AttributeError:
            return False

    @property
    def name(self):
        """The name of the gate."""
        return self._name

    @property
    def parameters(self):
        """
        What arguments (numbers, qubits, etc) the gate should be called with.
        """
        return self._parameters

    def parse_parameters(self, *args, **kwargs):
        """
        Create a :class:`GateStatement` that calls this gate.
        The arguments to this method will be the arguments the gate is called with.
        If all arguments are keyword arguments, their names should match the names of this
        gate's parameters, and the values will be passed accordingly.
        If all arguments are positional arguments, each value will be passed to the next
        parameter in sequence.
        For convenience, calling the AbstractGate like a function is equivalent to this.

        :returns: The new statement.
        :rtype: GateStatement
        :raises JaqalError: If both keyword and positional arguments are passed.
        :raises JaqalError: If the wrong number of arguments are passed.
        :raises JaqalError: If the parameter names don't match the parameters this gate
            takes.
        """
        params = OrderedDict()
        if args and not kwargs:
            if len(args) > len(self.parameters):
                raise JaqalError(f"Too many parameters for gate {self.name}.")
            elif len(args) > len(self.parameters):
                raise JaqalError(f"Insufficient parameters for gate {self.name}.")
            else:
                for name, arg in zip([param.name for param in self.parameters], args):
                    params[name] = arg
        elif kwargs and not args:
            try:
                for param in self.parameters:
                    params[param.name] = kwargs.pop(param.name)
            except KeyError as ex:
                raise JaqalError(
                    f"Missing parameter {param.name} for gate {self.name}."
                ) from ex
            if kwargs:
                raise JaqalError(
                    f"Invalid parameters {', '.join(kwargs)} for gate {self.name}."
                )
        elif kwargs and args:
            raise JaqalError(
                "Cannot mix named and positional parameters in call to gate."
            )
        if len(self.parameters) != len(params):
            raise JaqalError(
                f"Bad argument count: expected {len(self.parameters)}, found {len(params)}"
            )
        for param in self.parameters:
            param.validate(params[param.name])
        return params

    def call(self, *args, **kwargs):
        """(deprecated) use gatedef(*args) instead"""
        warnings.warn("Use gatedef() instead of gatedef.call()", DeprecationWarning)
        return self(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        params = self.parse_parameters(*args, **kwargs)
        return GateStatement(self, params)

    def copy(
        self,
        *,
        name=None,
        parameters=None,
        ideal_unitary=False,
        origin=False,
        unitary=None,
    ):
        """Returns a shallow copy of the gate or gate definition.

        :param name: (optional) change the name in the copy.
        :param parameters: (optional) change the parameters in the copy.
        """

        kls = type(self)
        copy = kls.__new__(kls)
        copy.__dict__ = self.__dict__.copy()
        if name is not None:
            copy._name = name
        if parameters is not None:
            copy._parameters = parameters
        if ideal_unitary is not False:
            copy._ideal_unitary = ideal_unitary
        if origin is not False:
            copy._origin = origin
        if unitary is not None:
            copy._unitary = unitary

        return copy


class GateDefinition(AbstractGate):
    """
    Base: :class:`AbstractGate`

    Represents a gate that's implemented by a pulse sequence in a gate definition file.
    :param bool unitary: Whether or not the gate represents a purely unitary action.
    """

    def __init__(self, *args, origin=None, unitary=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._origin = origin
        self._unitary = unitary

    @property
    def origin(self):
        """The Jaqal module in which this gate is defined."""
        return self._origin

    @property
    def unitary(self):
        """Whether or not the gate is (ideally) unitary."""
        return self._unitary

    @property
    def used_qubits(self):
        """Return the parameters in this gate that are qubits. Subclasses may
        return the special symbol `all` indicating they operate on all
        qubits. Otherwise this is identical to quantum_parameters."""
        for p in self.parameters:
            try:
                if not p.classical:
                    yield p
            except JaqalError:
                # This happens if we don't have a real gate definition.
                # Lean on the upper layers being able to infer the type.
                yield p

    @property
    def quantum_parameters(self):
        """The quantum parameters (qubits or registers) this gate takes.

        :raises JaqalError: If this gate has parameters without type annotations; for
            example, if it is a macro.
        """
        try:
            return [param for param in self.parameters if not param.classical]
        except JaqalError:
            pass
        raise JaqalError("Gate {self.name} has a parameter with unknown type")

    @property
    def classical_parameters(self):
        """The classical parameters (ints or floats) this gate takes.

        :raises JaqalError: If this gate has parameters without type annotations; for
            example, if it is a macro.

        """
        try:
            return [param for param in self.parameters if param.classical]
        except JaqalError:
            pass
        raise JaqalError("Gate {self.name} has a parameter with unknown type")

    @property
    def ideal_unitary(self):
        """(deprecated) The ideal unitary action of the gate on its target qubits"""
        warnings.warn("Use get_ideal_action instead", DeprecationWarning)
        from jaqalpaq.emulator._import import get_ideal_action

        return get_ideal_action(self)


class IdleGateDefinition(GateDefinition):
    """
    Base: :class:`GateDefinition`

    Represents a gate that merely idles for some duration.
    """

    def __init__(self, gate, name=None):
        # Special case handling of prepare and measure gates
        if gate.name in ("prepare_all", "measure_all"):
            raise JaqalError(f"Cannot make an idle gate for {gate.name}")
        super().__init__(
            name=name if name else f"I_{gate.name}",
            parameters=gate._parameters,
            origin=gate.origin,
            unitary=True,
        )
        self._parent_def = gate

    @property
    def used_qubits(self):
        """Iterates over the qubits used by an idle gate: nothing.

        The idle operation does not act on any qubits.
        """
        yield from ()

    def __call__(self, *args, **kwargs):
        params = self._parent_def.parse_parameters(*args, **kwargs)
        return GateStatement(self, params)


class BusyGateDefinition(GateDefinition):
    """
    Base: :class:`AbstractGate`

    Represents an operation that cannot be parallelized with any other operation.
    """

    @property
    def used_qubits(self):
        yield all


def add_idle_gates(active_gates):
    """Augments a dictionary of gates with associated idle gates.

    :param active_gates: A dictionary of GateDefinition objects representing the active
      gates available.
    :return:  A list of GateDefinitions including both the active gates passed, and their
      associated idle gates.
    """
    gates = {}
    for n, g in active_gates.items():
        gates[n] = g
        idle_gate = IdleGateDefinition(g)
        gates[idle_gate.name] = idle_gate

    return gates
