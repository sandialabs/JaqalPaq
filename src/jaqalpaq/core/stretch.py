# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
from .parameter import Parameter, ParamType
from .gatedef import IdleGateDefinition


def stretched_gates(gates, *, suffix=None, update=False, origin=None):
    """Generate stretched GateDefinitions from parent GateDefinitions

    :param gates: A dictionary of GateDefinition objects representing the gates
      available.  The keys are ignored, and the intrinsic gate names are processed.
    :param suffix str: (optional) A suffix to append to the names of the gates.
    :param update: (default False) If True, return gates after updating with the new
      stretched gates.
    :param origin: Assign an originating Jaqal module for the gate definitions
    :type origin: str

    :return dict: The stretched gates, with keys being the gate names.

    If an idle gate is passed in via gates, a stretched gate for its parent gate is
    automatically generated.
    """
    new_gates = {}
    for gate in gates.values():
        name = gate.name
        if name in new_gates:
            # We already processed the idle gate for this gate,
            # and generated the parent gate.
            continue

        if isinstance(gate, IdleGateDefinition):
            add_idle = True
            gate = gate._parent_def
        else:
            add_idle = False

        if suffix:
            new_name = gate.name + suffix
        else:
            new_name = None

        parameters = gate.parameters.copy()
        parameters.append(Parameter("stretch", ParamType.FLOAT))

        new_gate = gate.copy(
            name=new_name,
            parameters=parameters,
            origin=origin,
        )

        if hasattr(gate, "_ideal_unitary"):
            # (this process is deprecated)
            new_gate._ideal_unitary = do_stretch_as_noop(gate._ideal_unitary)

        new_gates[new_name] = new_gate
        if add_idle:
            new_name = name + suffix
            new_gate = IdleGateDefinition(new_gate, name=new_name)
            new_gates[new_name] = new_gate

    if update:
        gates.update(new_gates)
        return gates
    return new_gates


def do_stretch_as_noop(unitary):
    """Returns a wrapped function that takes (and ignores) a stretch
    factor as the last parameter.
    """
    if unitary is None:
        return None

    def stretched_unitary(*args):
        # Drop the last argument, which will be the stretch factor
        return unitary(*args[:-1])

    return stretched_unitary


def stretched_unitaries(unitaries, *, suffix=None, update=False):
    """Generate stretched unitaries from original unitaries and stretched GateDefinitions

    :param unitaries: A dictionary of original unitaries, with keys corresponding
      to the gate names.
    :param suffix str:  The suffix to append to the names of the gates to find the
      stretched gate name
    :param update: (default False) If True, return unitaries after updating with the
      new stretched gates.

    :return dict: The unitaries for the stretched gates, with keys being the gate names.

    If an idle gate is passed in via gates, a stretched gate for its parent gate is
    automatically generated.
    """

    new_unitaries = {}
    for name, unitary in unitaries.items():
        if suffix:
            name = name + suffix

        new_unitaries[name] = do_stretch_as_noop(unitary)

    if update:
        unitaries.update(new_unitaries)
        return unitaries
    return new_unitaries
