from jaqalpaq.error import JaqalError
from jaqalpaq._import import jaqal_import


def get_ideal_action(gate, jaqal_filename=None, *, default=None):
    if hasattr(gate, "_ideal_unitary"):
        return gate._ideal_unitary

    try:
        origin = gate.origin
    except AttributeError:
        raise JaqalError("Unknown gate origin")

    jg = jaqal_import(origin, "jaqal_action")
    return jg.IDEAL_ACTION.get(gate.name, default)
