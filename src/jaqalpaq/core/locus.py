# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
from .block import LoopStatement
from .branch import BranchStatement, CaseStatement
from .circuit import Circuit


class Locus:
    """A lightweight class capturing a spot inside of a circuit.
    :param obj: What object in the circuit is being referred to.
    :param Locus parent: The owner of the object referred to.
    :param index: The index of the object referred to, among its siblings.
    """

    __slots__ = ("_object", "_parent", "_index")

    @classmethod
    def from_address(klass, circuit, address):
        """Generate a Locus object from a list of indices

        :param Circuit circuit: The circuit the address is looking into.
        :param list address: A list of integers indexing the children at each step.
        """
        obj = klass(circuit)
        for term in address:
            obj = obj[term]
        return obj

    def __init__(self, obj, *, parent=None, index=None):
        self._object = obj
        self._parent = parent
        self._index = index

    @property
    def index(self):
        """The index of the object referred to, among its siblings."""
        return self._index

    @property
    def parent(self):
        """The owner of the object referred to."""
        return self._parent

    @property
    def object(self):
        """What object in the circuit is being referred to."""
        return self._object

    @property
    def lineage(self):
        """An iterator yielding all Locus ancestors."""
        if self._parent:
            yield from self._parent.lineage
        yield self

    @property
    def address(self):
        """A list of integers indexing the children at each step."""
        return tuple(link._index for link in self.lineage if link._index is not None)

    @property
    def children(self):
        """All child objects owned by the Locus's object."""
        obj = self._object
        if isinstance(obj, Circuit):
            return obj.body.statements
        elif isinstance(obj, (LoopStatement, CaseStatement)):
            return obj.statements.statements
        elif isinstance(obj, BranchStatement):
            return obj.cases
        else:
            assert isinstance(obj.statements, list)
            return obj.statements

    def __getitem__(self, index):
        """Returns a new Locus object referring to a child of a particular index."""
        return type(self)(self.children[index], parent=self, index=index)

    def __repr__(self):
        return f"Locus<{self.address}>"

    def __eq__(self, other):
        if self is other:
            return True
        return (
            (self._object == other._object)
            and (self._parent == other._parent)
            and (self._index == other._index)
        )
