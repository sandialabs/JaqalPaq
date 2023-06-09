jaqalpaq.transpilers.qiskit package
===================================

.. automodule:: jaqalpaq.transpilers.qiskit
   :members:
   :undoc-members:
   :show-inheritance:

qiskit.circuit.QuantumCircuit extensions
----------------------------------------

Importing the jaqalpaq.qiskit package also patches the
:class:`qiskit.circuit.QuantumCircuit` object, adding the following four methods. This
follows the standard Qiskit API, allowing users to add trapped-ion gates to circuits using
the same syntax as gates from the Qiskit standard extension.

.. function:: qiskit.circuit.QuantumCircuit.jaqalms(theta, phi, a, b)

    Add a two-parameter Mølmer-Sørensen gate to a circuit.

    :param float theta: The angle by which the gate rotates the state.
    :param float phi: The phase angle determining the mix of XX and YY rotation.
    :param a: The first qubit to act on.
    :type a: qiskit.circuit.Bit, qiskit.circuit.Register, int, slice, list, or range
    :param b: The second qubit to act on.
    :type b: qiskit.circuit.Bit, qiskit.circuit.Register, int, slice, list, or range

.. function:: qiskit.circuit.QuantumCircuit.jaqalr(theta, phi, q)

    Add a single-qubit gate representing arbitrary rotation around an axis in the X-Y
    plane to a circuit. Note that this is essentially a different parametrization of
    Qiskit's U2 gate.

    :param float theta: The angle that sets the planar axis to rotate around.
    :param float phi: The angle by which the gate rotates the state.
    :param q: The qubit to act on.
    :type q: qiskit.circuit.Bit, qiskit.circuit.Register, int, slice, list, or range

.. function:: qiskit.circuit.QuantumCircuit.sy(q)

    Add a sqrt(Y) gate to a circuit.

    :param q: The qubit to act on.
    :type q: qiskit.circuit.Bit, qiskit.circuit.Register, int, slice, list, or range

.. function:: qiskit.circuit.QuantumCircuit.sydg(q)

    Add an inverse sqrt(Y) gate to a circuit.

    :param q: The qubit to act on.
    :type q: qiskit.circuit.Bit, qiskit.circuit.Register, int, slice, list, or range
