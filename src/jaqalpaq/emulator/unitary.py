# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.
import numpy

from jaqalpaq.error import JaqalError
from jaqalpaq.run.cursor import SubcircuitCursor, State
from jaqalpaq.run import result
from jaqalpaq.emulator.backend import EmulatedIndependentSubcircuitsBackend
from ._import import get_ideal_action


def inplace_multiply(dsub, qind, vec, scratch):
    # The plan is to apply the associated unitary to vec for each gate.
    # now we need to sparse-multiply:
    # vec = U * inp
    # But! U isn't just dsub

    # The current state-vector becomes the input to the matrix multiplication
    inp, vec = vec, scratch
    # (Notice that this initializes inp, from above)
    vec[:] = 0

    # For every column in the output matrix, we need to compute the sum
    # over the nonzero rows of U.

    # However, this corresponds to a sum only over rows of dsub, with
    # a particular mapping between the rows and columns.
    for i in range(len(vec)):
        # Because we are only dealing with qubits, the binary representation
        # of the row is precisely the standard basis label corresponding to that
        # row.

        # We need to re-map the qubits that are being acted on by dsub to
        # the *column* of dsub.  Additionally, the qubits that are not being
        # acted on by dsub are unaffected, and therefore are the same in both
        # the input and output.

        # mask is precisely these bystander qubits --- the affected qubits are
        # set to zero, and shuffled into another variable, dsub_row
        mask = i

        # We initialize dsub_row to zero, and then build it up via bit-twiddling
        dsub_row = 0

        for dsub_bit, i_k in enumerate(qind):
            # We iterate over all the qubits acted on by dsub

            # Is this specific qubit high (for this row)?
            n_high = mask & (0b1 << i_k)

            # If it is high, lower it in the mask
            # (notice this is equivalent to subtraction)
            mask ^= n_high

            # If it is high, raise it in the row to be passed to dsub
            # (notice this is equivalent to addition)
            dsub_row |= (n_high >> i_k) << dsub_bit

        # We now have the row in dsub that corresponds to the row in U

        # Next, we need to iterate over the column of U, and simultaneously the
        # column of inp --- this mapping is the same as the above, backwards.
        for dsub_col in range(dsub.shape[0]):
            j = mask
            dsub_col_stack = dsub_col
            for dsub_bit, j_k in enumerate(qind):
                j |= (dsub_col_stack & 0b1) << j_k
                dsub_col_stack >>= 1

            # Suitably armed with the associated row and column, we
            # do the standard matrix accumulation sum step.

            vec[i] += dsub[dsub_row, dsub_col] * inp[j]

    return vec, inp


class UnitarySerializedEmulator(EmulatedIndependentSubcircuitsBackend):
    """Serialized emulator using unitary matrices

    This object should be treated as an opaque symbol to be passed to run_jaqal_circuit.
    """

    def _simulate_subcircuit(self, job, subcirc):
        """Generate the ProbabilisticSubcircuit associated with the trace of circuit
            being process in job.

        :param job: the job object controlling the emulation
        :param int index: the index of the trace in the circuit
        :param Trace trace: the trace of the subcircuit
        """
        circ = subcirc.filled_circuit
        gatedefs = circ.native_gates

        # fill_in_let modified the circuit here
        cursor = SubcircuitCursor(subcirc.start, subcirc.end)

        n_qubits = self.get_n_qubits(circ)

        # We serialize the subcircuit, obtaining a list of gates.
        # The plan is to apply the associated unitary to vec for each gate.
        hilb_dim = 2**n_qubits
        vec = numpy.zeros(hilb_dim, dtype=complex)
        vec[0] = 1

        def handle_final_measurement(cursor, vec, node):
            P = numpy.abs(vec) ** 2
            P = result.validate_probabilities(P)
            for i, prob in enumerate(P):
                if prob <= result.CUTOFF_ZERO:
                    continue
                meas_cursor = cursor.copy()
                meas = meas_cursor.next_measure()
                assert meas_cursor.state == State.shutdown
                sub = node.force_get(i, meas_cursor)
                assert sub.classical_state == meas_cursor
                sub.simulated_probability = prob
            assert node.classical_state == cursor
            node.state_vector = vec

        def handle_unitary(cursor, vec, node):
            # We don't need to initialize this yet.
            scratch = numpy.empty(hilb_dim, dtype=complex)

            while cursor.state == State.gate:
                gate = cursor.next_gate()
                cursor.report_gate_executed()

                # This captures the classical arguments to the gate
                argv = []
                # This capture the quantum arguments to the gate --- the qubit index
                qind = []
                gatedef = gatedefs[gate.name]
                ideal_unitary = get_ideal_action(gatedef)
                if ideal_unitary is None:
                    continue

                for param, val in zip(gatedef.parameters, gate.parameters.values()):
                    if param.classical:
                        argv.append(val)
                    else:
                        qind.append(val.alias_index)

                # This is the dense submatrix
                dsub = ideal_unitary(*argv)

                vec, scratch = inplace_multiply(dsub, qind, vec, scratch)

            if cursor.state == State.final_measurement:
                handle_final_measurement(cursor, vec, node)
            else:
                raise NotImplementedError()

        handle_unitary(cursor, vec, subcirc.tree)
