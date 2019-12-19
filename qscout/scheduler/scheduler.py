from qscout.core import GateBlock, LoopStatement, GateStatement, QUBIT_PARAMETER

def schedule_circuit(circ):
	schedule_block(circ, circ.gates)

def schedule_block(circ, block):
	force_flatten = False
	for instr in block.gates:
		if isinstance(instr, GateBlock):
			schedule_block(circ, instr)
			if not (instr.parallel or block.parallel):
				force_flatten = True
		elif isinstance(instr, LoopStatement):
			schedule_block(circ, instr.gates)
	if block.parallel is None:
		new_block = GateBlock()
		used_qubits = circ.used_qubit_indices(block)
		freeze_timestamps = {regname: {idx: -1 for idx in used_qubits[regname]} for regname in used_qubits}
		for instr in block:
			schedule_instr(circ, instr, new_block, freeze_timestamps)
	elif force_flatten:
		# The block we're trying to schedule is locked to sequential order, but one of the
		# sub-blocks of that block is also sequentially ordered. (This is usually a result
		# of a sequential block containing an unscheduled block, which we then schedule
		# into a sequential block.) Nesting blocks of the same type is forbidden by Jaqal,
		# so we need to flatten the inner block.
		i = 0
		while i < len(block.gates): # We can't do this with a for loop since block.gates
			# will get longer as we're iterating over it.
			inc = 1
			if isinstance(block.gates[i], GateBlock) and not block.gates[i].parallel:
				inc = len(block.gates[i].gates) - 1
				block.gates[i:i+1] = block.gates[i].gates
			i += inc			

def schedule_instr(circ, instr, target, freeze_timestamps)
	used_qubits = circ.used_qubit_indices(instr)
	is_block = isinstance(instr, GateBlock)
	is_gate = isinstance(instr, GateStatement)
	is_loop = isinstance(instr, LoopStatement)
	if (is_block and instr.parallel) or is_gate:
		defrost = 0
		for reg in used_qubits:
			for idx in used_qubits[reg]:
				defrost = max(defrost, freeze_timestamps[reg][idx] + 1)
		while defrost < len(target.gates) and not can_parallelize(circ, target.gates[defrost], instr, used_qubits):
			defrost += 1
		if defrost >= len(target.gates):
			target.gates.append(instr)
		elif is_block:
			target.gates[defrost].gates.extend(instr.gates)
		else:
			target.gates[defrost].gates.append(instr)
	elif is_block:
		# You can't nest two sequential blocks, so we flatten the block.
		for instr in instr.gates:
			schedule_instr(circ, instr, new_block, freeze_timestamps)
		return # We've frozen all the relevant qubits already.
	elif is_loop:
		# Loop statements can't be parallelized with anything; just stick it at the end
		defrost = len(target.gates) # Any qubit used in the loop shouldn't be touched
		target.gates.append(instr)  # Until after the loop finishes
	else:
		raise QSCOUTError("Can't schedule instruction %s." % str(instr))
	for reg in used_qubits:
		for idx in used_qubits[reg]:
			freeze_timestamps[reg][idx] = defrost

def can_parallelize(circ, block, instr, qubits):
	if isinstance(block, GateBlock) and block.parallel:
		block_used_qubits = circ.used_qubit_indices(block)
		for reg in block_used_qubits:
			if reg in qubits and not isdisjoint(block_used_qubits[reg], qubits[reg]):
				return False # Can't act on the same qubit twice simultaneously.
		for sub_instr in block:
			if not can_parallelize_sub_instr(circ, sub_instr):
				return False
		if isinstance(instr, GateBlock):
			for sub_instr in instr:
				if not can_parallelize_gate(circ, block, sub_instr, qubits):
					return False # If we can parallelize all the components, we can parallelize the block.
		elif isinstance(instr, GateStatement):
			return can_parallelize_gate(circ, block, instr, qubits)
		else:
			return False # We don't know what this is, so we can't parallelize it.
	else:
		return False # Not a parallel block; can't add more instructions in parallel.

def can_parallelize_gate(circ, block, instr, qubits):
	if not can_parallelize_subinstr(circ, instr):
		return False
	else:
		# We could do other checks here, but right now there's nothing we need to worry
		# about. Specifically, if there were two instructions that couldn't be in parallel
		# with each other, even on different qubits, but could be in parallel with other
		# instructions, this is where we'd test for it. I expect this and
		# can_parallelize_subinstr to change as our hardware evolves, whereas everything
		# above shouldn't need to, since it tests for limitations like qubit overlap that
		# aren't dependent on a specific hardware implementation.
		return True

def can_parallelize_subinstr(circ, sub_instr):
	if not isinstance(sub_instr, GateStatement):
		return False # Too much nested structure.
	if sub_instr.name not in circ.native_gates:
		return False # Can't do macros in parallel, because they could include anything.
	if len([p for p in circ.native_gates[sub_instr.name].parameters if p.kind == QUBIT_PARAMETER]) > 1:
		return False # Can't do multiple 2-qubit gates at once.
	if sub_instr.name in ['prepare_all', 'measure_all']:
		return False # Can't do gates while preparing or measuring ions.
	return True