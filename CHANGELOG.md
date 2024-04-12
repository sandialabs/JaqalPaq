Changes in 1.2
==============

 - NEW: ExecutionResults API
 - NEW: Batching and Let override controls
 - NEW: IPC framework for execution on actual hardware
 - CHANGED: `get_jaqal_action` for noiseless emulation
 - DEPRECATED: Pre-1.2 ExecutionResults API

Changes in 1.1
==============

 - NEW: qscout-gatemodels-ionsim package
   - Microscopic models of gate behavior
 - NEW: Q syntax
   - Simpler interface for writing Jaqal with Python
 - NEW: Subcircuits
   - Encapsulates prepare/measure blocks
 - NEW: JaqalPaw
   - Describes pulses and waveforms used for gates
 - NEW: Reverse transpilers
   - Convert from Qiskit and TKET to Jaqal
 - NEW: `relative_frequency_by_*` and `simulated_probability_by_*`
    added to disambiguate experimental vs. simulated output
 - CHANGED: Default UnitarySerializedEmulator
   - Does not use pyGSTi
   - Faster
   - Provides access to the full unitary of the gates in the circuit
 - CHANGED: Unified Jaqal name space hierarchy
 - CHANGED: Refreshed dependencies on external packages
 - REMOVED: `readouts`, which are not implemented as such on
    the hardware
 - FIXED: Broaden compatibility for JaqalPaw pulse definitions
