from .block import BlockStatement
from .gate import GateStatement
from .gatedef import AbstractGate
from jaqal import QSCOUTError

class Macro(AbstractGate):
	"""
	Base: :class:`AbstractGate`
	
	Represents a gate that's implemented by Jaqal macro statement.
	
	:param str name: The name of the gate.
	:param parameters: What arguments (numbers, qubits, etc) the gate should be called with. If None, the gate takes no parameters.
	:type parameters: list(Parameter) or None
	:param body: The implementation of the macro. If None, an empty sequential BlockStatement is created.
	:type body: BlockStatement or None
	"""
	def __init__(self, name, parameters=None, body=None):
		super().__init__(name, parameters)
		if body is None:
			self._body = BlockStatement()
		else:
			self._body = body
	
	@property
	def body(self):
		"""
		A :class:`GateBlock` that implements the macro.
		"""
		return self._body