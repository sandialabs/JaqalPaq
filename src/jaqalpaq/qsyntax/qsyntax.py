# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
# certain rights in this software.

from contextlib import contextmanager
import importlib
import functools
from typing import Any
from inspect import getfullargspec, currentframe
import warnings

from jaqalpaq.error import JaqalError
from jaqalpaq.core.circuitbuilder import build


class CircuitInterface:
    def __init__(self):
        self._blocks = {}
        # True when we are actively instantiating a circuit
        self._in_circuit_instantiation = False

    def clear(self):
        """Remove any stored function blocks. Mostly useful for unit testing,
        but could also be useful in a notebook."""
        self._blocks = {}

    @property
    def sequential(self):
        """Decorator that registers a function as a sequential block that can
        be inserted later."""
        self._verify_outside_circuit("sequential")
        return DecoratorOrContext(self._sequential_decorator, self._sequential_context)

    def _sequential_decorator(self, func):
        return _register_block_function(self._blocks, func, "sequential")

    def _sequential_context(self):
        raise JaqalError(f"Use the Q object to open a sequential context")

    @property
    def parallel(self):
        """Decorator that registers a function as a parallel block that can
        be inserted later."""
        self._verify_outside_circuit("parallel")
        return DecoratorOrContext(self._parallel_decorator, self._parallel_context)

    def _parallel_decorator(self, func):
        return _register_block_function(self._blocks, func, "parallel")

    def _parallel_context(self):
        raise JaqalError(f"Use the Q object to open a parallel context")

    @property
    def subcircuit(self):
        """Decorator that registers a function as a subcircuit block that can
        be inserted later."""
        self._verify_outside_circuit("subcircuit")
        return DecoratorOrContext(self._subcircuit_decorator, self._subcircuit_context)

    def _subcircuit_decorator(self, func, argument=1):
        return _register_block_function(self._blocks, func, "subcircuit", argument)

    def _subcircuit_context(self, *args):
        raise JaqalError(f"Use the Q object to open a subcircuit context")

    @property
    def loop(self):
        """Decorator that registers a function as a loop that can be inserted
        later."""
        self._verify_outside_circuit("loop")
        return DecoratorOrContext(self._loop_decorator, self._loop_context)

    def _loop_decorator(self, func, repeats):
        return _register_block_function(self._blocks, func, "loop", repeats)

    def _loop_context(self, *args):
        raise JaqalError(f"Use the Q object to open a loop context")

    def _verify_outside_circuit(self, name):
        if self._in_circuit_instantiation:
            msg = (
                f"@circuit.{name} decorators may not be nested or defined "
                + f"inside a circuit definition. Either move this definition to "
                + f"global scope, or use @Q.{name} when inside a circuit definition."
            )
            raise JaqalError(msg)

    def __call__(
        self,
        *args,
        inject_pulses=None,
        autoload_pulses="ignore",
        **kwargs,
    ):
        """Inner decorator function defining a Jaqal circuit by adding all
        statements defined inside of the function decorated with this
        decorator.

        :param inject_pulses: If given, use these pulses specifically.

        :param autoload_pulses: Whether to use the usepulses statement
        when parsing. Can be given the special value "ignore" (default) to
        use import pulses when possible, but continue in the case of
        failure.

        :rtype: QCircuit

        """

        context = ContextLookup(currentframe().f_back)

        def outer(func):
            @functools.wraps(func)
            def inner(*args, **kwargs):
                try:
                    self._in_circuit_instantiation = True
                    stack = Stack()
                    qsyntax = Q(stack, dict(self._blocks), context)
                    with stack.frame():
                        func(qsyntax, *args, **kwargs)
                        return circuit_from_stack(
                            qsyntax,
                            inject_pulses=inject_pulses,
                            autoload_pulses=autoload_pulses,
                        )
                finally:
                    self._in_circuit_instantiation = False

            argspec = getfullargspec(func)
            default_count = len(argspec.defaults) if argspec.defaults else 0
            return mark_qcircuit(inner, len(argspec.args) - default_count - 1)

        if len(args) > 0 and callable(args[0]):
            return outer(args[0])
        else:
            return outer


class ContextLookup:
    """Look up a particular symbol in the context given by a stack
    frame. Saves the original context, so that we can also detect
    changes in the context."""

    def __init__(self, frame):
        if frame is not None:
            self._locals = dict(frame.f_locals)
            self._globals = dict(frame.f_globals)
        else:
            self._locals = self._globals = {}

    def __contains__(self, key):
        return key in self._locals or key in self._globals

    def __getitem__(self, key):
        try:
            return self._locals[key]
        except Exception:
            pass
        return self._globals[key]


# The primary way the user will interface with Qsyntax. When used as a
# decorator on a function, creates a circuit. It can also be used to
# register certain global objects.
circuit = CircuitInterface()


class Q:
    """A local namespace for instructions that the user issues to create a
    Jaqal circuit. The framework creates this and automatically passes
    it as the first argument of a user-defined circuit.

    """

    def __init__(self, stack, blocks, context):
        # A stack mimicking the structure of the Jaqal program
        self._stack = stack
        # Stored blocks that are inserted later. Similar to macros but
        # not implemented using Jaqal macros.
        self._blocks = blocks
        # The context in which the circuit was declared
        self._context = context

    def __getattr__(self, name):
        """Return an object that represents a gate with an unknown
        definition.

        :param string name: The name of the gate
        """

        if name in self._blocks:
            self._check_block_context(name)
            return self._make_block(*self._blocks[name])

        return QGate(name, self._stack)

    def _check_block_context(self, name):
        """Check if the given block is found in the context, and if it
        is, but doesn't match the one we are about to use, issue a
        warning."""
        if name not in self._context:
            return
        ctx_obj = self._context[name]
        if not hasattr(ctx_obj, "__wrapped__"):
            # Something has the same name as the block, but isn't a block
            return
        blk = self._blocks[name][0]
        if ctx_obj.__wrapped__ is not blk:
            msg = (
                f"When looking up block {name}, the actual block differs "
                + "from the one in the calling context. This might mean you "
                + "are using a different block definition than you expect. If "
                + f"you intended to override {name}, consider deleting it in "
                + "the calling context."
            )
            warnings.warn(msg)

    def _make_block(self, func, context_name, *ctxargs):
        """Create a block from a function that was stored earlier."""

        @functools.wraps(func)
        def do_block(*funcargs):
            with self._block_context(context_name, ctxargs):
                func(self, *funcargs)

        return do_block

    @contextmanager
    def _block_context(self, context_name, args):
        context = getattr(self, context_name)
        with context(*args):
            yield

    def register(self, size, name=None):
        """Create a register with the given size.

        :param size: The number of qubits in the register.
        :type size: int or QConstant
        :rtype: QRegister
        """

        reg = QRegister(size, name=name)
        self._stack.set_register(reg)
        return reg

    def let(self, value, name=None):
        """Define a constant in Jaqal. This isn't strictly necessary, but can
        have some benefits such as allowing the hardware to override this
        value.

        :param value: The value that this constant will always be equal to.
        :type value: int, float, or QConstant
        :rtype: QConstant
        """

        let = QConstant(value, name=name)
        self._stack.set_let(let)
        return let

    @property
    def loop(self):
        """Mark a block or function as representing a Jaqal loop."""
        return DecoratorOrContext(self._loop_decorator, self._loop_context)

    def _loop_decorator(self, func, repeats):
        return _register_block_function(self._blocks, func, "loop", repeats)

    def _loop_context(self, repeats):
        with self._stack.frame():
            yield
            loop = QLoop.from_stack(self._stack, argument=repeats)
        self._stack.set_statement(loop)

    @property
    def sequential(self):
        """Mark a block or function as representing a Jaqal sequential
        block."""
        return DecoratorOrContext(self._sequential_decorator, self._sequential_context)

    def _sequential_decorator(self, func):
        return _register_block_function(self._blocks, func, "sequential")

    def _sequential_context(self):
        with self._stack.frame():
            yield
            block = QSequentialBlock.from_stack(self._stack)
        self._stack.set_statement(block)

    @property
    def parallel(self):
        """Mark a block or function as representing a Jaqal parallel
        block."""
        return DecoratorOrContext(self._parallel_decorator, self._parallel_context)

    def _parallel_decorator(self, func):
        return _register_block_function(self._blocks, func, "parallel")

    def _parallel_context(self):
        with self._stack.frame():
            yield
            block = QParallelBlock.from_stack(self._stack)
        self._stack.set_statement(block)

    @property
    def subcircuit(self):
        """Mark a block or function as representing a Jaqal subcircuit
        block."""
        return DecoratorOrContext(self._subcircuit_decorator, self._subcircuit_context)

    def _subcircuit_decorator(self, func, argument=1):
        return _register_block_function(self._blocks, func, "subcircuit", argument)

    def _subcircuit_context(self, argument=1):
        with self._stack.frame():
            yield
            block = QSubcircuitBlock.from_stack(self._stack, argument=argument)
        self._stack.set_statement(block)

    def usepulses(self, module, names="*"):
        """Instruct the Jaqal file to import its pulses from the given
        file."""

        usepulses = QUsePulses(module, names)
        self._stack.set_usepulses(usepulses)
        return usepulses

    @contextmanager
    def branch(self):
        with self._stack.frame():
            yield
            block = QBranch.from_stack(self._stack)
        self._stack.set_statement(block)

    @contextmanager
    def case(self, label):
        with self._stack.frame():
            yield
            block = QCase.from_stack(self._stack, argument=label)
        self._stack.set_statement(block)

    @property
    def registers(self):
        """Return a list of all registers registered so far."""
        return list(self._stack.iter_registers())

    @property
    def lets(self):
        """Return a list of all let constants registered so far."""
        return list(self._stack.iter_lets())


class QRegister:
    def __init__(self, size, name=None):
        self.size = self._validate_normalize_size(size)
        self.name = name

    @staticmethod
    def _validate_normalize_size(size):
        return validate_int(size)

    def __getitem__(self, index):
        # Note: we could check if index is a slice, and create a
        # register alias object. If the slice parameters are all
        # integers, we can do the mapping in this module, but if any
        # of them are let constants, we would have to resort to a map
        # statement. Otherwise, let overrides would not work properly.
        return QNamedQubit(self, index)


class QNamedQubit:
    def __init__(self, source, index):
        self.source = self._validate_source(source)
        self.index = self._validate_normalize_index(index)

    @staticmethod
    def _validate_source(source):
        if not isinstance(source, QRegister):
            # The user should not be able to trigger this error
            raise JaqalError(f"Cannot create named qubit with source {source}")
        return source

    @staticmethod
    def _validate_normalize_index(index):
        return validate_int(index)


class QConstant:
    def __init__(self, value, name=None):
        self.value = self._validate_normalize_constant(value)
        self.name = name

    @staticmethod
    def _validate_normalize_constant(value):
        # Resolve all values down to an integer or floating point
        # number. As a convenience to the user, we will allow them to
        # give us a constant, even though base Jaqal does/did not
        # allow this.
        if isinstance(value, QConstant):
            value = QConstant.value
        if not isinstance(value, (int, float)):
            raise JaqalError(f"Invalid let value {value}")
        return value


def circuit_from_stack(
    qsyntax,
    inject_pulses=None,
    autoload_pulses="ignore",
    prepare="prepare_all",
    measure="measure_all",
):
    """Construct a new QCircuit using the elements that have been pushed
    on the stack."""
    if qsyntax._stack.depth() > 1:
        raise JaqalError("Q stack corrupted: too many stack frames.")

    let_names = [let.name for let in qsyntax.lets]
    register_names = [reg.name for reg in qsyntax.registers]
    namer = Namer(let_names=let_names, register_names=register_names)
    sexpr = ["circuit"]

    imports = {}
    for usepulses in qsyntax._stack.iter_usepulses():
        imports[usepulses.module] = usepulses.names
        sexpr.append(["usepulses", usepulses.module, usepulses.names])

    # TODO: Make sure our generated let and register names don't
    # conflict with user-provided ones

    let_dict = {}
    for let in qsyntax._stack.iter_lets():
        name = namer.name_let(let)
        let_dict[let] = name
        sexpr.append(["let", name, let.value])

    # Gate objects for preparing and measuring all
    # qubits. Measurement and prepartion of a limited number of
    # qubits are treated like any other gate.
    prepare_gate = QGateCall(prepare, ())
    measure_gate = QGateCall(measure, ())

    def lookup_object(obj):
        if isinstance(obj, QConstant):
            return let_dict[obj]
        if isinstance(obj, QRegister):
            return register_dict[obj]
        if isinstance(obj, QNamedQubit):
            return [
                "array_item",
                lookup_object(obj.source),
                lookup_object(obj.index),
            ]

        return obj

    register_dict = {}
    for reg in qsyntax._stack.iter_registers():
        name = namer.name_register(reg)
        register_dict[reg] = name
        sexpr.append(["register", name, lookup_object(reg.size)])

    statements = list(qsyntax._stack.iter_statements())

    do_implicit_measure = len(statements) == 0 or not statements[0].starts_with_prepare(
        prepare
    )

    if do_implicit_measure:
        sexpr.append(prepare_gate.build(lookup_object))

    # Process all statements and fill in their arguments
    for stmt in statements:
        sexpr.append(stmt.build(lookup_object))

    if do_implicit_measure:
        sexpr.append(measure_gate.build(lookup_object))

    # If the user has not given us gates and has provided at least
    # one usepulses statement, and we can import that module, do
    # so when creating the circuit.
    if (
        autoload_pulses
        and inject_pulses is None
        and len(list(qsyntax._stack.iter_usepulses())) > 0
    ):
        for module in imports:
            try:
                importlib.import_module(module)
            except Exception:
                if autoload_pulses != "ignore":
                    raise JaqalError(f"Could not load pulses from `{module}'")
                else:
                    autoload_pulses = False
                    break

    if autoload_pulses == "ignore":
        # We can get here if either inject_pulses was given or all
        # modules could be imported. If there are no usepulses,
        # then we would not have engaged the above logic, and we
        # must set autoload_pulses to False or we will get an
        # error if the user attempts to use any gate.
        autoload_pulses = len(list(qsyntax._stack.iter_usepulses())) > 0

    if not isinstance(autoload_pulses, bool):
        # build() expects autoload_pulses to be Boolean, so we
        # ensure we've filtered out any special values by now.
        raise JaqalError(f"Bad value for autoload_pulses: {autoload_pulses}")

    circ = build(sexpr, inject_pulses=inject_pulses, autoload_pulses=autoload_pulses)
    if circ.native_gates:
        for block_name in qsyntax._blocks:
            if block_name in circ.native_gates:
                raise JaqalError(f"Attempting to redefine gate {block_name}")
    return circ


class QGate:
    """A gate without specific arguments that define it precisely."""

    def __init__(self, name, stack):
        self.name = name
        self._stack = stack

    def __call__(self, *args):
        # We do pretty late binding here, that way the user can create
        # a QGate object, bind it to a name in Python, then call it
        # multiple times with different arguments and create a
        # different gate each time.
        self._stack.set_statement(QGateCall(self.name, args))


class QGateCall:
    """A specific instantiation of a gate with arguments."""

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def build(self, lookup_object):
        """Using a builder object, add a gate with this gate's name and
        arguments. The arguments may need to be looked up with
        lookup_object."""
        return ["gate", self.name, *(lookup_object(arg) for arg in self.args)]

    def starts_with_prepare(self, name):
        return self.name == name


class QBlock:
    """A Jaqal block, either sequential, parallel, or a subcircuit. This
    is meant to be subclassed, although except for internal_name it has
    defaults that work for some of its subclasses."""

    # The name used by the circuit builder for a given block
    internal_name: str = ""

    # The number of arguments this block takes.
    arity: int = 0

    # If the user gives no argument, fill in this. A value of None
    # here indicates it is required.
    default_argument: Any = None

    # Some blocks are actually constructs containing a sequential
    # block as far as the circuit builder back end is
    # concerned. Accommodate them by setting this to True.
    wrap_statements = False

    @classmethod
    def from_stack(cls, stack, argument=None):
        statements = list(stack.iter_statements())
        return cls(statements, argument=argument)

    def __init__(self, statements, argument=None):
        assert self.internal_name != "", "QBlock must be subclassed"
        self.statements = statements
        self.argument = argument

    def build(self, lookup_object):
        ret = []

        ret.append(self.internal_name)

        assert self.arity in (0, 1), "Internal block arity badly implemented"
        if self.arity == 1:
            if self.argument is None:
                if self.default_argument is None:
                    raise JaqalError(f"{type(self).__name__} requires an argument")
                ret.append(self.default_argument)
            ret.append(lookup_object(self.argument))

        if self.wrap_statements:
            inner = [QSequentialBlock.internal_name]
            ret.append(inner)
        else:
            inner = ret

        for stmt in self.statements:
            self._validate_statement(stmt)
            inner.append(stmt.build(lookup_object))

        return ret

    def _validate_statement(self, statement):
        """Make sure a statement can exist inside this block given our white
        and blacklists."""

        # We only check blocks. There are cases (e.g. inside a branch
        # block) where non-blocks are not allowed, but we don't check
        # that here. We could, but I want to keep this logic from
        # getting too complicated as any checking here is really just
        # a courtesy to the user.
        if isinstance(statement, QBlock):
            if not self._validate_inner_block(statement):
                raise JaqalError(
                    f"{type(self).__name__}: cannot contain block {type(statement).__name__}"
                )

    def _validate_inner_block(self, blk):
        """Determine if a block can nest into this. Override this default
        action if needed."""
        return not isinstance(blk, QCase)

    def starts_with_prepare(self, name):
        return len(self.statements) > 0 and self.statements[0].starts_with_prepare(name)


class QSequentialBlock(QBlock):
    internal_name = "sequential_block"


class QParallelBlock(QBlock):
    internal_name = "parallel_block"


class QSubcircuitBlock(QBlock):
    internal_name = "subcircuit_block"
    arity = 1
    default_argument = 1

    def _validate_inner_block(self, blk):
        return not isinstance(blk, (QCase, QSubcircuitBlock))

    def starts_with_prepare(self, _name):
        return True


class QLoop(QBlock):
    internal_name = "loop"
    arity = 1
    wrap_statements = True


class QBranch(QBlock):
    internal_name = "branch"

    def _validate_inner_block(self, blk):
        return isinstance(blk, QCase)


class QCase(QBlock):
    internal_name = "case"
    arity = 1
    wrap_statements = True


class QUsePulses:
    @classmethod
    def __init__(self, module, names):
        self.module = module
        self.names = self._validate_normalize_names(names)

    @staticmethod
    def _validate_normalize_names(names):
        """Make sure the names imported are valid. For now, only importing all
        identifiers into the global namespace is supported, so names must be
        "*" or the special symbol `all'."""

        if names == all or names == "*":
            return "*"

        raise JaqalError(f"usepulses names must be '*' or all")


#
# Private implementation details
#


class Stack:
    """Implement a stack of where Jaqal objects are stored. Each frame
    corresponds to a new block."""

    def __init__(self):
        self.stack = []
        self.top_context = {
            "lets": [],
            "registers": [],
            "usepulses": [],
        }

    def set_let(self, item):
        self.top_context["lets"].append(item)

    def set_register(self, item):
        self.top_context["registers"].append(item)

    def set_statement(self, item):
        if self.depth() == 0:
            raise JaqalError("Cannot define statements outside a circuit")
        self.stack[-1].append(item)

    def set_usepulses(self, item):
        self.top_context["usepulses"].append(item)

    def iter_lets(self):
        return iter(self.top_context["lets"])

    def iter_registers(self):
        return iter(self.top_context["registers"])

    def iter_usepulses(self):
        return iter(self.top_context["usepulses"])

    def iter_statements(self):
        return iter(self.stack[-1])

    def push(self):
        self.stack.append([])

    def pop(self):
        if self.depth() == 0:
            raise JaqalError("Popping frame off empty stack")
        self.stack.pop()

    def depth(self):
        return len(self.stack)

    @contextmanager
    def frame(self):
        start_depth = self.depth()

        self.push()

        try:
            yield
        finally:
            self.pop()

        if start_depth != self.depth():
            raise JaqalError("Stack depth changed in block")


class Namer:
    """Internal class for assigning names to anonymous objects.

    Names are chosen by incrementing an index in a template. The
    template is chosen so as to be short, but unlikely to be chosen by
    a user. If the user chooses such a name anyway, we will be careful
    to avoid using it.

    """

    register_template = "__r{}"
    let_template = "__c{}"

    def __init__(self, *, let_names, register_names):
        self.let_names = let_names
        self.register_names = register_names
        self.next_let = 0
        self.next_register = 0

    def name_let(self, let):
        assert isinstance(let, QConstant)
        if let.name is not None:
            return let.name
        name, self.next_let = self._choose_name(
            self.let_template, self.next_let, self.let_names
        )
        return name

    def name_register(self, register):
        assert isinstance(register, QRegister)
        if register.name is not None:
            return register.name
        name, self.next_register = self._choose_name(
            self.register_template, self.next_register, self.register_names
        )
        return name

    def _choose_name(self, template, index, user_names):
        """Choose a new name for some object. Return the name chosen and the
        new value for the index. Avoids choosing any name in user_names."""

        while True:
            name = template.format(index)
            index += 1
            if name not in user_names:
                break
        return name, index


def validate_int(value):
    """Make sure this value is an int or a Constant that is an int."""
    if isinstance(value, QConstant):
        pre_value = value.value
        post_value = int(value.value)
    else:
        pre_value = value
        post_value = int(value)
    if pre_value != post_value:
        raise JaqalError(f"Invalid int value {value}")
    return value


def mark_qcircuit(func, argcount):
    """Internally-used function to help identify circuits by introspection
    tools."""
    func._QCIRCUIT_FUNCTION = True
    func._QCIRCUIT_ARG_COUNT = argcount
    return func


def is_qcircuit(func, argcount=None):
    """Return whether the given function is properly annotated Q circuit
    that when called will return a QCircuit object. If argcount is not
    None, also check whether it accepts the given number of arguments.

    """

    if not callable(func):
        return False
    if not hasattr(func, "_QCIRCUIT_FUNCTION"):
        return False
    if argcount is not None and getattr(func, "_QCIRCUIT_ARG_COUNT") != argcount:
        return False
    return True


def _register_block_function(blocks, func, *args):
    """Register a block function to a dictionary of stored blocks, and
    return a function that can be called with a Q instance as its
    first argument to create that block in a circuit."""

    func_name = _add_block(blocks, func, *args)
    context_name, *ctxargs = args

    @functools.wraps(func)
    def do_block(Qinst, *funcargs):
        if not isinstance(Qinst, Q):
            raise JaqalError(
                f"call {func} either as Q.{func_name}(...) or {func_name}(Q,...)"
            )
        with Qinst._block_context(context_name, ctxargs):
            func(Qinst, *funcargs)

    return do_block


def _add_block(blocks, func, source, *args):
    """Add a function block to a dictionary of stored blocks."""
    if not callable(func):
        raise JaqalError(
            f"Argument to {source}() must be callable, found {func} (type={type(func)})"
        )
    try:
        func_name = func.__name__
    except AttributeError:
        raise JaqalError(
            f"Could not determine name of function used with {source}(), if this is a class instance, add a '__name__' member."
        )
    # We could test to make sure the user isn't redefining a block, as
    # this is usually an error. However, Python silently allows this
    # behavior in general. Also, there are times when it is desirable,
    # such as locally overriding a block, or iterating on a circuit in
    # a Jupyter notebook without reloading the kernel.
    blocks[func_name] = (func, source, *args)

    return func_name


class DecoratorOrContext:
    """(internal) Create an object that can decorate a function or serve
    as a context for a with-statement. This allows a single property
    of a class to serve as both a decorator for functions and a
    context that can be opened. Additionally, it simplifies decorators
    so omitting their argument list is the same as providing an empty
    argument list. Decorators may take additional positional and
    keyword arguments.

    """

    def __init__(self, decorator, ctx_func):
        """Set up our context actions and decorator.

        ctx_func must be a generator in the style of a contextmanager,
        but not decorated with that decorator.

        The decorator must take the function to decorate as its first
        argument, then can take any additional arguments after. The
        additional arguments are drawn from the decorator's--not the
        function's--argument list.

        """

        self.ctx_func = ctx_func
        self.decorator = decorator

        self._args = []
        self._kwargs = {}

        self._cm = None

    def __call__(self, *args, **kwargs):
        # Let's consider the different ways this class can be used:
        #
        # 1) Are we a decorator?
        # 1a) Are we a decorator with arguments (including an empty arg list)?
        # 1b) Are we a decorator without arguments?
        # 2) Are we a context?
        # 2a) Are we a context with arguments (including an empty arg list)?
        # 2b) Are we a context without arguments?
        #
        # For simplicity, we assume that if our context function or
        # decorator takes positional arguments, the first one cannot
        # be a callable object. This allows us to make decisions about
        # whether we are called with and without arguments based on
        # the presence of a single function argument.
        #
        # We can immediately know we are not in case 2b because the
        # context protocol would be invoked immediately and we would
        # be in the __enter__ method.
        #
        # If our first argument is a function, then we are in case 1b
        # or we have recursed. Either way, decorate the function with
        # our stored args.
        #
        # Otherwise, collect the arguments, and recurse. The behavior
        # is initially the same for cases 1a and 2a.

        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Decorate the callable function. We are in case 1b or we
            # have already recursed and are in 1a.
            func = args[0]
            wrapped = self.decorator(func, *self._pop_args(), **self._pop_kwargs())
            functools.wraps(func)(wrapped)
            return wrapped
        else:
            # This assertion can only happen if this internal class is
            # used improperly. When actually used as a decorator or
            # context there is no way to recurse more than once.
            assert not self._args and not self._kwargs

            # Store our arguments and recurse. We are in case 1a or 2a.
            self._set_args(args)
            self._set_kwargs(kwargs)
            return self

    def __enter__(self):
        # Remove references to what is passed to the context manager.
        self._cm = contextmanager(self.ctx_func)(
            *self._pop_args(),
            **self._pop_kwargs(),
        )
        self.ctx_func = None
        return self._cm.__enter__()

    def __exit__(self, *args):
        assert self._cm, "Internal context manager not properly intialized"
        _cm = self._cm
        self._cm = None
        return _cm.__exit__(*args)

    def _set_args(self, args):
        if self._args is None:
            raise JaqalError(f"DecoratorOrContext object reused")
        self._args = args

    def _pop_args(self):
        if self._args is None:
            raise JaqalError(f"DecoratorOrContext object reused")
        args = self._args
        self._args = None
        return args

    def _set_kwargs(self, kwargs):
        if self._kwargs is None:
            raise JaqalError(f"DecoratorOrContext object reused")
        self._kwargs = kwargs

    def _pop_kwargs(self):
        if self._kwargs is None:
            raise JaqalError(f"DecoratorOrContext object reused")
        kwargs = self._kwargs
        self._kwargs = None
        return kwargs
