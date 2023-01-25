from functools import update_wrapper, wraps


class AccessorMeta(type):
    def __new__(metaklass, name, bases, dict):
        # The inner "Accessor" class (probably) contains nothing of value besides the
        # reference to the containing outer class. Instead of requiring every method to
        # be defined like
        #
        #   def inner_method(accessor, *args):
        #       self = accessor.instance
        #
        # We adjust it so that
        #
        #   def inner_method(self, *args):
        #
        # does what you would expect.  Use @Accessor.direct to disable this logic.
        # Aside: All accesses of special methods like `__getitem__` bypass all
        # __getattribute__, __getattr__ mechanisms [1], so we can't play games by
        # doing, e.g.,
        #
        #   def __getattribute__(self, name):
        #       instance = object.__getattr__(self, 'instance')
        #       return object.__getattr__(self, name).__get__(instance)
        #
        # [1] https://docs.python.org/3/reference/datamodel.html#special-lookup
        function = type(lambda: 0)

        def fix(v):
            if v is None:
                return
            return update_wrapper(
                lambda self, *args, **kwargs: v(self.instance, *args, **kwargs), v
            )

        def fix_property(v):
            return property(
                fget=fix(v.fget),
                fset=fix(v.fset),
                fdel=fix(v.fdel),
                doc=v.__doc__,
            )

        for k, v in list(dict.items()):
            if k in ("__new__", "__init__", "__init_subclass__"):
                continue
            elif isinstance(k, (classmethod, staticmethod)):
                continue
            elif isinstance(v, AccessorDirect):
                dict[k] = v.func
            elif isinstance(v, function):
                dict[k] = fix(v)
            elif isinstance(v, property):
                dict[k] = fix_property(v)

        if "__slots__" not in dict:
            dict["__slots__"] = ()

        return super(AccessorMeta, metaklass).__new__(metaklass, name, bases, dict)

    def __get__(klass, instance, owner=None):
        return klass(instance, owner=owner)


class AccessorDirect:
    def __init__(self, func):
        self.func = func


class Accessor(metaclass=AccessorMeta):
    __slots__ = ("instance",)

    def __init__(self, instance, owner=None):
        self.instance = instance

    @staticmethod
    def direct(func):
        return AccessorDirect(func)

    # Implement this by hand.
    def __iter__(self):
        raise TypeError(f"'{type(self).__name__}' is not iterable")


class Accessorgetitem(AccessorDirect):
    def __init__(self, func):
        self._func = func

    @property
    def func(container):
        @wraps(container._func)
        def _inner(acc, i):
            if not isinstance(i, int):
                raise TypeError(f"{type(acc)} indices must be integers, not {type(i)}")
            if (i >= len(acc)) or (i < 0):
                raise IndexError(f"{type(acc)} index out of range")
            return container._func(acc.instance, i)

        return _inner


class ArrayAccessor(Accessor):
    @Accessor.direct
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def getitem(func):
        return Accessorgetitem(func)


class cachedproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner=None):
        try:
            return instance.__dict__[self.func.__name__]
        except KeyError:
            ret = instance.__dict__[self.func.__name__] = self.func(instance)
            return ret
