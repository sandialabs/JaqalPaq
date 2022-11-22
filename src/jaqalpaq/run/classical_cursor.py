import bisect
import warnings
from jaqalpaq.error import JaqalError
from jaqalpaq.core.algorithm.walkers import walk_circuit
from ._view import Accessor, ArrayAccessor, cachedproperty


def get_index_broadcaster(index, subcirc_list):
    if index is None:
        if subcirc_list is not None:
            return _Indices_subcirc_list(subcirc_list)
        return None

    if not isinstance(index, list):
        index = [index]

    if not isinstance(index[0], list):
        if len(index) == 1:
            return _Indices_broadcast_item(index[0])
        else:
            return _Indices_singlebrace(index)

    if len(index) == 1:
        return _Indices_broadcast_list(index[0])
    else:
        # Remove this error to add support for outer lists with len > 1
        raise JaqalError(
            "Nested lists for '__index__' must have an outer list of length one: [[0,1,2,...]]"
        )
        return _Indices_doublebrace(index)


class _Indices_subcirc_list:
    __slots__ = ("_subcirc_list",)

    def __init__(self, subcirc_list):
        self._subcirc_list = subcirc_list

    def get(self, i, j):
        return self._subcirc_list[j]

    def len(self, i):
        return len(self._subcirc_list)


class _Indices_singlebrace:
    __slots__ = ("_index",)

    def __init__(self, index):
        self._index = index

    def get(self, i, j):
        return self._index[i]

    def len(self, i):
        return 1


class _Indices_broadcast_item:
    __slots__ = ("_item",)

    def __init__(self, item):
        self._item = item

    def get(self, i, j):
        return self._item

    def len(self, i):
        return 1


class _Indices_doublebrace:
    __slots__ = ("_index",)

    def __init__(self, index):
        self._index = index

    def get(self, i, j):
        return self._index[i][j]

    def len(self, i):
        return len(self._index[i])


class _Indices_broadcast_list:
    __slots__ = ("_items",)

    def __init__(self, items):
        self._items = items

    def get(self, i, j):
        return self._items[j]

    def len(self, i):
        return len(self._items)


def parse_override_dict(overrides):
    if overrides is None:
        overrides = {}

    maxlen = max(
        (1 if not isinstance(x, list) else len(x) for x in overrides.values()),
        default=1,
    )

    index = None
    filtered_overrides = {}

    for k, v in overrides.items():
        if k[:2] == "__" and k[-2:] == "__":
            dunder = k[2:-2]

            if dunder == "index":
                index = v
                continue
            elif dunder == "repeats":
                # we broadcast this as normal
                pass
            else:
                warnings.warn(f"Suppressing unsupported override '{k}'.")
                continue
        else:
            if k.startswith("j."):
                warnings.warn(f"Removing 'j.' prefix from '{k}'")
                k = k[2:]
            if k.startswith("pd."):
                warnings.warn(f"Ignoring 'pd.' prefix from '{k}'")
                continue

        if isinstance(v, list):
            if len(v) == 1:
                # Automatically broadcast lists of length 1 as if they were maxlen.
                # But, catch nested lists, by which the user probably intended to mean
                # something different.
                if isinstance(v[0], list):
                    raise JaqalError("Unsupported broadcast")
                (v,) = v
            elif len(v) != maxlen:
                raise JaqalError("Can't have mixed length lists in override")

        filtered_overrides[k] = v

    return maxlen, filtered_overrides, index


class ClassicalCursor:
    def __init__(self, overrides, subcirc_list=None):
        self.maxlen, self.filtered_overrides, index = parse_override_dict(overrides)

        self._indices = get_index_broadcaster(index, subcirc_list)
        if self._indices:
            self._calculate_skips()

    def _calculate_skips(self):
        sb_skips = self._sb_skips = []
        sc_skips = self._sc_skips = []
        spot = 0
        for subbatch in self.by_subbatch:
            bytime = subbatch.by_time
            spot += len(bytime)
            sb_skips.append(spot)

            per_sb_sc_skips = {}
            sc_skips.append(per_sb_sc_skips)
            for ci in bytime:
                sc_i = ci.subcircuit_i
                try:
                    sk = per_sb_sc_skips[sc_i]
                except KeyError:
                    sk = per_sb_sc_skips[sc_i] = []
                sk.append(ci.per_sb_time_i)

    class by_subbatch(ArrayAccessor):
        @ArrayAccessor.getitem
        def __getitem__(self, i):
            return SubbatchScan(self, i)

        def __len__(self):
            return self.maxlen

    def get_sb_i_from_time_i(self, time_index):
        sb_i = bisect.bisect(self._sb_skips, time_index)
        n = time_index
        if sb_i > 0:
            n -= self._sb_skips[sb_i - 1]
        return sb_i, n

    @property
    def sc_skips(self):
        """(internal) Return a (by-subbatch) list of (by-subcircuit) circuitindexes."""
        try:
            return self._sc_skips
        except AttributeError:
            raise JaqalError(
                "Indexing by subcircuit unsuported: cannot infer number of subcircuit"
            )

    def get_per_sc_time_i(self, subbatch_i, subcircuit_i, per_sb_time_i):
        sk = self.sc_skips[subbatch_i][subcircuit_i]
        spot = bisect.bisect_left(sk, per_sb_time_i)
        if sk[spot] != per_sb_time_i:
            raise IndexError(
                f"{per_sb_time_i} does not correspond to subcircuit {subcircuit_i}"
            )
        return spot

    def get_time_i(self, subbatch_i, per_sb_time_i):
        if subbatch_i:
            return per_sb_time_i + self._sb_skips[subbatch_i - 1]
        return per_sb_time_i

    class by_time(Accessor):
        @ArrayAccessor.getitem
        def __getitem__(self, time_i):
            sb_i, per_sb_time_i = self.get_sb_i_from_time_i(time_i)
            sb = self.by_subbatch[sb_i]
            return sb.by_time[per_sb_time_i]

        def __iter__(self):
            for sb in self.by_subbatch:
                yield from sb.by_time

        def __len__(self):
            return self._sb_skips[-1]


class CircuitIndex:
    def __init__(
        self, subbatch, *, per_sc_time_i=None, per_sb_time_i=None, subcircuit_i=None
    ):
        self.subbatch = subbatch

        if per_sc_time_i is not None:
            self.__dict__["per_sc_time_i"] = per_sc_time_i
        if per_sb_time_i is not None:
            self.__dict__["per_sb_time_i"] = per_sb_time_i
        if subcircuit_i is not None:
            self.__dict__["subcircuit_i"] = subcircuit_i

    def __repr__(self):
        return f"<CircuitIndex {self.time_i} ({self.subbatch.index}:{self.per_sb_time_i}, {self.subcircuit_i})>"

    @property
    def overrides(self):
        return self.subbatch.overrides

    @property
    def time_i(self):
        sb = self.subbatch
        return sb._classical_cursor.get_time_i(sb.index, self.per_sb_time_i)

    @cachedproperty
    def subcircuit_i(self):
        sb = self.subbatch
        return sb._classical_cursor._indices.get(sb.index, self.per_sb_time_i)

    @cachedproperty
    def per_sc_time_i(self):
        sb = self.subbatch
        return sb._classical_cursor.get_per_sc_time_i(
            sb.index, self.subcircuit_i, self.per_sb_time_i
        )

    @cachedproperty
    def per_sb_time_i(self):
        sb = self.subbatch
        skips = sb._classical_cursor.sc_skips
        return skips[sb.index][self.subcircuit_i][self.per_sc_time_i]


class SubcircuitCollation:
    def __init__(self, subbatch, subcircuit_index):
        self.subbatch = subbatch
        self.subcircuit_i = subcircuit_index

    class by_time(ArrayAccessor):
        def __len__(self):
            return len(self)

        @ArrayAccessor.getitem
        def __getitem__(self, i):
            return CircuitIndex(
                self.subbatch, per_sc_time_i=i, subcircuit_i=self.subcircuit_i
            )

    def __len__(self):
        sb = self.subbatch
        return len(sb._classical_cursor.sc_skips[sb.index][self.subcircuit_i])


class SubbatchScan:
    def __init__(self, classical_cursor, subbatch_i):
        self._classical_cursor = classical_cursor
        self.index = subbatch_i

    class by_time(ArrayAccessor):
        def __len__(self):
            return self._classical_cursor._indices.len(self.index)

        @ArrayAccessor.getitem
        def __getitem__(self, i):
            return CircuitIndex(self, per_sb_time_i=i)

    class by_subcircuit(Accessor):
        def __len__(self):
            return len(self._classical_cursor.sc_skips[self.index])

        def __getitem__(self, i):
            return SubcircuitCollation(self, i)

        def __iter__(self):
            for sc_i in self._classical_cursor.sc_skips[self.index]:
                yield SubcircuitCollation(self, sc_i)

    def get_override(self, name):
        over = self._classical_cursor.filtered_overrides
        val = over.get(name, None)
        if val is None:
            return
        if isinstance(val, list):
            return val[self.index]
        return val

    @cachedproperty
    def overrides(self):
        sb_i = self.index

        return {
            k: v[sb_i] if isinstance(v, list) else v
            for k, v in self._classical_cursor.filtered_overrides.items()
        }


__all__ = ("Accessor", "ArrayAccessor", "ClassicalCursor")
