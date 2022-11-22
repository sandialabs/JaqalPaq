import unittest, pytest

from jaqalpaq.run.classical_cursor import ClassicalCursor
from jaqalpaq.error import JaqalError

from .batching_reference import build_override_lists, ref_jaqal_batch


class CCTester(unittest.TestCase):
    def check(self, overrides, by_subbatch=None, by_time=None, subcirc_list=None):
        computed = ClassicalCursor(overrides, subcirc_list=subcirc_list)
        if overrides is not None:
            ref_expanded, ref_count = build_override_lists(overrides)
            self.assertEqual(computed.maxlen, ref_count)

        # Check by_time if the indices are somehow specified
        if subcirc_list or (overrides and ("__index__" in overrides)):
            if overrides is not None:
                natural_count = len(subcirc_list) if subcirc_list else None
                r_by_sb, r_by_time = ref_jaqal_batch(
                    natural_count, ref_expanded, ref_count
                )
                if by_time:
                    self.assertEqual(by_time, r_by_time)
                else:
                    by_time = r_by_time

                if by_subbatch:
                    self.assertEqual(by_subbatch, r_by_sb)
                else:
                    by_subbatch = r_by_sb

            # Check the array-accessor interfaces
            for i, one_batch in enumerate(by_subbatch):
                one_overrides, one_subcirc_indices = one_batch
                s = computed.by_subbatch[i]
                self.assertEqual(s.overrides, one_overrides)
                self.assertEqual(
                    [ss.subcircuit_i for ss in s.by_time], one_subcirc_indices
                )
                for n, subcirc_index in enumerate(one_batch[1]):
                    self.assertEqual(s.by_time[n].subcircuit_i, subcirc_index)
            for i, (exp_overrides, exp_subcircuitindex) in enumerate(by_time):
                computed_i = computed.by_time[i]
                self.assertEqual(computed_i.overrides, exp_overrides)
                self.assertEqual(computed_i.subcircuit_i, exp_subcircuitindex)
            # Check the iterator interfaces (and that they can be run twice).
            for i in range(2):
                self.assertEqual(
                    [
                        (s.overrides, [ss.subcircuit_i for ss in s.by_time])
                        for s in computed.by_subbatch
                    ],
                    by_subbatch,
                )
                self.assertEqual(
                    [(ss.overrides, ss.subcircuit_i) for ss in computed.by_time],
                    by_time,
                )
        else:
            # (as above, but with neither by_time nor indices)
            for i, one_batch in enumerate(by_subbatch):
                s = computed.by_subbatch[i]
                self.assertEqual(s.overrides, one_batch)
            for i in range(2):
                self.assertEqual(
                    [s.overrides for s in computed.by_subbatch], by_subbatch
                )

    def test_empty(self):
        self.check(None, [{}])

    def test_empty_subcircs(self):
        self.check(
            None,
            [({}, [0, 1, 2, 3])],
            [
                ({}, 0),
                ({}, 1),
                ({}, 2),
                ({}, 3),
            ],
            subcirc_list=[0, 1, 2, 3],
        )

    def test_passthrough(self):
        self.check(dict(a=10), [dict(a=10)])

    def test_pairing(self):
        self.check(
            dict(a=[10, 5, 8], b=[12, 10, 0], __repeats__=[1, 10, 100]),
            [
                dict(a=10, b=12, __repeats__=1),
                dict(a=5, b=10, __repeats__=10),
                dict(a=8, b=0, __repeats__=100),
            ],
        )

    def test_broadcasting(self):
        self.check(
            dict(a=[10, 5, 8], b=2),
            [
                dict(a=10, b=2),
                dict(a=5, b=2),
                dict(a=8, b=2),
            ],
        )

    def test_indexed(self):
        self.check(
            dict(__index__=[1, 2, 3]),
            [
                ({}, [1]),
                ({}, [2]),
                ({}, [3]),
            ],
            [
                ({}, 1),
                ({}, 2),
                ({}, 3),
            ],
        )

    def test_indexed_passthrough(self):
        self.check(
            dict(a=10, __index__=[[1, 2, 3]]),
            [(dict(a=10), [1, 2, 3])],
            [
                (dict(a=10), 1),
                (dict(a=10), 2),
                (dict(a=10), 3),
            ],
        )

    def test_mismatched_index(self):
        overrides = dict(a=[10, 11], __index__=[1, 2, 3])
        self.assertRaises(JaqalError, lambda: list(ClassicalCursor(overrides)))

    def test_extra_brackets(self):
        overrides = dict(a=[10, 5, 8], b=[[12, 10, 0]])
        self.assertRaises(JaqalError, lambda: list(ClassicalCursor(overrides)))

    def test_broadcasted_index(self):
        self.check(
            dict(a=[10, 5, 8], b=[0.1, 0.2, 0.1], __index__=[[12, 10, 0]]),
            [
                (dict(a=10, b=0.1), [12, 10, 0]),
                (dict(a=5, b=0.2), [12, 10, 0]),
                (dict(a=8, b=0.1), [12, 10, 0]),
            ],
            [
                (dict(a=10, b=0.1), 12),
                (dict(a=10, b=0.1), 10),
                (dict(a=10, b=0.1), 0),
                (dict(a=5, b=0.2), 12),
                (dict(a=5, b=0.2), 10),
                (dict(a=5, b=0.2), 0),
                (dict(a=8, b=0.1), 12),
                (dict(a=8, b=0.1), 10),
                (dict(a=8, b=0.1), 0),
            ],
        )

    def test_nested_scan(self):
        self.check({"__index__": [[0, 2]], "theta": [1.2, 2.3, 3.4], "phi": [4, 5, 6]})

    def test_flat_zipped(self):
        self.check({"__index__": [1, 2, 0], "theta": [1.2, 2.3, 3.4], "phi": [4, 5, 6]})

    def test_multiple_override_same_length(self):
        self.check({"__index__": 2, "theta": [1.2, 2.3, 3.4], "phi": [4, 5, 6]})

    def test_multiple_ovveride_theta_repeated(self):
        self.check({"__index__": 2, "theta": 1.2, "phi": [4, 5, 6]})

    def test_one_override_one_subcircuit(self):
        self.check({"__index__": 2, "theta": 1.2, "phi": 4})

    # per_subcircuit tests
    def test_per_subcircuit_simple(self):
        cc = ClassicalCursor(
            dict(a=[10, 5, 8], b=[0.1, 0.2, 0.1], __index__=[[12, 10, 0]])
        )
        self.assertEqual(cc.get_per_sc_time_i(2, 12, 0), 0)
        self.assertEqual(cc.get_per_sc_time_i(2, 10, 1), 0)
        self.assertEqual(cc.get_per_sc_time_i(2, 0, 2), 0)

    def test_several_subcircuits(self):
        cc = ClassicalCursor(
            dict(
                a=[10, 5, 8], b=[0.1, 0.2, 0.1], __index__=[[12, 10, 12, 12, 0, 10, 0]]
            )
        )
        self.assertEqual(cc.by_subbatch[0].by_subcircuit[12].by_time[2].time_i, 3)
        self.assertEqual(cc.by_subbatch[1].by_subcircuit[12].by_time[2].time_i, 10)
