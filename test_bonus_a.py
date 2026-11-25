import unittest
from unittest.mock import patch

from bonus_a import (
    CASES, cardinality, check, complement, difference, intersection, parse,
    powerset, subset, symmetric_difference, union,
)


class Tests(unittest.TestCase):
    def test_cases_and_certificates(self):
        for left, right, expected in CASES:
            k = len(set(left + right) & set("ABC"))
            for n in (1, 2):
                for fast in (True, False):
                    with self.subTest(left=left, right=right, n=n, fast=fast):
                        ok, count, witness = check(left, right, n, fast)
                        self.assertEqual(ok, expected)
                        if expected:
                            self.assertEqual(count, 1 << (k if fast else n * k))
                            self.assertIsNone(witness)
                        elif right == "A":
                            self.assertEqual(witness, {"A": {1}, "B": {1}})
                            self.assertNotEqual(witness["A"] - witness["B"], witness["A"])
                        else:
                            self.assertEqual(witness, {"A": {1}, "B": set(), "C": {1}})
                            a, b, c = (witness[name] for name in "ABC")
                            self.assertNotEqual((a - b) - c, a - (b - c))

    def test_bad_syntax(self):
        for expression in ("A)", "(A", "A|", "A B", "", "A@B"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    parse(expression)

    def test_precedence_and_associativity(self):
        self.assertEqual(parse("A-B&C"), ("&", ("-", "A", "B"), "C"))
        self.assertEqual(parse("A-(B&C)"), ("-", "A", ("&", "B", "C")))
        self.assertEqual(
            parse("¬A∩B△C∪D"), ("|", ("^", ("&", ("~", "A"), "B"), "C"), "D"),
        )

    def test_operations_against_sets(self):
        universe = {1, 2, 3, 4}
        decode = lambda mask: {i for i in universe if mask & (1 << (i - 1))}
        for a in range(16):
            x = decode(a)
            self.assertEqual(decode(complement(a, 4)), universe - x)
            self.assertEqual(cardinality(a), len(x))
            self.assertCountEqual(powerset(a), [b for b in range(16) if decode(b) <= x])
            for b in range(16):
                y = decode(b)
                self.assertEqual(decode(union(a, b)), x | y)
                self.assertEqual(decode(intersection(a, b)), x & y)
                self.assertEqual(decode(difference(a, b)), x - y)
                self.assertEqual(decode(symmetric_difference(a, b)), x ^ y)
                self.assertEqual(subset(a, b), x <= y)

    def test_empty_universe(self):
        for fast in (True, False):
            self.assertEqual(check("A-B", "A", 0, fast), (True, 1, None))

    def test_counterexample_after_first_batch(self):
        left, right = "(A-A)|(B-B)|A", "A-A"
        with patch("bonus_a.BATCH_SIZE", 64):
            self.assertEqual(check(left, right, 10, False),
                             (False, 1025, {"A": {1}, "B": set()}))


if __name__ == "__main__":
    unittest.main()
