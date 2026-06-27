"""Unit tests for calculator.py."""

import unittest
from calculator import add, subtract


class TestCalculator(unittest.TestCase):

    def test_add_buggy(self):
        """This test fails because add() has a bug."""
        self.assertEqual(add(5, 3), 2)  # Expects 5 - 3 = 2

    def test_subtract_correct(self):
        """This test passes - subtract() works correctly."""
        self.assertEqual(subtract(5, 3), 2)


if __name__ == '__main__':
    unittest.main()