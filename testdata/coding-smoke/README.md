# Coding Smoke Fixture

A minimal, deterministic coding task for benchmarking agents.

## Problem
The `add()` function in `calculator.py` returns `a + b` instead of `a - b`.

## Fix
Change `return a + b` to `return a - b` in `calculator.py`.

## Verify
```bash
python -m unittest test_calculator.py
```

Expected: 1 test passes, 1 test fails (before fix) → 2 tests pass (after fix)