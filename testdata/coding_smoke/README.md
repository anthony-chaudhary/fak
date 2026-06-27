# Coding Smoke Test Fixture

A minimal Python project for demonstrating agent coding capabilities.

## Problem

The `add()` function in `calculator.py` is buggy: it subtracts instead of adds.

## Verification

Before the fix, one test should fail:
```bash
python -m unittest test_calculator.py
# Expected output: .F. (1 pass, 1 fail)
```

After fixing the `add()` function, both tests should pass:
```bash
python -m unittest test_calculator.py
# Expected output: .. (2 passes)
```

## Files

- `calculator.py` - The buggy calculator implementation
- `test_calculator.py` - Unit tests (one failing, one passing)