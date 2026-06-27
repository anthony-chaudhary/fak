"""A tiny calculator with a bug."""

def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b  # BUG: should be a - b for the test

def subtract(a: int, b: int) -> int:
    """Return the difference of two integers."""
    return a - b