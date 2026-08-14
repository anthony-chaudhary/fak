import time
from collections import deque
from threading import Lock


class RateLimiter:
    """Allow at most max_calls requests per period seconds, per key."""

    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self._calls = {}
        self._lock = Lock()

    def allow(self, key):
        now = time.monotonic()
        cutoff = now - self.period

        with self._lock:
            calls = self._calls.setdefault(key, deque())
            while calls and calls[0] <= cutoff:
                calls.popleft()

            if len(calls) >= self.max_calls:
                return False

            calls.append(now)
            return True
