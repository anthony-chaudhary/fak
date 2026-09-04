"""Circuit breaker for the CAMA vLLM connector.

States:
    CLOSED    -> all CAMA ops attempted; transitions to OPEN on
                 ``failure_threshold`` consecutive failures, or when
                 ``CamaServerOverloadError`` rate exceeds ``overload_rate_per_sec``.
    OPEN      -> all CAMA ops short-circuit (no RPC) and report misses;
                 transitions to HALF_OPEN after ``probe_interval_s`` and a
                 successful probe RPC.
    HALF_OPEN -> ops are attempted; one failure -> OPEN; ``close_after_successes``
                 successes -> CLOSED.

Distinct from cama_storage._BackpressureGuard, which is per-batch sleep-jitter
backoff; this is process-wide hard short-circuiting.

Thread-safe.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


class State(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 10,
        overload_rate_per_sec: float = 3.0,
        probe_interval_s: float = 5.0,
        close_after_successes: int = 100,
        clock=time.monotonic,
    ):
        self._failure_threshold = failure_threshold
        self._overload_rate = overload_rate_per_sec
        self._probe_interval = probe_interval_s
        self._close_after = close_after_successes
        self._clock = clock

        self._lock = threading.Lock()
        self._state = State.CLOSED
        self._consec_failures = 0
        self._consec_successes = 0
        self._overload_ts: deque[float] = deque(maxlen=64)
        self._opened_at: float | None = None

    @property
    def state(self) -> State:
        return self._state

    def allow(self) -> bool:
        """Return True if an op may be attempted.

        Also advances OPEN -> HALF_OPEN once the probe interval has elapsed
        (the next op is the probe).
        """
        with self._lock:
            if self._state is State.CLOSED:
                return True
            if self._state is State.OPEN:
                if (
                    self._opened_at is not None
                    and self._clock() - self._opened_at >= self._probe_interval
                ):
                    self._state = State.HALF_OPEN
                    logger.info("cama circuit breaker: OPEN -> HALF_OPEN (probing)")
                    return True
                return False
            # HALF_OPEN
            return True

    def on_success(self) -> None:
        with self._lock:
            self._consec_failures = 0
            if self._state is State.HALF_OPEN:
                self._consec_successes += 1
                if self._consec_successes >= self._close_after:
                    self._state = State.CLOSED
                    self._consec_successes = 0
                    self._opened_at = None
                    logger.info("cama circuit breaker: HALF_OPEN -> CLOSED")

    def on_failure(self) -> None:
        with self._lock:
            if self._state is State.HALF_OPEN:
                self._trip_locked("probe failed")
                return
            self._consec_failures += 1
            if self._consec_failures >= self._failure_threshold:
                self._trip_locked(
                    f"{self._consec_failures} consecutive failures"
                )

    def on_overload(self) -> None:
        now = self._clock()
        with self._lock:
            self._overload_ts.append(now)
            # Count overloads in the last 1 second.
            cutoff = now - 1.0
            recent = sum(1 for t in self._overload_ts if t >= cutoff)
            if recent / 1.0 > self._overload_rate:
                self._trip_locked(
                    f"overload rate {recent}/s > {self._overload_rate}/s"
                )

    def _trip_locked(self, reason: str) -> None:
        if self._state is State.OPEN:
            return
        self._state = State.OPEN
        self._opened_at = self._clock()
        self._consec_failures = 0
        self._consec_successes = 0
        logger.warning("cama circuit breaker: -> OPEN (%s)", reason)
