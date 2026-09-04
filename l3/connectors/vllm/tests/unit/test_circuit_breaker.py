"""Tests for l3_vllm.circuit_breaker.CircuitBreaker."""

import time

import pytest

from l3_vllm.circuit_breaker import CircuitBreaker, State


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_starts_closed():
    cb = CircuitBreaker()
    assert cb.state is State.CLOSED
    assert cb.allow()


def test_trips_open_on_consecutive_failures():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        cb.on_failure()
    assert cb.state is State.OPEN
    assert not cb.allow()


def test_success_resets_failure_counter():
    cb = CircuitBreaker(failure_threshold=3)
    cb.on_failure(); cb.on_failure()
    cb.on_success()
    cb.on_failure(); cb.on_failure()
    # Still 2 consecutive after reset, not 4 — should be CLOSED.
    assert cb.state is State.CLOSED


def test_open_to_half_open_after_probe_interval():
    clock = FakeClock()
    cb = CircuitBreaker(failure_threshold=1, probe_interval_s=5.0, clock=clock)
    cb.on_failure()
    assert cb.state is State.OPEN
    assert not cb.allow()
    clock.advance(5.1)
    assert cb.allow()    # probe allowed
    assert cb.state is State.HALF_OPEN


def test_half_open_close_after_n_successes():
    clock = FakeClock()
    cb = CircuitBreaker(
        failure_threshold=1, probe_interval_s=1.0,
        close_after_successes=3, clock=clock,
    )
    cb.on_failure()
    clock.advance(1.1)
    cb.allow()  # -> HALF_OPEN
    cb.on_success(); cb.on_success(); cb.on_success()
    assert cb.state is State.CLOSED


def test_half_open_trip_back_to_open_on_failure():
    clock = FakeClock()
    cb = CircuitBreaker(failure_threshold=1, probe_interval_s=1.0, clock=clock)
    cb.on_failure()
    clock.advance(1.1)
    cb.allow()  # -> HALF_OPEN
    cb.on_failure()  # probe failed
    assert cb.state is State.OPEN


def test_overload_rate_trip():
    clock = FakeClock()
    cb = CircuitBreaker(overload_rate_per_sec=2.0, clock=clock)
    cb.on_overload(); clock.advance(0.1)
    cb.on_overload(); clock.advance(0.1)
    cb.on_overload(); clock.advance(0.1)
    assert cb.state is State.OPEN
