"""Tier 0 — Barrier synchronization logic tests.

Validates threading.Barrier behavior used by bench.py workers.
Pure Python, runs anywhere.
"""

import threading
import time

import pytest


class TestBarrierSynchronization:
    def test_all_threads_start_together(self):
        """N threads + 1 main thread arrive → all start within 50ms."""
        n_workers = 4
        barrier = threading.Barrier(n_workers + 1, timeout=5)
        start_times = [None] * n_workers

        def worker(idx):
            barrier.wait()
            start_times[idx] = time.monotonic()

        threads = []
        for i in range(n_workers):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Main thread is the +1 party
        barrier.wait()
        main_start = time.monotonic()

        for t in threads:
            t.join(timeout=5)

        # All start times should be within 50ms of each other
        all_times = [t for t in start_times if t is not None] + [main_start]
        assert len(all_times) == n_workers + 1
        spread = max(all_times) - min(all_times)
        assert spread < 0.050, f"Spread was {spread*1000:.1f}ms (expected < 50ms)"

    def test_broken_barrier_on_timeout(self):
        """Barrier with timeout raises BrokenBarrierError when a party never arrives."""
        barrier = threading.Barrier(3, timeout=0.1)
        # Only 1 party arrives (out of 3 required)
        with pytest.raises(threading.BrokenBarrierError):
            barrier.wait()
