"""Tier 1 — GIL release flag + empirical concurrency test.

Requires compiled _l3_rdma.so but no running server.
THE key validation that the GIL release fix actually works.
"""

import threading
import time

import pytest

_l3_rdma = pytest.importorskip("l3_client._l3_rdma")
RDMATransport = _l3_rdma.RDMATransport


# ======================================================================
# GIL_RELEASED flag checks
# ======================================================================

class TestGILReleasedFlag:
    def test_attribute_exists(self):
        assert hasattr(_l3_rdma, "GIL_RELEASED")

    def test_value_is_true(self):
        assert _l3_rdma.GIL_RELEASED is True

    def test_type_is_bool(self):
        assert isinstance(_l3_rdma.GIL_RELEASED, bool)


# ======================================================================
# Transport construction (no server needed)
# ======================================================================

class TestTransportConstruction:
    def test_default_construction_and_close(self):
        """Default construction + close() on unconnected transport is no-op."""
        t = RDMATransport()
        t.close()  # should not raise

    def test_custom_buffer_sizes(self):
        """Custom buffer sizes accepted without error."""
        t = RDMATransport(
            send_buf_size=1024 * 1024,
            recv_buf_size=1024 * 1024,
        )
        t.close()

    def test_connect_to_unreachable_raises(self):
        """connect() to RFC 5737 TEST-NET address raises RuntimeError."""
        t = RDMATransport()
        with pytest.raises(RuntimeError):
            t.connect("192.0.2.1", 19999)
        t.close()

    def test_double_close_is_safe(self):
        """Calling close() twice does not raise."""
        t = RDMATransport()
        t.close()
        t.close()


# ======================================================================
# Empirical GIL release test — THE MOST IMPORTANT TEST
# ======================================================================

class TestEmpiricalGILRelease:
    # RFC 5737 TEST-NET-1: guaranteed unreachable, causes rdma_getaddrinfo
    # to block for a measurable period before failing.
    UNREACHABLE = "192.0.2.1"
    PORT = 19999
    N_THREADS = 4

    def _timed_connect_attempt(self):
        """Attempt connect() to unreachable address, return wall-clock seconds."""
        t = RDMATransport()
        t0 = time.monotonic()
        try:
            t.connect(self.UNREACHABLE, self.PORT)
        except RuntimeError:
            pass
        elapsed = time.monotonic() - t0
        t.close()
        return elapsed

    def test_concurrent_connects_overlap(self):
        """If GIL is released, N concurrent connect() calls should overlap.

        Measurement:
          - baseline = single-thread connect() wall time
          - concurrent = N threads behind a barrier, measure wall time
          - If GIL released: wall ≈ baseline (threads overlap in rdma_getaddrinfo)
          - If GIL held: wall ≈ N × baseline (threads serialize)
        """
        # Measure baseline
        baseline = self._timed_connect_attempt()

        if baseline < 0.001:
            pytest.skip(
                f"connect() failed too fast ({baseline*1000:.1f}ms) — "
                "can't measure GIL contention"
            )

        # Run N threads concurrently
        barrier = threading.Barrier(self.N_THREADS + 1, timeout=30)
        errors = []
        elapsed_per_thread = [0.0] * self.N_THREADS

        def worker(idx):
            try:
                barrier.wait()
                elapsed_per_thread[idx] = self._timed_connect_attempt()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(self.N_THREADS):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        wall_start = time.monotonic()
        barrier.wait()  # release all threads
        for t in threads:
            t.join(timeout=60)
        wall_elapsed = time.monotonic() - wall_start

        assert not errors, f"Worker errors: {errors}"

        # Key assertion: if GIL released, wall_elapsed should be close to baseline
        # (all threads ran concurrently). If GIL held, wall_elapsed ≈ N * baseline.
        # Use generous 2.5x margin for scheduling jitter.
        limit = baseline * 2.5
        assert wall_elapsed < limit, (
            f"Wall time {wall_elapsed:.3f}s > {limit:.3f}s "
            f"(baseline={baseline:.3f}s × 2.5). "
            f"GIL may not be released! Per-thread: {elapsed_per_thread}"
        )
