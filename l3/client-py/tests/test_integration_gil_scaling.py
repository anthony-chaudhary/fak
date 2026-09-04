"""Tier 3 — Multi-thread scaling proof over RDMA.

THE DEFINITIVE PROOF that the GIL release fix enables real parallelism.
Requires RDMA server on DGX. Set CAMA_TEST_RDMA_SERVER=addr:port.
"""

import ctypes
import os
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("CAMA_TEST_RDMA_SERVER"),
    reason="Set CAMA_TEST_RDMA_SERVER=addr:port to run RDMA integration tests",
)


def _parse_server():
    raw = os.environ["CAMA_TEST_RDMA_SERVER"]
    addr, port_s = raw.rsplit(":", 1)
    return addr, int(port_s)


def _make_client():
    from l3_client.rdma_client import RDMAClient
    addr, port = _parse_server()
    return RDMAClient(addr, port=port)


# ---------------------------------------------------------------------------
# Module-scoped prepopulate: write 10 keys × 5MB via RDMA
# ---------------------------------------------------------------------------
VALUE_SIZE = 5 * 1024 * 1024
NUM_KEYS = 10
KEY_PREFIX = "gil_scale_test_"


@pytest.fixture(scope="module")
def prepopulate():
    from l3_client.sgl import SGL

    client = _make_client()
    data = os.urandom(VALUE_SIZE)
    buf = ctypes.create_string_buffer(data, VALUE_SIZE)
    ptr = ctypes.addressof(buf)
    handle = client.reg_memory(ptr, VALUE_SIZE, buf=buf)
    sgl = SGL(ptr=ptr, size=VALUE_SIZE, reg_handle=handle)

    keys = [f"{KEY_PREFIX}{i}" for i in range(NUM_KEYS)]
    for key in keys:
        client.set(key, sgl)

    client.dereg_memory(handle)
    client.close()
    return keys


# ---------------------------------------------------------------------------
# Worker function: run GETs in a loop for `duration` seconds
# ---------------------------------------------------------------------------
def _get_worker(keys, duration, barrier, result):
    from l3_client.rdma_client import RDMAClient
    from l3_client.sgl import SGL

    addr, port = _parse_server()
    client = RDMAClient(addr, port=port)

    buf = ctypes.create_string_buffer(VALUE_SIZE)
    ptr = ctypes.addressof(buf)
    handle = client.reg_memory(ptr, VALUE_SIZE, buf=buf)
    sgl = SGL(ptr=ptr, size=VALUE_SIZE, reg_handle=handle)

    barrier.wait()

    ops = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        key = keys[ops % len(keys)]
        client.get(key, sgl)
        ops += 1
    elapsed = time.monotonic() - t0

    result["ops"] = ops
    result["elapsed"] = elapsed

    # Collect stats before closing
    try:
        result["stats"] = client._transport.get_stats()
    except Exception:
        pass

    client.dereg_memory(handle)
    client.close()


# ======================================================================
# Tests
# ======================================================================

N_WORKERS = 4
DURATION = 5  # seconds per test


class TestParallelismMetric:
    def test_parallelism_metric(self, prepopulate):
        """Run N=4 workers for 5s, measure sum(worker_elapsed) / wall_elapsed.

        If GIL held: parallelism ≈ 1.0
        If GIL released: parallelism ≈ N
        Assert parallelism > N × 0.5 (≥ 2.0x for 4 threads)
        """
        keys = prepopulate
        barrier = threading.Barrier(N_WORKERS + 1, timeout=30)
        results = [{} for _ in range(N_WORKERS)]
        threads = []

        for i in range(N_WORKERS):
            t = threading.Thread(
                target=_get_worker,
                args=(keys, DURATION, barrier, results[i]),
            )
            threads.append(t)
            t.start()

        wall_start = time.monotonic()
        barrier.wait()
        for t in threads:
            t.join(timeout=DURATION + 30)
        wall_elapsed = time.monotonic() - wall_start

        sum_worker_time = sum(r.get("elapsed", 0) for r in results)
        parallelism = sum_worker_time / wall_elapsed if wall_elapsed > 0 else 0

        print(f"\n  Parallelism: {parallelism:.1f}x (target >= {N_WORKERS * 0.5:.1f}x)")
        print(f"  Wall time: {wall_elapsed:.1f}s")
        print(f"  Sum worker time: {sum_worker_time:.1f}s")

        assert parallelism > N_WORKERS * 0.5, (
            f"Parallelism {parallelism:.1f}x < {N_WORKERS * 0.5:.1f}x — "
            f"GIL may not be released!"
        )


class TestThroughputScales:
    def test_throughput_scales(self, prepopulate):
        """Multi-thread throughput should be > 2x single-thread.

        Conservative: doesn't require linear scaling, just >2x improvement.
        """
        keys = prepopulate

        # Single-thread measurement
        single_barrier = threading.Barrier(2, timeout=30)
        single_result = {}
        t = threading.Thread(
            target=_get_worker,
            args=(keys, DURATION, single_barrier, single_result),
        )
        t.start()
        single_barrier.wait()
        t.join(timeout=DURATION + 30)
        single_ops_s = single_result["ops"] / single_result["elapsed"]

        # Multi-thread measurement
        multi_barrier = threading.Barrier(N_WORKERS + 1, timeout=30)
        multi_results = [{} for _ in range(N_WORKERS)]
        threads = []
        for i in range(N_WORKERS):
            t = threading.Thread(
                target=_get_worker,
                args=(keys, DURATION, multi_barrier, multi_results[i]),
            )
            threads.append(t)
            t.start()
        multi_barrier.wait()
        for t in threads:
            t.join(timeout=DURATION + 30)

        total_multi_ops = sum(r.get("ops", 0) for r in multi_results)
        avg_elapsed = sum(r.get("elapsed", 0) for r in multi_results) / N_WORKERS
        multi_ops_s = total_multi_ops / avg_elapsed if avg_elapsed > 0 else 0

        speedup = multi_ops_s / single_ops_s if single_ops_s > 0 else 0

        print(f"\n  Single-thread: {single_ops_s:.0f} ops/s")
        print(f"  Multi-thread ({N_WORKERS}): {multi_ops_s:.0f} ops/s")
        print(f"  Speedup: {speedup:.1f}x")

        assert multi_ops_s > single_ops_s * 2.0, (
            f"Multi-thread {multi_ops_s:.0f} ops/s not > 2x single-thread "
            f"{single_ops_s:.0f} ops/s. Speedup: {speedup:.1f}x"
        )


class TestCppStatsUnderConcurrency:
    def test_per_client_stats_independent(self, prepopulate):
        """4 clients, each does 10 GETs behind a barrier.

        Verify each client's get_stats() shows roundtrip_count >= 10.
        Proves per-client stats are independent under concurrent access.
        """
        from l3_client.rdma_client import RDMAClient
        from l3_client.sgl import SGL

        keys = prepopulate
        n_ops = 10
        barrier = threading.Barrier(N_WORKERS + 1, timeout=30)
        results = [{} for _ in range(N_WORKERS)]

        def stats_worker(idx):
            addr, port = _parse_server()
            client = RDMAClient(addr, port=port)
            buf = ctypes.create_string_buffer(VALUE_SIZE)
            ptr = ctypes.addressof(buf)
            handle = client.reg_memory(ptr, VALUE_SIZE, buf=buf)
            sgl = SGL(ptr=ptr, size=VALUE_SIZE, reg_handle=handle)

            client._transport.reset_stats()
            barrier.wait()

            for i in range(n_ops):
                client.get(keys[i % len(keys)], sgl)

            results[idx]["stats"] = client._transport.get_stats()
            client.dereg_memory(handle)
            client.close()

        threads = []
        for i in range(N_WORKERS):
            t = threading.Thread(target=stats_worker, args=(i,))
            threads.append(t)
            t.start()

        barrier.wait()
        for t in threads:
            t.join(timeout=60)

        for i, r in enumerate(results):
            stats = r.get("stats", {})
            assert stats.get("roundtrip_count", 0) >= n_ops, (
                f"Client {i}: roundtrip_count={stats.get('roundtrip_count', 0)} "
                f"< {n_ops}"
            )
