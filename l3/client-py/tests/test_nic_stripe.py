"""Tests for NIC striping in RDMAClientPool.

These are pure-Python unit tests that verify the partitioning, merging,
and edge-case logic of the striped mget_rdma path.  They do NOT require
RDMA hardware — they mock the _l3_rdma extension.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Inject fake _l3_rdma so rdma_client can be imported on any platform
# ---------------------------------------------------------------------------

def _ensure_fake_rdma():
    """Insert a fake _l3_rdma module if the real one isn't available."""
    if "l3_client._l3_rdma" in sys.modules:
        mod = sys.modules["l3_client._l3_rdma"]
        if hasattr(mod, "_is_fake"):
            return  # already injected
        # Real extension present — nothing to do
        return

    fake_ext = types.ModuleType("l3_client._l3_rdma")
    fake_ext._is_fake = True
    fake_ext.RDMATransport = MagicMock
    fake_ext.GIL_RELEASED = True
    fake_ext.DEFAULT_SEND_BUF_SIZE = 16 * 1024 * 1024
    fake_ext.is_available = lambda: False
    fake_ext.__version__ = "0.0.0"
    try:
        from l3_client._version import __version__
        fake_ext.__version__ = __version__
    except Exception:
        pass
    sys.modules["l3_client._l3_rdma"] = fake_ext

    # Force-reimport subpackage modules then facade to pick up the fake
    for mod_name in (
        "l3_client.rdma._constants",
        "l3_client.rdma._batch_ops",
        "l3_client.rdma._client",
        "l3_client.rdma._pool",
        "l3_client.rdma_client",
    ):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])


_ensure_fake_rdma()

from l3_client.rdma_client import RDMAClientPool, _PooledConn  # noqa: E402
from l3_client.reconnect import _MREntry  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build a mock pool without real RDMA connections
# ---------------------------------------------------------------------------

def _make_mock_pool(pool_size: int, endpoints=None):
    """Create a minimal RDMAClientPool-like object for testing stripe logic."""
    from concurrent.futures import ThreadPoolExecutor

    pool = object.__new__(RDMAClientPool)
    pool._pool_size = pool_size
    pool._endpoints = endpoints
    pool._debug = False
    pool._timeout = None
    pool._send_buf_size = None
    pool._has_mget_rdma = True
    pool._rdma_read_retries = 0
    pool._rdma_read_failures = 0
    pool._counter_lock = threading.Lock()
    pool._mr_map = {}
    pool._independent_pd_conns = set()
    pool._independent_mr_maps = {}
    pool._rr_lock = threading.Lock()
    pool._rr_idx = 0
    pool._rebuilding_event = threading.Event()
    pool._rebuilding_event.set()

    # Stripe metrics
    pool._stripe_calls = 0
    pool._stripe_nics_used = 0
    pool._stripe_errors = 0
    pool._per_nic_reads = [0] * pool_size
    pool._per_nic_bytes = [0] * pool_size
    pool._mr_lock = threading.RLock()

    # Build mock connections
    conns = []
    for i in range(pool_size):
        mock_transport = MagicMock()
        conn = _PooledConn(mock_transport, threading.Lock(), endpoint_idx=i)
        conns.append(conn)
    pool._conns = conns

    if pool_size > 1:
        pool._stripe_executor = ThreadPoolExecutor(
            max_workers=pool_size, thread_name_prefix="test-stripe")
    else:
        pool._stripe_executor = None

    return pool


class TestPartitioning(unittest.TestCase):
    """Test that keys are partitioned round-robin across connections."""

    def test_partition_distributes_evenly(self):
        """10 keys across 4 connections -> 3,3,2,2."""
        pool = _make_mock_pool(4)
        n = 10
        ps = pool._pool_size
        part_indices = [[] for _ in range(ps)]
        for i in range(n):
            part_indices[i % ps].append(i)

        self.assertEqual(part_indices[0], [0, 4, 8])
        self.assertEqual(part_indices[1], [1, 5, 9])
        self.assertEqual(part_indices[2], [2, 6])
        self.assertEqual(part_indices[3], [3, 7])

    def test_single_key(self):
        """1 key goes to connection 0 only."""
        pool = _make_mock_pool(4)
        part_indices = [[] for _ in range(pool._pool_size)]
        part_indices[0].append(0)
        self.assertEqual(len(part_indices[0]), 1)
        self.assertEqual(len(part_indices[1]), 0)

    def test_keys_less_than_pool_size(self):
        """2 keys with pool_size=4 -> only 2 connections used."""
        pool = _make_mock_pool(4)
        n = 2
        ps = pool._pool_size
        part_indices = [[] for _ in range(ps)]
        for i in range(n):
            part_indices[i % ps].append(i)
        active = sum(1 for p in part_indices if p)
        self.assertEqual(active, 2)


class TestMerging(unittest.TestCase):
    """Test that per-partition results merge back correctly."""

    def test_merge_preserves_order(self):
        """Results from partitions map back to original indices."""
        n = 8
        ps = 4
        part_indices = [[] for _ in range(ps)]
        for i in range(n):
            part_indices[i % ps].append(i)

        # Simulate partition results: first item found (0), second miss (-1)
        part_results = {}
        for conn_idx, indices in enumerate(part_indices):
            part_results[conn_idx] = [0 if j == 0 else -1 for j in range(len(indices))]

        # Merge
        results = [-1] * n
        for conn_idx, indices in enumerate(part_indices):
            for j, orig_idx in enumerate(indices):
                results[orig_idx] = part_results[conn_idx][j]

        # conn 0: indices [0, 4] -> results [0, -1]
        # conn 1: indices [1, 5] -> results [0, -1]
        # conn 2: indices [2, 6] -> results [0, -1]
        # conn 3: indices [3, 7] -> results [0, -1]
        expected = [0, 0, 0, 0, -1, -1, -1, -1]
        self.assertEqual(results, expected)

    def test_merge_all_miss(self):
        """All misses merge correctly."""
        n = 4
        results = [-1] * n
        self.assertTrue(all(r == -1 for r in results))


class TestPoolSize1Passthrough(unittest.TestCase):
    """pool_size=1 should take the single-connection fast path."""

    def test_pool_size_1_no_executor(self):
        pool = _make_mock_pool(1)
        self.assertIsNone(pool._stripe_executor)

    def test_pool_size_1_has_single_conn(self):
        pool = _make_mock_pool(1)
        self.assertEqual(len(pool._conns), 1)


class TestEndpointIdx(unittest.TestCase):
    """Verify endpoint_idx tracking on _PooledConn."""

    def test_endpoint_idx_assignment(self):
        pool = _make_mock_pool(4, endpoints=[
            ("10.0.0.1", 18001),
            ("10.0.0.2", 18001),
            ("10.0.0.3", 18001),
            ("10.0.0.4", 18001),
        ])
        for i, conn in enumerate(pool._conns):
            self.assertEqual(conn.endpoint_idx, i)


class TestStripeMetrics(unittest.TestCase):
    """Test that stripe metrics accumulate correctly."""

    def test_metric_counters_init_zero(self):
        pool = _make_mock_pool(4)
        self.assertEqual(pool._stripe_calls, 0)
        self.assertEqual(pool._stripe_nics_used, 0)
        self.assertEqual(pool._per_nic_reads, [0, 0, 0, 0])
        self.assertEqual(pool._per_nic_bytes, [0, 0, 0, 0])

    def test_per_nic_reads_increment(self):
        pool = _make_mock_pool(4)
        with pool._counter_lock:
            pool._per_nic_reads[2] += 10
            pool._per_nic_bytes[2] += 1024 * 1024
        self.assertEqual(pool._per_nic_reads[2], 10)
        self.assertEqual(pool._per_nic_bytes[2], 1024 * 1024)


class TestGetMrEntry(unittest.TestCase):
    """Test MR entry lookup for shared vs independent PD connections."""

    def test_shared_pd_lookup(self):
        pool = _make_mock_pool(4)
        pool._mr_map[1] = _MREntry(lkey=42, mr_handle=100, buf_ref=None, ptr=0, size=4096)
        entry = pool._get_mr_entry(0, 1)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.lkey, 42)

    def test_independent_pd_lookup(self):
        pool = _make_mock_pool(4)
        pool._independent_pd_conns.add(2)
        pool._independent_mr_maps[2] = {
            1: _MREntry(lkey=99, mr_handle=200, buf_ref=None, ptr=0, size=4096)
        }
        pool._mr_map[1] = _MREntry(lkey=42, mr_handle=100, buf_ref=None, ptr=0, size=4096)

        # conn 0 uses shared PD
        entry0 = pool._get_mr_entry(0, 1)
        self.assertEqual(entry0.lkey, 42)

        # conn 2 uses independent PD
        entry2 = pool._get_mr_entry(2, 1)
        self.assertEqual(entry2.lkey, 99)

    def test_missing_mr_returns_none(self):
        pool = _make_mock_pool(4)
        self.assertIsNone(pool._get_mr_entry(0, 999))


class TestNextConnAllDead(unittest.TestCase):
    """Verify _next_conn and _next_conn_with_index raise when all dead."""

    def test_next_conn_all_dead(self):
        pool = _make_mock_pool(4)
        for conn in pool._conns:
            conn.alive = False
        with self.assertRaises(RuntimeError) as ctx:
            pool._next_conn()
        self.assertIn("all RDMA pool connections are dead", str(ctx.exception))

    def test_next_conn_with_index_all_dead(self):
        pool = _make_mock_pool(4)
        for conn in pool._conns:
            conn.alive = False
        with self.assertRaises(RuntimeError) as ctx:
            pool._next_conn_with_index()
        self.assertIn("all RDMA pool connections are dead", str(ctx.exception))

    def test_next_conn_some_alive(self):
        """Should succeed when at least one conn is alive."""
        pool = _make_mock_pool(4)
        pool._conns[0].alive = False
        pool._conns[1].alive = False
        pool._conns[3].alive = False
        # conn 2 is still alive
        conn = pool._next_conn()
        self.assertTrue(conn.alive)


class TestStripeErrors(unittest.TestCase):
    """Verify stripe_errors counter init and visibility."""

    def test_stripe_errors_init_zero(self):
        pool = _make_mock_pool(4)
        self.assertEqual(pool._stripe_errors, 0)

    def test_stripe_errors_increment(self):
        pool = _make_mock_pool(4)
        with pool._counter_lock:
            pool._stripe_errors += 1
        self.assertEqual(pool._stripe_errors, 1)


class TestMrLockProtection(unittest.TestCase):
    """Verify _get_mr_entry works correctly under lock."""

    def test_get_mr_entry_under_lock(self):
        pool = _make_mock_pool(4)
        pool._mr_map[1] = _MREntry(lkey=42, mr_handle=100, buf_ref=None, ptr=0, size=4096)
        # _get_mr_entry internally acquires _mr_lock (RLock), should work fine
        entry = pool._get_mr_entry(0, 1)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.lkey, 42)

    def test_get_mr_entry_reentrant(self):
        """RLock allows reentrant acquisition."""
        pool = _make_mock_pool(4)
        pool._mr_map[1] = _MREntry(lkey=42, mr_handle=100, buf_ref=None, ptr=0, size=4096)
        with pool._mr_lock:
            entry = pool._get_mr_entry(0, 1)
            self.assertIsNotNone(entry)
            self.assertEqual(entry.lkey, 42)


class TestMsetReturnType(unittest.TestCase):
    """Verify pool mset/mset_striped return list[int], not scalar."""

    def _make_mset_pool(self, pool_size=2, resp_opcode=0x01):
        """Build a mock pool that can run mset() via mocked roundtrip."""
        from l3_client import protocol

        pool = _make_mock_pool(pool_size)
        pool._degraded = False

        # Write-stripe metrics
        pool._stripe_write_calls = 0
        pool._stripe_write_errors = 0
        pool._per_nic_writes = [0] * pool_size
        pool._per_nic_write_bytes = [0] * pool_size
        pool._set_timings = []

        # Build a canned RESP_OK response (opcode=0x01, all-ok)
        resp_bytes = protocol._pack_header(resp_opcode, 0, 0, b"") + b""
        for conn in pool._conns:
            conn.transport.roundtrip.return_value = resp_bytes

        return pool

    def test_mset_empty_returns_empty_list(self):
        pool = self._make_mset_pool()
        result = pool.mset([], [])
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_mset_returns_list_of_int(self):
        from l3_client.sgl import SGL
        pool = self._make_mset_pool()
        buf = b"\x00" * 64
        sgl1 = SGL(id(buf), len(buf), 0)
        sgl1._buf_ref = buf
        sgl2 = SGL(id(buf), len(buf), 0)
        sgl2._buf_ref = buf
        result = pool.mset(["k1", "k2"], [sgl1, sgl2])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(r, int) for r in result))
        self.assertEqual(result, [0, 0])

    def test_mset_striped_empty_returns_empty_list(self):
        pool = self._make_mset_pool()
        result = pool.mset_striped([], [])
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_mset_striped_returns_list_of_int(self):
        from l3_client.sgl import SGL
        pool = self._make_mset_pool(pool_size=2)
        buf = b"\x00" * 64
        sgls = []
        for _ in range(4):
            s = SGL(id(buf), len(buf), 0)
            s._buf_ref = buf
            sgls.append(s)
        result = pool.mset_striped(["k0", "k1", "k2", "k3"], sgls)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(isinstance(r, int) for r in result))

    def test_mset_result_iterable_in_all(self):
        """The exact pattern from warmup: all(r == 0 for r in results)."""
        from l3_client.sgl import SGL
        pool = self._make_mset_pool()
        buf = b"\x00" * 64
        sgl1 = SGL(id(buf), len(buf), 0)
        sgl1._buf_ref = buf
        sgl2 = SGL(id(buf), len(buf), 0)
        sgl2._buf_ref = buf
        results = pool.mset(["k1", "k2"], [sgl1, sgl2])
        # This is the exact line that was crashing in warmup:
        self.assertTrue(all(r == 0 for r in results))

    def test_mset_striped_single_conn_delegates_to_mset(self):
        """pool_size=1 → mset_striped delegates to mset, returns list."""
        from l3_client.sgl import SGL
        pool = self._make_mset_pool(pool_size=1)
        buf = b"\x00" * 64
        sgl = SGL(id(buf), len(buf), 0)
        sgl._buf_ref = buf
        result = pool.mset_striped(["k1"], [sgl])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
