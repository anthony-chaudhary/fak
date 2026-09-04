"""Unit tests for backup queue coalescing — no live server required.

Tests cover BackupCoalescer.drain(), _merge_ops(), ack handling,
and observability stats.
"""

import threading
import time
import unittest
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Mock heavy deps before importing cache_controller.
import sys

_mock_torch = MagicMock()
_mock_torch.distributed.get_world_size.return_value = 1
# torch.cat must concatenate lists for merge_ops tests
_mock_torch.cat = lambda tensors: sum(tensors, [])
sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("sglang", MagicMock())
sys.modules.setdefault("sglang.srt", MagicMock())
sys.modules.setdefault("sglang.srt.mem_cache", MagicMock())
sys.modules.setdefault("sglang.srt.mem_cache.hicache_storage", MagicMock())
sys.modules.setdefault("sglang.srt.mem_cache.allocator", MagicMock())
sys.modules.setdefault("sglang.srt.mem_cache.memory_pool_host", MagicMock())
sys.modules.setdefault("sglang.srt.mem_cache.memory_pool", MagicMock())
sys.modules.setdefault("sglang.srt.distributed", MagicMock())
sys.modules.setdefault("sglang.srt.distributed.parallel_state", MagicMock())
sys.modules.setdefault("sglang.srt.layers", MagicMock())
sys.modules.setdefault("sglang.srt.layers.dp_attention", MagicMock())
sys.modules.setdefault("sglang.srt.utils", MagicMock())

from patches.cache_controller import BackupCoalescer, StorageOperation


def _make_op(num_pages=1, prefix_keys=None):
    """Create a minimal StorageOperation for testing."""
    return StorageOperation(
        host_indices=list(range(num_pages)),
        token_ids=list(range(num_pages * 10, num_pages * 10 + num_pages)),
        last_hash=f"hash_{num_pages}",
        hash_value=[f"h{i}" for i in range(num_pages)],
        prefix_keys=prefix_keys,
    )


# ---------------------------------------------------------------------------
# TestMergeOps — static merge logic
# ---------------------------------------------------------------------------
class TestMergeOps(unittest.TestCase):
    """Tests for BackupCoalescer._merge_ops()."""

    def test_single_op_passthrough(self):
        """Single-element list returns a merged op with same data."""
        op = _make_op(3, prefix_keys=["pk1"])
        merged = BackupCoalescer._merge_ops([op])
        self.assertEqual(merged.hash_value, op.hash_value)
        self.assertEqual(merged.token_ids, op.token_ids)
        self.assertIsNone(merged.prefix_keys)
        self.assertTrue(hasattr(merged, '_source_ops'))
        self.assertEqual(len(merged._source_ops), 1)

    def test_multi_op_concatenation(self):
        """Multiple ops have their hash_value and token_ids concatenated."""
        op1 = _make_op(2)
        op2 = _make_op(3)
        merged = BackupCoalescer._merge_ops([op1, op2])
        self.assertEqual(merged.hash_value, op1.hash_value + op2.hash_value)
        self.assertEqual(merged.token_ids, op1.token_ids + op2.token_ids)
        self.assertEqual(len(merged.host_indices), 2 + 3)

    def test_prefix_keys_from_first(self):
        """Multi-op merge sets prefix_keys to None (ops may have different prefixes)."""
        op1 = _make_op(1, prefix_keys=["a", "b"])
        op2 = _make_op(1, prefix_keys=["c", "d"])
        merged = BackupCoalescer._merge_ops([op1, op2])
        self.assertIsNone(merged.prefix_keys)

    def test_none_prefix_keys(self):
        """Multi-op merge always produces None prefix_keys regardless of inputs."""
        op1 = _make_op(1, prefix_keys=None)
        op2 = _make_op(1, prefix_keys=["x"])
        merged = BackupCoalescer._merge_ops([op1, op2])
        self.assertIsNone(merged.prefix_keys)

    def test_mixed_page_counts(self):
        """Ops with different page counts merge correctly."""
        op1 = _make_op(1)
        op2 = _make_op(5)
        op3 = _make_op(2)
        merged = BackupCoalescer._merge_ops([op1, op2, op3])
        self.assertEqual(len(merged.hash_value), 1 + 5 + 2)
        self.assertEqual(len(merged._source_ops), 3)


# ---------------------------------------------------------------------------
# TestDrain — drain behavior
# ---------------------------------------------------------------------------
class TestDrain(unittest.TestCase):
    """Tests for BackupCoalescer.drain()."""

    def test_single_op_returned(self):
        """Single op in queue is returned with _source_ops attached."""
        q = Queue()
        stop = threading.Event()
        c = BackupCoalescer(q, stop, max_pages=256, deadline_ms=0)
        op = _make_op(1)
        q.put(op)
        result = c.drain()
        self.assertIsNotNone(result)
        self.assertEqual(result.hash_value, op.hash_value)
        self.assertTrue(hasattr(result, '_source_ops'))

    def test_fills_to_max_pages(self):
        """Drain stops when max_pages is reached."""
        q = Queue()
        stop = threading.Event()
        c = BackupCoalescer(q, stop, max_pages=3, deadline_ms=50)
        for _ in range(10):
            q.put(_make_op(1))
        result = c.drain()
        # Should have merged 3 ops (3 pages = max_pages)
        self.assertEqual(len(result.hash_value), 3)
        self.assertEqual(len(result._source_ops), 3)
        # Remaining 7 should still be in the queue
        self.assertEqual(q.qsize(), 7)

    def test_deadline_zero_drains_available(self):
        """deadline_ms=0 drains only what's already in the queue (no waiting)."""
        q = Queue()
        stop = threading.Event()
        c = BackupCoalescer(q, stop, max_pages=256, deadline_ms=0)
        q.put(_make_op(1))
        q.put(_make_op(1))
        result = c.drain()
        # With deadline=0, phase 2 loop breaks immediately on remaining<=0,
        # but the first op was already taken in phase 1.
        # Whether we get 1 or 2 depends on timing — at minimum we get 1.
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result._source_ops), 1)

    def test_empty_queue_returns_none(self):
        """Empty queue returns None after timeout."""
        q = Queue()
        stop = threading.Event()
        c = BackupCoalescer(q, stop, max_pages=256, deadline_ms=0)
        t0 = time.monotonic()
        result = c.drain()
        elapsed = time.monotonic() - t0
        self.assertIsNone(result)
        # Should take ~1s (phase 1 blocking timeout)
        self.assertGreater(elapsed, 0.5)

    def test_stop_event_flushes(self):
        """stop_event causes drain to return immediately with what's collected."""
        q = Queue()
        stop = threading.Event()
        c = BackupCoalescer(q, stop, max_pages=256, deadline_ms=50)
        q.put(_make_op(1))
        stop.set()
        result = c.drain()
        # Should return the one op without waiting for more
        self.assertIsNotNone(result)
        self.assertEqual(len(result._source_ops), 1)

    def test_stop_event_on_empty(self):
        """stop_event + empty queue returns None quickly."""
        q = Queue()
        stop = threading.Event()
        stop.set()
        c = BackupCoalescer(q, stop, max_pages=256, deadline_ms=50)
        # Queue is empty, phase 1 get will timeout in 1s
        # but stop_event is already set so phase 2 won't wait
        t0 = time.monotonic()
        result = c.drain()
        # Returns None from the Empty exception in phase 1
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# TestAckHandling — ack all source ops
# ---------------------------------------------------------------------------
class TestAckHandling(unittest.TestCase):
    """Tests for _source_ops ack pattern."""

    def test_coalesced_ops_ack_all_sources(self):
        """getattr(_source_ops) returns all original ops for ack."""
        ops = [_make_op(1) for _ in range(5)]
        merged = BackupCoalescer._merge_ops(ops)
        sources = getattr(merged, '_source_ops', [merged])
        self.assertEqual(len(sources), 5)
        for orig, src in zip(ops, sources):
            self.assertIs(orig, src)

    def test_non_coalesced_single_ack(self):
        """Op with _source_ops=None falls back to [operation] for single ack."""
        op = _make_op(1)
        sources = getattr(op, '_source_ops', None) or [op]
        self.assertEqual(len(sources), 1)
        self.assertIs(sources[0], op)

    def test_non_coalesced_ack_fallback(self):
        """Op without _source_ops attr (non-coalesced path) uses getattr fallback."""
        op = _make_op(2)
        # Simulate non-coalesced path: op never went through drain/_merge_ops
        # so _source_ops is never set.  Delete it if StorageOperation.__init__
        # happens to set a default.
        if hasattr(op, '_source_ops'):
            delattr(op, '_source_ops')
        sources = getattr(op, '_source_ops', None) or [op]
        self.assertEqual(len(sources), 1)
        self.assertIs(sources[0], op)


# ---------------------------------------------------------------------------
# TestObservability — stats tracking
# ---------------------------------------------------------------------------
class TestMaxPagesRef(unittest.TestCase):
    """Tests for live max_pages tracking via max_pages_ref."""

    def test_max_pages_ref_overrides_static(self):
        """max_pages_ref callable is used instead of static max_pages."""
        q = Queue()
        stop = threading.Event()
        live_size = [5]  # mutable to simulate auto-tuning
        c = BackupCoalescer(q, stop, max_pages=3, deadline_ms=50,
                            max_pages_ref=lambda: live_size[0])
        for _ in range(10):
            q.put(_make_op(1))
        result = c.drain()
        # Should use live_size (5), not static max_pages (3)
        self.assertEqual(len(result.hash_value), 5)

    def test_max_pages_ref_tracks_changes(self):
        """Changing the value returned by max_pages_ref affects next drain."""
        q = Queue()
        stop = threading.Event()
        live_size = [2]
        c = BackupCoalescer(q, stop, max_pages=100, deadline_ms=50,
                            max_pages_ref=lambda: live_size[0])
        for _ in range(10):
            q.put(_make_op(1))
        result = c.drain()
        self.assertEqual(len(result.hash_value), 2)

        # "auto-tune" increases the batch size
        live_size[0] = 6
        result = c.drain()
        self.assertEqual(len(result.hash_value), 6)

    def test_no_ref_uses_static(self):
        """Without max_pages_ref, falls back to static max_pages."""
        q = Queue()
        stop = threading.Event()
        c = BackupCoalescer(q, stop, max_pages=4, deadline_ms=50)
        for _ in range(10):
            q.put(_make_op(1))
        result = c.drain()
        self.assertEqual(len(result.hash_value), 4)


class TestObservability(unittest.TestCase):
    """Tests for coalescer stats."""

    def test_avg_ops_per_batch(self):
        """avg_ops_per_batch tracks correctly across multiple drains."""
        q = Queue()
        stop = threading.Event()
        c = BackupCoalescer(q, stop, max_pages=10, deadline_ms=50)

        # Drain 1: 3 ops
        for _ in range(3):
            q.put(_make_op(1))
        c.drain()

        # Drain 2: 2 ops
        for _ in range(2):
            q.put(_make_op(1))
        c.drain()

        # avg = (3 + 2) / 2 = 2.5
        self.assertAlmostEqual(c.avg_ops_per_batch, 2.5)

    def test_reset_stats_zeroes(self):
        """reset_stats clears counters."""
        q = Queue()
        stop = threading.Event()
        c = BackupCoalescer(q, stop, max_pages=10, deadline_ms=0)
        for _ in range(3):
            q.put(_make_op(1))
        c.drain()
        self.assertGreater(c.avg_ops_per_batch, 0)

        c.reset_stats()
        self.assertEqual(c._coalesced_ops, 0)
        self.assertEqual(c._coalesced_batches, 0)
        self.assertEqual(c.avg_ops_per_batch, 0.0)


if __name__ == "__main__":
    unittest.main()
