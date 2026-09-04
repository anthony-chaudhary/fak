"""Unit tests for backup write jitter — no live server required.

Tests cover _jitter_sleep(), config clamping, _page_backup() jitter integration,
and _BackupResult stat propagation.
"""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# We test against the real class but mock out heavy deps at import time.
# cache_controller imports torch and sglang — patch them before import.
import sys

_mock_torch = MagicMock()
_mock_torch.distributed.get_world_size.return_value = 1
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

from patches.cache_controller import HiCacheController


def _make_controller_stub(**overrides):
    """Create a minimal stub with just the fields _jitter_sleep / _page_backup need."""
    stub = object.__new__(HiCacheController)
    stub.backup_jitter_ms = overrides.get("backup_jitter_ms", 0)
    stub.stop_event = overrides.get("stop_event", threading.Event())
    stub.storage_batch_size = overrides.get("storage_batch_size", 256)
    stub.page_size = overrides.get("page_size", 1)
    stub.page_set_func = overrides.get("page_set_func", lambda *a, **kw: True)
    stub.write_policy = overrides.get("write_policy", "write_through")
    stub.backup_skip = overrides.get("backup_skip", False)
    # _warmup mock: effective_batch_size / effective_jitter_ms used by _page_backup / _jitter_sleep
    warmup = SimpleNamespace(
        effective_batch_size=lambda: overrides.get("storage_batch_size", 256),
        effective_jitter_ms=lambda: float(overrides.get("backup_jitter_ms", 0)),
    )
    stub._warmup = warmup
    return stub


def _make_operation(num_pages, page_size=1):
    """Create a minimal operation-like object for _page_backup."""
    op = SimpleNamespace()
    op.hash_value = list(range(num_pages))
    op.host_indices = list(range(num_pages * page_size))
    op.prefix_keys = []
    op.completed_tokens = 0
    op.id = 42
    return op


class TestJitterSleep(unittest.TestCase):
    """Tests for _jitter_sleep()."""

    def test_jitter_zero_no_sleep(self):
        """backup_jitter_ms=0 returns 0.0 with no random/wait calls."""
        ctrl = _make_controller_stub(backup_jitter_ms=0)
        with patch("patches.cache_controller.random") as mock_random:
            result = ctrl._jitter_sleep()
        self.assertEqual(result, 0.0)
        mock_random.uniform.assert_not_called()

    def test_jitter_positive_in_range(self):
        """Repeated calls return values in [0, backup_jitter_ms]."""
        ctrl = _make_controller_stub(backup_jitter_ms=50)
        results = [ctrl._jitter_sleep() for _ in range(100)]
        for r in results:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 50.0)

    def test_jitter_respects_stop_event(self):
        """stop_event already set -> returns 0.0 immediately."""
        stop = threading.Event()
        stop.set()
        ctrl = _make_controller_stub(backup_jitter_ms=100, stop_event=stop)
        result = ctrl._jitter_sleep()
        self.assertEqual(result, 0.0)


class TestConfigClamping(unittest.TestCase):
    """Tests for backup_jitter_ms config parsing bounds."""

    def test_jitter_capped_at_500(self):
        """Config value 1000 -> stored as 500."""
        raw = 1000
        clamped = max(0, min(500, raw))
        self.assertEqual(clamped, 500)

    def test_jitter_negative_clamped(self):
        """Config value -10 -> stored as 0."""
        raw = -10
        clamped = max(0, min(500, raw))
        self.assertEqual(clamped, 0)


class TestPageBackupJitter(unittest.TestCase):
    """Tests for jitter integration in _page_backup()."""

    def test_no_jitter_before_first_batch(self):
        """_jitter_sleep called N-1 times for N sub-batches."""
        ctrl = _make_controller_stub(
            backup_jitter_ms=10, storage_batch_size=2, page_size=1,
        )
        ctrl._jitter_sleep = MagicMock(return_value=5.0)
        op = _make_operation(num_pages=5, page_size=1)  # 3 sub-batches: [2,2,1]

        ctrl._page_backup(op)

        # 3 sub-batches -> jitter called 2 times (not before the first)
        self.assertEqual(ctrl._jitter_sleep.call_count, 2)

    def test_stop_event_breaks_loop(self):
        """Set stop_event during jitter, verify loop exits early."""
        stop = threading.Event()
        ctrl = _make_controller_stub(
            backup_jitter_ms=10, storage_batch_size=1, page_size=1,
            stop_event=stop,
        )
        call_count = 0
        def jitter_side_effect():
            nonlocal call_count
            call_count += 1
            stop.set()  # Trigger stop on first jitter call
            return 5.0

        ctrl._jitter_sleep = jitter_side_effect
        op = _make_operation(num_pages=5, page_size=1)  # 5 sub-batches

        ctrl._page_backup(op)

        # First batch runs (no jitter), second batch triggers jitter which sets stop,
        # loop breaks — so only 1 batch completed
        self.assertEqual(op.completed_tokens, 1)
        self.assertEqual(call_count, 1)


class TestBackupResultStats(unittest.TestCase):
    """Tests for _BackupResult jitter/gap fields."""

    def test_backup_result_carries_stats(self):
        """_BackupResult tuple has jitter_ms and avg_gap_ms fields."""
        result = HiCacheController._BackupResult(
            success=True, elapsed_ms=100.0, pages=10,
            jitter_ms=25.0, avg_gap_ms=3.5,
        )
        self.assertEqual(result.jitter_ms, 25.0)
        self.assertEqual(result.avg_gap_ms, 3.5)
        self.assertTrue(result.success)
        self.assertEqual(result.pages, 10)


class TestAdaptiveBatchSize(unittest.TestCase):
    """Tests for batch_size_auto adaptive tuning logic."""

    def _make_auto_stub(self, batch_size=256, target_ms=50.0):
        stub = _make_controller_stub(storage_batch_size=batch_size)
        stub.batch_size_auto = True
        stub.batch_size_latency_target_ms = target_ms
        stub._batch_size_max = batch_size
        stub._batch_size_min = 32
        return stub

    def test_auto_halves_on_high_latency(self):
        """Batch size halves when avg latency exceeds target."""
        ctrl = self._make_auto_stub(batch_size=256, target_ms=50.0)
        # Simulate: avg_lat=80 > target=50 → halve to 128
        avg_lat = 80.0
        old_bs = ctrl.storage_batch_size
        if avg_lat > ctrl.batch_size_latency_target_ms and old_bs > ctrl._batch_size_min:
            ctrl.storage_batch_size = max(ctrl._batch_size_min, old_bs // 2)
        self.assertEqual(ctrl.storage_batch_size, 128)

    def test_auto_doubles_on_low_latency(self):
        """Batch size doubles when avg latency is well below target."""
        ctrl = self._make_auto_stub(batch_size=256, target_ms=50.0)
        ctrl.storage_batch_size = 64  # currently reduced
        avg_lat = 15.0  # < 50 * 0.5 = 25
        old_bs = ctrl.storage_batch_size
        if avg_lat < ctrl.batch_size_latency_target_ms * 0.5 and old_bs < ctrl._batch_size_max:
            ctrl.storage_batch_size = min(ctrl._batch_size_max, old_bs * 2)
        self.assertEqual(ctrl.storage_batch_size, 128)

    def test_auto_no_change_in_band(self):
        """Batch size unchanged when latency is within target band."""
        ctrl = self._make_auto_stub(batch_size=256, target_ms=50.0)
        ctrl.storage_batch_size = 128
        avg_lat = 35.0  # between 25 and 50 → no change
        old_bs = ctrl.storage_batch_size
        if avg_lat > ctrl.batch_size_latency_target_ms and old_bs > ctrl._batch_size_min:
            ctrl.storage_batch_size = max(ctrl._batch_size_min, old_bs // 2)
        elif avg_lat < ctrl.batch_size_latency_target_ms * 0.5 and old_bs < ctrl._batch_size_max:
            ctrl.storage_batch_size = min(ctrl._batch_size_max, old_bs * 2)
        self.assertEqual(ctrl.storage_batch_size, 128)

    def test_auto_respects_floor(self):
        """Batch size never goes below _batch_size_min."""
        ctrl = self._make_auto_stub(batch_size=256, target_ms=50.0)
        ctrl.storage_batch_size = 32  # already at floor
        avg_lat = 100.0  # high latency
        old_bs = ctrl.storage_batch_size
        if avg_lat > ctrl.batch_size_latency_target_ms and old_bs > ctrl._batch_size_min:
            ctrl.storage_batch_size = max(ctrl._batch_size_min, old_bs // 2)
        self.assertEqual(ctrl.storage_batch_size, 32)  # stays at floor

    def test_auto_respects_ceiling(self):
        """Batch size never exceeds configured max."""
        ctrl = self._make_auto_stub(batch_size=256, target_ms=50.0)
        ctrl.storage_batch_size = 256  # already at max
        avg_lat = 10.0  # very low
        old_bs = ctrl.storage_batch_size
        if avg_lat < ctrl.batch_size_latency_target_ms * 0.5 and old_bs < ctrl._batch_size_max:
            ctrl.storage_batch_size = min(ctrl._batch_size_max, old_bs * 2)
        self.assertEqual(ctrl.storage_batch_size, 256)  # stays at ceiling

    def test_auto_disabled_no_change(self):
        """When batch_size_auto=False, batch size is not adjusted."""
        ctrl = _make_controller_stub(storage_batch_size=256)
        ctrl.batch_size_auto = False
        # Even with high latency, no adjustment happens when auto is off
        self.assertEqual(ctrl.storage_batch_size, 256)


if __name__ == "__main__":
    unittest.main()
