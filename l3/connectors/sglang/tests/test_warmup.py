"""Unit tests for the data-driven warmup system.

Tests cover:
- INIT -> COLD transition on first data
- COLD -> STEADY on server readiness
- COLD -> STEADY on timeout fallback
- reset() on reconnect
- Backward-compat config mapping (aggressive_* -> cold_*)
- allow_adaptive_sizing() always returns True
- phase_info() in all states
- is_cache_cold() semantics
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from cama_module.warmup import (
    WarmupConfig,
    WarmupController,
    WarmupPhase,
    _apply_compat_mapping,
)


class TestWarmupPhaseEnum(unittest.TestCase):
    """Test phase enum values and backward-compat aliases."""

    def test_cold_value(self):
        self.assertEqual(WarmupPhase.COLD.value, "cold")

    def test_steady_value(self):
        self.assertEqual(WarmupPhase.STEADY.value, "steady")

    def test_aggressive_alias(self):
        self.assertEqual(WarmupPhase.AGGRESSIVE.value, WarmupPhase.COLD.value)

    def test_ramp_alias(self):
        self.assertEqual(WarmupPhase.RAMP.value, WarmupPhase.STEADY.value)


class TestBackwardCompatMapping(unittest.TestCase):
    """Test deprecated aggressive_* -> cold_* field mapping."""

    def test_maps_aggressive_batch_size(self):
        cfg = WarmupConfig(aggressive_batch_size=8192)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _apply_compat_mapping(cfg)
            self.assertEqual(result.cold_batch_size, 8192)
            self.assertTrue(any("deprecated" in str(x.message).lower() for x in w))

    def test_maps_aggressive_jitter_ms(self):
        cfg = WarmupConfig(aggressive_jitter_ms=5.0)
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _apply_compat_mapping(cfg)
            self.assertEqual(result.cold_jitter_ms, 5.0)

    def test_maps_aggressive_deadline_ms(self):
        cfg = WarmupConfig(aggressive_deadline_ms=10.0)
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _apply_compat_mapping(cfg)
            self.assertEqual(result.cold_deadline_ms, 10.0)

    def test_no_mapping_when_new_keys_set(self):
        cfg = WarmupConfig(
            cold_batch_size=2048,
            aggressive_batch_size=8192,
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _apply_compat_mapping(cfg)
            # New key takes precedence — old key ignored
            self.assertEqual(result.cold_batch_size, 2048)

    def test_no_warning_when_not_set(self):
        cfg = WarmupConfig()  # all defaults (aggressive_batch_size=0 means not set)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _apply_compat_mapping(cfg)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(dep_warnings), 0)


class TestInitPhase(unittest.TestCase):
    """Test INIT state (before first data)."""

    def setUp(self):
        self.ctrl = WarmupController(
            WarmupConfig(enabled=True),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )

    def test_starts_in_init(self):
        info = self.ctrl.phase_info()
        self.assertEqual(info["phase"], "init")

    def test_init_returns_steady_values(self):
        self.assertEqual(self.ctrl.effective_batch_size(), 2048)
        self.assertEqual(self.ctrl.effective_deadline_ms(), 20.0)

    def test_phase_returns_steady_in_init(self):
        self.assertIs(self.ctrl.phase(), WarmupPhase.STEADY)

    def test_is_cache_cold_false_in_init(self):
        self.assertFalse(self.ctrl.is_cache_cold())

    def test_allow_adaptive_sizing_always_true(self):
        self.assertTrue(self.ctrl.allow_adaptive_sizing())


class TestColdTransition(unittest.TestCase):
    """Test INIT -> COLD transition on first data."""

    def setUp(self):
        self.ctrl = WarmupController(
            WarmupConfig(
                enabled=True,
                cold_batch_size=4096,
                cold_jitter_ms=0.0,
                cold_deadline_ms=2.0,
                server_poll_timeout_s=0.5,  # short timeout for tests
            ),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )

    def test_record_pages_triggers_cold(self):
        self.ctrl.record_pages(100)
        info = self.ctrl.phase_info()
        self.assertEqual(info["phase"], "cold")
        self.assertTrue(self.ctrl.is_cache_cold())

    def test_cold_returns_cold_values(self):
        self.ctrl.record_pages(100)
        self.assertEqual(self.ctrl.effective_batch_size(), 4096)
        self.assertEqual(self.ctrl.effective_deadline_ms(), 2.0)

    def test_zero_pages_does_not_trigger_cold(self):
        self.ctrl.record_pages(0)
        info = self.ctrl.phase_info()
        self.assertEqual(info["phase"], "init")

    def test_pages_accumulate(self):
        self.ctrl.record_pages(50)
        self.ctrl.record_pages(50)
        info = self.ctrl.phase_info()
        self.assertEqual(info["pages_written"], 100)


class TestSteadyTransitionViaTimeout(unittest.TestCase):
    """Test COLD -> STEADY via timeout fallback (no conn_factory)."""

    def test_timeout_triggers_steady(self):
        ctrl = WarmupController(
            WarmupConfig(
                enabled=True,
                server_poll_timeout_s=0.2,  # very short for testing
            ),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        # Trigger COLD
        ctrl.record_pages(100)
        self.assertTrue(ctrl.is_cache_cold())

        # Wait for timeout fallback
        time.sleep(0.5)
        self.assertFalse(ctrl.is_cache_cold())
        info = ctrl.phase_info()
        self.assertEqual(info["phase"], "steady")


class TestSteadyTransitionViaServer(unittest.TestCase):
    """Test COLD -> STEADY via server readiness polling."""

    def test_server_ready_triggers_steady(self):
        mock_conn = MagicMock()
        mock_conn.maintenance_status.return_value = {
            "shards": [
                {"detection": {"status": "rebuilt"}},
                {"detection": {"status": "rebuilt"}},
            ]
        }

        ctrl = WarmupController(
            WarmupConfig(
                enabled=True,
                server_poll_interval_s=0.1,
                server_poll_timeout_s=5.0,
            ),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.set_conn_factory(lambda: mock_conn)
        ctrl.record_pages(100)

        # Give poller time to check
        time.sleep(0.5)
        self.assertFalse(ctrl.is_cache_cold())
        info = ctrl.phase_info()
        self.assertEqual(info["phase"], "steady")
        mock_conn.maintenance_status.assert_called()

    def test_server_warming_up_stays_cold(self):
        mock_conn = MagicMock()
        mock_conn.maintenance_status.return_value = {
            "shards": [
                {"detection": {"status": "warming_up"}},
                {"detection": {"status": "rebuilt"}},
            ]
        }

        ctrl = WarmupController(
            WarmupConfig(
                enabled=True,
                server_poll_interval_s=0.1,
                server_poll_timeout_s=0.5,
            ),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.set_conn_factory(lambda: mock_conn)
        ctrl.record_pages(100)

        # First check: should still be cold (one shard warming up)
        time.sleep(0.2)
        self.assertTrue(ctrl.is_cache_cold())

    def test_detected_and_disabled_count_as_ready(self):
        mock_conn = MagicMock()
        mock_conn.maintenance_status.return_value = {
            "shards": [
                {"detection": {"status": "detected"}},
                {"detection": {"status": "disabled"}},
                {"detection": {"status": "rebuilt"}},
            ]
        }

        ctrl = WarmupController(
            WarmupConfig(
                enabled=True,
                server_poll_interval_s=0.1,
                server_poll_timeout_s=5.0,
            ),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.set_conn_factory(lambda: mock_conn)
        ctrl.record_pages(100)

        time.sleep(0.5)
        self.assertFalse(ctrl.is_cache_cold())

    def test_conn_factory_error_retries(self):
        call_count = [0]
        mock_conn = MagicMock()

        def factory():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("not ready")
            return mock_conn

        mock_conn.maintenance_status.return_value = {
            "shards": [{"detection": {"status": "rebuilt"}}]
        }

        ctrl = WarmupController(
            WarmupConfig(
                enabled=True,
                server_poll_interval_s=0.1,
                server_poll_timeout_s=5.0,
            ),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.set_conn_factory(factory)
        ctrl.record_pages(100)

        time.sleep(1.0)
        self.assertFalse(ctrl.is_cache_cold())
        self.assertGreater(call_count[0], 2)


class TestReset(unittest.TestCase):
    """Test reset() for reconnect scenarios."""

    def test_reset_returns_to_init(self):
        ctrl = WarmupController(
            WarmupConfig(
                enabled=True,
                server_poll_timeout_s=60.0,
            ),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.record_pages(100)
        self.assertTrue(ctrl.is_cache_cold())

        ctrl.reset()
        info = ctrl.phase_info()
        self.assertEqual(info["phase"], "init")
        self.assertFalse(ctrl.is_cache_cold())
        self.assertEqual(info["pages_written"], 0)

    def test_reset_allows_re_entering_cold(self):
        ctrl = WarmupController(
            WarmupConfig(
                enabled=True,
                server_poll_timeout_s=0.2,
            ),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        # INIT -> COLD -> STEADY (timeout)
        ctrl.record_pages(100)
        time.sleep(0.5)
        self.assertFalse(ctrl.is_cache_cold())

        # Reset -> INIT
        ctrl.reset()
        self.assertEqual(ctrl.phase_info()["phase"], "init")

        # New data -> COLD again
        ctrl.record_pages(50)
        self.assertTrue(ctrl.is_cache_cold())

    def test_reset_disabled_stays_steady(self):
        ctrl = WarmupController(
            WarmupConfig(enabled=False),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.reset()
        self.assertFalse(ctrl.is_cache_cold())
        self.assertIs(ctrl.phase(), WarmupPhase.STEADY)


class TestDisabled(unittest.TestCase):
    """Test warmup disabled mode."""

    def test_disabled_starts_steady(self):
        ctrl = WarmupController(
            WarmupConfig(enabled=False),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        self.assertIs(ctrl.phase(), WarmupPhase.STEADY)
        self.assertFalse(ctrl.is_cache_cold())
        self.assertEqual(ctrl.effective_batch_size(), 2048)

    def test_record_pages_noop_when_disabled(self):
        ctrl = WarmupController(
            WarmupConfig(enabled=False),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.record_pages(1000)
        # Should still be STEADY, not COLD
        self.assertIs(ctrl.phase(), WarmupPhase.STEADY)
        self.assertFalse(ctrl.is_cache_cold())


class TestPhaseInfo(unittest.TestCase):
    """Test phase_info() output."""

    def test_init_phase_info(self):
        ctrl = WarmupController(
            WarmupConfig(enabled=True, tp_size=4),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        info = ctrl.phase_info()
        self.assertEqual(info["phase"], "init")
        self.assertEqual(info["pages_written"], 0)
        self.assertAlmostEqual(info["elapsed_s"], 0.0, places=0)
        self.assertEqual(info["tp_size"], 4)
        self.assertIn("tp_jitter_scale", info)

    def test_cold_phase_info(self):
        ctrl = WarmupController(
            WarmupConfig(enabled=True, server_poll_timeout_s=60.0),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.record_pages(42)
        info = ctrl.phase_info()
        self.assertEqual(info["phase"], "cold")
        self.assertEqual(info["pages_written"], 42)
        self.assertGreaterEqual(info["elapsed_s"], 0)


class TestTPJitterScale(unittest.TestCase):
    """Test TP-based jitter scaling."""

    def test_tp1_zero_jitter(self):
        ctrl = WarmupController(
            WarmupConfig(enabled=True, tp_size=1, cold_jitter_ms=10.0,
                         server_poll_timeout_s=60.0),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.record_pages(1)  # enter COLD
        self.assertAlmostEqual(ctrl.effective_jitter_ms(), 0.0)

    def test_tp8_near_full_jitter(self):
        ctrl = WarmupController(
            WarmupConfig(enabled=True, tp_size=8, cold_jitter_ms=10.0,
                         server_poll_timeout_s=60.0),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        ctrl.record_pages(1)  # enter COLD
        # (8-1)/4 = 1.75, clamped to 1.0
        self.assertAlmostEqual(ctrl.effective_jitter_ms(), 10.0)


class TestConcurrentAccess(unittest.TestCase):
    """Test thread safety."""

    def test_concurrent_record_and_query(self):
        ctrl = WarmupController(
            WarmupConfig(enabled=True, server_poll_timeout_s=0.3),
            steady_batch_size=2048,
            steady_jitter_ms=10.0,
            steady_deadline_ms=20.0,
        )
        errors = []

        def writer():
            try:
                for _ in range(100):
                    ctrl.record_pages(1)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    ctrl.effective_batch_size()
                    ctrl.is_cache_cold()
                    ctrl.phase_info()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
