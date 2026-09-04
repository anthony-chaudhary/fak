"""Unit tests for prewarm.py — no live PrisKV server required.

Tests cover the PrewarmRegistry (store/claim/expiry/mismatch/double-claim),
config fingerprinting, and the CamaPrewarmProvider error path.
"""

import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import cama_module.prewarm as pw
from cama_module.prewarm import (
    PrewarmRegistry,
    PrewarmResult,
    _cama_config_fingerprint,
    claim_prewarmed_connection,
    start_cama_prewarm,
    _PREWARM_TTL_S,
)


class TestConfigFingerprint(unittest.TestCase):
    """Test deterministic config fingerprinting."""

    def test_same_config_same_fingerprint(self):
        fp1 = _cama_config_fingerprint("10.0.0.1", 18001, 4, 0)
        fp2 = _cama_config_fingerprint("10.0.0.1", 18001, 4, 0)
        self.assertEqual(fp1, fp2)

    def test_different_addr_different_fingerprint(self):
        fp1 = _cama_config_fingerprint("10.0.0.1", 18001, 4, 0)
        fp2 = _cama_config_fingerprint("10.0.0.2", 18001, 4, 0)
        self.assertNotEqual(fp1, fp2)

    def test_different_port_different_fingerprint(self):
        fp1 = _cama_config_fingerprint("10.0.0.1", 18001, 4, 0)
        fp2 = _cama_config_fingerprint("10.0.0.1", 18002, 4, 0)
        self.assertNotEqual(fp1, fp2)

    def test_different_pool_size_different_fingerprint(self):
        fp1 = _cama_config_fingerprint("10.0.0.1", 18001, 4, 0)
        fp2 = _cama_config_fingerprint("10.0.0.1", 18001, 8, 0)
        self.assertNotEqual(fp1, fp2)

    def test_different_buf_size_different_fingerprint(self):
        fp1 = _cama_config_fingerprint("10.0.0.1", 18001, 4, 0)
        fp2 = _cama_config_fingerprint("10.0.0.1", 18001, 4, 8388608)
        self.assertNotEqual(fp1, fp2)


class TestPrewarmRegistry(unittest.TestCase):
    """Test the thread-safe store/claim registry."""

    def setUp(self):
        # Reset singleton for test isolation
        PrewarmRegistry._instance = None

    def tearDown(self):
        # Clean up singleton
        PrewarmRegistry._instance = None

    def _make_result(self, provider_name="cama", fingerprint="abc123", conn=None):
        if conn is None:
            conn = MagicMock()
        return PrewarmResult(
            provider_name=provider_name,
            connection=conn,
            fingerprint=fingerprint,
        )

    def test_store_and_claim(self):
        reg = PrewarmRegistry.get()
        result = self._make_result()
        reg.store(result)

        conn = reg.claim("cama")
        self.assertIsNotNone(conn)

    def test_claim_returns_none_when_empty(self):
        reg = PrewarmRegistry.get()
        conn = reg.claim("cama")
        self.assertIsNone(conn)

    def test_double_claim_returns_none(self):
        reg = PrewarmRegistry.get()
        result = self._make_result()
        reg.store(result)

        conn1 = reg.claim("cama")
        self.assertIsNotNone(conn1)

        conn2 = reg.claim("cama")
        self.assertIsNone(conn2)

    def test_claim_wrong_provider_returns_none(self):
        reg = PrewarmRegistry.get()
        result = self._make_result(provider_name="cama")
        reg.store(result)

        conn = reg.claim("other")
        self.assertIsNone(conn)

    def test_claim_with_matching_fingerprint(self):
        reg = PrewarmRegistry.get()
        result = self._make_result(fingerprint="fp123")
        reg.store(result)

        conn = reg.claim("cama", match_fn=lambda r: r.fingerprint == "fp123")
        self.assertIsNotNone(conn)

    def test_claim_with_mismatched_fingerprint(self):
        reg = PrewarmRegistry.get()
        mock_conn = MagicMock()
        result = self._make_result(fingerprint="fp123", conn=mock_conn)
        reg.store(result)

        conn = reg.claim("cama", match_fn=lambda r: r.fingerprint == "fp999")
        self.assertIsNone(conn)
        # Connection should have been closed
        mock_conn.close.assert_called_once()

    def test_claim_discards_error_result(self):
        reg = PrewarmRegistry.get()
        result = PrewarmResult(
            provider_name="cama",
            connection=None,
            fingerprint="abc",
            error=RuntimeError("connect failed"),
        )
        reg.store(result)

        conn = reg.claim("cama")
        self.assertIsNone(conn)

    def test_claim_discards_expired_result(self):
        reg = PrewarmRegistry.get()
        mock_conn = MagicMock()
        result = self._make_result(conn=mock_conn)
        # Backdate creation time to force expiry
        result.created_at = time.monotonic() - _PREWARM_TTL_S - 10
        reg.store(result)

        conn = reg.claim("cama")
        self.assertIsNone(conn)
        mock_conn.close.assert_called_once()

    def test_store_closes_previous_unclaimed(self):
        reg = PrewarmRegistry.get()
        mock_conn1 = MagicMock()
        result1 = self._make_result(conn=mock_conn1)
        reg.store(result1)

        mock_conn2 = MagicMock()
        result2 = self._make_result(conn=mock_conn2)
        reg.store(result2)

        # First connection should have been closed
        mock_conn1.close.assert_called_once()

        # Second connection should be claimable
        result = reg.claim("cama")
        self.assertIsNotNone(result)
        self.assertIs(result.connection, mock_conn2)

    def test_singleton(self):
        reg1 = PrewarmRegistry.get()
        reg2 = PrewarmRegistry.get()
        self.assertIs(reg1, reg2)

    def test_concurrent_store_claim(self):
        """Multiple threads storing and claiming should not corrupt state."""
        reg = PrewarmRegistry.get()
        results = []
        errors = []

        def store_and_claim(i):
            try:
                conn = MagicMock()
                conn.id = i
                result = self._make_result(
                    provider_name=f"provider_{i}",
                    conn=conn,
                )
                reg.store(result)
                claimed = reg.claim(f"provider_{i}")
                results.append(claimed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_and_claim, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        # All should have been claimed successfully
        for r in results:
            self.assertIsNotNone(r)


class TestPrewarmResult(unittest.TestCase):
    """Test PrewarmResult dataclass."""

    def test_age_increases(self):
        result = PrewarmResult(
            provider_name="cama",
            connection=MagicMock(),
            fingerprint="abc",
        )
        self.assertGreaterEqual(result.age_s, 0)

    def test_close_sets_connection_none(self):
        mock_conn = MagicMock()
        result = PrewarmResult(
            provider_name="cama",
            connection=mock_conn,
            fingerprint="abc",
        )
        result.close()
        self.assertIsNone(result.connection)
        mock_conn.close.assert_called_once()

    def test_close_handles_exception(self):
        mock_conn = MagicMock()
        mock_conn.close.side_effect = RuntimeError("close failed")
        result = PrewarmResult(
            provider_name="cama",
            connection=mock_conn,
            fingerprint="abc",
        )
        # Should not raise
        result.close()
        self.assertIsNone(result.connection)


class TestPublicAPI(unittest.TestCase):
    """Test start_cama_prewarm and claim_prewarmed_connection."""

    def setUp(self):
        PrewarmRegistry._instance = None
        pw._prewarm_ready = None
        pw._prewarm_start_time = None

    def tearDown(self):
        PrewarmRegistry._instance = None
        pw._prewarm_ready = None
        pw._prewarm_start_time = None

    @patch("cama_module.prewarm.CamaPrewarmProvider.create_connection")
    def test_start_and_claim(self, mock_create):
        mock_conn = MagicMock()
        fp = _cama_config_fingerprint("127.0.0.1", 18001, 4, 0)
        mock_create.return_value = PrewarmResult(
            provider_name="cama",
            connection=mock_conn,
            fingerprint=fp,
        )

        start_cama_prewarm("127.0.0.1", 18001, "")

        # claim_prewarmed_connection now waits for the prewarm event
        # so we don't need to sleep — the wait_timeout handles it.
        result = claim_prewarmed_connection("127.0.0.1", 18001, 4, 0, wait_timeout=2.0)
        self.assertIsNotNone(result)
        self.assertIs(result.connection, mock_conn)

    def test_claim_returns_none_when_not_started(self):
        result = claim_prewarmed_connection("127.0.0.1", 18001, 4, 0, wait_timeout=0.1)
        self.assertIsNone(result)

    @patch("cama_module.prewarm.CamaPrewarmProvider.create_connection")
    def test_claim_returns_none_on_config_mismatch(self, mock_create):
        mock_conn = MagicMock()
        fp = _cama_config_fingerprint("127.0.0.1", 18001, 4, 0)
        mock_create.return_value = PrewarmResult(
            provider_name="cama",
            connection=mock_conn,
            fingerprint=fp,
        )

        start_cama_prewarm("127.0.0.1", 18001, "")

        # Claim with different pool_size → fingerprint mismatch
        result = claim_prewarmed_connection("127.0.0.1", 18001, 8, 0, wait_timeout=2.0)
        self.assertIsNone(result)
        mock_conn.close.assert_called_once()


class TestPrewarmWait(unittest.TestCase):
    """Test the wait-for-prewarm synchronization in claim_prewarmed_connection."""

    def setUp(self):
        PrewarmRegistry._instance = None
        pw._prewarm_ready = None
        pw._prewarm_start_time = None

    def tearDown(self):
        PrewarmRegistry._instance = None
        pw._prewarm_ready = None
        pw._prewarm_start_time = None

    @patch("cama_module.prewarm.CamaPrewarmProvider.create_connection")
    def test_claim_waits_for_slow_prewarm(self, mock_create):
        """claim should wait for an in-progress prewarm instead of returning None."""
        mock_conn = MagicMock()
        fp = _cama_config_fingerprint("127.0.0.1", 18001, 4, 0)

        # Simulate a slow prewarm (200ms)
        def slow_create():
            time.sleep(0.2)
            return PrewarmResult(
                provider_name="cama",
                connection=mock_conn,
                fingerprint=fp,
            )

        mock_create.side_effect = slow_create

        start_cama_prewarm("127.0.0.1", 18001, "")

        # Claim immediately — should wait for the prewarm to finish
        result = claim_prewarmed_connection("127.0.0.1", 18001, 4, 0, wait_timeout=5.0)
        self.assertIsNotNone(result)
        self.assertIs(result.connection, mock_conn)

    @patch("cama_module.prewarm.CamaPrewarmProvider.create_connection")
    def test_claim_times_out_on_very_slow_prewarm(self, mock_create):
        """claim should give up after wait_timeout if prewarm is too slow."""
        def very_slow_create():
            time.sleep(10.0)  # much longer than timeout
            return PrewarmResult(
                provider_name="cama",
                connection=MagicMock(),
                fingerprint="abc",
            )

        mock_create.side_effect = very_slow_create

        start_cama_prewarm("127.0.0.1", 18001, "")

        # Short timeout — should give up
        result = claim_prewarmed_connection("127.0.0.1", 18001, 4, 0, wait_timeout=0.1)
        self.assertIsNone(result)


class TestHardening(unittest.TestCase):
    """Tests for hardening fixes (H1, H2, H4)."""

    def setUp(self):
        PrewarmRegistry._instance = None
        pw._prewarm_ready = None
        pw._prewarm_start_time = None

    def tearDown(self):
        PrewarmRegistry._instance = None
        pw._prewarm_ready = None
        pw._prewarm_start_time = None

    @patch("cama_module.prewarm.CamaPrewarmProvider.create_connection")
    def test_claim_mismatch_logs_warning(self, mock_create):
        """H4: Config mismatch should log at WARNING level."""
        mock_conn = MagicMock()
        fp = _cama_config_fingerprint("127.0.0.1", 18001, 4, 0)
        mock_create.return_value = PrewarmResult(
            provider_name="cama",
            connection=mock_conn,
            fingerprint=fp,
        )

        start_cama_prewarm("127.0.0.1", 18001, "")
        # Wait for daemon to finish before patching logger
        time.sleep(0.2)

        with patch("cama_module.prewarm.logger") as mock_logger:
            # Claim with different pool_size → fingerprint mismatch
            result = claim_prewarmed_connection("127.0.0.1", 18001, 8, 0, wait_timeout=0.1)
            self.assertIsNone(result)
            # Should have logged at WARNING, not INFO
            mock_logger.warning.assert_called()
            fmt_string = mock_logger.warning.call_args[0][0]
            self.assertIn("config mismatch", fmt_string)

    def test_close_logs_exception_at_debug(self):
        """H2: close() should log exceptions at DEBUG level."""
        mock_conn = MagicMock()
        mock_conn.close.side_effect = RuntimeError("close failed")
        result = PrewarmResult(
            provider_name="cama",
            connection=mock_conn,
            fingerprint="abc",
        )

        with patch("cama_module.prewarm.logger") as mock_logger:
            result.close()
            self.assertIsNone(result.connection)
            mock_logger.debug.assert_called()
            args = mock_logger.debug.call_args
            self.assertTrue(args[1].get("exc_info", False))

    def test_reaper_survives_close_exception(self):
        """H1: Reaper thread should survive exceptions from close()."""
        reg = PrewarmRegistry.get()

        # Store an expired entry with a broken close()
        broken_conn = MagicMock()
        broken_conn.close.side_effect = RuntimeError("close exploded")
        expired_result = PrewarmResult(
            provider_name="broken",
            connection=broken_conn,
            fingerprint="abc",
        )
        expired_result.created_at = time.monotonic() - _PREWARM_TTL_S - 10
        reg.store(expired_result)

        # Store a fresh entry that should survive the reaper cycle
        fresh_conn = MagicMock()
        fresh_result = PrewarmResult(
            provider_name="fresh",
            connection=fresh_conn,
            fingerprint="def",
        )
        reg.store(fresh_result)

        # Trigger reaper by lowering interval and waiting
        import cama_module.prewarm as pw
        original_interval = pw._REAPER_INTERVAL_S
        pw._REAPER_INTERVAL_S = 0.1
        try:
            # The reaper was already started by store(); wait for it to run
            time.sleep(0.5)

            # Fresh entry should still be claimable (reaper didn't die)
            result = reg.claim("fresh")
            self.assertIsNotNone(result)
            self.assertIs(result.connection, fresh_conn)
        finally:
            pw._REAPER_INTERVAL_S = original_interval


class TestPerRankPrewarm(unittest.TestCase):
    """Tests for per-rank prewarm via env-var signaling."""

    ENV_KEY = "_SGLANG_CAMA_PREWARM_SIGNAL"

    def setUp(self):
        PrewarmRegistry._instance = None
        pw._prewarm_ready = None
        pw._prewarm_start_time = None
        os.environ.pop(self.ENV_KEY, None)

    def tearDown(self):
        PrewarmRegistry._instance = None
        pw._prewarm_ready = None
        pw._prewarm_start_time = None
        os.environ.pop(self.ENV_KEY, None)

    def test_env_signal_roundtrip(self):
        """Set env var, call _maybe_start_rank_prewarm, verify correct args."""
        import json
        signal = json.dumps({
            "addr": "10.0.0.5",
            "port": 18001,
            "password": "secret",
            "pool_size": 8,
            "send_buf_size": 16777216,
            "nic_striping": False,
        })
        os.environ[self.ENV_KEY] = signal

        with patch("cama_module.prewarm.start_cama_prewarm") as mock_start:
            # Import and call the function
            from patches.environ import _maybe_start_rank_prewarm
            _maybe_start_rank_prewarm()

            mock_start.assert_called_once_with(
                "10.0.0.5", 18001, "secret",
                pool_size=8,
                send_buf_size=16777216,
                nic_striping=False,
                model_page_bytes=0,
            )

    def test_env_signal_absent_is_noop(self):
        """No env var → no prewarm started."""
        with patch("cama_module.prewarm.start_cama_prewarm") as mock_start:
            from patches.environ import _maybe_start_rank_prewarm
            _maybe_start_rank_prewarm()
            mock_start.assert_not_called()

    def test_env_signal_invalid_json_is_noop(self):
        """Bad JSON → no crash, no prewarm."""
        os.environ[self.ENV_KEY] = "not-valid-json{{"

        with patch("cama_module.prewarm.start_cama_prewarm") as mock_start:
            from patches.environ import _maybe_start_rank_prewarm
            _maybe_start_rank_prewarm()  # Should not raise
            mock_start.assert_not_called()

    def test_env_signal_import_error_is_noop(self):
        """Import failure → no crash."""
        import json
        os.environ[self.ENV_KEY] = json.dumps({
            "addr": "10.0.0.1", "port": 18001,
        })

        with patch.dict("sys.modules", {"cama_module.prewarm": None, "sglang.srt.mem_cache.storage.cama.prewarm": None}):
            from patches.environ import _maybe_start_rank_prewarm
            _maybe_start_rank_prewarm()  # Should not raise

    def test_env_signal_consumed(self):
        """Env var is removed after consumption."""
        import json
        os.environ[self.ENV_KEY] = json.dumps({
            "addr": "10.0.0.1", "port": 18001,
        })

        with patch("cama_module.prewarm.start_cama_prewarm"):
            from patches.environ import _maybe_start_rank_prewarm
            _maybe_start_rank_prewarm()

        self.assertNotIn(self.ENV_KEY, os.environ)


if __name__ == "__main__":
    unittest.main()
